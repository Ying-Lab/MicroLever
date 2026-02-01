# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_geometric.nn import GCNConv, global_mean_pool

# -------------------------------
# 位置编码器
# -------------------------------
class PositionalEncoder(nn.Module):
    def __init__(self, dropout=0.1, max_seq_len=5000, d_model=512, batch_first=True):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        position = torch.arange(max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        if self.batch_first:
            seq_len = x.size(1)
            pe = self.pe[:seq_len, :].unsqueeze(0)
        else:
            seq_len = x.size(0)
            pe = self.pe[:seq_len, :].unsqueeze(1).unsqueeze(2)
        x = x + pe
        return self.dropout(x)

# -------------------------------
# 动力学编码器
# -------------------------------
class DynamicsEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_layers, num_heads, seq_len):
        super(DynamicsEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = PositionalEncoder(
            d_model=embed_dim, dropout=0.1, 
            max_seq_len=seq_len, batch_first=True
        )
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim*4, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        batch_size, num_nodes, seq_length, input_dim = x.shape
        x = x.view(batch_size * num_nodes, seq_length, input_dim)
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        # x = x.mean(dim=1)
        x = x[:, -1, :]                      # 只取最后一个时间步 -> (B·N, D)
        x = x.view(batch_size, num_nodes, -1)
        return self.layer_norm(x)

# -------------------------------
# 特征调制器模块（全局FiLM + 交叉注意力）
# -------------------------------
class HierarchicalModulator(nn.Module):
    """
    Δ 改动一览
    1. cross_attn 的 need_weights=True
    2. forward 中将 gamma / beta 和 attn 权重缓存到实例属性
    3. 新增参数 return_attn: 若 True 则一并返回注意力权重
    """
    def __init__(self, receptor_dim, donor_dim, num_heads=4):
        super().__init__()
        # 全局 FiLM 调制
        self.global_film = nn.Sequential(
            nn.Linear(donor_dim, 2 * receptor_dim),
            nn.LayerNorm(2 * receptor_dim)
        )

        # 节点级交叉注意力
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=receptor_dim,
            num_heads=num_heads,
            kdim=donor_dim,
            vdim=donor_dim,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(receptor_dim)

        # 特征融合
        self.fusion = nn.Linear(receptor_dim * 2, receptor_dim)

        # ====== 用于解释的缓存 ======
        self.last_gamma: torch.Tensor | None = None   # (B, N, D_r)
        self.last_beta:  torch.Tensor | None = None   # (B, N, D_r)
        self.attn_weights: torch.Tensor | None = None # (B, N, M)

    def forward(self, receptor, donor, *, return_attn: bool = False):
        """
        receptor : (B, N, D_r)
        donor    : (B, M, D_d)   —— 已经过 donor_proj
        """
        B, N, D_r = receptor.shape

        # === 全局 FiLM 调制 ===
        film_params = self.global_film(donor)         # (B, M, 2*D_r)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        # 缓存 γ/β 供外部解释
        self.last_gamma = gamma.detach()
        self.last_beta  = beta.detach()

        # 将 γ/β 广播到受体维度（在此设计中 donor 和 receptor 结点数一致，直接相乘）
        modulated_global = gamma * receptor + beta

        # === 交叉注意力 ===
        attn_output, attn_w = self.cross_attn(         # need_weights=True (Δ)
            query=receptor,
            key=donor,
            value=donor,
            need_weights=True
        )
        # 缓存注意力权重
        self.attn_weights = attn_w.detach()            # (B, N, M)
        modulated_attn = self.attn_norm(attn_output + receptor)

        # === 融合 ===
        combined = torch.cat([modulated_global, modulated_attn], dim=-1)
        fused = self.fusion(combined)                  # (B, N, D_r)

        return (fused, attn_w) if return_attn else fused

# -------------------------------
# 拓扑编码器
# -------------------------------
class TopologyEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(TopologyEncoder, self).__init__()
        self.convs_pos = nn.ModuleList()
        self.convs_neg = nn.ModuleList()

        self.convs_pos.append(GCNConv(input_dim, hidden_dim))
        self.convs_neg.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs_pos.append(GCNConv(hidden_dim, hidden_dim))
            self.convs_neg.append(GCNConv(hidden_dim, hidden_dim))
        
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_weight, batch):
        pos_mask = edge_weight > 0
        neg_mask = edge_weight < 0

        pos_edge_index = edge_index[:, pos_mask]
        pos_edge_weight = edge_weight[pos_mask]
        neg_edge_index = edge_index[:, neg_mask]
        neg_edge_weight = torch.abs(edge_weight[neg_mask])

        for conv_p, conv_n in zip(self.convs_pos, self.convs_neg):
            pos_x = conv_p(x, pos_edge_index, pos_edge_weight) if pos_edge_index.size(1) > 0 else 0
            neg_x = conv_n(x, neg_edge_index, neg_edge_weight) if neg_edge_index.size(1) > 0 else 0
            x = torch.relu(pos_x) - torch.relu(neg_x)

        x = global_mean_pool(x, batch)
        return self.layer_norm(x)

# -------------------------------
# 拓扑编码器版本1：仅考虑正负号，忽略权重
# -------------------------------
class TopologyEncoderSignOnly(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.convs_pos = nn.ModuleList()
        self.convs_neg = nn.ModuleList()

        # 初始化双通道卷积层
        for _ in range(num_layers):
            self.convs_pos.append(GCNConv(input_dim if _ ==0 else hidden_dim, hidden_dim))
            self.convs_neg.append(GCNConv(input_dim if _ ==0 else hidden_dim, hidden_dim))
        
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_weight, batch):
        # 符号分离处理（权重强制设为1）
        pos_mask = edge_weight > 0
        neg_mask = edge_weight < 0
        
        # 创建符号掩码
        pos_edge_index = edge_index[:, pos_mask]
        neg_edge_index = edge_index[:, neg_mask]
        
        # 强制设置权重为1
        pos_edge_weight = torch.ones(pos_mask.sum(), device=x.device)
        neg_edge_weight = torch.ones(neg_mask.sum(), device=x.device)

        # 双通道符号传播
        for conv_p, conv_n in zip(self.convs_pos, self.convs_neg):
            pos_x = conv_p(x, pos_edge_index, pos_edge_weight) if pos_edge_index.size(1) > 0 else 0
            neg_x = conv_n(x, neg_edge_index, neg_edge_weight) if neg_edge_index.size(1) > 0 else 0
            x = torch.relu(pos_x) - torch.relu(neg_x)  # 符号特征融合

        x = global_mean_pool(x, batch)
        return self.layer_norm(x)

# -------------------------------
# 拓扑编码器版本2：忽略正负号和权重
# -------------------------------
class TopologyEncoderNoSign(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        
        # 初始化单通道卷积层
        for _ in range(num_layers):
            self.convs.append(GCNConv(input_dim if _ ==0 else hidden_dim, hidden_dim))
        
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_weight, batch):
        # 统一处理所有边（强制设置权重为1）
        edge_weight = torch.ones(edge_index.size(1), device=x.device)
        
        # 单通道传播
        for conv in self.convs:
            x = torch.relu(conv(x, edge_index, edge_weight))
        
        x = global_mean_pool(x, batch)
        return self.layer_norm(x)


# -------------------------------
# K-space投影器
# -------------------------------
class KSpaceProjector(nn.Module):
    def __init__(self, hidden_dim, embed_dim):
        super().__init__()
        self.mlp_avg = nn.Linear(hidden_dim, hidden_dim)
        self.mlp_max = nn.Linear(hidden_dim, hidden_dim)
        self.attention = nn.Sigmoid()

    def forward(self, Z_star):
        Z_avg = Z_star.mean(dim=1)
        Z_max, _ = Z_star.max(dim=1)
        Att = self.attention(self.mlp_avg(Z_avg) + self.mlp_max(Z_max))
        return (Att.unsqueeze(1) * Z_star).sum(dim=1)

# -------------------------------
# 韧性预测器
# -------------------------------
class ResiliencePredictor(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fc_k = nn.Linear(embed_dim, 1)
        self.fc_y = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, eG):
        k = self.fc_k(eG)
        return k, self.sigmoid(self.fc_y(k))

# -------------------------------
# 整体模型
# -------------------------------
class ResInfModel(nn.Module):
    def __init__(self, dynamics_input_dim, dynamics_embed_dim, dynamics_num_layers, dynamics_num_heads,
                 topology_input_dim, topology_hidden_dim, topology_num_layers, embed_dim,
                 donor_dim=1, desired_seq_len=5, topology_type="sign_only"):  # 新增参数
        super().__init__()
        # 显式定义所有属性
        self.desired_seq_len = desired_seq_len  # 关键修复点
        self.donor_dim = donor_dim
        
        # 动力学编码器
        self.dynamics_encoder = DynamicsEncoder(
            input_dim=dynamics_input_dim,
            embed_dim=dynamics_embed_dim,
            num_layers=dynamics_num_layers,
            num_heads=dynamics_num_heads,
            seq_len=desired_seq_len  # 参数传递修正
        )
        
        # 供体特征嵌入层
        self.donor_proj = nn.Linear(donor_dim, dynamics_embed_dim)
        
        # 特征调制器
        self.modulator = HierarchicalModulator(
            receptor_dim=dynamics_embed_dim,
            donor_dim=dynamics_embed_dim
        )
        
        # 拓扑编码器
        if topology_type == "original":
            self.topology_encoder = TopologyEncoder(
                input_dim=topology_input_dim,
                hidden_dim=topology_hidden_dim,
                num_layers=topology_num_layers
            )
        elif topology_type == "sign_only":
            self.topology_encoder = TopologyEncoderSignOnly(
                input_dim=topology_input_dim,
                hidden_dim=topology_hidden_dim,
                num_layers=topology_num_layers
            )
        elif topology_type == "no_sign":
            self.topology_encoder = TopologyEncoderNoSign(
                input_dim=topology_input_dim,
                hidden_dim=topology_hidden_dim,
                num_layers=topology_num_layers
            )
        else:
            raise ValueError(f"Invalid topology type: {topology_type}")

        # 其他模块保持不变
        self.k_space_projector = KSpaceProjector(topology_hidden_dim, embed_dim)
        self.resilience_predictor = ResiliencePredictor(embed_dim)

    def forward(self, node_activity, donor_data, edge_index, edge_weight, batch):
        # 维度处理（保持原有逻辑）
        batch_size = batch.max().item() + 1
        total_nodes = node_activity.size(0)
        num_nodes_per_graph = total_nodes // batch_size
        
        # 显式声明所需序列长度
        desired_seq_len = self.desired_seq_len  # 使用类属性
        
        # 维度调整
        node_activity = node_activity.view(batch_size, num_nodes_per_graph, -1, 1)
        node_activity = node_activity[:, :, :desired_seq_len, :]  # 使用局部变量

        
        receptor_features = self.dynamics_encoder(node_activity)
        
        # 供体特征处理
        donor = donor_data.view(batch_size, num_nodes_per_graph, -1)
        donor = self.donor_proj(donor)
        
        # 特征调制
        modulated = self.modulator(receptor_features, donor)
        
        # 拓扑编码
        topo_input = modulated.view(batch_size*num_nodes_per_graph, -1)
        virtual_node = self.topology_encoder(topo_input, edge_index, edge_weight, batch)
        
        # K-space投影
        Z_star = torch.cat([
            modulated, 
            virtual_node.unsqueeze(1).expand(-1, num_nodes_per_graph, -1)
        ], dim=1)
        
        # 预测
        k, y = self.resilience_predictor(self.k_space_projector(Z_star))
        return k, y