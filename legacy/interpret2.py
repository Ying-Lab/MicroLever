#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
interpret.py  ──  ResInfModel 可解释性分析（纯 Matplotlib 版，含 CSV 导出 & 多种分布可视化）
调用示例：
    # 自动读取 config.json
    python interpret.py --best_model experiments/.../best_model.pth
    # 或手动指定
    python interpret.py --best_model best.pth --config experiments/.../config.json
"""
from __future__ import annotations
import argparse, json, pathlib, sys, csv
from typing import List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from scipy.stats import gaussian_kde

# ---------- ① 关闭不兼容的高性能 Attention 内核 ----------
# 这 3 行必须在任何 Transformer 调用之前执行！
torch.backends.cuda.enable_flash_sdp(False)          # 关 Flash‑Attention
torch.backends.cuda.enable_mem_efficient_sdp(False)  # 关 Memory‑Efficient
torch.backends.cuda.enable_math_sdp(True)            # 强制回退到安全 math 实现
# --------------------------------------------------------

from data_loader_DM import create_loaders
from model import ResInfModel

# ------------------------- CLI -------------------------
parser = argparse.ArgumentParser("ResInfModel interpretation")
parser.add_argument("--best_model", required=True, help="验证集最佳权重 .pth")
parser.add_argument("--config",     default=None,   help="config.json（可省略）")
args = parser.parse_args()

best_model_path = pathlib.Path(args.best_model).resolve()
if not best_model_path.is_file():
    sys.exit(f"[ERROR] 权重文件不存在: {best_model_path}")

if args.config is None:
    cfg_guess = best_model_path.parents[1] / "config.json"
    if not cfg_guess.is_file():
        sys.exit(f"[ERROR] 未指定 --config，且未找到 {cfg_guess}")
    config_path = cfg_guess
else:
    config_path = pathlib.Path(args.config).resolve()
    if not config_path.is_file():
        sys.exit(f"[ERROR] config.json 不存在: {config_path}")

print(f"[INFO] 读取配置: {config_path}")
with open(config_path, "r", encoding="utf-8") as f:
    CFG = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] 使用设备: {device}")

# ------------------------- 加载测试集 -------------------------
print("[STEP] 加载测试集 …")
_, _, test_loader = create_loaders(
    h5_path               = CFG["data"]["h5_path"],
    adjacency_path        = CFG["data"]["adjacency_path"],
    cdiff_label_path      = CFG["data"]["cdiff_label_path"],
    donor_final_path      = CFG["data"]["donor_final_path"],
    receptor_ts_group     = CFG["data"]["receptor_ts_group"],
    sampling_mode         = CFG["data"]["sampling"]["mode"],
    num_samples           = CFG["data"]["sampling"]["num_samples"],
    test_size             = CFG["training"]["split"]["test"],
    val_size              = CFG["training"]["split"]["val"],
    noise_strength_train  = CFG["training"]["noise"]["train"],
    noise_strength_val    = CFG["training"]["noise"]["val"],
    noise_strength_test   = CFG["training"]["noise"]["test"],
    sparsity_config       = CFG["training"]["sparsity"],
    batch_size            = CFG["training"]["batch_size"],
    seed                  = CFG["training"]["seed"],
    new_h5_path           = CFG["data"]["new_h5_path"],
    new_cdiff_label_path  = CFG["data"]["new_cdiff_label_path"],
    new_donor_final_path  = CFG["data"]["new_donor_final_path"],
    new_receptor_ts_group = CFG["data"]["new_receptor_ts_group"]
)

# ------------------------- 构建并加载模型 -------------------------
print("[STEP] 构建并加载模型 …")
m_cfg = CFG["model"]
model = ResInfModel(
    dynamics_input_dim  = m_cfg["dynamics"]["input_dim"],
    dynamics_embed_dim  = m_cfg["dynamics"]["embed_dim"],
    dynamics_num_layers = m_cfg["dynamics"]["layers"],
    dynamics_num_heads  = m_cfg["dynamics"]["heads"],
    topology_input_dim  = m_cfg["topology"]["input_dim"],
    topology_hidden_dim = m_cfg["topology"]["hidden_dim"],
    topology_num_layers = m_cfg["topology"]["layers"],
    embed_dim           = m_cfg["embed_dim"],
    donor_dim           = m_cfg["donor_dim"],
    desired_seq_len     = CFG["data"]["sampling"]["num_samples"],
    topology_type       = m_cfg["topology_type"]
).to(device)

# ---------- ② 安全地加载权重，消除 FutureWarning ----------
_state = torch.load(best_model_path, map_location=device, weights_only=True)
sd = _state.get("model_state_dict", _state)
model.load_state_dict(sd)
# --------------------------------------------------------

model.eval()
print(f"[INFO] 权重已加载: {best_model_path}")

# ------------------------- 汇总样本节点张量 -------------------------
print("[STEP] 收集所有样本节点 …")
dataset = test_loader.dataset
num_samples = len(dataset)
sample0 = dataset[0]
num_species = sample0.donor_data.shape[0]
seq_len     = sample0.node_activity.shape[1]

node_act_ls = [d.node_activity for d in dataset]
donor_ls    = [d.donor_data    for d in dataset]
node_act  = torch.cat(node_act_ls, 0).to(device)
donor_dat = torch.cat(donor_ls,    0).to(device)
batches   = torch.arange(num_samples, device=device).repeat_interleave(num_species)

tpl_idx, tpl_wt = sample0.edge_index, sample0.edge_attr
edge_idx_parts, edge_wt_parts = [], []
for g in range(num_samples):
    off = g * num_species
    edge_idx_parts.append(tpl_idx + off)
    edge_wt_parts.append(tpl_wt)
edge_index  = torch.cat(edge_idx_parts, 1).to(device)
edge_weight = torch.cat(edge_wt_parts).to(device)

species_names = [f"Sp{str(i+1).zfill(2)}" for i in range(num_species)]
print(f"[INFO] 样本数={num_samples}·物种数={num_species}·seq_len={seq_len}")

out_dir = pathlib.Path(CFG["output"]["result_dir"])
out_dir.mkdir(parents=True, exist_ok=True)

# ------------------------- 1. FiLM γ/β & CSV -------------------------
print("\n[分析 1] FiLM γ/β")
with torch.no_grad():
    _ = model(node_act, donor_dat, edge_index, edge_weight, batches)

gamma = model.modulator.last_gamma.cpu()          # [B, N, L]
beta  = model.modulator.last_beta.cpu()           # [B, N, L]

# ❶ 直接取均值，保留正负号
mean_g = gamma.mean(dim=(0, -1)).numpy()          # shape [N]
mean_b = beta.mean(dim=(0, -1)).numpy()           # shape [N]

# ❷ 按 |γ| 从大到小排序（只是为了视觉上把最重要的排前面）
order = np.argsort(-np.abs(mean_g))

# ----- 条形图 -----
plt.figure(figsize=(10, 4))
bar_w = 0.4
x = np.arange(num_species)

plt.bar(x - bar_w / 2, mean_g[order], width=bar_w, label="γ")
plt.bar(x + bar_w / 2, mean_b[order], width=bar_w, label="β", alpha=0.7)

plt.axhline(0, color="k", linewidth=.8)           # 零线，分隔正负
plt.xticks(x, [species_names[i] for i in order], rotation=90, fontsize=6)
plt.ylabel("mean value")
plt.title("FiLM γ/β importance (signed)")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "film_params_importance_signed.png")
plt.close()
print(f"[保存] film_params_importance_signed.png → {out_dir}")

# ----- CSV 导出 -----
with open(out_dir / "film_importance_signed.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["species", "gamma_mean", "beta_mean"])
    for idx in order:
        w.writerow([species_names[idx], mean_g[idx], mean_b[idx]])
print(f"[导出] film_importance_signed.csv → {out_dir}")
# # ------------------------- 1. FiLM γ/β & CSV -------------------------
# print("\n[分析 1] FiLM γ/β")
# with torch.no_grad():
#     _ = model(node_act, donor_dat, edge_index, edge_weight, batches)
# gamma = model.modulator.last_gamma.cpu()
# beta  = model.modulator.last_beta.cpu()
# abs_g = gamma.abs().mean(dim=(0,-1)).numpy()
# abs_b = beta.abs().mean(dim=(0,-1)).numpy()
# order = abs_g.argsort()[::-1]

# # 条形图
# plt.figure(figsize=(10,4))
# plt.bar(range(num_species), abs_g[order], label="|γ|")
# plt.bar(range(num_species), abs_b[order], alpha=.5, label="|β|")
# plt.xticks(range(num_species), [species_names[i] for i in order], rotation=90, fontsize=6)
# plt.legend(); plt.ylabel("mean |value|"); plt.title("FiLM γ/β importance")
# plt.tight_layout(); plt.savefig(out_dir/"film_params_importance.png"); plt.close()
# print(f"[保存] film_params_importance.png → {out_dir}")

# # CSV 导出
# with open(out_dir/"film_importance.csv","w",newline="") as f:
#     w=csv.writer(f); w.writerow(["species","gamma_imp","beta_imp"])
#     for idx in order: w.writerow([species_names[idx], abs_g[idx], abs_b[idx]])
# print(f"[导出] film_importance.csv → {out_dir}")

# gamma_dist = [gamma.numpy()[:,i,:].flatten() for i in range(num_species)]
# beta_dist  = [beta.numpy()[:,i,:].flatten() for i in range(num_species)]
# with open(out_dir/"film_gamma_distribution.csv","w",newline="") as f:
#     w=csv.writer(f); w.writerow(["species","gamma_value"])
#     for i,vals in enumerate(gamma_dist):
#         for v in vals: w.writerow([species_names[i], float(v)])
# print(f"[导出] film_gamma_distribution.csv → {out_dir}")
# with open(out_dir/"film_beta_distribution.csv","w",newline="") as f:
#     w=csv.writer(f); w.writerow(["species","beta_value"])
#     for i,vals in enumerate(beta_dist):
#         for v in vals: w.writerow([species_names[i], float(v)])
# print(f"[导出] film_beta_distribution.csv → {out_dir}")

# ------------------------- 2. 交叉注意力 & CSV -------------------------
print("\n[分析 2] 交叉注意力")
attn = model.modulator.attn_weights.mean(dim=0).cpu().numpy()
plt.figure(figsize=(7,6))
plt.imshow(attn, cmap="viridis"); plt.colorbar(label="avg weight")
plt.xticks(range(num_species), species_names, rotation=90, fontsize=5)
plt.yticks(range(num_species), species_names, fontsize=5)
plt.title("Average Donor→Recipient Attention")
plt.tight_layout(); plt.savefig(out_dir/"cross_attention_heatmap.png"); plt.close()
print(f"[保存] cross_attention_heatmap.png → {out_dir}")

with open(out_dir/"attention_matrix_full.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["recipient","donor","avg_attention"])
    for i in range(num_species):
        for j in range(num_species):
            w.writerow([species_names[i], species_names[j], float(attn[i,j])])
print(f"[导出] attention_matrix_full.csv → {out_dir}")

flat = attn.flatten()
top_pairs = flat.argsort()[-10:][::-1]
with open(out_dir/"attention_top_pairs.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["rank","donor","recipient","weight"])
    for r,idx in enumerate(top_pairs,1):
        i,j = divmod(idx, num_species)
        w.writerow([r, species_names[j], species_names[i], float(attn[i,j])])
print(f"[导出] attention_top_pairs.csv → {out_dir}")

# ---------- 2‑b 交叉注意力：按 label 拆分 ----------
print("\n[分析 2‑b] 交叉注意力（按 label 拆分）")

attn_sum_valid   = np.zeros((num_species, num_species), dtype=np.float64)
attn_sum_invalid = np.zeros_like(attn_sum_valid)
cnt_valid = cnt_invalid = 0

with torch.no_grad():
    for d in tqdm(dataset, desc="aggregate by label"):
        na = d.node_activity.to(device).unsqueeze(0)   # [1, N, T, 1]
        dd = d.donor_data.to(device).unsqueeze(0)      # [1, N, 1]
        rec = model.dynamics_encoder(na)
        don = model.donor_proj(dd)
        _, attn_w = model.modulator(rec, don, return_attn=True)  # [1,N,M]
        mat = attn_w.squeeze(0).cpu().numpy()

        if int(d.y.item()) == 1:       # 有效
            attn_sum_valid += mat
            cnt_valid      += 1
        else:                          # 无效
            attn_sum_invalid += mat
            cnt_invalid      += 1

attn_valid   = attn_sum_valid   / max(cnt_valid,   1)
attn_invalid = attn_sum_invalid / max(cnt_invalid, 1)

def draw_heat(mat, fname, title):
    plt.figure(figsize=(7,6))
    plt.imshow(mat, cmap="viridis")
    plt.colorbar(label="avg weight")
    plt.xticks(range(num_species), species_names, rotation=90, fontsize=5)
    plt.yticks(range(num_species), species_names, fontsize=5)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_dir/fname); plt.close()
    print(f"[保存] {fname} → {out_dir}")

draw_heat(attn_valid,   "cross_attention_valid.png",   "Valid‑label Donor→Recipient Attention")
draw_heat(attn_invalid, "cross_attention_invalid.png", "Invalid‑label Donor→Recipient Attention")



# ------------------------- 3. GCN 边/节点 & CSV -------------------------
print("\n[分析 3] GCN 边/节点 perturbation")
adj = torch.tensor(np.loadtxt(CFG["data"]["adjacency_path"],delimiter=","), device=device)
with torch.no_grad():
    base_out = model(node_act, donor_dat, edge_index, edge_weight, batches)[1].view(-1).cpu()

edge_scores: List[Tuple[Tuple[int,int],float]]=[] 
rows,cols = torch.where(adj!=0)
for u,v in tqdm(list(zip(rows.tolist(),cols.tolist())),leave=False):
    if u>=v: continue
    mask = ~(((edge_index[0]%num_species==u)&(edge_index[1]%num_species==v))|
             ((edge_index[0]%num_species==v)&(edge_index[1]%num_species==u)))
    idx2,wt2 = edge_index[:,mask], edge_weight[mask]
    with torch.no_grad():
        out = model(node_act, donor_dat, idx2, wt2, batches)[1].view(-1).cpu()
    edge_scores.append(((u,v), float((out-base_out).abs().mean().item())))
edge_scores.sort(key=lambda x:x[1], reverse=True)
with open(out_dir/"edge_importance.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["rank","species_u","species_v","importance"])
    for r,((u,v),s) in enumerate(edge_scores,1):
        w.writerow([r, species_names[u], species_names[v], s])
print(f"[导出] edge_importance.csv → {out_dir}")

node_scores: List[Tuple[int,float]]=[]
for n in tqdm(range(num_species),leave=False):
    mask = ~((edge_index[0]%num_species==n)|(edge_index[1]%num_species==n))
    idx2,wt2 = edge_index[:,mask], edge_weight[mask]
    d0 = donor_dat.clone().view(num_samples,num_species,-1); d0[:,n]=0; d0=d0.view(-1,1)
    r0 = node_act.clone().view(num_samples,num_species,seq_len,1); r0[:,n]=0; r0=r0.view(-1,seq_len,1)
    with torch.no_grad():
        out = model(r0, d0, idx2, wt2, batches)[1].view(-1).cpu()
    node_scores.append((n, float((out-base_out).abs().mean().item())))
node_scores.sort(key=lambda x:x[1], reverse=True)
with open(out_dir/"node_importance.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["rank","species","importance"])
    for r,(n,s) in enumerate(node_scores,1):
        w.writerow([r, species_names[n], s])
print(f"[导出] node_importance.csv → {out_dir}")

# ------------------------- 4. FiLM 分布可视化 -------------------------
# print("\n[分析 4] FiLM 分布可视化")

# # 简易 Raincloud for γ
# plt.figure(figsize=(12,8))
# for i, vals in enumerate(gamma_dist):
#     kde = gaussian_kde(vals)
#     xs = np.linspace(min(vals), max(vals), 200)
#     plt.fill_betweenx(xs, i - kde(xs)*0.5, i + kde(xs)*0.5, alpha=0.3)
#     plt.plot([i-0.2, i+0.2], [np.median(vals)]*2, 'k-')
# plt.yticks(range(num_species), species_names, fontsize=6)
# plt.title("Raincloud-like Plot of FiLM γ")
# plt.xlabel("Density offset"); plt.tight_layout()
# plt.savefig(out_dir/"film_gamma_raincloud_simple.png"); plt.close()
# print(f"[保存] film_gamma_raincloud_simple.png → {out_dir}")

# # Beeswarm for γ
# plt.figure(figsize=(12,6))
# for i, vals in enumerate(gamma_dist):
#     x = np.random.normal(i, 0.08, size=len(vals))
#     plt.scatter(x, vals, s=4, alpha=0.5)
# plt.xticks(range(num_species), species_names, rotation=90, fontsize=6)
# plt.title("Beeswarm Plot of FiLM γ"); plt.ylabel("γ value")
# plt.tight_layout()
# plt.savefig(out_dir/"film_gamma_beeswarm.png"); plt.close()
# print(f"[保存] film_gamma_beeswarm.png → {out_dir}")

# # Ridge-like for γ
# plt.figure(figsize=(12,8))
# for i, vals in enumerate(gamma_dist):
#     kde = gaussian_kde(vals)
#     xs = np.linspace(min(vals), max(vals), 200)
#     plt.plot(kde(xs)+i, xs, linewidth=1)
# plt.yticks(range(num_species), species_names, fontsize=6)
# plt.title("Ridge Plot of FiLM γ"); plt.xlabel("Density + offset")
# plt.tight_layout()
# plt.savefig(out_dir/"film_gamma_ridge_simple.png"); plt.close()
# print(f"[保存] film_gamma_ridge_simple.png → {out_dir}")

# # Horizontal Box+Swarm for γ
# plt.figure(figsize=(8,12))
# plt.boxplot(gamma_dist, vert=False, widths=0.6, notch=True, showfliers=False)
# for i, vals in enumerate(gamma_dist, start=1):
#     y = np.random.normal(i, 0.1, size=len(vals))
#     plt.scatter(vals, y, s=3, alpha=0.3)
# plt.yticks(range(1,num_species+1), species_names, fontsize=6)
# plt.title("Horizontal Box+Swarm of FiLM γ")
# plt.tight_layout()
# plt.savefig(out_dir/"film_gamma_boxswarm.png"); plt.close()
# print(f"[保存] film_gamma_boxswarm.png → {out_dir}")

# # 同理 for β
# plt.figure(figsize=(12,8))
# for i, vals in enumerate(beta_dist):
#     kde = gaussian_kde(vals)
#     xs = np.linspace(min(vals), max(vals), 200)
#     plt.fill_betweenx(xs, i - kde(xs)*0.5, i + kde(xs)*0.5, alpha=0.3)
# plt.yticks(range(num_species), species_names, fontsize=6)
# plt.title("Raincloud-like Plot of FiLM β")
# plt.tight_layout()
# plt.savefig(out_dir/"film_beta_raincloud_simple.png"); plt.close()
# print(f"[保存] film_beta_raincloud_simple.png → {out_dir}")

# ------------------------- 5. Attention 降维可视化 -------------------------
print("\n[分析 5] Attention 降维")
attn_list, label_list = [], []
for d in tqdm(dataset, desc="Collect attention"):
    na = d.node_activity.to(device).unsqueeze(0)
    dd = d.donor_data.to(device).unsqueeze(0)
    rec = model.dynamics_encoder(na)
    emb = model.donor_proj(dd)
    _, at = model.modulator(rec, emb, return_attn=True)
    mat = at.squeeze(0).detach().cpu().numpy()
    attn_list.append(mat.flatten())
    label_list.append(int(d.y.item()))

X = np.vstack(attn_list); y = np.array(label_list)

print("  PCA...")
X_pca = PCA(n_components=2, random_state=42).fit_transform(X)
plt.figure(figsize=(6,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='viridis', alpha=0.7)
plt.title("PCA of cross-attention"); plt.tight_layout()
plt.savefig(out_dir/"attention_pca.png"); plt.close()
print(f"[保存] attention_pca.png → {out_dir}")

print("  UMAP...")
X_umap = umap.UMAP(n_components=2, random_state=42).fit_transform(X)
plt.figure(figsize=(6,6))
plt.scatter(X_umap[:,0], X_umap[:,1], c=y, cmap='viridis', alpha=0.7)
plt.title("UMAP of cross-attention"); plt.tight_layout()
plt.savefig(out_dir/"attention_umap.png"); plt.close()
print(f"[保存] attention_umap.png → {out_dir}")

print("  t-SNE...")
X_tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto').fit_transform(X)
plt.figure(figsize=(6,6))
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y, cmap='viridis', alpha=0.7)
plt.title("t-SNE of cross-attention"); plt.tight_layout()
plt.savefig(out_dir/"attention_tsne.png"); plt.close()
print(f"[保存] attention_tsne.png → {out_dir}")

print("\n✓ 全部可解释性分析完成，结果保存在:", out_dir)
