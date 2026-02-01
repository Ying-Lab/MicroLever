#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
interpret_attention_advanced.py

生成 ResInfModel 的多种 cross-attention 可视化：
  1) 原始 attention 平均热图（overall + 按标签分开）
  2) Z-score attention 热图（overall + 按标签分开）
  3) 加权 Z-score 热图（overall + 按标签分开，颜色=Z-score，透明度=原始 attention）
并使用 overall 的加权 Z-score 作为依据，选出 top-k pair 做局部注意力 mask 消融，
报告 AUROC 下降情况。

用法示例：
    python interpret_attention_advanced.py \
        --best_model path/to/best_epXXX_aucYYYY.pth \
        --config    path/to/config.json \
        --top_frac  0.5
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import types as _types

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# ------------------------------------------------------------------
# 补丁 1：伪造一个最小的 onnxscript 模块，避免 torch.onnx 导入时报错
# ------------------------------------------------------------------
try:
    import onnxscript  # type: ignore[import]
except ModuleNotFoundError:
    dummy_onnxscript = _types.ModuleType("onnxscript")
    dummy_ir = _types.ModuleType("onnxscript.ir")
    dummy_onnxscript.ir = dummy_ir  # type: ignore[attr-defined]
    sys.modules["onnxscript"] = dummy_onnxscript
    sys.modules["onnxscript.ir"] = dummy_ir

# ------------------------------------------------------------------
# 补丁 2：提前“占坑”一个假的 torch._dynamo 模块，避免真模块触发循环导入
# ------------------------------------------------------------------
_dummy_dynamo = _types.ModuleType("torch._dynamo")

def _dummy_disable(fn=None, *args, **kwargs):
    # 支持 @torch._dynamo.disable 和 @torch._dynamo.disable()
    if fn is None:
        def deco(f):
            return f
        return deco
    return fn

_dummy_dynamo.disable = _dummy_disable  # type: ignore[attr-defined]
sys.modules["torch._dynamo"] = _dummy_dynamo

# 现在再正式 import torch，后续 "import torch._dynamo" 会直接用上面的假模块
import torch

# 和其他脚本一致：禁用有时会冲突的 Flash/Mem-efficient attention 内核
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

from data_loader_DM import create_loaders
from model import ResInfModel

# --------------------- 载入配置 + 模型 + test_loader --------------------- #
def load_cfg_and_model(best_model_path: str,
                       config_path: str | None,
                       device: torch.device):
    best_model_path = pathlib.Path(best_model_path).resolve()
    if not best_model_path.is_file():
        raise FileNotFoundError(f"best_model 不存在: {best_model_path}")

    if config_path is None:
        config_path = best_model_path.parents[1] / "config.json"
    else:
        config_path = pathlib.Path(config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"config.json 不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        CFG = json.load(f)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] 使用设备: {device}")
    print(f"[INFO] 读取配置: {config_path}")

    # 与 main_seed.py 一致的 create_loaders 调用
    _, _, test_loader = create_loaders(
        h5_path=CFG["data"]["h5_path"],
        adjacency_path=CFG["data"]["adjacency_path"],
        cdiff_label_path=CFG["data"]["cdiff_label_path"],
        donor_final_path=CFG["data"]["donor_final_path"],
        receptor_ts_group=CFG["data"]["receptor_ts_group"],
        sampling_mode=CFG["data"]["sampling"]["mode"],
        num_samples=CFG["data"]["sampling"]["num_samples"],
        test_size=CFG["training"]["split"]["test"],
        val_size=CFG["training"]["split"]["val"],
        noise_strength_train=(
            CFG["training"]["noise_theta"]["train"]
            if CFG["training"]["use_dm_noise"]
            else CFG["training"]["noise"]["train"]
        ),
        noise_strength_val=(
            CFG["training"]["noise_theta"]["val"]
            if CFG["training"]["use_dm_noise"]
            else CFG["training"]["noise"]["val"]
        ),
        noise_strength_test=(
            CFG["training"]["noise_theta"]["test"]
            if CFG["training"]["use_dm_noise"]
            else CFG["training"]["noise"]["test"]
        ),
        use_dm_noise=CFG["training"]["use_dm_noise"],
        sparsity_config=CFG["training"]["sparsity"],
        batch_size=CFG["training"]["batch_size"],
        seed=CFG["training"]["seed"],
        corruption_mode=CFG["data"]["topology_corruption"]["mode"],
        corruption_rates=CFG["data"]["topology_corruption"]["rates"],
        new_h5_path=CFG["data"]["new_h5_path"],
        new_cdiff_label_path=CFG["data"]["new_cdiff_label_path"],
        new_donor_final_path=CFG["data"]["new_donor_final_path"],
        new_receptor_ts_group=CFG["data"]["new_receptor_ts_group"],
    )

    dataset = test_loader.dataset
    num_samples = len(dataset)
    if num_samples == 0:
        raise RuntimeError("测试集为空。")

    sample0 = dataset[0]
    num_species = sample0.donor_data.shape[0]
    seq_len = sample0.node_activity.shape[1]

    print(f"[INFO] 样本数={num_samples} · 物种数={num_species} · seq_len={seq_len}")

    print("[STEP] 构建并加载模型 …")
    model = ResInfModel(
        dynamics_input_dim=CFG["model"]["dynamics"]["input_dim"],
        dynamics_embed_dim=CFG["model"]["dynamics"]["embed_dim"],
        dynamics_num_layers=CFG["model"]["dynamics"]["layers"],
        dynamics_num_heads=CFG["model"]["dynamics"]["heads"],
        topology_input_dim=CFG["model"]["topology"]["input_dim"],
        topology_hidden_dim=CFG["model"]["topology"]["hidden_dim"],
        topology_num_layers=CFG["model"]["topology"]["layers"],
        embed_dim=CFG["model"]["embed_dim"],
        donor_dim=CFG["model"].get("donor_dim", 1),
        desired_seq_len=CFG["data"]["sampling"]["num_samples"],
        topology_type=CFG["model"]["topology_type"],
    ).to(device)

    try:
        state = torch.load(best_model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(best_model_path, map_location=device)
    sd = state.get("model_state_dict", state)
    model.load_state_dict(sd)
    model.eval()
    print(f"[INFO] 权重已加载: {best_model_path}")

    # species 名称（暂用 SpXX）
    # species_names = [f"Sp{str(i+1).zfill(2)}" for i in range(num_species)]
    species_names = [f"OTU{str(i+1).zfill(2)}" for i in range(num_species)]
    # 输出目录
    if "output" in CFG and "result_dir" in CFG["output"]:
        out_root = pathlib.Path(CFG["output"]["result_dir"])
    else:
        out_root = best_model_path.parents[1] / "results"
    out_dir = out_root / "attn_advanced"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 输出目录: {out_dir}")

    return CFG, model, test_loader, species_names, out_dir


# --------------------- 遍历 test_loader 收集 attn + 预测 --------------------- #
def collect_attn_and_preds(model, test_loader, device):
    raw_list = []
    y_true_list = []
    y_pred_list = []

    print("[STEP] 遍历 test_loader 收集 attention 和预测 …")
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            _, y = model(batch.node_activity,
                         batch.donor_data,
                         batch.edge_index,
                         batch.edge_attr,
                         batch.batch)
            attn = model.modulator.attn_weights.detach().cpu().numpy()  # (B_batch, N, M)
            raw_list.append(attn)
            y_true_list.append(batch.y.detach().cpu().numpy())
            y_pred_list.append(y.detach().cpu().numpy())

    raw_attn = np.concatenate(raw_list, axis=0)                  # (B, N, M)
    y_true = np.concatenate(y_true_list, axis=0).ravel()         # (B,)
    y_pred = np.concatenate(y_pred_list, axis=0).ravel()         # (B,)

    print(f"[INFO] raw_attn shape = {raw_attn.shape}")
    return raw_attn, y_true, y_pred


# --------------------- 统计函数 --------------------- #
def compute_stats(raw_attn: np.ndarray):
    """
    对 raw_attn[B, N, M] 计算：
      - mean_attn: (N, M)
      - z_mean:    (N, M)  —— 在样本维度做 z-score 后再平均
      - z_all:     (B, N, M)
    """
    mu = raw_attn.mean(axis=0, keepdims=True)
    sigma = raw_attn.std(axis=0, keepdims=True) + 1e-6
    z_all = (raw_attn - mu) / sigma
    z_mean = z_all.mean(axis=0)
    mean_attn = raw_attn.mean(axis=0)
    return mean_attn, z_mean, z_all


def normalize_alpha(mean_attn: np.ndarray):
    """把 mean_attn 归一化到 [0,1] 作为 alpha。"""
    a_min = float(mean_attn.min())
    a_max = float(mean_attn.max())
    if a_max - a_min < 1e-8:
        return np.ones_like(mean_attn)
    return (mean_attn - a_min) / (a_max - a_min + 1e-8)


def draw_heat(mat, species_names, fname, title, out_dir,
              cmap="viridis", color_label="value",
              alpha_mat: np.ndarray | None = None,
              vlim_symmetric: bool = False):
    """画二维热图，支持 alpha 权重。"""
    plt.figure(figsize=(7, 6))
    if vlim_symmetric:
        vmax = float(np.nanmax(np.abs(mat)))
        vmin = -vmax
    else:
        vmin = vmax = None

    im = plt.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
    if alpha_mat is not None:
        im.set_alpha(alpha_mat)

    cbar = plt.colorbar(im)
    cbar.set_label(color_label)

    n_rec, n_don = mat.shape
    plt.xticks(range(n_don), species_names[:n_don], rotation=90, fontsize=5)
    plt.yticks(range(n_rec), species_names[:n_rec], fontsize=5)
    plt.title(title)
    plt.tight_layout()

    out_path = out_dir / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"[保存] {fname} → {out_dir}")

# ===================== 额外可视化模块 ===================== #

def draw_weighted_zscore_masked(z_mean: np.ndarray,
                                mean_attn: np.ndarray,
                                species_names,
                                out_dir: pathlib.Path,
                                fname: str,
                                title: str,
                                attn_quantile: float = 0.8):
    """
    改良版加权 Z-score 热图：
      - 输入:
          z_mean:      (N_rec, N_don)，样本维度 z-score 平均值
          mean_attn:   (N_rec, N_don)，原始 attention 平均值
          species_names: 物种名列表，用于坐标轴
          out_dir:     输出文件夹 Path
          fname:       输出文件名 (例如 'attn_weighted_zscore_masked_overall.pdf')
          title:       图片标题
          attn_quantile: raw attention 的分位数阈值，如 0.8 表示只显示 top 20% raw 边
      - 做法:
          1) 找出 mean_attn 高于给定分位数的格子
          2) 仅对这些格子用 z_mean 上色
          3) 其他格子设为 NaN，imshow 不画（白色）
    """
    assert z_mean.shape == mean_attn.shape, "z_mean 与 mean_attn 形状需一致"
    N_rec, N_don = z_mean.shape

    # 1) 用 raw attention 分位数做 mask
    thr = np.quantile(mean_attn, attn_quantile)
    mat = z_mean.copy()
    mask_low = mean_attn < thr
    mat[mask_low] = np.nan  # 低于阈值的边不显示

    # 2) 对称 vlim，突出正负 z
    vmax = float(np.nanmax(np.abs(mat)))
    vmin = -vmax if vmax > 0 else 0.0

    plt.figure(figsize=(7, 6))
    im = plt.imshow(mat, cmap="coolwarm", vmin=vmin, vmax=vmax)

    cbar = plt.colorbar(im)
    cbar.set_label("avg z-score (masked by raw attention)")

    plt.xticks(range(N_don), species_names[:N_don], rotation=90, fontsize=5)
    plt.yticks(range(N_rec), species_names[:N_rec], fontsize=5)
    plt.title(title)
    plt.tight_layout()

    out_path = out_dir / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"[保存] {fname} → {out_dir}")


def draw_bubble_topk(z_mean: np.ndarray,
                     mean_attn: np.ndarray,
                     species_names,
                     out_dir: pathlib.Path,
                     fname: str,
                     title: str,
                     top_k: int = 200):
    """
    Top-k 气泡图：
      - 只画 score = |z_mean| * mean_attn 最大的 top_k 条边
      - x 轴: donor 物种索引
      - y 轴: recipient 物种索引
      - 颜色: z_mean (表示异常程度, z-score)
      - 点大小: mean_attn (表示模型权重大小)

      输入:
        z_mean:    (N_rec, N_don)
        mean_attn: (N_rec, N_don)
        species_names: 物种名列表
    """
    assert z_mean.shape == mean_attn.shape, "z_mean 与 mean_attn 形状需一致"

    N_rec, N_don = z_mean.shape
    score = np.abs(z_mean) * mean_attn       # |z| × raw attention
    num_pairs = score.size
    top_k = min(top_k, num_pairs)

    # 按 score 从大到小取 top_k
    flat_idx = np.argsort(score.flatten())[::-1][:top_k]
    rec_idx, don_idx = np.unravel_index(flat_idx, score.shape)

    z_sel = z_mean[rec_idx, don_idx]
    a_sel = mean_attn[rec_idx, don_idx]

    # 点大小归一化
    a_min = float(a_sel.min())
    a_max = float(a_sel.max())
    sizes = (a_sel - a_min) / (a_max - a_min + 1e-8)
    sizes = (sizes * 2000.0) + 50.0   # 这两个数字可以微调点大小范围

    x = don_idx
    y = rec_idx

    plt.figure(figsize=(7, 6))

    vmax = float(np.max(np.abs(z_sel)))
    vmin = -vmax if vmax > 0 else 0.0

    sc = plt.scatter(
        x, y,
        s=sizes,
        c=z_sel,
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        alpha=0.85,
        edgecolors="k",
        linewidths=0.4,
    )

    cbar = plt.colorbar(sc)
    cbar.set_label("z-score")

    plt.xticks(range(N_don), species_names[:N_don], rotation=90, fontsize=5)
    plt.yticks(range(N_rec), species_names[:N_rec], fontsize=5)
    plt.xlim(-0.5, N_don - 0.5)
    plt.ylim(-0.5, N_rec - 0.5)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.tight_layout()

    out_path = out_dir / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"[保存] {fname} → {out_dir}")


# --------------------- main --------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Advanced cross-attention visualization for ResInfModel "
                    "(raw / z-score / weighted z-score + top-k ablation)."
    )
    parser.add_argument("--best_model", required=True, help="Path to best_model .pth")
    parser.add_argument("--config", default=None, help="Path to config.json (optional)")
    parser.add_argument("--top_frac", type=float, default=0.5,
                        help="Fraction of pairs to mask for ablation, based on weighted z-score (0~1).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CFG, model, test_loader, species_names, out_dir = load_cfg_and_model(
        args.best_model, args.config, device
    )

    # 1) 收集全体样本的 attention 和预测
    raw_attn, y_true, y_orig = collect_attn_and_preds(model, test_loader, device)
    B, N_rec, N_don = raw_attn.shape

    # 2) overall raw/z/weighted-z
    mean_attn_all, z_mean_all, z_all = compute_stats(raw_attn)

    draw_heat(mean_attn_all, species_names,
              "attn_raw_overall.pdf",
              "Average Donor→Recipient Attention (all samples)",
              out_dir, cmap="viridis", color_label="avg weight")

    draw_heat(z_mean_all, species_names,
              "attn_zscore_overall.pdf",
              "Z-scored Donor→Recipient Attention (all samples)",
              out_dir, cmap="coolwarm", color_label="avg z-score",
              vlim_symmetric=True)

    alpha_all = normalize_alpha(mean_attn_all)
    draw_heat(z_mean_all, species_names,
              "attn_weighted_zscore_overall.pdf",
              "Weighted Z-scored Attention (color=z, alpha=raw)",
              out_dir, cmap="coolwarm", color_label="avg z-score",
              alpha_mat=alpha_all, vlim_symmetric=True)

    # ==== 1) 改良版 masked weighted Z-score 热图（整体） ====
    draw_weighted_zscore_masked(
        z_mean=z_mean_all,
        mean_attn=mean_attn_all,
        species_names=species_names,
        out_dir=out_dir,
        fname="attn_weighted_zscore_masked_overall.pdf",
        title="Masked Weighted Z-scored Attention (top raw edges)",
        attn_quantile=0.8,   # 只显示 raw attention 最高 20% 的边，可按需调节
    )

    # ==== 2) Top-k 气泡图（整体） ====
    draw_bubble_topk(
        z_mean=z_mean_all,
        mean_attn=mean_attn_all,
        species_names=species_names,
        out_dir=out_dir,
        fname="attn_bubble_top200_overall.pdf",
        title="Top-200 edges by |z|×raw attention (overall)",
        top_k=200,   # 可以改成 100、300 等
    )

    # 3) 按标签分组的 raw / z-score / weighted-z
    print("\n[STEP] 按标签分组的 attention 统计 …")
    mask_R = (y_true == 1)
    mask_NR = (y_true == 0)

    def safe_group_stats(mask, name):
        if mask.sum() == 0:
            print(f"[WARN] 组 {name} 没有样本，跳过该组热图。")
            return None, None, None
        mean_attn, z_mean, _ = compute_stats(raw_attn[mask])
        alpha = normalize_alpha(mean_attn)
        return mean_attn, z_mean, alpha

    mean_R, z_R, alpha_R = safe_group_stats(mask_R, "Responder")
    mean_NR, z_NR, alpha_NR = safe_group_stats(mask_NR, "Non-responder")

    # raw by label
    if mean_R is not None:
        draw_heat(mean_R, species_names,
                  "attn_raw_responder.pdf",
                  "Average Attention (Responders)",
                  out_dir, cmap="viridis", color_label="avg weight")
    if mean_NR is not None:
        draw_heat(mean_NR, species_names,
                  "attn_raw_nonresponder.pdf",
                  "Average Attention (Non-responders)",
                  out_dir, cmap="viridis", color_label="avg weight")

    # z-score by label
    if z_R is not None:
        draw_heat(z_R, species_names,
                  "attn_zscore_responder.pdf",
                  "Z-scored Attention (Responders)",
                  out_dir, cmap="coolwarm", color_label="avg z-score",
                  vlim_symmetric=True)
    if z_NR is not None:
        draw_heat(z_NR, species_names,
                  "attn_zscore_nonresponder.pdf",
                  "Z-scored Attention (Non-responders)",
                  out_dir, cmap="coolwarm", color_label="avg z-score",
                  vlim_symmetric=True)

    # weighted z-score by label
    if z_R is not None:
        draw_heat(z_R, species_names,
                  "attn_weighted_zscore_responder.pdf",
                  "Weighted Z-scored Attention (Responders)",
                  out_dir, cmap="coolwarm", color_label="avg z-score",
                  alpha_mat=alpha_R, vlim_symmetric=True)
    if z_NR is not None:
        draw_heat(z_NR, species_names,
                  "attn_weighted_zscore_nonresponder.pdf",
                  "Weighted Z-scored Attention (Non-responders)",
                  out_dir, cmap="coolwarm", color_label="avg z-score",
                  alpha_mat=alpha_NR, vlim_symmetric=True)

    # ---------------- 4&5) top-k 消融：weighted z-score & z-only ---------------- #

        # ---------------- 4–6) top-k 消融：weighted z-score / |z| / z_mean ---------------- #

    print("\n[STEP] 基于 weighted z-score / |z| / z_mean 的 top-k 消融 …")

    # baseline AUROC
    auc_orig = roc_auc_score(y_true, y_orig)
    print(f"[RESULT] 原始 AUROC = {auc_orig:.4f}")

    top_frac = float(args.top_frac)
    if not (0.0 < top_frac < 1.0):
        raise ValueError("--top_frac 必须在 (0,1) 区间内")

    # 保存原始 cross_attn
    orig_attn_module = model.modulator.cross_attn

    class MaskedCrossAttn(torch.nn.Module):
        """
        使用 MultiheadAttention 自带的 attn_mask，在 softmax 前屏蔽指定 (i,j) pair。
        mask_2d: (N_rec, N_don) float32，0 表示保留，-1e9 表示屏蔽。
        """
        def __init__(self, orig, mask_2d):
            super().__init__()
            self.orig = orig
            self.register_buffer("attn_mask", mask_2d)

        def forward(self, query, key, value, need_weights: bool = True):
            # attn_mask: [L, S] = [N_rec, N_don]
            attn_mask = self.attn_mask.to(dtype=query.dtype, device=query.device)
            out, w = self.orig(query, key, value,
                               need_weights=need_weights,
                               attn_mask=attn_mask)
            return out, w

    # ---- 4) weighted z-score: score = |z_mean| * mean_attn ----
    score_matrix_w = np.abs(z_mean_all) * mean_attn_all  # (N_rec, N_don)
    num_pairs_w = score_matrix_w.size
    top_k_w = max(int(num_pairs_w * top_frac), 1)

    flat_idx_w = np.argsort(score_matrix_w.flatten())[::-1][:top_k_w]
    rec_idx_w, don_idx_w = np.unravel_index(flat_idx_w, score_matrix_w.shape)

    mask_w = np.zeros_like(score_matrix_w, dtype=np.float32)
    mask_w[rec_idx_w, don_idx_w] = -1e9          # 这些 edge logits 加 -1e9 → softmax 后 ~0
    mask_w_tensor = torch.from_numpy(mask_w)     # (N_rec, N_don)

    model.modulator.cross_attn = MaskedCrossAttn(orig_attn_module, mask_w_tensor)

    y_mask_w_list = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            _, y = model(batch.node_activity,
                         batch.donor_data,
                         batch.edge_index,
                         batch.edge_attr,
                         batch.batch)
            y_mask_w_list.append(y.detach().cpu().numpy())
    y_mask_w = np.concatenate(y_mask_w_list, axis=0).ravel()
    auc_mask_w = roc_auc_score(y_true, y_mask_w)
    drop_pct_w = (auc_orig - auc_mask_w) / max(auc_orig, 1e-6) * 100.0

    print(f"[RESULT] (weighted z = |z|×raw) Mask top {top_frac*100:.1f}% pairs "
          f"→ AUROC: {auc_orig:.4f} → {auc_mask_w:.4f}, drop {drop_pct_w:.1f}%")

    # ---- 5) 纯 |z|: score = |z_mean| ----
    score_matrix_absz = np.abs(z_mean_all)              # (N_rec, N_don)
    num_pairs_absz = score_matrix_absz.size
    top_k_absz = max(int(num_pairs_absz * top_frac), 1)

    flat_idx_absz = np.argsort(score_matrix_absz.flatten())[::-1][:top_k_absz]
    rec_idx_absz, don_idx_absz = np.unravel_index(flat_idx_absz, score_matrix_absz.shape)

    mask_absz = np.zeros_like(score_matrix_absz, dtype=np.float32)
    mask_absz[rec_idx_absz, don_idx_absz] = -1e9
    mask_absz_tensor = torch.from_numpy(mask_absz)

    model.modulator.cross_attn = MaskedCrossAttn(orig_attn_module, mask_absz_tensor)

    y_mask_absz_list = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            _, y = model(batch.node_activity,
                         batch.donor_data,
                         batch.edge_index,
                         batch.edge_attr,
                         batch.batch)
            y_mask_absz_list.append(y.detach().cpu().numpy())
    y_mask_absz = np.concatenate(y_mask_absz_list, axis=0).ravel()
    auc_mask_absz = roc_auc_score(y_true, y_mask_absz)
    drop_pct_absz = (auc_orig - auc_mask_absz) / max(auc_orig, 1e-6) * 100.0

    print(f"[RESULT] (|z|-only) Mask top {top_frac*100:.1f}% pairs "
          f"→ AUROC: {auc_orig:.4f} → {auc_mask_absz:.4f}, drop {drop_pct_absz:.1f}%")

    # ---- 6) 老版本风格：用 z_mean 选出 z-score 最高的 top_frac (例如 10%) 边 ----
    # 这里不取绝对值，保留符号，更接近 draw_k_space_3.py 里用 z_mean 排序的做法
    score_matrix_zmean = z_mean_all                    # (N_rec, N_don)
    num_pairs_zmean = score_matrix_zmean.size
    top_k_zmean = max(int(num_pairs_zmean * top_frac), 1)

    flat_idx_zmean = np.argsort(score_matrix_zmean.flatten())[::-1][:top_k_zmean]
    rec_idx_zmean, don_idx_zmean = np.unravel_index(flat_idx_zmean, score_matrix_zmean.shape)

    mask_zmean = np.zeros_like(score_matrix_zmean, dtype=np.float32)
    mask_zmean[rec_idx_zmean, don_idx_zmean] = -1e9
    mask_zmean_tensor = torch.from_numpy(mask_zmean)

    model.modulator.cross_attn = MaskedCrossAttn(orig_attn_module, mask_zmean_tensor)

    y_mask_zmean_list = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            _, y = model(batch.node_activity,
                         batch.donor_data,
                         batch.edge_index,
                         batch.edge_attr,
                         batch.batch)
            y_mask_zmean_list.append(y.detach().cpu().numpy())
    y_mask_zmean = np.concatenate(y_mask_zmean_list, axis=0).ravel()
    auc_mask_zmean = roc_auc_score(y_true, y_mask_zmean)
    drop_pct_zmean = (auc_orig - auc_mask_zmean) / max(auc_orig, 1e-6) * 100.0

    print(f"[RESULT] (z-mean old-style) Mask top {top_frac*100:.1f}% pairs "
          f"→ AUROC: {auc_orig:.4f} → {auc_mask_zmean:.4f}, drop {drop_pct_zmean:.1f}%")

    # 恢复原始 cross_attn
    model.modulator.cross_attn = orig_attn_module



if __name__ == "__main__":
    main()
