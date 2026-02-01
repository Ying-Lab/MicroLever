# test_clr_spline.py
# -*- coding: utf-8 -*-
"""
在 CLR 空间做三次样条插值，然后逆 CLR，推理真实数据。
其它流程（CLR、归一化、模型加载、结果保存）与原 test_new_dataset_np.py 一致。
"""

import os, csv, json, logging
import numpy as np
import h5py
import torch
import sys, types
dynamo_stub = types.ModuleType("torch._dynamo")

def _dummy_disable(fn=None, recursive=True):
    if fn is None:
        def deco(f):
            return f
        return deco
    return fn

dynamo_stub.disable = _dummy_disable

import sys as _sys
_sys.modules["torch._dynamo"] = dynamo_stub
torch._dynamo = dynamo_stub


from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader

from scipy.interpolate import CubicSpline
from skbio.stats.composition import clr, clr_inv    # pip install scikit-bio

# ---------------- 1. 配置 ----------------
CONFIG = {
    "paths": {
        # ---------- 模型 ----------
        "model_weights": "/home/data/FMT/code/1026/seed/nosign_seed/experiments/mfront_n3_noise0_s1_no_sign_dmFalse_theta0_corrnone_0_seed4220/20250523_204533/experiments/mfront_n3_noise0_s1_no_sign_dmFalse_theta0_corrnone_0_seed4220/20250523_204533/experiments/mfront_n3_noise0_s1_no_sign_dmFalse_theta0_corrnone_0_seed4220/models/best_ep141_auc0.8477.pth",
        # "model_weights": "/home/data/FMT/code/1026/seed/ori_seed/experiments/mrandom_n3_noise0_s1_original_dmFalse_theta0_corrnone_0_seed2025/20250524_064307/experiments/mrandom_n3_noise0_s1_original_dmFalse_theta0_corrnone_0_seed2025/20250524_064307/experiments/mrandom_n3_noise0_s1_original_dmFalse_theta0_corrnone_0_seed2025/models/best_ep150_auc0.9616.pth",
        # "model_weights":"/home/data/FMT/code/1026/seed/ori_seed/experiments/mrandom_n5_noise0_s1_original_dmFalse_theta0_corrnone_0_seed2025/20250524_082039/experiments/mrandom_n5_noise0_s1_original_dmFalse_theta0_corrnone_0_seed2025/20250524_082039/experiments/mrandom_n5_noise0_s1_original_dmFalse_theta0_corrnone_0_seed2025/models/best_ep145_auc0.9596.pth",
        # "model_weights": "/home/data/FMT/code/1026/seed/ml_seed/lstm_seed/experiments/mrandom_n3_noise0_s1_original_dmFalse_theta0_corrnone_0_seed4220/20250526_024832/experiments/mrandom_n3_noise0_s1_original_dmFalse_theta0_corrnone_0_seed4220/20250526_024832/experiments/mrandom_n3_noise0_s1_original_dmFalse_theta0_corrnone_0_seed4220/models/best_ep150_auc0.7902.pth",
        # "model_weights": "/home/data/FMT/code/1026/seed/nosign_seed/experiments/mfront_n3_noise0_s1_no_sign_dmFalse_theta0_corrnone_0_seed4220/20250523_204533/experiments/mfront_n3_noise0_s1_no_sign_dmFalse_theta0_corrnone_0_seed4220/20250523_204533/experiments/mfront_n3_noise0_s1_no_sign_dmFalse_theta0_corrnone_0_seed4220/models/best_ep141_auc0.8477.pth",
        #"model_weights": "/home/data/FMT/code/1026/seed/nosign_seed/experiments/mfront_n3_noise0_s1_no_sign_dmFalse_theta0_corrnone_0_seed2025/20250523_202137/experiments/mfront_n3_noise0_s1_no_sign_dmFalse_theta0_corrnone_0_seed2025/20250523_202137/experiments/mfront_n3_noise0_s1_no_sign_dmFalse_theta0_corrnone_0_seed2025/models/best_ep146_auc0.8485.pth",
        # "model_weights": "/home/data/FMT/code/1026/seed/ml_seed/lstm_seed/experiments/mrandom_n3_noise0_s1_original_dmFalse_theta0_corrnone_0_seed4220/20250526_024832/experiments/mrandom_n3_noise0_s1_original_dmFalse_theta0_corrnone_0_seed4220/20250526_024832/experiments/mrandom_n3_noise0_s1_original_dmFalse_theta0_corrnone_0_seed4220/models/best_ep150_auc0.7902.pth",
        #mlp#"model_weights":"/home/data/FMT/code/1026/seed/ml_seed/mlp_seed/experiments/mrandom_n3_noise0_s1_original_dmFalse_theta0_corrnone_0_seed2025/20250526_010951/experiments/mrandom_n3_noise0_s1_original_dmFalse_theta0_corrnone_0_seed2025/20250526_010951/experiments/mrandom_n3_noise0_s1_original_dmFalse_theta0_corrnone_0_seed2025/models/best_ep149_auc0.7882.pth",
        # "model_weights":"/home/data/FMT/code/1026/seed/sign_seed/experiments/mrandom_n3_noise0_s1_sign_only_dmFalse_theta0_corrnone_0_seed2025/20250524_064905/experiments/mrandom_n3_noise0_s1_sign_only_dmFalse_theta0_corrnone_0_seed2025/20250524_064905/experiments/mrandom_n3_noise0_s1_sign_only_dmFalse_theta0_corrnone_0_seed2025/models/best_ep149_auc0.9509.pth",
        # "model_weights":"/home/data/FMT/code/1026/seed/nosign_seed/experiments/mrandom_n3_noise0_s1_no_sign_dmFalse_theta0_corrnone_0_seed2025/20250524_091447/experiments/mrandom_n3_noise0_s1_no_sign_dmFalse_theta0_corrnone_0_seed2025/20250524_091447/experiments/mrandom_n3_noise0_s1_no_sign_dmFalse_theta0_corrnone_0_seed2025/models/best_ep137_auc0.8485.pth",
        # ---------- 测试集 ----------
        # "h5_path": "/home/data/FMT/code/1026/seed/zz_real_test/real_data/real_was/FMT_real_was.h5",
        # "adjacency_csv": "/home/data/FMT/data/Mij/adjacency_matrix.csv",
        # "cdiff_label_csv": "/home/data/FMT/code/1026/seed/zz_real_test/real_data/real_was/Cdiff_labels_table.csv",
        # "donor_final_csv": "/home/data/FMT/code/1026/seed/zz_real_test/real_data/real_was/data/donor_final_abundance.csv",
        # "receptor_ts_group": "/recipients"

        # Damman
        "h5_path": "/home/data/FMT/code/1026/seed/zz_real_test/Damman/real_Damman/FMT_real_data_Damman.h5",
        "adjacency_csv": "/home/data/FMT/data/Mij/adjacency_matrix.csv",
        "cdiff_label_csv": "/home/data/FMT/code/1026/seed/zz_real_test/Damman/real_Damman/Cdiff_labels_table.csv",
        "donor_final_csv": "/home/data/FMT/code/1026/seed/zz_real_test/Damman/real_Damman/data/donor_final_abundance.csv",
        "receptor_ts_group": "/recipients"
    },
    "model": {
        "dynamics": {"input_dim": 1, "embed_dim": 64, "layers": 2, "heads": 4},
        "topology": {"input_dim": 64, "hidden_dim": 64, "layers": 2},
        "embed_dim": 64,
        "donor_dim": 1,
        "desired_seq_len": 3,      # 训练时 num_samples
        "topology_type": "no_sign"
    },
    "loader": {
        "batch_size": 128,
        "down_sample_step": 1,
        "num_samples": 3,
        "classification_threshold": 0.5
    },
    "output_dir": "test_results_no_sign_front_4220_Damman"
}

# ---------------- 2. 工具函数 ----------------
def clr_spline_sampling(rec_seq, target_len=3, down_step=10):
    """
    先 down‑sample，再在 CLR 空间做三次样条插值，最后逆 CLR 返回相对丰度。
    输出 shape: (target_len, n_taxa)
    """
    rec_ds = rec_seq[::down_step]                 # (T', n_taxa)
    T, n_taxa = rec_ds.shape
    if T < 2:                                     # 样本太少无法拟合样条
        raise ValueError("Sequence too short for spline interpolation")

    # a) CLR 变换（先加极小值防 0）
    clr_ds = clr(rec_ds + 1e-8)                   # (T', n_taxa)

    # b) 三次样条插值
    xi    = np.arange(T)
    x_new = np.linspace(0, T-1, target_len)
    out_clr = np.zeros((target_len, n_taxa), dtype=np.float32)
    for j in range(n_taxa):
        cs = CubicSpline(xi, clr_ds[:, j], bc_type='natural')
        out_clr[:, j] = cs(x_new)

    # c) 逆 CLR → 相对丰度
    out = clr_inv(out_clr)
    return out.astype(np.float32)

def robust_clr_transform(mat, eps=1e-8, min_pos=1e-12):
    mat = np.where(mat < min_pos, min_pos, mat)
    mat = np.where(mat == 0, eps, mat)
    log_m = np.log(mat)
    gm = np.exp(log_m.mean(axis=1, keepdims=True))
    return np.clip(log_m - np.log(gm), -50, 50).astype(np.float32)

def read_numeric_matrix(path, skip_header=True, skip_first_col=True):
    import csv
    with open(path, newline='') as f:
        rdr = csv.reader(f)
        if skip_header:
            next(rdr, None)
        rows = []
        for row in rdr:
            if skip_first_col:
                row = row[1:]
            rows.append([float(x) if x != '' else np.nan for x in row])
    return np.array(rows, dtype=np.float32)

def read_label_matrix(path):
    import csv
    names, mat = [], []
    with open(path, newline='') as f:
        rdr = csv.reader(f)
        header = next(rdr)
        for row in rdr:
            names.append(row[0])
            mat.append([float(x) if x != '' else np.nan for x in row[1:]])
    return names, np.array(mat, dtype=np.float32)

# ---------------- 3. Dataset ----------------
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, A, rec_seqs, donor_states, labels):
        self.edge_index = torch.tensor(np.array(np.nonzero(A)).astype(np.int64),
                                       dtype=torch.long)
        self.edge_attr = torch.tensor(A[self.edge_index[0], self.edge_index[1]],
                                      dtype=torch.float32)
        self.samples = []
        for rec, don, y in zip(rec_seqs, donor_states, labels):
            # ---- CLR‑空间三次样条插值 ----
            rec_proc = clr_spline_sampling(
                rec_seq=rec,
                target_len=CONFIG["loader"]["num_samples"],
                down_step=CONFIG["loader"]["down_sample_step"]
            )
            # 训练时模型收到的是 CLR 特征 → 再做 robust CLR
            rec_proc = robust_clr_transform(rec_proc / (rec_proc.sum(1, keepdims=True)+1e-8))

            # donor 处理同旧逻辑
            don_proc = don / (don.sum() + 1e-8)
            don_proc = robust_clr_transform(don_proc.reshape(1, -1)).flatten()

            self.samples.append((rec_proc, don_proc, float(y)))

    def __len__(self):  return len(self.samples)

    def __getitem__(self, idx):
        rec, don, label = self.samples[idx]
        node_activity = torch.tensor(rec.T, dtype=torch.float32).unsqueeze(-1)
        donor_data    = torch.tensor(don, dtype=torch.float32).view(-1, 1)
        return Data(
            x=node_activity[:, -1],
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            y=torch.tensor([label], dtype=torch.float32),
            node_activity=node_activity,
            donor_data=donor_data
        )

# ---------------- 4. 其它流程（日志、加载、推理、保存） ----------------
def setup_logger():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    logger = logging.getLogger("TEST_CLR_SPLINE")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(os.path.join(CONFIG["output_dir"], "test.log"))
    sh = logging.StreamHandler()
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger

def load_test_loader():
    p = CONFIG["paths"]
    A = read_numeric_matrix(p["adjacency_csv"], skip_header=False, skip_first_col=False)
    donor_final = read_numeric_matrix(p["donor_final_csv"])
    rec_names, lbl_mat = read_label_matrix(p["cdiff_label_csv"])

    pairs = np.where(~np.isnan(lbl_mat))
    rec_seqs, don_states, labels = [], [], []
    with h5py.File(p["h5_path"], "r") as f:
        g = f[p["receptor_ts_group"]]
        for r, c in zip(*pairs):
            rid = rec_names[r]
            seq = g[f'{rid}/data'][:].astype(np.float32)
            rec_seqs.append(seq)
            don_states.append(donor_final[:, c].astype(np.float32))
            labels.append(lbl_mat[r, c])

    ds = TestDataset(A, rec_seqs, don_states, labels)
    return GeoDataLoader(ds, CONFIG["loader"]["batch_size"], shuffle=False), np.array(labels)

from model import ResInfModel
# from lstm_model import SimpleLSTMModel
# from mlp_model import SimpleMLPModel


@torch.no_grad()
def infer(model, loader, device):
    model.eval(); preds, labels = [], []
    for batch in tqdm(loader, desc="Testing"):
        batch = batch.to(device)
        _, y_hat = model(batch.node_activity, batch.donor_data,
                         batch.edge_index, batch.edge_attr, batch.batch)
        preds.extend(y_hat.squeeze().cpu().numpy())
        labels.extend(batch.y.squeeze().cpu().numpy())
    return np.array(preds), np.array(labels)

def main():
    logger = setup_logger(); logger.info("加载测试集…")
    test_loader, raw_labels = load_test_loader()
    logger.info(f"样本数: {len(raw_labels)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"设备: {device}")

    # ---- 模型 ----
    m = CONFIG["model"]
    # model = SimpleLSTMModel(
    #     dynamics_input_dim=m["dynamics"]["input_dim"],
    #     dynamics_embed_dim=m["dynamics"]["embed_dim"],
    #     lstm_layers=CONFIG["model"]["dynamics"]["layers"],  # 使用原配置的层数参数
    #     donor_dim=m["donor_dim"],
    #     desired_seq_len=m["desired_seq_len"]
    # ).to(device)

    # model = SimpleMLPModel(
    #     dynamics_input_dim=m["dynamics"]["input_dim"],
    #     dynamics_embed_dim=m["dynamics"]["embed_dim"],
    #     donor_dim=m["donor_dim"],
    #     desired_seq_len=m["desired_seq_len"]
    # ).to(device)

    model = ResInfModel(
        dynamics_input_dim=m["dynamics"]["input_dim"],
        dynamics_embed_dim=m["dynamics"]["embed_dim"],
        dynamics_num_layers=m["dynamics"]["layers"],
        dynamics_num_heads=m["dynamics"]["heads"],
        topology_input_dim=m["topology"]["input_dim"],
        topology_hidden_dim=m["topology"]["hidden_dim"],
        topology_num_layers=m["topology"]["layers"],
        embed_dim=m["embed_dim"],
        donor_dim=m["donor_dim"],
        desired_seq_len=m["desired_seq_len"],
        topology_type=m["topology_type"],
    ).to(device)

    ckpt = torch.load(CONFIG["paths"]["model_weights"], map_location=device)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    logger.info("权重已加载")

    logger.info("开始推理…")
    y_pred, y_true = infer(model, test_loader, device)
    logger.info("—— Per‑sample results (index, pred_prob, label) ——")
    for idx, (p, l) in enumerate(zip(y_pred, y_true)):
        logger.info(f"{idx:04d}\t{p:.6f}\t{int(l)}")
    thr = CONFIG["loader"]["classification_threshold"]
    acc = float(np.mean((y_true > thr) == (y_pred > thr)))
    logger.info(f"Accuracy (thr={thr}): {acc:.4f}")

    # ---- 保存 ----
    out_dir = CONFIG["output_dir"]
    csv_path = os.path.join(out_dir, "per_sample_results.csv")
    with open(csv_path, "w", newline='') as f:
        csv.writer(f).writerows([["pred_prob", "label"], *zip(y_pred, y_true)])
    json.dump(CONFIG, open(os.path.join(out_dir, "test_config.json"), "w"), indent=4)

if __name__ == "__main__":
    main()
