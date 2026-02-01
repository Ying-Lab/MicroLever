# main.py – CLI 支持 DM 噪声 + 拓扑伪造开关 + 额外指标/日志/结果保存
# -*- coding: utf-8 -*-
import os, json, logging, random, argparse, sys
from datetime import datetime

import numpy as np
import torch

# Allow running from repo root without installation
import os as _os, sys as _sys
_sys.path.append(_os.path.join(_os.path.dirname(__file__), '..', 'src'))

import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, roc_auc_score, confusion_matrix, recall_score,
    accuracy_score, precision_score
)

from microlever.data_loader import create_loaders   # 需按实际文件名修改
from microlever.model import ResInfModel

import torch.multiprocessing as mp

# 切换启动方式与共享策略，避免 Bus Error
mp.set_start_method('forkserver', force=True)
mp.set_sharing_strategy('file_descriptor')

# ---------------------------  基础 CONFIG  ---------------------------
CONFIG = {
    "data": {
        "h5_path": "data/sim/train/FMT_structured_data.h5",
        "adjacency_path": "data/adjacency/fake_A_53_dHOMO.csv",
        "cdiff_label_path": "data/sim/train/csv_reports/Cdiff_labels_table.csv",
        "donor_final_path": "data/sim/train/csv_reports/donor_final_abundance.csv",
        "receptor_ts_group": "/recipients",
        "sampling": {"mode": "random",
                     "num_samples": 5,
                     "down_sample_step": 10},
        # --- 拓扑伪造默认 ---
        "topology_corruption": {"mode": "none",            # "none" | "sign" | "value" | "both"
                                "rates": {"train": 0.0,
                                          "val":   0.0,
                                          "test":  0.0}},
        # ---- 新测试集 ----
        "new_h5_path": None,
        "new_cdiff_label_path": None,
        "new_donor_final_path": None,
        "new_receptor_ts_group": "/recipients"
    },
    "training": {
        "seed": 42,
        "batch_size": 128,
        "epochs": 150,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        # ---- 高斯噪声 (备用) ----
        "noise": {"train": 0.0, "val": 0.0, "test": 0.0},
        # ---- DM 噪声 ----
        "use_dm_noise": True,
        "noise_theta": {"train": 0.0, "val": 0.0, "test": 0.0},
        "split": {"test": 0.2, "val": 0.5},
        "sparsity": {"train": (1, 1),
                     "val":   (1, 1),
                     "test":  (1, 1)}
    },
    "model": {
        "dynamics": {"input_dim": 1, "embed_dim": 64, "layers": 2, "heads": 4},
        "topology": {"input_dim": 64, "hidden_dim": 64, "layers": 2},
        "embed_dim": 64,
        "donor_dim": 1,
        "topology_type": "original"          # ["original","sign_only","no_sign"]
    },
    "output": {"root_dir": "experiments",
               "log_dir":  "logs",
               "model_dir": "models",
               "result_dir": "results"}
}

# ---------------------------  环境 & 日志 ---------------------------
def setup_env(cfg):
    """Create an experiment folder and return the torch device.

    Directory layout:
        {output.root_dir}/{output.run_name}/{timestamp}/
            config.json
            logs/
            models/
            results/
    """
    exp_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_root = cfg.get("output", {}).get("root_dir", "experiments")
    run_name = cfg.get("output", {}).get("run_name", "microlever_run")
    exp_root = os.path.join(base_root, run_name, exp_time)

    log_dir = os.path.join(exp_root, "logs")
    model_dir = os.path.join(exp_root, "models")
    result_dir = os.path.join(exp_root, "results")
    for p in (exp_root, log_dir, model_dir, result_dir):
        os.makedirs(p, exist_ok=True)

    cfg.setdefault("output", {})
    cfg["output"]["root_dir"] = exp_root
    cfg["output"]["log_dir"] = log_dir
    cfg["output"]["model_dir"] = model_dir
    cfg["output"]["result_dir"] = result_dir

    seed = int(cfg.get("training", {}).get("seed", 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_logger(cfg):
    lg = logging.getLogger()
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    fh = logging.FileHandler(os.path.join(cfg["output"]["log_dir"], "train.log"))
    fh.setFormatter(fmt); lg.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt); lg.addHandler(sh)
    return lg

# --------------------------- Train / Eval ---------------------------
def train_epoch(model, loader, crit, opt, dev):
    model.train()
    tot = 0; p, l = [], []
    for d in tqdm(loader, desc="Train"):
        d = d.to(dev)
        opt.zero_grad()
        _, y = model(d.node_activity, d.donor_data, d.edge_index, d.edge_attr, d.batch)
        loss = crit(y.squeeze(), d.y.squeeze())
        loss.backward(); opt.step()
        tot += loss.item()
        p.extend(y.detach().cpu().numpy().ravel())
        l.extend(d.y.cpu().numpy().ravel())
    return tot/len(loader), np.array(p), np.array(l)

@torch.no_grad()
def evaluate(model, loader, crit, dev):
    model.eval()
    tot = 0; p, l, k = [], [], []
    for d in tqdm(loader, desc="Eval"):
        d = d.to(dev)
        K, y = model(d.node_activity, d.donor_data, d.edge_index, d.edge_attr, d.batch)
        tot += crit(y.squeeze(), d.y.squeeze()).item()
        p.extend(y.squeeze().cpu().numpy().ravel())
        l.extend(d.y.squeeze().cpu().numpy().ravel())
        k.extend(K.squeeze().cpu().numpy().ravel())
    return tot/len(loader), np.array(p), np.array(l), np.array(k)

# --------------------------- 主流程 ---------------------------
def main():
    dev = setup_env(CONFIG)
    logger = get_logger(CONFIG)

    # ----- 统一基础信息日志（来自代码 1） -----
    logger.info("运行配置：\n%s", json.dumps(CONFIG, indent=4, ensure_ascii=False))
    logger.info(f"使用的拓扑编码器类型: {CONFIG['model']['topology_type']}")
    logger.info(f"设备类型: {dev}")
    logger.info(f"随机种子: {CONFIG['training']['seed']}")
    logger.info(f"输出目录: {CONFIG['output']['root_dir']}")
    logger.info("正在加载数据集...")

    # ---- 数据加载 ----
    train_loader, val_loader, test_loader = create_loaders(
        h5_path=CONFIG["data"]["h5_path"],
        adjacency_path=CONFIG["data"]["adjacency_path"],
        cdiff_label_path=CONFIG["data"]["cdiff_label_path"],
        donor_final_path=CONFIG["data"]["donor_final_path"],
        receptor_ts_group=CONFIG["data"]["receptor_ts_group"],
        sampling_mode=CONFIG["data"]["sampling"]["mode"],
        num_samples=CONFIG["data"]["sampling"]["num_samples"],

        test_size=CONFIG["training"]["split"]["test"],
        val_size=CONFIG["training"]["split"]["val"],

        noise_strength_train=(CONFIG["training"]["noise_theta"]["train"]
                              if CONFIG["training"]["use_dm_noise"]
                              else CONFIG["training"]["noise"]["train"]),
        noise_strength_val  =(CONFIG["training"]["noise_theta"]["val"]
                              if CONFIG["training"]["use_dm_noise"]
                              else CONFIG["training"]["noise"]["val"]),
        noise_strength_test =(CONFIG["training"]["noise_theta"]["test"]
                              if CONFIG["training"]["use_dm_noise"]
                              else CONFIG["training"]["noise"]["test"]),
        use_dm_noise=CONFIG["training"]["use_dm_noise"],

        sparsity_config=CONFIG["training"]["sparsity"],
        batch_size=CONFIG["training"]["batch_size"],
        seed=CONFIG["training"]["seed"],

        corruption_mode=CONFIG["data"]["topology_corruption"]["mode"],
        corruption_rates=CONFIG["data"]["topology_corruption"]["rates"],

        new_h5_path=CONFIG["data"]["new_h5_path"],
        new_cdiff_label_path=CONFIG["data"]["new_cdiff_label_path"],
        new_donor_final_path=CONFIG["data"]["new_donor_final_path"],
        new_receptor_ts_group=CONFIG["data"]["new_receptor_ts_group"]
    )
    logger.info("Noise mode: %s",
                "DM" if CONFIG["training"]["use_dm_noise"] else "Gaussian")
    logger.info("Topo corruption: mode=%s rate=%s",
                CONFIG["data"]["topology_corruption"]["mode"],
                CONFIG["data"]["topology_corruption"]["rates"])

    # ---- 模型 ----
    model = ResInfModel(
        dynamics_input_dim = CONFIG["model"]["dynamics"]["input_dim"],
        dynamics_embed_dim = CONFIG["model"]["dynamics"]["embed_dim"],
        dynamics_num_layers= CONFIG["model"]["dynamics"]["layers"],
        dynamics_num_heads = CONFIG["model"]["dynamics"]["heads"],
        topology_input_dim = CONFIG["model"]["topology"]["input_dim"],
        topology_hidden_dim= CONFIG["model"]["topology"]["hidden_dim"],
        topology_num_layers= CONFIG["model"]["topology"]["layers"],
        embed_dim          = CONFIG["model"]["embed_dim"],
        donor_dim          = CONFIG["model"]["donor_dim"],
        desired_seq_len    = CONFIG["data"]["sampling"]["num_samples"],
        topology_type      = CONFIG["model"]["topology_type"]
    ).to(dev)

    opt  = torch.optim.Adam(model.parameters(),
                            lr=CONFIG["training"]["lr"],
                            weight_decay=CONFIG["training"]["weight_decay"])
    crit = torch.nn.BCELoss()

    best_auc = 0.0
    tr_losses, va_losses = [], []

    logger.info("开始训练 ...")
    for ep in range(CONFIG["training"]["epochs"]):
        tl, tp, tlbl = train_epoch(model, train_loader, crit, opt, dev)
        vl, vp, vlbl, _ = evaluate(model, val_loader, crit, dev)

        tr_losses.append(tl); va_losses.append(vl)
        tauc = roc_auc_score(tlbl, tp); vauc = roc_auc_score(vlbl, vp)
        tf1  = f1_score(tlbl > .5, tp > .5)
        vf1  = f1_score(vlbl > .5, vp > .5)

        logger.info(
            f"[{ep+1}/{CONFIG['training']['epochs']}] "
            f"TrainLoss {tl:.4f} F1 {tf1:.4f} AUC {tauc:.4f} | "
            f"ValLoss {vl:.4f} F1 {vf1:.4f} AUC {vauc:.4f}"
        )

        if vauc > best_auc:
            best_auc = vauc
            pth = os.path.join(CONFIG["output"]["model_dir"],
                               f"best_ep{ep+1}_auc{vauc:.4f}.pth")
            torch.save(model.state_dict(), pth)
            logger.info("发现新的最佳模型，已保存至: %s", pth)

    # ---- 测试 ----
    _, te_p, te_l, k_vals = evaluate(model, test_loader, crit, dev)

    te_bin = (te_p > 0.5).astype(int)
    tl_bin = (te_l > 0.5).astype(int)
    te_auc  = roc_auc_score(te_l, te_p)
    te_acc  = accuracy_score(tl_bin, te_bin)
    te_prec = precision_score(tl_bin, te_bin)
    te_rec  = recall_score(tl_bin, te_bin)
    te_f1   = f1_score(tl_bin, te_bin)
    tn, fp, fn, tp = confusion_matrix(tl_bin, te_bin).ravel()
    te_spec = tn / (tn + fp)

    logger.info(
        "[Test] AUC %.4f ACC %.4f F1 %.4f Precision %.4f Recall %.4f Spec %.4f",
        te_auc, te_acc, te_f1, te_prec, te_rec, te_spec
    )

    # ---- 结果可视化与保存 ----
    plt.figure(figsize=(8, 5))
    plt.plot(tr_losses, label='Train')
    plt.plot(va_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.title('Training / Validation Loss')
    plt.legend(); plt.grid(True)

    loss_png = os.path.join(CONFIG["output"]["result_dir"], "loss.png")
    plt.savefig(loss_png, bbox_inches='tight')
    plt.close()
    logger.info("Loss 曲线已保存至: %s", loss_png)

    np.save(os.path.join(CONFIG["output"]["result_dir"], "test_preds.npy"),  te_p)
    np.save(os.path.join(CONFIG["output"]["result_dir"], "test_labels.npy"), te_l)
    np.save(os.path.join(CONFIG["output"]["result_dir"], "k_values.npy"),    k_vals)
    logger.info("测试结果数组已保存至: %s", CONFIG["output"]["result_dir"])


# ---------------------------  CLI ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--gpu")
    parser.add_argument("--seed", type=int,help="随机种子（覆盖 config 中 training.seed）")
    # --- 旧参数 ---
    parser.add_argument("--mode", choices=["front","back","random"])
    parser.add_argument("--num_samples", type=int)
    parser.add_argument("--noise", type=float,
                        help="Gaussian σ² (use_dm_noise False)")
    parser.add_argument("--dm", action="store_true",
                        help="启用 Dirichlet‑Multinomial 噪声")
    parser.add_argument("--theta", type=float, help="DM θ")
    parser.add_argument("--sparsity", type=float)
    parser.add_argument("--topology_type",
                        choices=["original","sign_only","no_sign"])
    # --- 新增拓扑伪造 ---
    parser.add_argument("--corr_mode",
                        choices=["none","sign","value","both"])
    parser.add_argument("--corr_rate", type=float,
                        help="0‑1 之间，所有 split 共用")

    args = parser.parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.config:
        with open(args.config) as f:
            CONFIG.update(json.load(f))

    # CLI 覆盖
    if args.mode:
        CONFIG["data"]["sampling"]["mode"] = args.mode
    if args.num_samples:
        CONFIG["data"]["sampling"]["num_samples"] = args.num_samples
    if args.sparsity is not None:
        CONFIG["training"]["sparsity"] = {k:(args.sparsity,args.sparsity)
                                          for k in ("train","val","test")}
    if args.topology_type:
        CONFIG["model"]["topology_type"] = args.topology_type
    # ---------- 噪声开关 ----------
    if args.dm:
        CONFIG["training"]["use_dm_noise"] = True
        if args.theta:
            CONFIG["training"]["noise_theta"] = {k:args.theta
                                                 for k in ("train","val","test")}
    elif args.noise is not None:
        CONFIG["training"]["use_dm_noise"] = False
        CONFIG["training"]["noise"] = {k:args.noise for k in ("train","val","test")}
    # ---------- 拓扑伪造 ----------
    if args.corr_mode:
        CONFIG["data"]["topology_corruption"]["mode"] = args.corr_mode
    if args.corr_rate is not None:
        CONFIG["data"]["topology_corruption"]["rates"] = {k:args.corr_rate
                                                          for k in ("train","val","test")}
    # ---------- 随机种子 ----------
    if args.seed is not None:
        CONFIG["training"]["seed"] = args.seed
        
    main()
