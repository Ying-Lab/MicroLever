###运行脚本时使用 python eval_real_test_merge.py --output-dir /你的目录 验证输出位置。


import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix
)

# ============ 1. 配置：两个结果文件路径 ============

RESULT_FILES = {
    "Damman": "/home/data/FMT/code/1026/seed/zz_real_test/test_results_Damman_mlp_2025/per_sample_results.csv",
    "Other":  "/home/data/FMT/test_results_3_mlp/per_sample_results.csv",
}

# 分类阈值（和你 test_clr_spline.py 中保持一致）
THRESHOLD = 0.5

# 解析命令行参数，允许指定输出目录
parser = argparse.ArgumentParser(description="合并不同队列的评估结果")
parser.add_argument(
    "--output-dir",
    type=str,
    default="eval_real_merged",
    help="所有评估输出文件保存的目录（默认：eval_real_merged）",
)
args = parser.parse_args()

# 合并结果的输出目录（可通过参数指定）
OUT_DIR = args.output_dir
os.makedirs(OUT_DIR, exist_ok=True)

# ============ 2. 读取并合并两个队列的结果 ============

dfs = []
for name, path in RESULT_FILES.items():
    df = pd.read_csv(path)
    if "pred_prob" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{path} 中缺少 'pred_prob' 或 'label' 列")
    df["cohort"] = name
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

y_true = df_all["label"].values.astype(int)
y_score = df_all["pred_prob"].values.astype(float)

print(f"Total samples: {len(y_true)}")
print(df_all["cohort"].value_counts())

# ============ 3. 计算分类指标 ============

# 1) 连续概率上的指标
auroc = roc_auc_score(y_true, y_score)
auprc = average_precision_score(y_true, y_score)

# 2) 按阈值 0.5（二分类标签）
y_pred = (y_score >= THRESHOLD).astype(int)

acc = accuracy_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)       # 召回率 Recall
prec = precision_score(y_true, y_pred)   # 精确率 Precision
f1 = f1_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)    # [[TN, FP], [FN, TP]]

print("=== Combined metrics (two cohorts merged) ===")
print(f"AUROC : {auroc:.4f}")
print(f"AUPRC : {auprc:.4f}")
print(f"ACC   : {acc:.4f}")
print(f"Recall: {rec:.4f}")
print(f"Prec. : {prec:.4f}")
print(f"F1    : {f1:.4f}")
print("Confusion matrix [[TN, FP], [FN, TP]]:")
print(cm)

# 保存成一个 csv 方便写论文时查数
metrics_df = pd.DataFrame({
    "metric": ["AUROC", "AUPRC", "ACC", "Recall", "Precision", "F1",
               "TN", "FP", "FN", "TP"],
    "value": [
        auroc, auprc, acc, rec, prec, f1,
        cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1],
    ]
})
metrics_df.to_csv(os.path.join(OUT_DIR, "combined_metrics.csv"), index=False)

# ============ 4. 画 ROC 曲线 ============

fpr, tpr, _ = roc_curve(y_true, y_score)

plt.figure()
plt.plot(fpr, tpr, label=f"Combined ROC (AUROC = {auroc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve (two cohorts merged)")
plt.legend(loc="lower right")
plt.grid(True)
roc_path = os.path.join(OUT_DIR, "combined_roc.pdf")
plt.savefig(roc_path, format="pdf", bbox_inches="tight")
plt.close()

print(f"ROC curve saved to: {roc_path}")

# ============ 5. 画 PR 曲线 ============

precision, recall_vals, _ = precision_recall_curve(y_true, y_score)

plt.figure()
plt.step(recall_vals, precision, where="post", label=f"Combined PR (AUPRC = {auprc:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall curve (two cohorts merged)")
plt.legend(loc="lower left")
plt.grid(True)
pr_path = os.path.join(OUT_DIR, "combined_pr.pdf")
plt.savefig(pr_path, format="pdf", bbox_inches="tight")
plt.close()

print(f"PR curve saved to: {pr_path}")
