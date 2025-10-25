#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_reports.py
---------------
从 sklearn classification_report 的文本文件 (report.txt)
和 confusion_matrix.csv 生成两张图：
1. Per-class Precision/Recall/F1 柱状图
2. Confusion Matrix 热力图
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_report_text(path):
    """解析 sklearn 的 classification_report 文本格式"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.lower().startswith(("accuracy", "macro avg", "weighted avg", "micro avg")):
            continue
        match = re.match(r"(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
        if match:
            cls, prec, rec, f1 = match.groups()
            rows.append({
                "class": cls,
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1)
            })
    if not rows:
        raise ValueError("无法从 report.txt 中解析出类别行，请确认格式正确。")
    return pd.DataFrame(rows)


def plot_per_class_metrics(df, save_path):
    """绘制 Precision / Recall / F1 柱状图"""
    classes = df["class"]
    x = np.arange(len(classes))
    width = 0.25

    plt.figure(figsize=(8, 5))
    plt.bar(x - width, df["precision"], width, label="Precision")
    plt.bar(x, df["recall"], width, label="Recall")
    plt.bar(x + width, df["f1"], width, label="F1")

    plt.xticks(x, classes)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Per-class Metrics (Val)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_confusion_matrix(csv_path, labels, save_path):
    """绘制混淆矩阵"""
    cm = pd.read_csv(csv_path, header=None).values
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="viridis")
    plt.title("Confusion Matrix (Validation)")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, help="Path to report.txt")
    parser.add_argument("--confusion", required=True, help="Path to confusion_matrix.csv")
    parser.add_argument("--outdir", default="figs", help="Output folder")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = parse_report_text(args.report)

    plot_per_class_metrics(df, os.path.join(args.outdir, "perclass_metrics.png"))
    plot_confusion_matrix(args.confusion, df["class"].tolist(),
                          os.path.join(args.outdir, "confusion_matrix.png"))
    print("✅ 已保存图像到文件夹:", args.outdir)


if __name__ == "__main__":
    main()
