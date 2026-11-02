"""Evaluate a trained run on FER-2013 test or RAF-DB test (cross-dataset), incl. subgroup audit.

    python -m src.evaluate --run results/cnn_xxx --dataset fer2013
    python -m src.evaluate --run results/cnn_xxx --dataset rafdb
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .common import get_device, get_frames, load_run, make_loader
from .datasets import CLASSES

SUBGROUP_COLS = ["gender", "race", "age"]


@torch.no_grad()
def predict(model, loader, device):
    probs, idx = [], []
    for x, _, i in loader:
        probs.append(torch.softmax(model(x.to(device)), 1).cpu())
        idx.append(i)
    return torch.cat(probs).numpy(), torch.cat(idx).numpy()


def bootstrap_ci(y, p, metric, n=1000, seed=0):
    """95% bootstrap confidence interval for a metric(y_true, y_pred)."""
    rng = np.random.default_rng(seed)
    scores = [metric(y[s], p[s]) for s in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def macro_f1(y, p):
    return f1_score(y, p, average="macro", labels=range(len(CLASSES)), zero_division=0)


def subgroup_audit(df):
    """Accuracy / macro-F1 with CIs per demographic subgroup, plus the best-worst gap."""
    rows = []
    for col in SUBGROUP_COLS:
        if col not in df:
            continue
        for group, g in df.dropna(subset=[col]).groupby(col):
            y, p = g["label"].values, g["pred"].values
            rows.append({
                "attribute": col, "group": group, "n": len(g),
                "acc": accuracy_score(y, p), "acc_ci": bootstrap_ci(y, p, accuracy_score),
                "macro_f1": macro_f1(y, p), "macro_f1_ci": bootstrap_ci(y, p, macro_f1),
            })
    audit = pd.DataFrame(rows)
    gaps = {
        col: float(g["acc"].max() - g["acc"].min())
        for col, g in audit.groupby("attribute")
    } if len(audit) else {}
    return audit, gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dataset", choices=["fer2013", "rafdb"], default="fer2013")
    args = ap.parse_args()

    device = get_device()
    model, cfg = load_run(args.run, device)
    df = get_frames(cfg, args.dataset)["test"].reset_index(drop=True)
    probs, idx = predict(model, make_loader(cfg, df, train=False), device)
    df.loc[idx, "pred"] = probs.argmax(1)
    df["pred"] = df["pred"].astype(int)

    out = Path(args.run) / f"eval_{args.dataset}"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "predictions.csv", index=False)

    y, p = df["label"].values, df["pred"].values
    metrics = {
        "dataset": args.dataset, "n": len(df),
        "acc": accuracy_score(y, p), "macro_f1": macro_f1(y, p),
        "per_class": classification_report(y, p, labels=range(len(CLASSES)), target_names=CLASSES,
                                           output_dict=True, zero_division=0),
    }

    audit, gaps = subgroup_audit(df)
    if len(audit):
        audit.to_csv(out / "subgroup_audit.csv", index=False)
        metrics["subgroup_acc_gap"] = gaps
        print(audit[["attribute", "group", "n", "acc", "macro_f1"]].to_string(index=False))

    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    cm = confusion_matrix(y, p, labels=range(len(CLASSES)), normalize="true")
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("predicted"), plt.ylabel("true"), plt.title(f"{cfg['model']['name']} on {args.dataset}")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=150)

    print(f"{args.dataset}: acc {metrics['acc']:.4f}  macro-F1 {metrics['macro_f1']:.4f}  -> {out}")


if __name__ == "__main__":
    main()
