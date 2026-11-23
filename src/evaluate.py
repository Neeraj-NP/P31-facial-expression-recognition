"""Evaluate a trained run on FER-2013 test or RAF-DB test (cross-dataset), incl. subgroup audit.

    python -m src.evaluate --run results/cnn_xxx --dataset fer2013
    python -m src.evaluate --run results/cnn_xxx --dataset rafdb
"""
# =====================================================================
# FILE ROLE: tests a trained model on the FER-2013 test set or on RAF-DB (the cross-dataset test) and, on RAF-DB, runs the
# fairness audit by gender / race / age.
# HOW IT IS RUN: python -m src.evaluate --run results/resnet50 --dataset rafdb
# INPUT: results/<run>/ (config.yaml + best.pt).
# OUTPUT: results/<run>/eval_<dataset>/ with predictions.csv, metrics.json, confusion.csv, confusion_matrix.png, subgroup_audit.csv.
# GOES TO: bias_report.py, cross_dataset.py and class_mix_check.py read these files.
# =====================================================================
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw plots to files without needing a screen
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .common import get_device, get_frames, load_run, make_loader
from .datasets import CLASSES

# The demographic attributes that RAF-DB provides (FER-2013 has none, so the audit is skipped there).
SUBGROUP_COLS = ["gender", "race", "age"]


# WHAT: runs the model over a DataLoader and collects its probabilities.
# INPUT: model, DataLoader, device.  RETURNS: (probabilities of shape images x 7, row indexes).  GOES TO: main().
@torch.no_grad()  # no gradients are needed to predict: saves memory and time
def predict(model, loader, device):
    probs, idx = [], []
    for x, _, i in loader:
        # softmax turns the 7 raw scores into probabilities that sum to 1
        probs.append(torch.softmax(model(x.to(device)), 1).cpu())
        idx.append(i)  # remember which DataFrame row each prediction belongs to
    return torch.cat(probs).numpy(), torch.cat(idx).numpy()


# INPUT: true labels, predicted labels, a metric function.  RETURNS: (low, high) of the 95% interval.
# GOES TO: subgroup_audit(), and finally the acc_ci columns of subgroup_audit.csv.
# Q: Why 1000 bootstrap resamples? A: A common choice that gives stable 2.5th and 97.5th percentiles at low cost;
#    the fixed seed makes the intervals reproducible.
# Q: Why bootstrap and not a formula? A: It works for any metric, including macro-F1, which has no simple formula.
def bootstrap_ci(y, p, metric, n=1000, seed=0):
    """95% bootstrap confidence interval for a metric(y_true, y_pred)."""
    # Bootstrap = resample the test images WITH replacement 1000 times, recompute the metric each time, and take
    # the middle 95% of those values (2.5th to 97.5th percentile). It shows how uncertain a score is,
    # especially for small groups (e.g. only 89 images aged 70+). Fixed seed -> reproducible intervals.
    rng = np.random.default_rng(seed)
    scores = [metric(y[s], p[s]) for s in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


# INPUT: true labels and predicted labels.  RETURNS: one number, the average F1 over the 7 emotions.
def macro_f1(y, p):
    # labels=range(7) + zero_division=0: a bootstrap sample may miss a class entirely; count that as 0
    # instead of raising an error or silently shrinking the average.
    return f1_score(y, p, average="macro", labels=range(len(CLASSES)), zero_division=0)


# WHAT: the fairness audit. INPUT: a DataFrame with label, pred and (for RAF-DB) gender / race / age columns.
# RETURNS: (a table with one row per group, a dict with the best-minus-worst gap per attribute).
# GOES TO: saved as subgroup_audit.csv, which bias_report.py reads.
def subgroup_audit(df):
    """Accuracy / macro-F1 with CIs per demographic subgroup, plus the best-worst gap."""
    # The fairness audit: how accurate is the model for each gender / race / age group?
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
    # "Gap" = accuracy of the best group minus the worst group for each attribute.
    gaps = {
        col: float(g["acc"].max() - g["acc"].min())
        for col, g in audit.groupby("attribute")
    } if len(audit) else {}
    return audit, gaps


# FLOW: load the saved model -> predict on the test split -> write predictions.csv -> compute accuracy / macro-F1 / per-class
#  report -> subgroup audit (RAF-DB only) -> write metrics.json -> write the confusion matrix (csv and png).
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dataset", choices=["fer2013", "rafdb"], default="fer2013")
    args = ap.parse_args()

    device = get_device()
    model, cfg = load_run(args.run, device)
    df = get_frames(cfg, args.dataset)["test"].reset_index(drop=True)
    probs, idx = predict(model, make_loader(cfg, df, train=False), device)
    # Put each prediction back on its own row (idx), then store the most probable class as an integer.
    df.loc[idx, "pred"] = probs.argmax(1)
    df["pred"] = df["pred"].astype(int)

    out = Path(args.run) / f"eval_{args.dataset}"
    out.mkdir(exist_ok=True)
    # Per-image predictions. Kept local only: for RAF-DB they are per-image labels, which the dataset
    # licence does not allow us to publish (hence predictions.csv is gitignored).
    # Q: Why are per-image predictions not in the repo? A: For RAF-DB they reveal per-image labels, and the RAF-DB
    #    licence forbids redistribution; only aggregate numbers (metrics, confusion counts, group tables) are committed.
    df.to_csv(out / "predictions.csv", index=False)

    y, p = df["label"].values, df["pred"].values
    metrics = {
        "dataset": args.dataset, "n": len(df),
        "acc": accuracy_score(y, p), "macro_f1": macro_f1(y, p),
        # precision / recall / F1 for every emotion separately
        "per_class": classification_report(y, p, labels=range(len(CLASSES)), target_names=CLASSES,
                                           output_dict=True, zero_division=0),
    }

    # The fairness audit: on RAF-DB, accuracy and macro-F1 for every gender, race and age group, each with a 95%
    # bootstrap interval. FER-2013 has no demographic columns, so there the audit table is simply empty.
    audit, gaps = subgroup_audit(df)
    if len(audit):  # only RAF-DB has demographic columns
        audit.to_csv(out / "subgroup_audit.csv", index=False)
        metrics["subgroup_acc_gap"] = gaps
        print(audit[["attribute", "group", "n", "acc", "macro_f1"]].to_string(index=False))

    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # raw counts (aggregate, safe to commit for RAF-DB); src.cross_dataset compares these across datasets
    # confusion matrix: a 7x7 table where row = true emotion, column = predicted emotion, cell = number of images.
    # Example (ViT-B on RAF-DB): row "disgust", column "sad" = 57, i.e. 57 of the 160 disgust faces were called sad.
    counts = confusion_matrix(y, p, labels=range(len(CLASSES)))
    pd.DataFrame(counts, index=CLASSES, columns=CLASSES).to_csv(out / "confusion.csv", index_label="true")
    # Confusion matrix picture: rows = true emotion, columns = predicted emotion, normalised so each row sums to 1.
    # The diagonal is the per-class recall; off-diagonal cells show which emotions get mixed up.
    cm = confusion_matrix(y, p, labels=range(len(CLASSES)), normalize="true")
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("predicted"), plt.ylabel("true"), plt.title(f"{cfg['model']['name']} on {args.dataset}")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=150)

    print(f"{args.dataset}: acc {metrics['acc']:.4f}  macro-F1 {metrics['macro_f1']:.4f}  -> {out}")


if __name__ == "__main__":
    main()
