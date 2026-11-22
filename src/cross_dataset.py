"""Cross-dataset comparison: models trained on FER-2013, tested on FER-2013 and RAF-DB (#9, #13).

Reads each run's eval_fer2013/ and eval_rafdb/ (metrics.json, confusion.csv) and writes to --out:
  cross_dataset.csv / .md   overall accuracy, macro-F1 and balanced accuracy on both datasets, and the drop
  per_class.csv             per-emotion F1 and recall on both datasets, and the drop
  per_class_drop.png        per-emotion F1 drop for the --focus models
  confusion_compare.png     row-normalised confusion on FER-2013, on RAF-DB, and the difference
  confusion_shift.csv       off-diagonal cells that grow the most from FER-2013 to RAF-DB

    python -m src.cross_dataset                      # default runs
    python -m src.cross_dataset --runs results/cnn_best results/resnet50 results/vit \
        --focus results/cnn_best results/resnet50 results/vit

confusion.csv is written by src.evaluate. For older runs, --from-predictions builds it from a local
predictions.csv first (the per-image file is never committed, only the aggregate counts).
"""
# =====================================================================
# FILE ROLE: compares how each model does on FER-2013 (in-domain) and on RAF-DB (a different dataset), overall and per emotion.
# HOW IT IS RUN: python -m src.cross_dataset
# INPUT: each run's eval_fer2013/ and eval_rafdb/ (metrics.json, confusion.csv).
# OUTPUT: results/cross_dataset/ (tables as csv/md, per-emotion chart, confusion comparison chart, confusion_shift.csv).
# GOES TO: results/cross_dataset_analysis.md, the README results table and the report quote these numbers.
# =====================================================================
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .datasets import CLASSES

DATASETS = ("fer2013", "rafdb")
DEFAULT_RUNS = ["cnn", "cnn_best", "resnet", "resnet_224", "resnet_frozen", "resnet50", "vit_small", "vit_small_lr1e4", "vit"]
DEFAULT_FOCUS = ["cnn_best", "resnet50", "vit"]  # the three selected models
# Friendly names for the tables and charts (the run folder name is used when a run is not listed here).
LABELS = {
    "cnn": "CNN baseline", "cnn_best": "CNN (tuned)", "resnet": "ResNet-18",
    "resnet_224": "ResNet-18 224px", "resnet_frozen": "ResNet-18 frozen", "resnet50": "ResNet-50",
    "vit_small": "ViT-S baseline", "vit_small_lr1e4": "ViT-S lr 1e-4", "vit": "ViT-B",
}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]  # fixed order: one colour per focus model
INK, MUTED = "#0b0b0b", "#52514e"  # text and secondary-text colours for the charts


def label(run):
    return LABELS.get(Path(run).name, Path(run).name)


# INPUT: an eval_<dataset> folder.  RETURNS: True if it rebuilt confusion.csv from a local predictions.csv, else False.
# WHY: older runs have no confusion.csv; counts are safe to commit but per-image predictions are not.
def confusion_from_predictions(eval_dir):
    # Rebuild the confusion-matrix COUNTS from the per-image predictions (only possible where the local
    # predictions.csv exists). Counts are aggregate numbers, so unlike per-image data they are safe to commit.
    pred = eval_dir / "predictions.csv"
    if not pred.exists():
        return False
    df = pd.read_csv(pred)
    counts = confusion_matrix(df["label"], df["pred"], labels=range(len(CLASSES)))
    pd.DataFrame(counts, index=CLASSES, columns=CLASSES).to_csv(eval_dir / "confusion.csv", index_label="true")
    return True


# INPUT: a run folder and a dataset name.  RETURNS: (metrics dict, row-normalised confusion matrix or None).  GOES TO: the table and plot functions.
def load_eval(run, dataset):
    d = Path(run) / f"eval_{dataset}"
    with open(d / "metrics.json") as f:
        m = json.load(f)
    cm = None
    if (d / "confusion.csv").exists():
        cm = pd.read_csv(d / "confusion.csv", index_col=0).loc[CLASSES, CLASSES].values.astype(float)
        if cm.sum() > 2 * len(CLASSES):  # counts -> row-normalised; files already normalised are used as-is
            # Divide each row by its total so a cell reads "of the true <row> faces, this share was predicted <column>".
            cm = cm / cm.sum(1, keepdims=True)
    return m, cm


# INPUT: the loaded evaluations.  RETURNS: one row per model with accuracy, macro-F1 and balanced accuracy on both datasets
# and the drop between them.  GOES TO: cross_dataset.csv and cross_dataset.md.
def overall_table(evals):
    rows = []
    for run, e in evals.items():
        f, r = e["fer2013"][0], e["rafdb"][0]
        # Balanced accuracy = the mean per-class recall (sklearn calls it the "macro avg" recall). Unlike plain
        # accuracy it is NOT inflated when one class dominates: RAF-DB is 39% happy, so a model that is good only at
        # "happy" still gets a decent plain accuracy but a poor balanced accuracy.
        bal = lambda m: m["per_class"]["macro avg"]["recall"]  # noqa: E731  (mean per-class recall)
        rows.append({
            "run": Path(run).name, "model": label(run),
            "fer_acc": f["acc"], "fer_macro_f1": f["macro_f1"], "fer_bal_acc": bal(f),
            "raf_acc": r["acc"], "raf_macro_f1": r["macro_f1"], "raf_bal_acc": bal(r),
        })
    df = pd.DataFrame(rows)
    # "Drop" = FER-2013 score minus RAF-DB score = how much performance is lost when the model meets new data.
    for k in ("acc", "macro_f1", "bal_acc"):
        df[f"{k}_drop"] = df[f"fer_{k}"] - df[f"raf_{k}"]
    return df


# INPUT: the loaded evaluations.  RETURNS: one row per model and emotion with F1 / recall on both datasets and the drop.  GOES TO: per_class.csv.
def per_class_table(evals):
    # Same comparison, but for each emotion separately: which emotions transfer well and which collapse?
    rows = []
    for run, e in evals.items():
        f, r = e["fer2013"][0]["per_class"], e["rafdb"][0]["per_class"]
        for c in CLASSES:
            rows.append({
                "run": Path(run).name, "model": label(run), "emotion": c,
                "fer_f1": f[c]["f1-score"], "raf_f1": r[c]["f1-score"], "f1_drop": f[c]["f1-score"] - r[c]["f1-score"],
                "fer_recall": f[c]["recall"], "raf_recall": r[c]["recall"],
                "recall_drop": f[c]["recall"] - r[c]["recall"],
                "fer_n": int(f[c]["support"]), "raf_n": int(r[c]["support"]),
            })
    return pd.DataFrame(rows)


# INPUT: the two tables, the focus models and an output path.  RETURNS: nothing; it writes cross_dataset.md.
def write_markdown(overall, per_class, focus, path):
    # Writes the human-readable results/cross_dataset/cross_dataset.md (tables only, generated from the CSVs).
    pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
    lines = [
        "# Cross-dataset results: trained on FER-2013, tested on RAF-DB",
        "",
        "Generated by `python -m src.cross_dataset`. RAF-DB is never used for training or model selection.",
        "Balanced accuracy is the mean per-class recall; unlike accuracy it is not inflated by RAF-DB being",
        "39% happy. Drops are FER-2013 minus RAF-DB (accuracy in points, F1 as a fraction).",
        "",
        "| Model | FER-2013 acc / F1 / bal. acc | RAF-DB acc / F1 / bal. acc | Acc drop | F1 drop | Bal. acc drop |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in overall.iterrows():
        lines.append(
            f"| {r.model} (`{r.run}`) | {pct(r.fer_acc)} / {r.fer_macro_f1:.3f} / {pct(r.fer_bal_acc)} "
            f"| {pct(r.raf_acc)} / {r.raf_macro_f1:.3f} / {pct(r.raf_bal_acc)} "
            f"| {100 * r.acc_drop:.1f} | {r.macro_f1_drop:.3f} | {100 * r.bal_acc_drop:.1f} |")
    lines += ["", "## Per-emotion F1 (FER-2013 -> RAF-DB)", "",
              "| Emotion | RAF-DB n | " + " | ".join(label(f) for f in focus) + " |",
              "|---|---|" + "---|" * len(focus)]
    pc = per_class.set_index(["run", "emotion"])
    for c in CLASSES:
        cells = []
        for f in focus:
            x = pc.loc[(Path(f).name, c)]
            cells.append(f"{x.fer_f1:.2f} -> {x.raf_f1:.2f} ({-x.f1_drop:+.2f})")
        lines.append(f"| {c} | {int(pc.loc[(Path(focus[0]).name, c)].raf_n)} | " + " | ".join(cells) + " |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# INPUT: the per-class table, focus models, output path.  RETURNS: nothing; it saves per_class_drop.png.
def plot_per_class_drop(per_class, focus, path):
    # Bar chart: for each emotion, how much F1 each focus model loses on RAF-DB (taller bar = worse transfer).
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(CLASSES))
    w = 0.8 / len(focus)
    pc = per_class.set_index(["run", "emotion"])
    for i, f in enumerate(focus):
        drops = [pc.loc[(Path(f).name, c)].f1_drop for c in CLASSES]
        ax.bar(x + (i - (len(focus) - 1) / 2) * w, drops, w * 0.9, color=SERIES[i], label=label(f))
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x, CLASSES)
    ax.set_ylabel("F1 drop (FER-2013 minus RAF-DB)", color=INK)
    ax.set_title("Which emotions transfer worst? Per-class F1 drop from FER-2013 to RAF-DB", color=INK)
    ax.grid(axis="y", color="#e4e3df", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# INPUT: the loaded evaluations, focus models, output path.  RETURNS: nothing; it saves confusion_compare.png.
def plot_confusion_compare(evals, focus, path):
    # One row per model with three panels: confusion on FER-2013, confusion on RAF-DB, and RAF-DB minus FER-2013.
    # In the difference panel, red cells are mistakes that become MORE common on the new dataset.
    fig, axes = plt.subplots(len(focus), 3, figsize=(15, 4.6 * len(focus)), squeeze=False)
    for row, f in enumerate(focus):
        fer, raf = evals[f]["fer2013"][1], evals[f]["rafdb"][1]
        panels = [(fer, "FER-2013 (in-domain)", "Blues", 0, 1), (raf, "RAF-DB (cross-dataset)", "Blues", 0, 1),
                  (raf - fer, "RAF-DB minus FER-2013", "RdBu_r", -0.6, 0.6)]
        for ax, (m, title, cmap, lo, hi) in zip(axes[row], panels):
            im = ax.imshow(m, cmap=cmap, vmin=lo, vmax=hi)
            for i in range(len(CLASSES)):
                for j in range(len(CLASSES)):
                    v = m[i, j]
                    # Write the number in white on dark cells and dark on light cells so it stays readable.
                    dark = (v > 0.55) if cmap == "Blues" else abs(v) > 0.35
                    ax.text(j, i, f"{v:+.2f}" if cmap != "Blues" else f"{v:.2f}", ha="center", va="center",
                            fontsize=8, color="white" if dark else INK)
            ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
            ax.set_yticks(range(len(CLASSES)), CLASSES)
            ax.set_title(f"{label(f)}: {title}", color=INK)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# INPUT: the loaded evaluations and the focus models.  RETURNS: a table of the mistakes that grow most on RAF-DB.  GOES TO: confusion_shift.csv.
def confusion_shift(evals, focus, top=8):
    # List the specific mistakes (true emotion -> wrongly predicted emotion) that grow the most from FER-2013 to
    # RAF-DB, e.g. "disgust is read as angry". Only off-diagonal cells (i != j) are mistakes.
    rows = []
    for f in focus:
        diff = evals[f]["rafdb"][1] - evals[f]["fer2013"][1]
        for i, t in enumerate(CLASSES):
            for j, p in enumerate(CLASSES):
                if i != j:
                    rows.append({"run": Path(f).name, "model": label(f), "true": t, "predicted": p,
                                 "fer_rate": evals[f]["fer2013"][1][i, j], "raf_rate": evals[f]["rafdb"][1][i, j],
                                 "increase": diff[i, j]})
    df = pd.DataFrame(rows).sort_values(["run", "increase"], ascending=[True, False])
    return df.groupby("run").head(top)  # keep the `top` biggest increases per model


# FLOW: keep only runs evaluated on both datasets -> (optionally) rebuild confusion.csv -> load everything -> write the tables
#  -> draw the charts -> print a summary.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[f"results/{r}" for r in DEFAULT_RUNS])
    ap.add_argument("--focus", nargs="+", default=[f"results/{r}" for r in DEFAULT_FOCUS],
                    help="models for the per-class chart and confusion comparison (at most 4)")
    ap.add_argument("--out", default="results/cross_dataset")
    ap.add_argument("--from-predictions", action="store_true",
                    help="(re)build confusion.csv from a local predictions.csv where one exists")
    args = ap.parse_args()

    # A run can only be compared if it was evaluated on BOTH datasets; other runs are skipped (and reported).
    runs = [r for r in args.runs if all((Path(r) / f"eval_{d}" / "metrics.json").exists() for d in DATASETS)]
    skipped = sorted(set(args.runs) - set(runs))
    if skipped:
        print("skipping (no eval on both datasets):", ", ".join(skipped))
    if args.from_predictions:
        for r in runs:
            for d in DATASETS:
                if confusion_from_predictions(Path(r) / f"eval_{d}"):
                    print("confusion.csv <-", Path(r) / f"eval_{d}" / "predictions.csv")

    evals = {r: {d: load_eval(r, d) for d in DATASETS} for r in runs}
    # Only 4 chart colours are defined, so at most 4 focus models can be drawn.
    focus = [f for f in args.focus if f in evals][:len(SERIES)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    overall = overall_table(evals)
    per_class = per_class_table(evals)
    overall.to_csv(out / "cross_dataset.csv", index=False, float_format="%.4f")
    per_class.to_csv(out / "per_class.csv", index=False, float_format="%.4f")
    write_markdown(overall, per_class, focus, out / "cross_dataset.md")
    plot_per_class_drop(per_class, focus, out / "per_class_drop.png")

    # The confusion comparison needs confusion.csv for BOTH datasets; models without it are skipped, not an error.
    with_cm = [f for f in focus if all(evals[f][d][1] is not None for d in DATASETS)]
    if with_cm:
        plot_confusion_compare(evals, with_cm, out / "confusion_compare.png")
        confusion_shift(evals, with_cm).to_csv(out / "confusion_shift.csv", index=False, float_format="%.3f")
    missing = sorted(set(focus) - set(with_cm))
    if missing:
        print("no confusion.csv (confusion comparison skipped):", ", ".join(missing))

    print(overall[["model", "fer_acc", "raf_acc", "acc_drop", "fer_macro_f1", "raf_macro_f1",
                   "macro_f1_drop", "bal_acc_drop"]].round(3).to_string(index=False))
    print("->", out)


if __name__ == "__main__":
    main()
