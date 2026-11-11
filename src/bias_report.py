"""Aggregate subgroup bias audits across models: gap table + significance + plots.

Uses each run's eval_<dataset>/subgroup_audit.csv (produced by src.evaluate).

    python -m src.bias_report --runs results/cnn results/resnet results/vit --dataset rafdb
"""
import argparse
import ast
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_audit(run_dir, dataset):
    run_dir = Path(run_dir)
    model_name = run_dir.name
    df = pd.read_csv(run_dir / f"eval_{dataset}" / "subgroup_audit.csv")
    df["acc_ci"] = df["acc_ci"].apply(ast.literal_eval)
    df["macro_f1_ci"] = df["macro_f1_ci"].apply(ast.literal_eval)
    df.insert(0, "model", model_name)
    df.insert(1, "run", run_dir.name)
    return df


def gap_table(df):
    """Best-worst accuracy gap per model/attribute. Flagged significant when the best and
    worst groups' 95% bootstrap CIs don't overlap."""
    rows = []
    for (model, attribute), g in df.groupby(["model", "attribute"]):
        best = g.loc[g["acc"].idxmax()]
        worst = g.loc[g["acc"].idxmin()]
        best_lo, _ = best["acc_ci"]
        _, worst_hi = worst["acc_ci"]
        rows.append({
            "model": model, "attribute": attribute,
            "best_group": best["group"], "best_acc": best["acc"],
            "worst_group": worst["group"], "worst_acc": worst["acc"],
            "gap": best["acc"] - worst["acc"], "significant": worst_hi < best_lo,
        })
    return pd.DataFrame(rows).sort_values(["attribute", "gap"], ascending=[True, False])


def plot_attribute(df, attribute, out):
    sub = df[df["attribute"] == attribute]
    models, groups = list(sub["model"].unique()), list(sub["group"].unique())
    width = 0.8 / len(models)
    x = range(len(groups))

    fig, ax = plt.subplots(figsize=(1.6 * len(groups) * max(len(models), 1) + 2, 4))
    for i, model in enumerate(models):
        m = sub[sub["model"] == model].set_index("group").reindex(groups)
        acc = m["acc"].values
        lo = [ci[0] for ci in m["acc_ci"]]
        hi = [ci[1] for ci in m["acc_ci"]]
        err = [[a - l for a, l in zip(acc, lo)], [h - a for a, h in zip(acc, hi)]]
        pos = [xi + i * width for xi in x]
        ax.bar(pos, acc, width=width, yerr=err, capsize=3, label=model)

    ax.set_xticks([xi + width * (len(models) - 1) / 2 for xi in x])
    ax.set_xticklabels(groups)
    ax.set_ylabel("accuracy")
    ax.set_title(f"RAF-DB accuracy by {attribute} (95% bootstrap CI)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="results/<run> dirs, one per model")
    ap.add_argument("--dataset", default="rafdb")
    args = ap.parse_args()

    combined = pd.concat([load_audit(r, args.dataset) for r in args.runs], ignore_index=True)

    out = Path("results/bias_report")
    out.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out / "combined_subgroup_audit.csv", index=False)

    gaps = gap_table(combined)
    gaps.to_csv(out / "gap_table.csv", index=False)
    print(gaps.to_string(index=False))

    for attribute in combined["attribute"].unique():
        plot_attribute(combined, attribute, out / f"{attribute}_by_model.png")

    print(f"-> {out}")


if __name__ == "__main__":
    main()
