"""Aggregate subgroup bias audits across models: gap table + interval/p-value checks + plots.

Uses each run's eval_<dataset>/subgroup_audit.csv (produced by src.evaluate).

    python -m src.bias_report --runs results/cnn results/resnet results/vit --dataset rafdb
"""
import argparse
import ast
import math
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


def two_proportion_p(acc1, n1, acc2, n2):
    """Two-sided p-value of a pooled two-proportion z-test between two groups' accuracies."""
    pooled = (acc1 * n1 + acc2 * n2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    return math.erfc(abs(acc1 - acc2) / se / math.sqrt(2)) if se else 1.0


def gap_table(df):
    """Best-worst accuracy gap per model/attribute, with two ways to judge it:
    - ci_separated: the best and worst groups' 95% bootstrap CIs do not overlap. Conservative, so a
      gap can be real even when this is False.
    - p_value: two-proportion z-test between the best and worst group. Neither measure accounts for
      the best and worst groups being picked out of several, so both flatter the gap a little."""
    rows = []
    # "unsure" (n=199) is not a gender group in the female-vs-male comparison; it stays in the per-group tables
    df = df[~((df["attribute"] == "gender") & (df["group"] == "unsure"))]
    for (model, attribute), g in df.groupby(["model", "attribute"]):
        best = g.loc[g["acc"].idxmax()]
        worst = g.loc[g["acc"].idxmin()]
        best_lo, _ = best["acc_ci"]
        _, worst_hi = worst["acc_ci"]
        rows.append({
            "model": model, "attribute": attribute,
            "best_group": best["group"], "best_acc": best["acc"],
            "worst_group": worst["group"], "worst_acc": worst["acc"],
            "gap": best["acc"] - worst["acc"], "ci_separated": worst_hi < best_lo,
            "p_value": two_proportion_p(best["acc"], best["n"], worst["acc"], worst["n"]),
        })
    return pd.DataFrame(rows).sort_values(["attribute", "gap"], ascending=[True, False])


def plot_attribute(df, attribute, out):
    sub = df[df["attribute"] == attribute]
    models, groups = list(sub["model"].unique()), list(sub["group"].unique())
    age_order = ["0-3", "4-19", "20-39", "40-69", "70+"]
    if attribute == "age":
        groups = [g for g in age_order if g in groups]
    width = 0.8 / len(models)
    x = range(len(groups))

    fig, ax = plt.subplots(figsize=(min(12, 0.5 * len(groups) * len(models) + 3), 4.5))
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
