"""Aggregate subgroup bias audits across models: gap table + interval/p-value checks + plots.

Uses each run's eval_<dataset>/subgroup_audit.csv (produced by src.evaluate).

    python -m src.bias_report --runs results/cnn results/resnet results/vit --dataset rafdb
"""
# =====================================================================
# FILE ROLE: combines the fairness audits of several models into one gap table and bar charts.
# HOW IT IS RUN: python -m src.bias_report --runs results/cnn_best results/resnet50 results/vit
# INPUT: each run's eval_rafdb/subgroup_audit.csv (written by evaluate.py).
# OUTPUT: results/bias_report/ with combined_subgroup_audit.csv, gap_table.csv and one chart per attribute.
# GOES TO: the report tables and slides 8-9 use gap_table.csv; the results notebook (02_results) imports gap_table / load_audit.
# =====================================================================
import argparse
import ast
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# INPUT: a results/<run> folder and a dataset name.  RETURNS: that run's audit table with its confidence intervals as tuples.
# GOES TO: main() (and notebooks/02_results.ipynb), which stacks all runs together.
def load_audit(run_dir, dataset):
    run_dir = Path(run_dir)
    model_name = run_dir.name
    df = pd.read_csv(run_dir / f"eval_{dataset}" / "subgroup_audit.csv")
    # The confidence intervals were saved as text like "(0.61, 0.67)"; literal_eval turns them back into tuples.
    df["acc_ci"] = df["acc_ci"].apply(ast.literal_eval)
    df["macro_f1_ci"] = df["macro_f1_ci"].apply(ast.literal_eval)
    df.insert(0, "model", model_name)  # label every row with its model so all runs can be stacked into one table
    df.insert(1, "run", run_dir.name)
    return df


# INPUT: accuracy and number of images for two groups.  RETURNS: a two-sided p-value between 0 and 1.
# GOES TO: gap_table(), which stores it in the p_value column.
def two_proportion_p(acc1, n1, acc2, n2):
    """Two-sided p-value of a pooled two-proportion z-test between two groups' accuracies."""
    # Question answered: "if both groups truly had the same accuracy, how likely is a difference this big
    # just by chance?" A small p (e.g. < 0.05) means the gap is unlikely to be pure luck.
    # pooled = the accuracy if the two groups were one; se = the standard error of the difference;
    # the last line converts the z-score |acc1 - acc2| / se into a two-sided p-value (erfc is the normal tail).
    # Worked example (ResNet-50, gender): women 67.5% of 1,620 images vs men 59.4% of 1,249 images.
    #   pooled = (0.675 * 1620 + 0.594 * 1249) / 2869 ~ 0.640
    #   se     = sqrt(0.640 * 0.360 * (1/1620 + 1/1249)) ~ 0.018
    #   z      = (0.675 - 0.594) / 0.018 ~ 4.5  -> p ~ 8e-6, so a gap this large is very unlikely to be chance.
    pooled = (acc1 * n1 + acc2 * n2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    return math.erfc(abs(acc1 - acc2) / se / math.sqrt(2)) if se else 1.0


# INPUT: the combined audit table of all models.  RETURNS: one row per model and attribute with best group, worst group, gap,
# ci_separated and p_value.  GOES TO: gap_table.csv and the printed table.
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
        # The gap is between the highest- and lowest-scoring group of each attribute.
        best = g.loc[g["acc"].idxmax()]
        worst = g.loc[g["acc"].idxmin()]
        # "Separated" = the lower end of the best group's interval is above the upper end of the worst group's.
        # Example: ResNet-50 women [0.651, 0.698] vs men [0.568, 0.619]: 0.619 < 0.651, so the intervals are separated.
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


# INPUT: the combined table, one attribute name and an output path.  RETURNS: nothing; it saves a bar chart file.
def plot_attribute(df, attribute, out):
    # Grouped bar chart: one cluster per group (e.g. female / male), one bar per model, with error bars = 95% CI.
    sub = df[df["attribute"] == attribute]
    models, groups = list(sub["model"].unique()), list(sub["group"].unique())
    age_order = ["0-3", "4-19", "20-39", "40-69", "70+"]
    if attribute == "age":
        # Age groups are ordered, so show them young -> old instead of alphabetically.
        groups = [g for g in age_order if g in groups]
    width = 0.8 / len(models)
    x = range(len(groups))

    fig, ax = plt.subplots(figsize=(min(12, 0.5 * len(groups) * len(models) + 3), 4.5))
    for i, model in enumerate(models):
        m = sub[sub["model"] == model].set_index("group").reindex(groups)
        acc = m["acc"].values
        lo = [ci[0] for ci in m["acc_ci"]]
        hi = [ci[1] for ci in m["acc_ci"]]
        # matplotlib wants error bars as distances below and above the bar top
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


# FLOW: load every run's audit -> stack into one table -> write combined_subgroup_audit.csv -> compute and write gap_table.csv
#  -> draw one chart per attribute.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="results/<run> dirs, one per model")
    ap.add_argument("--dataset", default="rafdb")
    args = ap.parse_args()

    # Stack every model's subgroup audit into one big table.
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
