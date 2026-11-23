"""Does an accuracy gap between two groups come from a different emotion mix, or from the model?

Reads <run>/eval_rafdb/predictions.csv (written by src.evaluate; not committed because it holds
per-image RAF-DB labels) and compares, for one attribute, each group's emotion mix and per-emotion
accuracy. It also re-weights the lower-accuracy group's per-emotion accuracy to the other group's
emotion mix: the part of the gap that remains is a within-emotion difference.

    python -m src.class_mix_check --run results/resnet50 --attribute gender --a female --b male
"""
# =====================================================================
# FILE ROLE: a small diagnostic for the gender gap. It asks whether the gap comes from the two groups having different emotion
# mixes, or from the model being worse for one group even within the same emotion.
# HOW IT IS RUN: python -m src.class_mix_check --run results/resnet50 --attribute gender --a female --b male
# INPUT: results/<run>/eval_rafdb/predictions.csv (local only, never committed).  OUTPUT: printed tables, no files written.
# GOES TO: the "Why is accuracy lower for men?" section of the report quotes this result.
# =====================================================================
import argparse
from pathlib import Path

import pandas as pd

from .datasets import CLASSES


# FLOW: load predictions -> keep the two groups -> compute each group's emotion mix and per-emotion accuracy -> re-weight group b
#  to group a's mix and print how much of the gap that explains.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--attribute", default="gender", choices=["gender", "race", "age"])
    ap.add_argument("--a", default="female", help="reference group (the better-scoring one)")
    ap.add_argument("--b", default="male", help="group whose accuracy is re-weighted to a's emotion mix")
    args = ap.parse_args()

    d = pd.read_csv(Path(args.run) / "eval_rafdb" / "predictions.csv")
    d = d[d[args.attribute].isin([args.a, args.b])].copy()
    d["ok"] = d["label"] == d["pred"]  # True when the prediction was correct
    d["emotion"] = d["label"].map(dict(enumerate(CLASSES)))

    # IDEA: groups can differ in accuracy simply because they show different emotions (e.g. more "happy" faces,
    # which are easy). This script separates that "emotion mix" effect from a real difference in model quality.
    # mix = share of each emotion within each group; acc = accuracy per emotion within each group.
    mix = pd.crosstab(d[args.attribute], d["emotion"], normalize="index").mul(100).round(1)[CLASSES]
    acc = d.pivot_table(index=args.attribute, columns="emotion", values="ok").mul(100).round(1)[CLASSES]
    print("emotion mix (% of each group's images)")
    print(mix.to_string())
    print("\naccuracy per emotion (%)")
    print(acc.to_string())

    # Re-weighting: take group B's accuracy for each emotion, but weight the emotions by group A's mix.
    # That is the accuracy B WOULD have if it had A's emotion mix. Whatever gap remains afterwards is NOT explained
    # by the emotion mix, so it is a within-emotion difference.
    # Q: Why re-weight men's accuracy to the women's mix and not the reverse? A: Either direction works (it is a
    #    standardisation); we use the better-scoring group as the reference so the result reads as "how much of
    #    the gap would close if men showed the same emotions as women".
    mix_a = d[d[args.attribute] == args.a]["emotion"].value_counts(normalize=True)
    b_acc = d[d[args.attribute] == args.b].groupby("emotion")["ok"].mean()
    a_overall = d[d[args.attribute] == args.a]["ok"].mean() * 100
    b_overall = d[d[args.attribute] == args.b]["ok"].mean() * 100
    # Result for ResNet-50: men 59.4%, re-weighted to the women's emotion mix 61.5%, so the mix closes only 2.1 of
    # the 8.1-point gap (about a quarter); the rest is lower accuracy within the same emotions.
    b_reweighted = (b_acc * mix_a).sum() * 100
    gap = a_overall - b_overall
    print(f"\n{args.a}: {a_overall:.1f}%   {args.b}: {b_overall:.1f}%   gap {gap:.1f} pts")
    print(f"{args.b} re-weighted to {args.a}'s emotion mix: {b_reweighted:.1f}% "
          f"(closes {b_reweighted - b_overall:.1f} of {gap:.1f} pts)")


if __name__ == "__main__":
    main()
