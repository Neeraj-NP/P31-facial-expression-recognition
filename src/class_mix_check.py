"""Does an accuracy gap between two groups come from a different emotion mix, or from the model?

Reads <run>/eval_rafdb/predictions.csv (written by src.evaluate; not committed because it holds
per-image RAF-DB labels) and compares, for one attribute, each group's emotion mix and per-emotion
accuracy. It also re-weights the lower-accuracy group's per-emotion accuracy to the other group's
emotion mix: the part of the gap that remains is a within-emotion difference.

    python -m src.class_mix_check --run results/resnet50 --attribute gender --a female --b male
"""
import argparse
from pathlib import Path

import pandas as pd

from .datasets import CLASSES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--attribute", default="gender", choices=["gender", "race", "age"])
    ap.add_argument("--a", default="female", help="reference group (the better-scoring one)")
    ap.add_argument("--b", default="male", help="group whose accuracy is re-weighted to a's emotion mix")
    args = ap.parse_args()

    d = pd.read_csv(Path(args.run) / "eval_rafdb" / "predictions.csv")
    d = d[d[args.attribute].isin([args.a, args.b])].copy()
    d["ok"] = d["label"] == d["pred"]
    d["emotion"] = d["label"].map(dict(enumerate(CLASSES)))

    mix = pd.crosstab(d[args.attribute], d["emotion"], normalize="index").mul(100).round(1)[CLASSES]
    acc = d.pivot_table(index=args.attribute, columns="emotion", values="ok").mul(100).round(1)[CLASSES]
    print("emotion mix (% of each group's images)")
    print(mix.to_string())
    print("\naccuracy per emotion (%)")
    print(acc.to_string())

    mix_a = d[d[args.attribute] == args.a]["emotion"].value_counts(normalize=True)
    b_acc = d[d[args.attribute] == args.b].groupby("emotion")["ok"].mean()
    a_overall = d[d[args.attribute] == args.a]["ok"].mean() * 100
    b_overall = d[d[args.attribute] == args.b]["ok"].mean() * 100
    b_reweighted = (b_acc * mix_a).sum() * 100
    gap = a_overall - b_overall
    print(f"\n{args.a}: {a_overall:.1f}%   {args.b}: {b_overall:.1f}%   gap {gap:.1f} pts")
    print(f"{args.b} re-weighted to {args.a}'s emotion mix: {b_reweighted:.1f}% "
          f"(closes {b_reweighted - b_overall:.1f} of {gap:.1f} pts)")


if __name__ == "__main__":
    main()
