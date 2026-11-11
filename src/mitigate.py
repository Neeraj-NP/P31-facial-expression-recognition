"""Mitigation experiment: fine-tune a trained model on RAF-DB with subgroup-balanced sampling.

    python -m src.mitigate --run results/resnet50 --balance-by gender --epochs 10
    python -m src.mitigate --run results/resnet50 --balance-by none --epochs 10   # control

Produces results/<run>_mit_<attribute>/ in the same layout as a normal run (config.yaml,
history.csv, best.pt), so src.evaluate / src.explain / src.bias_report work on it
unmodified. Fine-tuning on RAF-DB alone raises RAF-DB accuracy, so the `none` control is
needed to tell the effect of balancing apart from the effect of the extra training:

    python -m src.evaluate --run results/resnet50_mit_gender --dataset rafdb
    python -m src.bias_report --runs results/resnet50 results/resnet50_mit_none results/resnet50_mit_gender --dataset rafdb
"""
import argparse
import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler

from .common import ROOT, get_device, load_run, make_loader, seed_everything
from .datasets import rafdb_frame
from .train import run_epoch


def subgroup_sampler(df, attribute):
    """Inverse-frequency sampling weights so every group of `attribute` is drawn equally often."""
    counts = df[attribute].value_counts()
    weights = df[attribute].map(lambda g: 1.0 / counts[g]).to_numpy()
    return WeightedRandomSampler(weights, num_samples=len(df), replacement=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="results/<run> of the already-trained model")
    ap.add_argument("--balance-by", choices=["gender", "race", "age", "none"], default="gender",
                    help="'none' = same fine-tuning with ordinary shuffling, the control for the balancing effect")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-5, help="low LR: fine-tuning, not retraining from scratch")
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()

    device = get_device()
    model, cfg = load_run(args.run, device)
    seed_everything(cfg["seed"])

    root = ROOT / cfg["data"]["rafdb_root"]
    balance = args.balance_by != "none"
    df = rafdb_frame(root, "train")
    if balance:
        df = df.dropna(subset=[args.balance_by])
    df = df.reset_index(drop=True)
    train_df, val_df = train_test_split(
        df, test_size=args.val_frac, stratify=df["label"], random_state=cfg["seed"]
    )
    print(f"mitigation train={len(train_df)} val={len(val_df)} balance_by={args.balance_by} device={device}")
    if balance:
        print(train_df[args.balance_by].value_counts().to_string())

    sampler = subgroup_sampler(train_df, args.balance_by) if balance else None
    train_loader = make_loader(cfg, train_df, train=True, sampler=sampler)
    val_loader = make_loader(cfg, val_df, train=False)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["train"].get("label_smoothing", 0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=cfg["train"]["weight_decay"])
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: 0.5 * (1 + math.cos(math.pi * s / max(1, total_steps)))
    )
    scaler = torch.amp.GradScaler() if cfg["train"].get("amp", True) and device.type == "cuda" else None

    out = ROOT / "results" / f"{Path(args.run).name}_mit_{args.balance_by}"
    out.mkdir(parents=True, exist_ok=True)
    mcfg = {**cfg, "mitigation": {
        "source_run": str(args.run), "balance_by": args.balance_by, "epochs": args.epochs, "lr": args.lr,
    }}
    yaml.safe_dump(mcfg, open(out / "config.yaml", "w"))

    best_f1, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, device, criterion, optimizer, scaler, scheduler)
        va = run_epoch(model, val_loader, device, criterion, scaler=scaler)
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()},
                        **{f"val_{k}": v for k, v in va.items()}})
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print(f"[{epoch:03d}] train loss {tr['loss']:.3f} acc {tr['acc']:.3f} | "
              f"val loss {va['loss']:.3f} acc {va['acc']:.3f} f1 {va['macro_f1']:.3f}")
        if va["macro_f1"] > best_f1:
            best_f1 = va["macro_f1"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": va}, out / "best.pt")

    print(f"best val macro-F1 {best_f1:.4f} -> {out / 'best.pt'}")
    print(f"next: python -m src.evaluate --run {out.relative_to(ROOT)} --dataset rafdb")


if __name__ == "__main__":
    main()
