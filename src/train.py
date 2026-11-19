"""Train any model on FER-2013.

    python -m src.train --config configs/cnn.yaml
    python -m src.train --config configs/resnet.yaml --epochs 2   # quick check
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm.optim
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from .common import ROOT, get_device, get_frames, load_config, make_loader, model_kwargs, seed_everything
from .models import build_model


def class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    w = counts.sum() / (num_classes * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)


def build_optimizer(model, t):
    """AdamW; with train.layer_decay (ViT) earlier blocks get a smaller LR than the head."""
    if not t.get("layer_decay"):
        return torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    opt = timm.optim.create_optimizer_v2(
        model, opt="adamw", lr=t["lr"], weight_decay=t["weight_decay"], layer_decay=t["layer_decay"])
    # timm only stores the per-group scale in "lr_scale" and applies it in its own scheduler.
    # We use torch's LambdaLR, so bake the scale into each group's lr here.
    for g in opt.param_groups:
        g["lr"] = t["lr"] * g.get("lr_scale", 1.0)
    return opt


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None, scheduler=None):
    train = optimizer is not None
    model.train(train)
    total_loss, preds, targets = 0.0, [], []
    for x, y, _ in tqdm(loader, leave=False, desc="train" if train else "eval"):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train), torch.autocast(device.type, enabled=scaler is not None):
            logits = model(x)
            loss = criterion(logits, y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            scheduler.step()
        total_loss += loss.item() * x.size(0)
        preds.append(logits.argmax(1).cpu())
        targets.append(y.cpu())
    preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    return {
        "loss": total_loss / len(targets),
        "acc": accuracy_score(targets, preds),
        "macro_f1": f1_score(targets, preds, average="macro"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, help="override config epochs")
    ap.add_argument("--run-name", help="override run name (default: model name + timestamp)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    seed_everything(cfg["seed"])
    device = get_device()
    t = cfg["train"]

    run_name = args.run_name or f"{cfg['model']['name']}_{time.strftime('%Y%m%d-%H%M%S')}"
    out = ROOT / "results" / run_name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    frames = get_frames(cfg, "fer2013")
    train_loader = make_loader(cfg, frames["train"], train=True)
    val_loader = make_loader(cfg, frames["val"], train=False)
    print(f"train={len(frames['train'])} val={len(frames['val'])} test={len(frames['test'])} device={device}")

    m = cfg["model"]
    model = build_model(m["name"], m["num_classes"], m.get("pretrained", True),
                        in_channels=cfg["data"]["channels"], **model_kwargs(m)).to(device)

    weight = class_weights(frames["train"]["label"].values, m["num_classes"]).to(device) \
        if t.get("class_weights") else None
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=t.get("label_smoothing", 0.0))
    optimizer = build_optimizer(model, t)

    # linear warmup then cosine decay, stepped per batch
    total_steps = t["epochs"] * len(train_loader)
    warmup = t.get("warmup_epochs", 0) * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (s + 1) / max(1, warmup) if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup))),
    )
    scaler = torch.amp.GradScaler() if t.get("amp", True) and device.type == "cuda" else None

    best_f1, patience, history = -1.0, 0, []
    for epoch in range(1, t["epochs"] + 1):
        tr = run_epoch(model, train_loader, device, criterion, optimizer, scaler, scheduler)
        va = run_epoch(model, val_loader, device, criterion, scaler=scaler)
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()},
                        **{f"val_{k}": v for k, v in va.items()}})
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print(f"[{epoch:03d}] train loss {tr['loss']:.3f} acc {tr['acc']:.3f} | "
              f"val loss {va['loss']:.3f} acc {va['acc']:.3f} f1 {va['macro_f1']:.3f}")

        if va["macro_f1"] > best_f1:
            best_f1, patience = va["macro_f1"], 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": va}, out / "best.pt")
        else:
            patience += 1
            if patience >= t.get("early_stop_patience", 10):
                print("early stopping")
                break

    print(f"best val macro-F1 {best_f1:.4f} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
