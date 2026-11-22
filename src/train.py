"""Train any model on FER-2013.

    python -m src.train --config configs/cnn.yaml
    python -m src.train --config configs/resnet.yaml --epochs 2   # quick check
"""
# =====================================================================
# FILE ROLE: the training script. Given a YAML config it builds the data loaders and the model, trains for several
# epochs, checks validation macro-F1 after every epoch and saves the best weights.
# HOW IT IS RUN: python -m src.train --config configs/resnet50.yaml --run-name resnet50
# INPUT: a YAML config + the FER-2013 images.   OUTPUT: results/<run>/ with config.yaml, history.csv and best.pt.
# GOES TO: evaluate.py and explain.py load best.pt next; mitigate.py reuses run_epoch() from here.
# =====================================================================
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


# INPUT: all training labels.  RETURNS: a tensor of 7 weights.  GOES TO: nn.CrossEntropyLoss(weight=...) in main().
def class_weights(labels, num_classes):
    # FER-2013 is very imbalanced (happy has ~16x more images than disgust). Without correction the model
    # can score well by mostly predicting the big classes. Inverse-frequency weights make a mistake on a rare
    # class cost more in the loss: weight = total / (num_classes * count_of_that_class).
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    w = counts.sum() / (num_classes * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)


# INPUT: the model and the "train" part of the config.  RETURNS: an optimizer object.  GOES TO: main() and the LR scheduler.
def build_optimizer(model, t):
    """AdamW; with train.layer_decay (ViT) earlier blocks get a smaller LR than the head."""
    # AdamW = Adam with proper weight decay (a small penalty on large weights that reduces overfitting).
    if not t.get("layer_decay"):
        return torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    # Layer-wise learning-rate decay (used in one ViT tuning run): early layers hold general features, so they
    # get a smaller learning rate than the later layers and the new head.
    opt = timm.optim.create_optimizer_v2(
        model, opt="adamw", lr=t["lr"], weight_decay=t["weight_decay"], layer_decay=t["layer_decay"])
    # timm only stores the per-group scale in "lr_scale" and applies it in its own scheduler.
    # We use torch's LambdaLR, so bake the scale into each group's lr here.
    for g in opt.param_groups:
        g["lr"] = t["lr"] * g.get("lr_scale", 1.0)
    return opt


# WHAT: one full pass over a DataLoader (training when an optimizer is given, otherwise just evaluation).
# INPUT: model, DataLoader, loss function and (for training) optimizer, scaler, scheduler.
# RETURNS: a dict {loss, acc, macro_f1}.  GOES TO: main() prints it, saves it to history.csv and uses val macro_f1 to pick
# the best epoch. mitigate.py calls this same function.
def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None, scheduler=None):
    # One pass over a DataLoader. The same function does training (optimizer given) and evaluation
    # (optimizer=None), so both report their numbers in exactly the same way.
    train = optimizer is not None
    model.train(train)  # train mode switches dropout/BatchNorm on; eval mode switches them off
    total_loss, preds, targets = 0.0, [], []
    for x, y, _ in tqdm(loader, leave=False, desc="train" if train else "eval"):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        # set_grad_enabled(train): no gradients needed when only evaluating (faster, less memory).
        # autocast: mixed precision (float16 where safe) on GPU, which speeds training up; only on if a scaler exists.
        with torch.set_grad_enabled(train), torch.autocast(device.type, enabled=scaler is not None):
            logits = model(x)
            loss = criterion(logits, y)
        if train:
            optimizer.zero_grad(set_to_none=True)  # clear the gradients left from the previous batch
            if scaler:
                # GradScaler multiplies the loss so tiny float16 gradients do not underflow to zero.
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()   # compute gradients
                optimizer.step()  # update the weights
            # The learning-rate schedule moves forward after EVERY batch, not every epoch.
            scheduler.step()
        total_loss += loss.item() * x.size(0)
        preds.append(logits.argmax(1).cpu())  # predicted class = the output with the highest score
        targets.append(y.cpu())
    preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    # Accuracy is reported, but MACRO-F1 (the average F1 over the 7 classes, each class counted equally) is the
    # main metric because accuracy rewards ignoring rare classes on an imbalanced dataset.
    return {
        "loss": total_loss / len(targets),
        "acc": accuracy_score(targets, preds),
        "macro_f1": f1_score(targets, preds, average="macro"),
    }


# FLOW: read config -> make the results folder -> load data -> build model, loss, optimizer, scheduler
#  -> for each epoch: train, validate, log -> save best.pt when validation macro-F1 improves -> stop early if it stalls.
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

    # Every run gets its own folder results/<run_name>/ and keeps a copy of the exact config used.
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

    # Class weights are computed from the TRAINING labels only (never from validation/test) and are optional.
    weight = class_weights(frames["train"]["label"].values, m["num_classes"]).to(device) \
        if t.get("class_weights") else None
    # Cross-entropy is the standard classification loss. Label smoothing (e.g. 0.1) softens the "hard" 0/1
    # targets slightly, which stops the model becoming over-confident and helps with FER-2013's noisy labels.
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=t.get("label_smoothing", 0.0))
    optimizer = build_optimizer(model, t)

    # linear warmup then cosine decay, stepped per batch
    # Warmup: start with a tiny learning rate and ramp up, so the first updates do not wreck the pretrained weights.
    # Cosine decay: then lower the rate smoothly to ~0, so the model settles into a good solution at the end.
    total_steps = t["epochs"] * len(train_loader)
    warmup = t.get("warmup_epochs", 0) * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (s + 1) / max(1, warmup) if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup))),
    )
    # Mixed-precision scaler only exists on a CUDA GPU (it is not used on CPU).
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

        # Model selection: keep only the weights of the epoch with the best VALIDATION macro-F1. The test set
        # is never used to choose anything.
        if va["macro_f1"] > best_f1:
            best_f1, patience = va["macro_f1"], 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": va}, out / "best.pt")
        else:
            # Early stopping: if validation macro-F1 has not improved for `early_stop_patience` epochs, stop.
            # The model has started to overfit (train score keeps rising, validation does not).
            patience += 1
            if patience >= t.get("early_stop_patience", 10):
                print("early stopping")
                break

    print(f"best val macro-F1 {best_f1:.4f} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
