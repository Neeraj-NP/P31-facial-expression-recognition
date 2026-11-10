"""Helpers shared by train / evaluate / explain."""
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .datasets import FrameDataset, build_transforms, fer2013_frames, rafdb_frame
from .models import build_model

ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_frames(cfg, dataset):
    """Returns {split: DataFrame} for the requested dataset."""
    d = cfg["data"]
    if dataset == "fer2013":
        train, val, test = fer2013_frames(ROOT / d["fer2013_root"], d["val_frac"], cfg["seed"])
        return {"train": train, "val": val, "test": test}
    if dataset == "rafdb":
        root = ROOT / d["rafdb_root"]
        return {"train": rafdb_frame(root, "train"), "test": rafdb_frame(root, "test")}
    raise ValueError(dataset)


def make_loader(cfg, df, train):
    d = cfg["data"]
    tf = build_transforms(d["img_size"], d["channels"], train,
                          augment=d.get("augment", True), erasing=d.get("random_erasing", 0.25))
    return DataLoader(
        FrameDataset(df, tf),
        batch_size=cfg["train"]["batch_size"],
        shuffle=train,
        num_workers=d["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
    )


def model_kwargs(m):
    """Optional architecture settings from the config's model section."""
    return {k: m[k] for k in ("dropout", "block_dropout", "freeze_backbone") if k in m}


def load_run(run_dir, device):
    """Rebuild a trained model from results/<run>/ (config.yaml + best.pt)."""
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    m = cfg["model"]
    model = build_model(m["name"], m["num_classes"], pretrained=False, in_channels=cfg["data"]["channels"],
                        **model_kwargs(m))
    state = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(state["model"])
    return model.to(device).eval(), cfg
