"""Helpers shared by train / evaluate / explain."""
# =====================================================================
# FILE ROLE: small helper functions shared by train.py, evaluate.py, explain.py and mitigate.py, so they all load the
# config, the data and the model in exactly the same way.
# BIG PICTURE OF THE PROJECT (there is no web server or API here; everything is command-line scripts):
#   configs/*.yaml  -> train.py     -> results/<run>/  (best.pt, history.csv, config.yaml)
#   results/<run>/  -> evaluate.py  -> eval_fer2013/ and eval_rafdb/  (metrics.json, confusion.csv, subgroup_audit.csv, predictions.csv)
#   those eval files -> bias_report.py (fairness tables/charts) and cross_dataset.py (FER-2013 vs RAF-DB comparison)
#   results/<run>/best.pt -> explain.py (Grad-CAM) and mitigate.py (fine-tune on RAF-DB, writes a new run folder)
# =====================================================================
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .datasets import FrameDataset, build_transforms, fer2013_frames, rafdb_frame
from .models import build_model

# Project root folder (the parent of src/). Dataset paths in the YAML configs are relative to this.
ROOT = Path(__file__).resolve().parents[1]


# INPUT: path to a YAML file.  RETURNS: a Python dict (cfg).
# GOES TO: every script, which reads cfg["data"], cfg["model"] and cfg["train"] from it.
def load_config(path):
    # Every experiment is described by a YAML file (model, image size, learning rate, ...), so runs differ only
    # in config and are easy to reproduce.
    with open(path) as f:
        return yaml.safe_load(f)


# INPUT: an integer seed.  RETURNS: nothing; it sets the random state of Python, NumPy and PyTorch globally.
def seed_everything(seed):
    # Fix every random number generator so a rerun gives (nearly) the same result.
    # It makes the experiments reproducible; GPU maths can still differ slightly between machines.
    random.seed(seed)
    np.random.seed(seed)
    # Q: Why seed 42 and only one seed? A: 42 is an arbitrary fixed value used by every run. One seed per run was
    #    all our compute allowed, which is why we call differences of about one point noise (see the limitations).
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# RETURNS: a torch device ("cuda" or "cpu"). It is passed to .to(device) calls so tensors and the model sit on the same device.
def get_device():
    # Use the GPU when there is one (much faster for training), otherwise fall back to the CPU.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# INPUT: the config dict and the dataset name ("fer2013" or "rafdb").
# RETURNS: a dict of DataFrames, e.g. {"train":..., "val":..., "test":...}.  GOES TO: make_loader.
def get_frames(cfg, dataset):
    """Returns {split: DataFrame} for the requested dataset."""
    d = cfg["data"]
    if dataset == "fer2013":
        # train / validation / test; validation is carved out of train (see datasets.fer2013_frames)
        train, val, test = fer2013_frames(ROOT / d["fer2013_root"], d["val_frac"], cfg["seed"])
        return {"train": train, "val": val, "test": test}
    if dataset == "rafdb":
        # RAF-DB has no validation split here: its TEST set is only ever used for the cross-dataset test and
        # the bias audit. Its TRAIN split is used only by the mitigation experiment.
        root = ROOT / d["rafdb_root"]
        return {"train": rafdb_frame(root, "train"), "test": rafdb_frame(root, "test")}
    raise ValueError(dataset)


# INPUT: config, a DataFrame, a train flag and an optional custom sampler.
# RETURNS: a DataLoader that yields batches of (images, labels, row indexes).  GOES TO: run_epoch in train.py and predict in evaluate.py.
def make_loader(cfg, df, train, sampler=None):
    """sampler overrides shuffling, e.g. a WeightedRandomSampler for subgroup-balanced training."""
    d = cfg["data"]
    # The transform differs for training (with augmentation) and evaluation (clean images only).
    tf = build_transforms(d["img_size"], d["channels"], train,
                          augment=d.get("augment", True), erasing=d.get("random_erasing", 0.25))
    return DataLoader(
        FrameDataset(df, tf),
        batch_size=cfg["train"]["batch_size"],
        # Shuffle the order every epoch during training, unless a custom sampler decides the order
        # (a DataLoader cannot use both at once).
        shuffle=train and sampler is None,
        sampler=sampler,
        # Q: What is num_workers? A: The number of background processes that load and transform images in parallel,
        #    so the GPU does not wait for the disk. It changes speed only, never the results.
        num_workers=d["num_workers"],
        # pin_memory makes copying batches to the GPU faster.
        pin_memory=torch.cuda.is_available(),
        # drop_last: during training drop an incomplete final batch so every step has the same batch size.
        drop_last=train,
    )


# RETURNS: a small dict of optional settings that is forwarded as **kwargs to build_model().
def model_kwargs(m):
    """Optional architecture settings from the config's model section."""
    # Pass only the optional settings that the config actually contains on to build_model.
    return {k: m[k] for k in ("dropout", "block_dropout", "freeze_backbone", "freeze_blocks") if k in m}


# INPUT: a results/<run> folder and a device.  RETURNS: (the model in eval mode, its config).
# GOES TO: evaluate.py, explain.py, mitigate.py and the demo notebook.
# IF best.pt IS MISSING: torch.load raises FileNotFoundError (checkpoints are not in git; they live on the team Drive).
def load_run(run_dir, device):
    """Rebuild a trained model from results/<run>/ (config.yaml + best.pt)."""
    run_dir = Path(run_dir)
    # Each run saves the exact config it was trained with, so we can rebuild the identical architecture...
    cfg = load_config(run_dir / "config.yaml")
    m = cfg["model"]
    # pretrained=False: no need to download ImageNet weights, we overwrite them with our own right below.
    model = build_model(m["name"], m["num_classes"], pretrained=False, in_channels=cfg["data"]["channels"],
                        **model_kwargs(m))
    # ...and load the weights of the epoch that scored best on validation macro-F1 (saved by train.py).
    state = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(state["model"])
    # .eval() switches off dropout and uses BatchNorm's stored statistics, as required when predicting.
    return model.to(device).eval(), cfg
