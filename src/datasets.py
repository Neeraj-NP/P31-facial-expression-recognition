"""Dataset loaders for FER-2013 and RAF-DB with one shared label space."""
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

# Canonical label order used everywhere in the project.
CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# RAF-DB basic labels are 1..7 = Surprise, Fear, Disgust, Happiness, Sadness, Anger, Neutral
RAFDB_TO_IDX = {1: 5, 2: 2, 3: 1, 4: 3, 5: 4, 6: 0, 7: 6}

# RAF-DB manual attribute codes
RAFDB_GENDER = {0: "male", 1: "female", 2: "unsure"}
RAFDB_RACE = {0: "caucasian", 1: "african-american", 2: "asian"}
RAFDB_AGE = {0: "0-3", 1: "4-19", 2: "20-39", 3: "40-69", 4: "70+"}

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build_transforms(img_size: int, channels: int, train: bool, augment: bool = True, erasing: float = 0.25):
    """Grayscale 1-channel for the scratch CNN, 3-channel ImageNet-normalised for pretrained models.

    augment=False turns off flip/affine/jitter; erasing is the RandomErasing probability (0 = off).
    """
    train_aug = train and augment
    ops = [transforms.Grayscale(num_output_channels=channels), transforms.Resize((img_size, img_size))]
    if train_aug:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
        ]
    ops.append(transforms.ToTensor())
    if channels == 3:
        ops.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    else:
        ops.append(transforms.Normalize((0.5,), (0.5,)))
    if train_aug and erasing > 0:
        ops.append(transforms.RandomErasing(p=erasing))
    return transforms.Compose(ops)


class FrameDataset(Dataset):
    """Generic dataset over a DataFrame with columns `path`, `label` (+ optional metadata)."""

    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        return self.transform(img), int(row["label"]), i


def fer2013_frames(root, val_frac=0.1, seed=42):
    """Returns train/val/test DataFrames. Val is a stratified split of the official train set."""
    root = Path(root)

    def scan(split):
        rows = [
            {"path": str(p), "label": CLASS_TO_IDX[cls_dir.name]}
            for cls_dir in sorted((root / split).iterdir()) if cls_dir.name in CLASS_TO_IDX
            for p in cls_dir.glob("*.*")
        ]
        return pd.DataFrame(rows)

    train_full, test = scan("train"), scan("test")
    train, val = train_test_split(
        train_full, test_size=val_frac, stratify=train_full["label"], random_state=seed
    )
    return train, val, test


def _read_rafdb_attributes(path: Path):
    # *_manu_attri.txt: 5 landmark lines, then gender, race, age (one integer per line)
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    gender, race, age = (int(x) for x in lines[5:8])
    return RAFDB_GENDER[gender], RAFDB_RACE[race], RAFDB_AGE[age]


def rafdb_frame(root, split="test"):
    """RAF-DB basic split ('train' or 'test') with gender/race/age columns when available."""
    root = Path(root)
    rows = []
    for line in (root / "EmoLabel" / "list_patition_label.txt").read_text().splitlines():
        if not line.strip():
            continue
        name, label = line.split()
        if not name.startswith(split):
            continue
        stem = Path(name).stem
        row = {
            "path": str(root / "Image" / "aligned" / f"{stem}_aligned.jpg"),
            "label": RAFDB_TO_IDX[int(label)],
        }
        attr = root / "Annotation" / "manual" / f"{stem}_manu_attri.txt"
        if attr.exists():
            row["gender"], row["race"], row["age"] = _read_rafdb_attributes(attr)
        rows.append(row)
    return pd.DataFrame(rows)
