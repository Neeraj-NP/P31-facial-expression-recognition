"""Dataset loaders for FER-2013 and RAF-DB with one shared label space."""
# =====================================================================
# FILE ROLE: turns the raw image files on disk into data that PyTorch can train on.
#  - Reads FER-2013 (one folder per emotion) and RAF-DB (text files that hold the labels) and builds a table
#    (a pandas DataFrame) with: image path, emotion label and, for RAF-DB only, gender / race / age.
#  - Defines the image preprocessing (build_transforms) and the Dataset class that opens one image at a time.
# HOW IT CONNECTS: common.py calls these functions (get_frames, make_loader). train.py, evaluate.py, explain.py and
#  mitigate.py then get their data through common.py.
# IF THE FOLDERS ARE MISSING OR LAID OUT DIFFERENTLY: the folder scan fails with a FileNotFoundError.
# IF THIS FILE WAS REMOVED: no model could load any image, so nothing could be trained or tested.
# =====================================================================
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

# Canonical label order used everywhere in the project.
# WHY: the model outputs a number 0..6, so every dataset must map its emotion names to the SAME numbers,
# otherwise "happy" in one dataset could be scored as "surprise" in the other without any error message.
CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# RAF-DB basic labels are 1..7 = Surprise, Fear, Disgust, Happiness, Sadness, Anger, Neutral
# WHY: RAF-DB numbers its emotions in a different order from ours, so this dict translates RAF-DB's label
# into our index (e.g. RAF-DB 4 = Happiness -> our index 3 = "happy").
RAFDB_TO_IDX = {1: 5, 2: 2, 3: 1, 4: 3, 5: 4, 6: 0, 7: 6}

# RAF-DB manual attribute codes
# WHY: RAF-DB ships manual annotations of gender, race and age. FER-2013 has none, which is why the
# fairness (bias) audit can only be done on RAF-DB.
RAFDB_GENDER = {0: "male", 1: "female", 2: "unsure"}
RAFDB_RACE = {0: "caucasian", 1: "african-american", 2: "asian"}
RAFDB_AGE = {0: "0-3", 1: "4-19", 2: "20-39", 3: "40-69", 4: "70+"}

# The pixel mean/std of ImageNet. WHY: the pretrained ResNet/ViT weights were trained on images normalised
# with exactly these numbers, so feeding them the same statistics lets the pretrained features work as intended.
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# WHAT: builds the chain of image preprocessing steps (grey, resize, augment, convert to tensor, normalise).
# INPUT: image size, number of channels, whether this is training data, and the augmentation settings from the YAML config.
# RETURNS: one torchvision Compose object.  GOES TO: FrameDataset applies it to every image it loads.
# WHY IT MATTERS: a model only works if every image has the same size, channels and number range as in training.
def build_transforms(img_size: int, channels: int, train: bool, augment: bool = True, erasing: float = 0.25):
    """Grayscale 1-channel for the scratch CNN, 3-channel ImageNet-normalised for pretrained models.

    augment=False turns off flip/affine/jitter; erasing is the RandomErasing probability (0 = off).
    """
    # Augmentation is only ever applied to the TRAINING split; validation/test must see clean images,
    # otherwise the scores would be random and not comparable between runs.
    train_aug = train and augment
    # Step 1: convert EVERY image (both datasets, every model) to greyscale first.
    #   WHY: FER-2013 is greyscale only; RAF-DB is colour. Forcing both to grey means the FER-2013 -> RAF-DB
    #   gap we measure cannot be blamed on colour. channels=1 for the CNN; channels=3 simply COPIES the grey
    #   image into 3 channels because pretrained ResNet/ViT expect a 3-channel input.
    # Step 2: resize to the size the model was configured for (48 for CNN, 112/224 for ResNet/ViT).
    ops = [transforms.Grayscale(num_output_channels=channels), transforms.Resize((img_size, img_size))]
    if train_aug:
        ops += [
            # Mirror image: a face flipped left/right is still the same emotion -> free extra data.
            transforms.RandomHorizontalFlip(),
            # Small rotation / shift / zoom: mimics different head poses and crops, so the model does not
            # memorise exact face positions.
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            # Brightness/contrast jitter: mimics different lighting. (Only these two because the image is grey.)
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
        ]
    # PIL image -> tensor with values in [0, 1].
    ops.append(transforms.ToTensor())
    if channels == 3:
        # Pretrained models: use the ImageNet statistics they were trained with.
        ops.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    else:
        # Scratch CNN: no pretrained stats to match, so just centre [0,1] to [-1,1].
        ops.append(transforms.Normalize((0.5,), (0.5,)))
    if train_aug and erasing > 0:
        # Randomly blanks a rectangle: forces the model not to rely on one facial region and to cope with
        # occlusion (hands, glasses, hair). Applied after normalisation, with probability `erasing`.
        ops.append(transforms.RandomErasing(p=erasing))
    return transforms.Compose(ops)


# WHAT: a PyTorch Dataset. PyTorch's DataLoader asks it for image number i and gets back (image tensor, label, i).
# INPUT: a DataFrame (one row per image) and the transform from build_transforms.
# GOES TO: wrapped by a DataLoader (see common.make_loader), which groups the items into batches for the model.
class FrameDataset(Dataset):
    """Generic dataset over a DataFrame with columns `path`, `label` (+ optional metadata)."""

    # WHY a DataFrame: it keeps the file path, the label AND extra columns (gender/race/age) together in one
    # row, so after prediction we can group results by demographic subgroup for the bias audit.
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        # .convert("RGB") gives a uniform 3-channel starting point whatever the file format is;
        # the Grayscale transform then reduces it to what the model needs.
        img = Image.open(row["path"]).convert("RGB")
        # We also return the row index i, so predictions can be matched back to their DataFrame row
        # (the DataLoader may process items in a different order).
        return self.transform(img), int(row["label"]), i


# WHAT: scans the FER-2013 folders and returns three tables: train, val and test (path + label for each image).
# INPUT: dataset folder path, validation fraction, random seed.  RETURNS: three DataFrames.
# GOES TO: common.get_frames, then train.py (train and val) and evaluate.py / explain.py (test).
def fer2013_frames(root, val_frac=0.1, seed=42):
    """Returns train/val/test DataFrames. Val is a stratified split of the official train set."""
    root = Path(root)

    # FER-2013 is stored as folders: <split>/<emotion name>/<image files>. The folder name is the label.
    def scan(split):
        rows = [
            {"path": str(p), "label": CLASS_TO_IDX[cls_dir.name]}
            for cls_dir in sorted((root / split).iterdir()) if cls_dir.name in CLASS_TO_IDX
            for p in cls_dir.glob("*.*")
        ]
        return pd.DataFrame(rows)

    train_full, test = scan("train"), scan("test")
    # A validation set is carved out of the TRAINING data (10%). We use it to pick the best epoch and the best
    # configuration, so the test set is never used for any choice.
    #   stratify=label  -> every split keeps the same class proportions (important: disgust is ~16x rarer than happy)
    #   random_state=seed -> identical split for every model and every rerun, so models are comparable.
    train, val = train_test_split(
        train_full, test_size=val_frac, stratify=train_full["label"], random_state=seed
    )
    return train, val, test


# WHAT: reads one RAF-DB annotation text file and converts the gender / race / age numbers into words.
# INPUT: path of that file.  RETURNS: (gender, race, age).
# GOES TO: rafdb_frame stores them as columns; evaluate.subgroup_audit later groups results by them (the bias audit).
def _read_rafdb_attributes(path: Path):
    # *_manu_attri.txt: 5 landmark lines, then gender, race, age (one integer per line)
    # (the first 5 lines are facial landmark coordinates we do not need, hence lines[5:8])
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    gender, race, age = (int(x) for x in lines[5:8])
    return RAFDB_GENDER[gender], RAFDB_RACE[race], RAFDB_AGE[age]


# WHAT: builds the RAF-DB table for the "train" or "test" split.
# INPUT: dataset folder and the split name.  RETURNS: DataFrame with path, label (in our label order), gender, race, age.
# GOES TO: common.get_frames -> evaluate.py (cross-dataset test and fairness audit) and mitigate.py (fine-tuning).
def rafdb_frame(root, split="test"):
    """RAF-DB basic split ('train' or 'test') with gender/race/age columns when available."""
    root = Path(root)
    rows = []
    # list_patition_label.txt (the typo is in the dataset itself) lists "<image name> <label>" for every image;
    # image names start with "train_" or "test_", which is how we pick the requested split.
    for line in (root / "EmoLabel" / "list_patition_label.txt").read_text().splitlines():
        if not line.strip():
            continue
        name, label = line.split()
        if not name.startswith(split):
            continue
        stem = Path(name).stem
        row = {
            # We use the "aligned" images (faces already cropped and aligned by the dataset authors).
            "path": str(root / "Image" / "aligned" / f"{stem}_aligned.jpg"),
            # Translate RAF-DB's label numbering into our shared label order.
            "label": RAFDB_TO_IDX[int(label)],
        }
        attr = root / "Annotation" / "manual" / f"{stem}_manu_attri.txt"
        if attr.exists():
            row["gender"], row["race"], row["age"] = _read_rafdb_attributes(attr)
        rows.append(row)
    return pd.DataFrame(rows)
