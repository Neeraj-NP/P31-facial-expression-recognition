"""Grad-CAM saliency maps (works for CNN, ResNet and ViT).

    python -m src.explain --run results/resnet18_xxx --dataset fer2013 --per-class 4
    python -m src.explain --run results/resnet18_xxx --dataset rafdb --group-by gender
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from .common import get_device, get_frames, load_run
from .datasets import CLASSES, FrameDataset, build_transforms
from .models import gradcam_target_layer


def vit_reshape(tensor, h=14, w=14):
    # drop CLS token, [B, N, C] -> [B, C, H, W]
    return tensor[:, 1:, :].reshape(tensor.size(0), h, w, tensor.size(2)).permute(0, 3, 1, 2)


def to_display(x, channels):
    """Undo normalisation so the face can be shown under the heatmap."""
    img = x.permute(1, 2, 0).numpy()
    if channels == 3:
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    else:
        img = np.repeat(img * 0.5 + 0.5, 3, axis=2)
    return np.clip(img, 0, 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dataset", choices=["fer2013", "rafdb"], default="fer2013")
    ap.add_argument("--per-class", type=int, default=4, help="example images per class")
    ap.add_argument("--group-by", choices=["gender", "race", "age"],
                    help="RAF-DB only: also save mean CAM per subgroup")
    args = ap.parse_args()

    device = get_device()
    model, cfg = load_run(args.run, device)
    name, d = cfg["model"]["name"], cfg["data"]
    df = get_frames(cfg, args.dataset)["test"].reset_index(drop=True)
    ds = FrameDataset(df, build_transforms(d["img_size"], d["channels"], train=False))

    cam = GradCAM(
        model=model,
        target_layers=[gradcam_target_layer(model, name)],
        reshape_transform=(lambda t: vit_reshape(t, d["img_size"] // 16, d["img_size"] // 16))
        if name.startswith("vit") else None,
    )
    out = Path(args.run) / f"explain_{args.dataset}"
    out.mkdir(exist_ok=True)

    # grid: rows = classes, cols = examples (CAM for the predicted class)
    k = args.per_class
    fig, axes = plt.subplots(len(CLASSES), k, figsize=(2 * k, 2 * len(CLASSES)), squeeze=False)
    rng = np.random.default_rng(0)
    for c, cls in enumerate(CLASSES):
        ids = df.index[df["label"] == c].to_numpy()
        for j, i in enumerate(rng.choice(ids, size=min(k, len(ids)), replace=False)):
            x, _, _ = ds[i]
            heat = cam(input_tensor=x.unsqueeze(0).to(device), targets=None)[0]
            pred = model(x.unsqueeze(0).to(device)).argmax(1).item()
            axes[c, j].imshow(show_cam_on_image(to_display(x, d["channels"]), heat, use_rgb=True))
            axes[c, j].set_title(f"{cls} -> {CLASSES[pred]}", fontsize=8)
        for ax in axes[c]:
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(out / "gradcam_grid.png", dpi=150)
    print(f"saved {out / 'gradcam_grid.png'}")

    # mean CAM per demographic subgroup (where does the model look for each group?)
    if args.group_by:
        if args.group_by not in df:
            raise SystemExit(f"'{args.group_by}' not available for {args.dataset}")
        groups = df.dropna(subset=[args.group_by]).groupby(args.group_by)
        fig, axes = plt.subplots(1, len(groups), figsize=(3 * len(groups), 3), squeeze=False)
        for ax, (g, sub) in zip(axes[0], groups):
            ids = rng.choice(sub.index.to_numpy(), size=min(200, len(sub)), replace=False)
            batch = torch.stack([ds[i][0] for i in ids]).to(device)
            heats = np.concatenate([cam(input_tensor=b, targets=None) for b in batch.split(32)])
            ax.imshow(heats.mean(0), cmap="jet")
            ax.set_title(f"{g} (n={len(ids)})")
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(out / f"mean_cam_by_{args.group_by}.png", dpi=150)
        print(f"saved {out / f'mean_cam_by_{args.group_by}.png'}")


if __name__ == "__main__":
    main()
