"""Grad-CAM saliency maps (works for CNN, ResNet and ViT).

    python -m src.explain --run results/resnet18_xxx --dataset fer2013 --per-class 4
    python -m src.explain --run results/resnet18_xxx --dataset rafdb --group-by gender
"""
# =====================================================================
# FILE ROLE: makes Grad-CAM heat maps that show which part of a face the model used for its prediction.
# HOW IT IS RUN: python -m src.explain --run results/vit --dataset fer2013
# INPUT: results/<run>/ (config.yaml + best.pt).  OUTPUT: results/<run>/explain_<dataset>/gradcam_grid.png
#  (and mean_cam_by_<attribute>.png when --group-by is used on RAF-DB).
# LIBRARY: pytorch-grad-cam supplies the GradCAM class.
# =====================================================================
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

# WHAT IS GRAD-CAM? A heat map showing which image regions pushed the model towards its prediction. It takes
# the gradient of the predicted class's score with respect to a late layer's feature maps, averages it into
# weights, and uses them to blend the maps. Red = the output was most sensitive there. It shows WHERE the output
# is sensitive, not WHY the model decided, so a hotspot on the mouth does not prove the model "understood" the smile.


# INPUT: ViT token outputs of shape [batch, tokens, channels].  RETURNS: a [batch, channels, height, width] map.
# GOES TO: GradCAM as its reshape_transform, because Grad-CAM needs a 2D map and a ViT gives a flat list of tokens.
def vit_reshape(tensor, h=14, w=14):
    # drop CLS token, [B, N, C] -> [B, C, H, W]
    # A ViT outputs a flat list of tokens, not a 2D map. Token 0 is the special CLS token (no position in the image),
    # so we drop it and fold the remaining 14x14 = 196 patch tokens back into a grid that Grad-CAM can use.
    # Example for ViT-B: [B, 197, 768] -> drop CLS -> [B, 196, 768] -> reshape -> [B, 14, 14, 768]
    # -> permute (move channels first, as Grad-CAM expects) -> [B, 768, 14, 14].
    return tensor[:, 1:, :].reshape(tensor.size(0), h, w, tensor.size(2)).permute(0, 3, 1, 2)


# INPUT: one normalised image tensor and its channel count.  RETURNS: a float image in [0, 1] that matplotlib can draw.
# GOES TO: show_cam_on_image(), which lays the heat map over the face.
def to_display(x, channels):
    """Undo normalisation so the face can be shown under the heatmap."""
    # The model sees normalised numbers; to draw the face we must invert that normalisation to get back to [0, 1].
    img = x.permute(1, 2, 0).numpy()
    if channels == 3:
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    else:
        img = np.repeat(img * 0.5 + 0.5, 3, axis=2)
    return np.clip(img, 0, 1).astype(np.float32)


# FLOW: load model -> build the Grad-CAM object for the right layer -> for each emotion pick a few random test images,
#  compute and draw the heat map with the title "true -> predicted" -> save the grid -> optionally average heat maps per subgroup.
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

    # img_size // 16 = patches per side for a ViT with 16x16 patches (224 // 16 = 14).
    # Grad-CAM step by step, for one image and the predicted class c:
    #   1. Forward pass; keep the target layer's feature maps A_k (k = channel; e.g. 2048 maps of 4x4 for ResNet-50 at 112px).
    #   2. Backward pass from the score of class c; get the gradient dScore_c / dA_k for every map.
    #   3. Average each map's gradient over its pixels -> one weight alpha_k per channel ("how much map k matters for c").
    #   4. Heat map = ReLU(sum over k of alpha_k * A_k): keep only regions that push the score UP, then resize it to
    #      the image size and scale to 0..1. The library does all of this; we only choose the layer.
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
    # Fixed seed -> the SAME images are drawn for every model, so the grids can be compared side by side.
    rng = np.random.default_rng(0)
    for c, cls in enumerate(CLASSES):
        ids = df.index[df["label"] == c].to_numpy()
        for j, i in enumerate(rng.choice(ids, size=min(k, len(ids)), replace=False)):
            x, _, _ = ds[i]
            # targets=None -> explain the class the model actually predicted (not the true label)
            heat = cam(input_tensor=x.unsqueeze(0).to(device), targets=None)[0]
            pred = model(x.unsqueeze(0).to(device)).argmax(1).item()
            # show_cam_on_image colours the heat map (blue = low, red = high) and blends it half-and-half with the face.
            axes[c, j].imshow(show_cam_on_image(to_display(x, d["channels"]), heat, use_rgb=True))
            # Title reads "true -> predicted", so wrong predictions are easy to spot.
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
            # Up to 200 random images per group, processed in batches of 32 to keep memory low.
            ids = rng.choice(sub.index.to_numpy(), size=min(200, len(sub)), replace=False)
            batch = torch.stack([ds[i][0] for i in ids]).to(device)
            heats = np.concatenate([cam(input_tensor=b, targets=None) for b in batch.split(32)])
            # Average all heat maps of the group: a "typical" attention pattern for that group.
            ax.imshow(heats.mean(0), cmap="jet")
            ax.set_title(f"{g} (n={len(ids)})")
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(out / f"mean_cam_by_{args.group_by}.png", dpi=150)
        print(f"saved {out / f'mean_cam_by_{args.group_by}.png'}")


if __name__ == "__main__":
    main()
