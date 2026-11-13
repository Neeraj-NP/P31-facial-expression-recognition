"""Model zoo: scratch CNN baseline, pretrained ResNet, pretrained ViT."""
import timm
import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    """VGG-style CNN for 48x48 grayscale faces (~5M params)."""

    def __init__(self, num_classes=7, in_channels=1, dropout=0.4, block_dropout=0.25):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2), nn.Dropout(block_dropout),
            )

        self.features = nn.Sequential(
            block(in_channels, 64), block(64, 128), block(128, 256), block(256, 512)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(name, num_classes=7, pretrained=True, **kwargs):
    if name == "cnn":
        return SimpleCNN(num_classes, in_channels=kwargs.get("in_channels", 1),
                         dropout=kwargs.get("dropout", 0.4), block_dropout=kwargs.get("block_dropout", 0.25))
    if name in ("resnet18", "resnet50"):
        weights = "DEFAULT" if pretrained else None
        m = getattr(models, name)(weights=weights)
        if kwargs.get("freeze_backbone"):
            for p in m.parameters():
                p.requires_grad = False
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, num_classes))
        if kwargs.get("freeze_backbone"):
            # frozen BatchNorm layers must stay in eval mode, otherwise their running stats keep drifting
            module_train = m.train

            def train(mode=True):
                module_train(mode)
                for name, child in m.named_children():
                    if name != "fc":
                        child.eval()
                return m

            m.train = train
        return m
    if name.startswith("vit"):
        # e.g. vit_base_patch16_224, vit_small_patch16_224
        return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")


def gradcam_target_layer(model, name):
    """Layer used by Grad-CAM for each architecture."""
    if name == "cnn":
        return model.features[-1][3]  # last conv of last block
    if name.startswith("resnet"):
        return model.layer4[-1]
    if name.startswith("vit"):
        return model.blocks[-1].norm1
    raise ValueError(name)
