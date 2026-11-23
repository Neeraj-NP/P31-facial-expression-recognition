"""Model zoo: scratch CNN baseline, pretrained ResNet, pretrained ViT."""
# =====================================================================
# FILE ROLE: defines the three model families that are compared: a small CNN built from scratch, a pretrained ResNet
# (torchvision) and a pretrained Vision Transformer (timm library). build_model() is the single entry point.
# HOW IT CONNECTS: train.py builds a fresh model with build_model(); evaluate.py, explain.py and mitigate.py rebuild a
#  saved model through common.load_run(), which also calls build_model().
# LIBRARIES: torch.nn builds layers; torchvision.models gives ResNet and its ImageNet weights; timm gives the ViT.
# =====================================================================
import timm
import torch.nn as nn
from torchvision import models


# WHAT: the from-scratch baseline network.
# INPUT (forward): a batch of 1-channel 48x48 images.  RETURNS: 7 raw scores (logits), one per emotion.
# WHY IT EXISTS: it shows what pretraining adds when compared with the ResNet and the ViT.
class SimpleCNN(nn.Module):
    """VGG-style CNN for 48x48 grayscale faces (~5M params)."""

    def __init__(self, num_classes=7, in_channels=1, dropout=0.4, block_dropout=0.25):
        super().__init__()

        # One "block" = the repeating unit of a VGG-style network:
        #   Conv -> BatchNorm -> ReLU, twice, then MaxPool (halves the image size) and Dropout.
        #   - Conv 3x3: learns local patterns (edges -> eyes/mouth shapes as blocks get deeper).
        #   - BatchNorm: keeps activations well-scaled, so training is faster and more stable.
        #   - ReLU: the non-linearity that lets the network learn more than a straight line.
        #   - MaxPool: shrinks the map (48 -> 24 -> 12 -> 6 -> 3), keeping the strongest responses.
        #   - Dropout: randomly switches off units so the model cannot just memorise the training set.
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2), nn.Dropout(block_dropout),
            )

        # Channels double each block (64 -> 128 -> 256 -> 512): as the picture gets smaller the network
        # keeps more feature maps, so deeper layers can describe more complex face parts.
        # Shape walk-through for one batch of B images (B = batch size):
        #   input            [B,   1, 48, 48]
        #   after block 1    [B,  64, 24, 24]   (MaxPool halves 48 -> 24)
        #   after block 2    [B, 128, 12, 12]
        #   after block 3    [B, 256,  6,  6]
        #   after block 4    [B, 512,  3,  3]
        # padding=1 in every 3x3 conv keeps height and width unchanged; only MaxPool shrinks them.
        self.features = nn.Sequential(
            block(in_channels, 64), block(64, 128), block(128, 256), block(256, 512)
        )
        # Head: average each of the 512 feature maps to one number (AdaptiveAvgPool, so it works for any input
        # size and adds no parameters), then a small 512 -> 256 -> 7 classifier. 7 = number of emotions.
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    # forward() is what runs when we call model(x). It returns LOGITS: 7 unnormalised scores, one per emotion.
    # They are not probabilities yet: softmax (in evaluate.py) turns them into probabilities that sum to 1, and
    # CrossEntropyLoss (in train.py) applies softmax internally, so the model itself needs no softmax layer.
    def forward(self, x):
        return self.classifier(self.features(x))


# WHAT: the one place that creates any model from the name in the YAML config ("cnn", "resnet18", "resnet50", "vit_...").
# INPUT: model name, number of classes (7), pretrained flag, and optional settings (freeze_backbone, freeze_blocks, dropout).
# RETURNS: a PyTorch model that outputs 7 scores per image.
# GOES TO: train.py (to train it) and common.load_run (to load saved weights into it).
# IF THE NAME IS UNKNOWN: it raises ValueError at the bottom.
def build_model(name, num_classes=7, pretrained=True, **kwargs):
    # One factory function for all three model families, so train / evaluate / explain / mitigate all build
    # models the same way and the only thing that differs between experiments is the YAML config.
    if name == "cnn":
        # Trained from scratch: no pretrained knowledge, the "baseline" that shows what pretraining adds.
        return SimpleCNN(num_classes, in_channels=kwargs.get("in_channels", 1),
                         dropout=kwargs.get("dropout", 0.4), block_dropout=kwargs.get("block_dropout", 0.25))
    if name in ("resnet18", "resnet50"):
        # Transfer learning: start from weights learned on ImageNet ("DEFAULT") instead of random weights.
        # pretrained=False is used when we only need the architecture, e.g. to load our own saved weights.
        # What a ResNet is: a deep CNN built from "residual blocks", where each block ADDS its input back to its output
        # (x + F(x)). These shortcut connections let the gradient flow through very deep networks, which is why
        # ResNet-50 (50 layers, ~25M parameters) still trains well. ResNet-18 has 18 layers and ~11M parameters.
        # "DEFAULT" = the best ImageNet weights torchvision provides (ImageNet: ~1.28M photos in 1000 classes).
        weights = "DEFAULT" if pretrained else None
        m = getattr(models, name)(weights=weights)
        if kwargs.get("freeze_backbone"):
            # "Linear probe" experiment: lock all pretrained layers so only the new final layer learns.
            # Compares "reuse features as they are" against full fine-tuning.
            for p in m.parameters():
                p.requires_grad = False
        # Replace ImageNet's 1000-class output layer with our own: dropout (regularisation) + a linear layer
        # for 7 emotions. This new layer is created AFTER the freezing loop, so it stays trainable.
        # m.fc.in_features = size of the feature vector after global average pooling: 512 for ResNet-18, 2048 for ResNet-50.
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, num_classes))
        if kwargs.get("freeze_backbone"):
            # frozen BatchNorm layers must stay in eval mode, otherwise their running stats keep drifting
            # (This was a real bug we found: calling model.train() puts BatchNorm back into training mode, which
            #  updates its running mean/variance even though its weights are frozen. The first frozen run scored
            #  about 30% before this fix. We override .train() so every layer except the new head stays in eval mode.)
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
        # Vision Transformer from the `timm` library, pretrained on ImageNet. It cuts the image into 16x16 patches
        # and lets them attend to each other (self-attention). num_classes=7 swaps in a new 7-way output layer.
        # How a ViT reads a 224x224 image:
        #   1. patch_embed cuts it into 16x16-pixel patches: (224/16) x (224/16) = 14 x 14 = 196 patches, and turns each
        #      patch into a vector (768 numbers for ViT-B, 384 for ViT-S).
        #   2. A learnable CLS token is put in front (197 tokens), and pos_embed adds a learned position vector to every
        #      token, so the model knows where each patch came from (self-attention on its own ignores order).
        #   3. 12 transformer blocks follow. In each, every token looks at every other token (self-attention), so a patch
        #      on the mouth can use information from a patch on the eyes; then a small MLP processes each token.
        #   4. The final CLS token summarises the whole image, and the new 7-way head turns it into emotion scores.
        # Sizes: ViT-B ~86M parameters, ViT-S ~22M.
        m = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
        n_freeze = kwargs.get("freeze_blocks", 0)
        if n_freeze:
            # freeze the patch embedding and the first n transformer blocks, train the rest
            # (Partial fine-tuning: early blocks hold general features, later blocks are more task-specific.)
            # requires_grad = False means: compute no gradient for this weight, so the optimizer never changes it.
            frozen = [m.patch_embed, m.blocks[:n_freeze]]
            for p in (m.cls_token, m.pos_embed):
                p.requires_grad = False
            for mod in frozen:
                for p in mod.parameters():
                    p.requires_grad = False
        return m
    raise ValueError(f"Unknown model: {name}")


# WHAT: tells Grad-CAM which layer to look at for each model type.
# INPUT: the model and its name.  RETURNS: one layer object.  GOES TO: explain.py and the demo notebook.
def gradcam_target_layer(model, name):
    """Layer used by Grad-CAM for each architecture."""
    # Grad-CAM needs a layer that is still "spatial" (it keeps a map of where things are) but already holds
    # high-level information: the LAST convolutional layer is the usual choice. Each architecture needs its own.
    if name == "cnn":
        # features[-1] is the last block; inside it index 0 = Conv, 1 = BatchNorm, 2 = ReLU, 3 = the second Conv.
        # Its output is a [512, 3, 3] map, so the CNN's heat map is very coarse (3x3) before being enlarged to 48x48.
        return model.features[-1][3]  # last conv of last block
    if name.startswith("resnet"):
        # layer4[-1] = the last residual block. Its output map is 4x4 for a 112x112 input and 7x7 for 224x224
        # (the image is shrunk 32 times), with 2048 channels for ResNet-50 and 512 for ResNet-18.
        return model.layer4[-1]
    if name.startswith("vit"):
        # A ViT has no conv maps; we use the token outputs of the last transformer block and (in explain.py)
        # reshape them back into a 14x14 grid of patches.
        # norm1 = the LayerNorm at the start of the last block. Its output has shape [B, 197, 768] for ViT-B
        # (1 CLS token + 196 patch tokens, 768 numbers each).
        return model.blocks[-1].norm1
    raise ValueError(name)
