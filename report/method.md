# Method

*Draft of the Method section. Results and discussion are in `results_and_discussion.md`.*

## Models

We compare three architectures of increasing size and pretraining, trained on the same
7-class label space (`angry, disgust, fear, happy, sad, surprise, neutral`):

- **CNN (scratch).** A small VGG-style network (4 conv blocks, 64→512 channels, ~5M
  parameters) trained from random initialisation on 48×48 grayscale faces. No
  pretrained knowledge of natural images; serves as the from-scratch baseline.
- **ResNet-18/50 (transfer learning).** ImageNet-pretrained `torchvision` ResNet with
  the final fully-connected layer replaced by a dropout + linear head for 7 classes.
  Inputs are resized to 112×112 (or 224×224, see tuning below) and 3-channel, using
  ImageNet normalisation, so the pretrained features are used as intended.
- **ViT (fine-tuning).** An ImageNet-pretrained Vision Transformer (`timm`,
  `vit_small`/`vit_base_patch16_224`), fine-tuned end-to-end at 224×224.

All three share one training script (`src/train.py`) and one evaluation script
(`src/evaluate.py`), differing only through a YAML config, so results are directly
comparable and not an artefact of different training code paths.

## Training setup

Every model is trained with the same recipe, parameterised per-model in
`configs/*.yaml`:

- **Optimiser:** AdamW, with per-model learning rate and weight decay.
- **Schedule:** linear warmup for `warmup_epochs`, then cosine decay to 0, stepped
  every batch.
- **Loss:** cross-entropy with label smoothing (0.1) and inverse-frequency class
  weights, to counter FER-2013's class imbalance (disgust is ~17x rarer than happy).
- **Data augmentation:** random horizontal flip, affine (rotation/translation/scale),
  colour jitter, and random erasing on the training split only; grayscale 1-channel
  for the CNN, 3-channel ImageNet-normalised for ResNet/ViT.
- **Mixed precision (AMP)** on GPU, for speed.
- **Model selection:** the checkpoint with the best validation macro-F1 is kept
  (`best.pt`); training stops early if macro-F1 doesn't improve for
  `early_stop_patience` epochs.
- **Validation split:** a stratified 10% held out from FER-2013's official training
  set (seeded, so it's identical across all models and reruns).

### ResNet tuning experiments

Three experiments isolate one variable each against the `configs/resnet.yaml`
baseline, so differences in outcome can be attributed to that one change:

| Experiment | Change vs. baseline |
|---|---|
| `resnet_frozen` | `freeze_backbone: true` — only the new classifier head is trained (linear probe); the pretrained backbone's weights are untouched |
| `resnet_224` | Input resized to 224×224 instead of 112×112 |
| `resnet50` | ResNet-50 instead of ResNet-18 (batch size halved for memory) |

`freeze_backbone` is implemented by setting `requires_grad=False` on every backbone
parameter before the new classifier head (which keeps `requires_grad=True`) is
attached, so the optimiser only updates the head. The frozen backbone is also kept in eval mode
during training: otherwise PyTorch's `model.train()` puts its BatchNorm layers back in
training mode and their running statistics drift away from the pretrained ones (our first frozen
run suffered from exactly this and scored about 30% before the fix).

## Evaluation protocol

Every trained model goes through the same three checks (`src/evaluate.py`,
`src/explain.py`):

1. **In-domain test:** accuracy and macro-F1 on the FER-2013 held-out test set — the
   standard "how good is it on the data it was trained for" number.
2. **Cross-dataset test:** the same model, unmodified, evaluated on RAF-DB's test
   split — a dataset it has never seen, collected differently (in-the-wild vs. the
   FER-2013 protocol). This measures whether the model learned "emotion" in a way
   that generalises, rather than memorising dataset-specific artefacts.
3. **Subgroup bias audit:** RAF-DB's manual annotations (gender, race,
   5-bucket age group) let us break the cross-dataset accuracy down by demographic
   subgroup. For each attribute we report accuracy and macro-F1 per group with a 95%
   bootstrap confidence interval (1000 resamples), and the best-worst group gap. We judge a
   gap two ways: whether the best and worst groups' 95% intervals are **separated** (don't
   overlap; a simple, conservative check, so overlapping intervals do not show the gap is
   zero), and a two-proportion z-test **p-value** between those two groups. Neither accounts
   for the best and worst groups being picked from several, so both flatter a gap somewhat. For gender the gap is female vs male; the 199 "unsure"
   images stay in the per-group tables but are not used for the gap. `src/bias_report.py` aggregates this across all three
   models into one comparison table and a bar chart with CI error bars per attribute.
4. **Grad-CAM saliency:** for qualitative interpretation, Grad-CAM heatmaps show which
   region of the face each model attended to for its prediction, including a
   mean-heatmap comparison across subgroups (does the model look in the same place
   for every gender/race/age group?).

### Mitigation experiment

For the two attributes with the largest gaps, gender and age, we fine-tune
the already-trained model further on RAF-DB's own training split, using a
`WeightedRandomSampler` with inverse-frequency weights so every group of that
attribute is drawn with equal total probability during training (i.e. the rarer
group is oversampled, the common group undersampled, in expectation; every attribute value
counts as a group, so gender's small "unsure" group is sampled about 5x more often than its share
of the data). A control run uses identical fine-tuning with ordinary shuffling, so the effect of
balancing can be told apart from the effect of simply training on RAF-DB. Fine-tuning
uses a low learning rate (1e-5) relative to original training, since the goal is to
adapt the existing model rather than retrain it. We then repeat the bias audit on
the fine-tuned model and compare gaps and overall accuracy before vs. after, to see
whether the mitigation closed the gap without an unacceptable accuracy cost.
