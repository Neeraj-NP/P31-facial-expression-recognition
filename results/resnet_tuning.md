# ResNet tuning (issue #7)

Each experiment changes one thing against the `resnet` baseline (ResNet-18, 112px, full
fine-tuning, `configs/resnet.yaml`). All runs: Kaggle T4, seed 42, up to 30 epochs with
early stopping on validation macro-F1. The best run is chosen on **validation macro-F1
only**; RAF-DB is reported but never used for selection.

| Run | Change vs baseline | Epochs (best) | Val F1 | FER-2013 acc / F1 | RAF-DB acc / F1 |
|---|---|---|---|---|---|
| `resnet` | baseline | 30 (24) | 0.672 | 69.1% / 0.664 | 60.4% / 0.449 |
| `resnet_frozen` | backbone frozen, head only | 16 (8) | 0.324 | 34.5% / 0.316 | 36.5% / 0.233 |
| `resnet_224` | 224px input | 29 (21) | 0.659 | 70.3% / 0.675 | 61.3% / 0.465 |
| `resnet50` | ResNet-50 | 30 (22) | **0.681** | 70.1% / **0.687** | **64.2% / 0.493** |

**Selected: `resnet50`** (highest validation macro-F1), saved as `configs/resnet_best.yaml`.

## RAF-DB subgroup accuracy gaps (best group minus worst group, points)

| Run | Age | Gender | Race |
|---|---|---|---|
| `resnet` | 18.2 | 9.0 | 3.1 |
| `resnet_frozen` | 19.4 | 9.8 | 4.5 |
| `resnet_224` | 9.2 | 8.2 | 7.1 |
| `resnet50` | 11.7 | 8.1 | 4.7 |

Gender is female minus male, as in `cnn_tuning.md`. Two checks are reported by `python -m
src.bias_report` (`gap_table.csv`): whether the best and worst groups' 95% bootstrap intervals are
separated (a conservative test: overlapping intervals do not mean the gap is zero), and a two-proportion
z-test p-value between those two groups (uncorrected: the best and worst of several groups are
picked, which makes it optimistic). Gender gaps are separated and p < 1e-5 for every run. Age gaps
are separated for `resnet`, `resnet_frozen` and `cnn_best`; for `resnet50` the intervals
overlap but p = 0.03, and for `resnet_224` p = 0.09. No race gap reaches p < 0.05. Per-group numbers
with intervals are in each run's `eval_rafdb/subgroup_audit.csv`.

## Findings

- **Full fine-tuning beats a frozen backbone by a wide margin in our setup.** Our frozen run
  (linear head, same recipe as the baseline, early-stopped at epoch 16 of 30 with the learning rate
  still high) reaches 34.5% vs 69.1%. That is a statement about this head trained this way, not an
  upper bound on frozen features: a logistic-regression probe on the untouched backbone reached about
  42% on a 6,000-image subset, so a better-trained linear head would close part, but probably not
  most, of the gap. (The first frozen run was worse still because of a BatchNorm bug in
  `freeze_backbone`, fixed in 607fa8d; the numbers above are from the fixed code.)
- **224px input gives little.** +1.2 pts FER-2013 accuracy but a lower validation F1
  (0.659 vs 0.672) and about 40% more training time. The differences are within what a single
  seed can produce, so we treat it as no clear benefit.
- **ResNet-50 is the best ResNet.** Same in-domain accuracy as 224px, but 64.2% on RAF-DB
  (+3.8 pts over the ResNet-18 baseline) and the highest macro-F1 on both datasets among the ResNets.
  Against the CNN, which was not the selection criterion here: Neeraj's untuned CNN baseline scores
  65.1% / 0.480 on RAF-DB, so ResNet-50's transfer advantage over it is small (macro-F1 +0.013, one seed).
- **Bigger and higher-resolution models narrow the age gap but not the gender gap.** The age gap
  falls from 18.2 pts (baseline) to 11.7 (ResNet-50) and 9.2 (224px); the intervals no longer
  separate for those two (ResNet-50 is borderline, p = 0.03; 224px p = 0.09). The gender gap barely
  moves (9.0, 8.2, 8.1) and is clear in every run: women are recognised 8-10 points more accurately than men regardless of
  architecture, so better transfer alone does not remove it.
- **Frozen has similar gaps at a much lower accuracy level** (22-42% per age group), so its
  larger age gap reflects a weak model rather than a specific bias.

## Caveats

- One seed per configuration; differences under about 1 point should not be over-read.
- The 199 "unsure" gender images are excluded from the female-vs-male gap but stay in the
  per-group tables. The age gap compares the single best and worst of five groups, and the 70+
  group is small (n=89), so it has a wide confidence interval.
- `resnet_frozen` has no Grad-CAM figure: the explain step produced none for that run. We did not
  investigate; a likely cause is that nothing before the target layer requires gradients, which
  setting `requires_grad` on the input should fix.
- `best.pt` checkpoints are not in git.
