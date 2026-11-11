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
| `resnet` | 18.2 | 13.6 | 3.1 |
| `resnet_frozen` | 19.4 | 20.2 | 4.5 |
| `resnet_224` | 9.2 | 8.2 | 7.1 |
| `resnet50` | 11.7 | 8.4 | 4.7 |

Full per-group numbers with 95% CIs are in each run's `eval_rafdb/subgroup_audit.csv`.

## Findings

- **Fine-tuning matters far more than anything else.** Freezing the ImageNet backbone drops
  FER-2013 accuracy from 69.1% to 34.5%. ImageNet features alone do not separate 48x48
  grayscale faces well; a logistic-regression probe on the untouched backbone reached only
  about 42% on a 6,000-image subset, so frozen features are a poor fit regardless of how the
  head is trained. (The first frozen run was worse still because of a BatchNorm bug in
  `freeze_backbone`, fixed in 607fa8d; the numbers above are from the fixed code.)
- **224px input gives little.** +1.2 pts FER-2013 accuracy but a lower validation F1
  (0.659 vs 0.672) and about 40% more training time. The differences are within what a single
  seed can produce, so we treat it as no clear benefit.
- **ResNet-50 is best on every transfer metric.** Same in-domain accuracy as 224px, but 64.2% on
  RAF-DB (+3.8 pts over the baseline) and the highest macro-F1 on both datasets.
- **Better transfer came with smaller subgroup gaps.** ResNet-50's age and gender gaps are not
  statistically significant (the best and worst groups' 95% CIs overlap), unlike the baseline's.
  The ranking by RAF-DB accuracy and by smallest gap agree, which the CNN runs did not.
- **Frozen is also the least fair** (gender gap 20.2 pts), but it is the least accurate overall,
  so this is a weak model rather than a biased one specifically.

## Caveats

- One seed per configuration; differences under about 1 point should not be over-read.
- The gender gap is measured against the "unsure" group (n=199) when that group is best; male vs
  female alone is smaller.
- `resnet_frozen` has no Grad-CAM figure: Grad-CAM needs gradients through the backbone, which
  a frozen backbone does not provide.
- `best.pt` checkpoints are not in git.
