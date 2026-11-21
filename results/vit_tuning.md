# ViT tuning (#8)

All runs: same seed, FER-2013 train/val split, 15 epochs max with early stopping (patience 5), model chosen on
FER-2013 **val** macro-F1. RAF-DB test (n=3068) is only used for reporting, never for choosing a config.
One seed per run.

| Run | Config | Best epoch | Val macro-F1 | FER-2013 test acc | FER-2013 test macro-F1 | RAF-DB acc | RAF-DB macro-F1 | Gender gap | Age gap | Race gap |
|---|---|---|---|---|---|---|---|---|---|---|
| `vit_small_lr1e5` | ViT-S, lr 1e-5 | 12/15 | 0.610 | 66.9% | 0.620 | 62.3% | 0.500 | 8.5 | 11.3 | 7.5 |
| `vit_small_lr3e5` | ViT-S, lr 3e-5 | 11/15 | 0.649 | 69.5% | 0.658 | **65.3%** | **0.504** | 4.6 | 15.9 | 5.3 |
| `vit_small` | ViT-S, lr 5e-5 (baseline) | 12/15 | 0.649 | 70.1% | 0.661 | 62.8% | 0.475 | 5.4 | 14.6 | 2.5 |
| `vit_small_lr1e4` | ViT-S, lr 1e-4 | 10/15 | 0.669 | **71.1%** | **0.682** | 63.8% | 0.485 | 4.8 | 14.0 | 2.1 |
| `vit_small_lld` | ViT-S, lr 1e-4, layer decay 0.75 | 8/13 | 0.638 | 68.6% | 0.651 | 62.9% | 0.473 | 5.5 | 12.5 | 4.3 |
| **`vit`** | ViT-B, lr 3e-5, batch 32 | 8/13 | **0.679** | 70.7% | 0.679 | 64.7% | 0.490 | 7.3 | 12.7 | 7.6 |

Gaps are best-minus-worst subgroup accuracy on RAF-DB in percentage points (gender: female minus male, the
"unsure" group is left out). We did not test whether gaps differ significantly between runs.
Whole-notebook wall time on a Kaggle T4 (train, evaluate and Grad-CAM; some runs shared the machine with another
notebook): ViT-S runs 3,020-3,620 s, ViT-B 5,577 s. GPU memory was not recorded.

## Findings

- **Learning rate.** Val macro-F1 rises with the learning rate: 0.610 (1e-5), 0.649 (3e-5, 5e-5), 0.669 (1e-4).
  1e-5 under-trains (train accuracy 0.70 at its best epoch). The best ViT-S value is at the top of the range we
  tried, so a higher rate may do better; we did not test one.
- **Layer-wise decay hurt.** With the same base lr (1e-4), decay 0.75 gives 0.638 against 0.669 without it, and
  stops early at epoch 13. We did not investigate why; one possibility is that the earliest blocks get too small a
  step for FER-2013, which is far from ImageNet.
- **ViT-B has the best val macro-F1** (0.679), 0.010 above the best ViT-S run. That margin is one seed. On the
  FER-2013 test set it is not better (0.679 against 0.682 for ViT-S at 1e-4). It overfits sooner (train accuracy
  0.95 by epoch 13), has about four times the parameters (about 86M against 22M) and took about 1.5-1.8 times as long as the ViT-S runs.
- **In-domain gains do not carry over to RAF-DB**, as with the CNN. RAF-DB macro-F1 ranges only 0.473-0.504, and
  the best values (3e-5, 1e-5) belong to runs that were not the best on validation. We do not choose on RAF-DB.
- **Subgroup gaps** differ between runs (gender 4.6-8.5, race 2.1-7.6), but with one seed and no significance
  test we do not rank the runs by fairness.
- `vit_small_freeze6` (freeze the first six blocks) was not run; the issue asks for layer decay *or* freezing.

## Selected config

By the selection rule (FER-2013 val macro-F1), the winner is `vit` (ViT-B, lr 3e-5, batch 32), which is what
`configs/vit.yaml` already contains, so that file is unchanged. The runner-up on validation is `vit_small_lr1e4`
(ViT-S, 0.669), which is smaller, trains faster and has the smaller bias gaps; it is the alternative if
model size matters more than the 0.010 val difference.
