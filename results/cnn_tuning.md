# CNN tuning (#6)

All runs: same seed, FER-2013 train/val split, model selected on FER-2013 **val** macro-F1.
RAF-DB test (n=3068) is only used for reporting, never for choosing a config.

| Run | Change vs baseline | Best epoch | Val macro-F1 | FER-2013 test acc | FER-2013 test macro-F1 | RAF-DB acc | RAF-DB macro-F1 | Gender gap | Age gap |
|---|---|---|---|---|---|---|---|---|---|
| `cnn` | baseline | 50/60 | 0.653 | 67.6% | 0.648 | 65.1% | **0.480** | 8.6 | 21.2 |
| `cnn_long` | 100 epochs (stopped at 93) | 78/93 | **0.667** | 69.6% | 0.667 | 64.3% | 0.473 | 8.0 | 19.1 |
| `cnn_lightreg` | dropout 0.2/0.1, no RandomErasing | 55/60 | 0.665 | 68.8% | 0.667 | 63.6% | 0.461 | 9.4 | 19.0 |
| `cnn_noweights` | no class weights | 56/60 | 0.660 | **69.9%** | **0.674** | **65.4%** | 0.450 | 8.3 | 19.6 |
| `cnn_noaug` | no augmentation | 35/47 | 0.623 | 66.5% | 0.627 | 58.2% | 0.411 | 9.0 | 17.6 |

Gaps are best-minus-worst subgroup accuracy on RAF-DB in percentage points (gender: female vs male).

## Findings

- **The baseline under-fitted.** Training longer (+1.4 val F1) or regularising less (+1.3) both help; with
  lighter regularisation train accuracy rises from 66% to 79% while val still improves.
- **Augmentation matters most.** Without it the model memorises the training set (99% train acc, stops at
  epoch 47) and loses 2 F1 points in-domain and 7 points on RAF-DB.
- **Class weights trade in-domain accuracy for rare-class transfer.** Without them FER-2013 test numbers are the
  best of all runs, but the model almost never predicts *disgust* on RAF-DB (3 times in 3068 images,
  F1 0.00 vs 0.18).
- **In-domain gains do not transfer.** Every change that improved FER-2013 left RAF-DB macro-F1 flat or lower
  (0.480 baseline vs 0.45-0.47): tuning on FER-2013 fits FER-2013-specific cues. Subgroup gaps barely move.
- Final config `configs/cnn_best.yaml` = `cnn_long` + `cnn_lightreg`, keeping class weights and augmentation.
