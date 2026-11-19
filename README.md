# P31 — Facial Expression Recognition: Accuracy, Attention and Bias

Team 31: Neeraj Parmar, Rakshita K Biradar, Pawan Kushwah

Classify 7 facial emotions (FER-2013) and audit performance across demographic subgroups (RAF-DB).
Approach: CNN from scratch → pretrained ResNet → ViT fine-tuning; cross-dataset test; saliency maps.

## Setup (Windows)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Put the datasets in `data/` as described in [data/README.md](data/README.md).

## Usage

```powershell
# 1. train: results/<run-name>/ gets config.yaml, history.csv, best.pt
python -m src.train --config configs/cnn_best.yaml --run-name cnn_best
python -m src.train --config configs/resnet50.yaml --run-name resnet50
python -m src.train --config configs/vit_small.yaml --run-name vit_small

# 2. evaluate: in-domain, then cross-dataset + subgroup audit
python -m src.evaluate --run results/cnn_best --dataset fer2013
python -m src.evaluate --run results/cnn_best --dataset rafdb

# 3. compare models: cross-dataset table, per-emotion drop, confusion shift; bias gap table
python -m src.cross_dataset
python -m src.bias_report --runs results/cnn_best results/resnet50 results/vit_small

# 4. saliency maps
python -m src.explain --run results/cnn_best --dataset fer2013
python -m src.explain --run results/cnn_best --dataset rafdb --group-by gender
```

Without `--run-name`, runs are saved as `results/<model>_<timestamp>/`. Every evaluation writes
`metrics.json`, `confusion.csv`, `confusion_matrix.png`, a local `predictions.csv` (not committed) and,
for RAF-DB, `subgroup_audit.csv` with accuracy / macro-F1 and 95% bootstrap CIs per gender, race and
age group. Training needs a GPU; we trained on Kaggle T4s with `notebooks/kaggle_train.ipynb`.

## Experiments

Each experiment is its own file in `configs/` and is trained with `--run-name <config name>`, so results
land in `results/<config name>/`. On Kaggle, set `CONFIG = "<config name>"` in `notebooks/kaggle_train.ipynb`.
Optional keys (defaults reproduce the baselines):

| Key | Default | Effect |
|---|---|---|
| `data.augment` | `true` | flip / affine / colour jitter on training images |
| `data.random_erasing` | `0.25` | RandomErasing probability, `0` = off |
| `model.dropout` | `0.4` | CNN classifier dropout |
| `model.block_dropout` | `0.25` | CNN dropout after each conv block |
| `train.class_weights` | `true` | inverse-frequency class weights in the loss |

## Results

Trained on FER-2013, tested on the FER-2013 test set (in-domain) and the RAF-DB test set (cross-dataset,
never used for training or model selection). The CNN and ResNet rows are the configurations selected on
FER-2013 validation macro-F1; ViT-S is a single untuned baseline run, so its row is indicative.

| Model | FER-2013 acc / macro-F1 | RAF-DB acc / macro-F1 | Balanced-acc drop | Gender gap (F-M) | Age gap |
|---|---|---|---|---|---|
| CNN from scratch (`cnn_best`) | 70.2% / 0.684 | 61.2% / 0.443 | 21.7 pts | 10.4 pts | 15.2 pts |
| ResNet-50, ImageNet-pretrained (`resnet50`) | 70.1% / 0.687 | 64.2% / 0.493 | 15.2 pts | 8.1 pts | 11.7 pts |
| ViT-S, ImageNet-pretrained, untuned (`vit_small`) | 70.1% / 0.661 | 62.8% / 0.475 | 18.4 pts | 5.4 pts | 14.6 pts |

- **Cross-dataset transfer is the main weakness.** Balanced accuracy (mean per-class recall) falls by
  15-22 points on RAF-DB; plain accuracy hides much of this because RAF-DB is 39% happy, the class that
  transfers best. Disgust collapses (F1 0.66 -> 0.13 for the CNN; it is read as anger), and surprise,
  fear and anger also drop sharply. Details: [results/cross_dataset_analysis.md](results/cross_dataset_analysis.md).
- **In-domain gains did not transfer.** Tuning the CNN raised FER-2013 accuracy from 67.6% to 70.2% but
  lowered RAF-DB accuracy from 65.1% to 61.2% ([results/cnn_tuning.md](results/cnn_tuning.md)).
- **Every model is less accurate on men than on women** (8-10 points for the CNN and ResNets, 5.4 for
  ViT-S; all p < 0.01). Fine-tuning on RAF-DB
  halves the gap; subgroup-balanced sampling adds little on top
  ([results/resnet_mitigation.md](results/resnet_mitigation.md)).

Gaps are best-minus-worst group accuracy on RAF-DB; see [results/bias_report/](results/bias_report/) for
intervals and p-values, and [results/resnet_tuning.md](results/resnet_tuning.md) for the ResNet runs.
FER-2013 test numbers are slightly optimistic because about 7% of its test images also appear in training.

## Labels

All code uses one label order (`src/datasets.py`):
`angry, disgust, fear, happy, sad, surprise, neutral`. RAF-DB labels are remapped to it.

## Plan and owners

Deadline: **22 Nov 2026**. Tasks are tracked as GitHub Issues, one milestone per phase.

| Dates | Phase | Neeraj | Rakshita | Pawan |
|---|---|---|---|---|
| 2-5 Nov | Data & baselines | RAF-DB access, EDA, CNN baseline | ResNet-18 baseline | ViT-S baseline |
| 6-9 Nov | Tuning | CNN experiments | ResNet experiments | ViT experiments |
| 10-14 Nov | Research | Cross-dataset test | Subgroup bias audit | Grad-CAM attention analysis |
| 15-17 Nov | Extension & results | Per-class analysis, README results | Mitigation experiment | Comparison table, demo |
| 18-21 Nov | Report & slides | Intro, related work, data | Method, bias results, slides | Attention results, conclusion |

## References

1. Goodfellow et al., *Challenges in Representation Learning* (FER-2013), 2013. https://arxiv.org/abs/1307.0414
2. Dosovitskiy et al., *An Image is Worth 16x16 Words* (ViT), ICLR 2021. https://arxiv.org/abs/2010.11929
3. Li, Deng & Du, *Reliable Crowdsourcing and Deep Locality-Preserving Learning for Expression Recognition in the Wild* (RAF-DB), CVPR 2017.
