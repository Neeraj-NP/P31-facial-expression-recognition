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
# 1. train (results/<model>_<timestamp>/ gets config.yaml, history.csv, best.pt)
python -m src.train --config configs/cnn.yaml
python -m src.train --config configs/resnet.yaml
python -m src.train --config configs/vit.yaml

# 2. evaluate: in-domain, then cross-dataset + subgroup audit
python -m src.evaluate --run results/<run> --dataset fer2013
python -m src.evaluate --run results/<run> --dataset rafdb

# 3. saliency maps
python -m src.explain --run results/<run> --dataset fer2013
python -m src.explain --run results/<run> --dataset rafdb --group-by gender
```

Every evaluation writes `metrics.json`, `predictions.csv`, `confusion_matrix.png` and (RAF-DB)
`subgroup_audit.csv` with accuracy / macro-F1 and 95% bootstrap CIs per gender, race and age group.

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
