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

## Labels

All code uses one label order (`src/datasets.py`):
`angry, disgust, fear, happy, sad, surprise, neutral`. RAF-DB labels are remapped to it.

## Plan and owners

| Week | Phase | Owner |
|---|---|---|
| 1 | Setup, data, EDA, label mapping | Neeraj |
| 2 | CNN baseline | Neeraj |
| 3 | ResNet transfer learning | Rakshita |
| 4 | ViT fine-tuning (Kaggle/Colab) | Pawan |
| 5 | Cross-dataset test + subgroup audit + saliency | Rakshita (audit), Pawan (saliency, cross-dataset) |
| 6 | Report, slides, demo | All |

## References

1. Goodfellow et al., *Challenges in Representation Learning* (FER-2013), 2013. https://arxiv.org/abs/1307.0414
2. Dosovitskiy et al., *An Image is Worth 16x16 Words* (ViT), ICLR 2021. https://arxiv.org/abs/2010.11929
3. Li, Deng & Du, *Reliable Crowdsourcing and Deep Locality-Preserving Learning for Expression Recognition in the Wild* (RAF-DB), CVPR 2017.
