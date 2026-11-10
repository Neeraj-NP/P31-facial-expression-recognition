# Slides outline (issue #19)

12 slides, ~10-12 minutes. Figures come from `notebooks/02_results.ipynb` (#14) unless
noted. **Status:** ready = content and figure exist now; needs = waiting on a run.

| # | Slide | Key points | Figure | Speaker | Status |
|---|---|---|---|---|---|
| 1 | Title | P31: Facial Expression Recognition: Accuracy, Attention and Bias; team names | none | any | ready |
| 2 | Problem | 7-emotion classification from faces; why accuracy alone isn't enough (generalisation, fairness, interpretability) | example faces per class | Neeraj | ready |
| 3 | Data | FER-2013 (35,887 imgs, 48×48 gray) for training; RAF-DB (in-the-wild, gender/race/age labels) for cross-dataset + bias; 16.6x class imbalance, 7.4% test leakage | `results/eda/fer_class_distribution.png`, `fer_samples.png` | Neeraj | ready |
| 4 | Models | CNN from scratch → ResNet-18/50 (ImageNet transfer) → ViT fine-tuning; one shared training/eval pipeline | architecture diagram (simple boxes) | Rakshita | ready |
| 5 | Training setup | AdamW, warmup + cosine, class-weighted loss + label smoothing, augmentation, early stop on val macro-F1 | none (bullet slide) | Rakshita | ready |
| 6 | Tuning | CNN: 5 runs, best 70.2% (longer training + lighter reg help, augmentation essential). ResNet: frozen vs fine-tune, 112 vs 224, R18 vs R50 | tuning table | Neeraj + Rakshita | needs ResNet (#7) |
| 7 | Results: in-domain | FER-2013 test accuracy / macro-F1 for all 3 models; per-class F1 (fear hardest, happy easiest) | comparison bar chart + confusion matrix of best model | Pawan | needs #4, #5, #14 |
| 8 | Results: cross-dataset | FER-2013 → RAF-DB drop for each model; best in-domain ≠ best transfer (CNN: 70.2% → 61.2%) | in-domain vs cross-dataset bars | Neeraj | needs #9 |
| 9 | Bias audit | Accuracy per gender/race/age with 95% CIs; significant gaps (CNN: age 15pt, gender 10.4pt significant; race not) | `results/bias_report/*_by_model.png`, gap table | Rakshita | needs #10 (all models) |
| 10 | Mitigation | Subgroup-balanced fine-tuning on RAF-DB; gap and accuracy before vs after | before/after gap bars | Rakshita | needs #12 |
| 11 | Attention | Grad-CAM: models focus on mouth/eyes; fear errors attend to hands/background; mean CAM per subgroup | `gradcam_grid.png`, `mean_cam_by_gender.png` | Pawan | needs #11 |
| 12 | Conclusion | Best model + number; accuracy vs generalisation vs fairness trade-off; limitations (label noise, leakage, small subgroups like 70+) | none | Pawan | needs all |

## Notes for building the deck

- Keep one message per slide; put numbers in large text, details in speaker notes.
- Always show CIs on the bias slide, otherwise small groups (70+, n=89) look more
  certain than they are.
- Slides 9-10 are the project's main contribution beyond accuracy; give them the most time.
- Rehearsal and final timing are part of #20.
