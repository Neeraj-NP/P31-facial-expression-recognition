# Bias results and discussion

*Draft for issue #17. Numbers come from the committed `results/*/eval_*/` files and are
reproducible with `python -m src.bias_report`. Items marked **TODO** wait on a run that has not
finished: the mitigation experiment (#12) and Pawan's ViT results (#5, #8, #11).*

## Cross-dataset performance

All models are trained on FER-2013 only. RAF-DB (3,068 test images) is never used for training or
model selection, so the drop from FER-2013 to RAF-DB measures how well each model transfers.

| Model | FER-2013 acc / macro-F1 | RAF-DB acc / macro-F1 | Accuracy drop |
|---|---|---|---|
| CNN from scratch (`cnn_best`) | 70.2% / 0.684 | 61.2% / 0.443 | 9.0 pts |
| ResNet-18 (`resnet`) | 69.1% / 0.664 | 60.4% / 0.449 | 8.7 pts |
| ResNet-18, 224px (`resnet_224`) | 70.3% / 0.675 | 61.3% / 0.465 | 9.0 pts |
| ResNet-18, frozen backbone | 34.5% / 0.316 | 36.5% / 0.233 | -2.0 pts* |
| **ResNet-50** (`resnet50`) | 70.1% / **0.687** | **64.2% / 0.493** | **5.9 pts** |
| ViT | TODO | TODO | TODO |

\*The frozen model is weak on both datasets, so its small "drop" is not evidence of good transfer.

ResNet-50 is the best model on both datasets and loses the least when moving to RAF-DB. The CNN
shows the opposite pattern: its tuning improved FER-2013 but made RAF-DB worse (`cnn_tuning.md`).

## Subgroup bias audit

RAF-DB's manual annotations give gender, race and a five-bucket age group. For each attribute we
report accuracy per group with a 95% bootstrap confidence interval, and the gap between the best and
worst group. A gap is called **significant** when the two groups' confidence intervals do not
overlap. Gender is compared as female vs male (the 199 "unsure" images are left out of the gap).

| Model | Age gap | Gender gap (F-M) | Race gap |
|---|---|---|---|
| CNN (`cnn_best`) | 15.2, significant | 10.4, significant | 5.3 |
| ResNet-18 | 18.2, significant | 9.0, significant | 3.1 |
| ResNet-18, 224px | 9.2 | 8.2, significant | 7.1 |
| ResNet-50 | 11.7 | 8.1, significant | 4.7 |
| ViT | TODO | TODO | TODO |

Gaps are in accuracy points. Where "significant" is missing, the gap is within what sampling noise
explains.

Per-group accuracy for ResNet-50 (n in brackets): female 67.5% (1,620), male 59.4% (1,249);
age 0-3 66.0% (329), 4-19 66.0% (486), 20-39 63.1% (1,662), 40-69 66.7% (502), 70+ 55.1% (89);
caucasian 63.8% (2,351), asian 67.1% (483), african-american 62.4% (234).

### Findings

1. **Gender is the one consistent bias.** Every model recognises women 8-10 points more accurately
   than men, and the gap is significant in all of them. Larger or higher-resolution models barely
   change it (9.0 for ResNet-18, 8.2 at 224px, 8.1 for ResNet-50).
2. **Age gaps shrink with better models.** The age gap falls from 18.2 (ResNet-18) to 11.7 (ResNet-50)
   and 9.2 (224px), and stops being significant for those two. The weakest group is always 70+, which
   has only 89 images, so its interval is wide and its estimate is the least certain.
3. **Race gaps are not significant for any model**, but the smallest group (african-american, n=234)
   also has wide intervals, so this is an absence of evidence rather than evidence of fairness.
4. **Fairness did not track accuracy.** The most accurate transfer model (ResNet-50) is fairer by
   age but not by gender. Tuning the CNN widened its gender gap (8.6 to 10.4) while narrowing its
   age gap (21.2 to 15.2), so one change moved the two in opposite directions.

### Why is accuracy lower for men?

We checked one explanation on ResNet-50's RAF-DB predictions: that the two genders simply have
different emotion mixes. They do differ (men: 34% happy, 9% angry; women: 44% happy, 3% angry), but
re-weighting men's accuracy to the women's emotion mix raises it only from 59.4% to 61.5%, closing
about a quarter of the 8.1-point gap. The rest comes from lower accuracy *within* emotions, most
clearly neutral (54% for men, 69% for women), and also happy, fear and angry. So the emotion mix is a
minor part and the model is genuinely less accurate on men's faces. We did not test why; possible
causes such as differences in how the two datasets show expression across gender are hypotheses.
This check used one model and the test set only.

## Mitigation: subgroup-balanced fine-tuning

**TODO (issue #12; Kaggle run in progress).** Procedure: fine-tune ResNet-50 on RAF-DB's training
split for 10 epochs at a low learning rate, under three conditions, and re-audit each on the RAF-DB
test set.

| Condition | Sampling |
|---|---|
| Control (`none`) | ordinary shuffling |
| Gender-balanced | equal total sampling probability for each gender group |
| Age-balanced | equal total sampling probability for each age group |

The control matters: fine-tuning on RAF-DB raises RAF-DB accuracy on its own, so only the *difference*
between the balanced runs and the control is evidence that balancing helped. Results table: TODO.

## Discussion

**Accuracy, transfer and fairness are separate questions.** Within the CNN, the model that fit
FER-2013 best transferred worst. Across architectures, pretraining and depth helped transfer most:
ResNet-50 loses 5.9 points on RAF-DB where the others lose about 9. Fine-tuning was essential, since
freezing the backbone cost 35 points. Fairness followed neither pattern cleanly, as the persistent
gender gap shows.

**Limitations.**
- Demographic labels exist only for RAF-DB, so the audit measures bias in cross-dataset transfer. FER-2013
  has no demographic labels, so we cannot say whether the bias is already present in-domain.
- One training seed per configuration; differences under about one point are within noise.
- Subgroups are coarse and unevenly sized (70+: 89 images; african-american: 234), so several
  comparisons have wide intervals. "Not significant" is not "no bias".
- Labels come from crowd annotation of faces and are themselves a possible source of bias.
- FER-2013 contains duplicated and conflicting-label images and 7.4% of its test set also appears in
  training (see the EDA, PR #21). In-domain scores are therefore somewhat optimistic; the cross-dataset
  numbers are not affected by this leakage.
- Gender is a three-way label in RAF-DB (female, male, unsure) and race a three-way label, which
  simplifies a continuous reality.
- ViT results (**TODO**) are not yet included, so conclusions about "all three model families" are
  provisional.
