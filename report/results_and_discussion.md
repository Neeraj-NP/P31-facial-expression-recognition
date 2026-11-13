# Bias results and discussion

*Draft. Numbers come from the committed `results/*/eval_*/` files and are reproducible with
`python -m src.bias_report`. Items marked **TODO** are waiting on the ViT results.*

## Cross-dataset performance

All models are trained on FER-2013 only. RAF-DB (3,068 test images) is never used for training or
model selection, so the drop from FER-2013 to RAF-DB measures how well each model transfers. `cnn_best` and `resnet50` are the configurations selected on FER-2013 validation macro-F1;
the other rows are baselines or tuning runs, shown for comparison.

| Model | FER-2013 acc / macro-F1 | RAF-DB acc / macro-F1 | Accuracy drop | Macro-F1 drop |
|---|---|---|---|---|
| CNN baseline (untuned, `cnn`) | 67.6% / 0.648 | 65.1% / 0.480 | 2.5 pts | 0.168 |
| CNN, tuned (`cnn_best`) | 70.2% / 0.684 | 61.2% / 0.443 | 9.0 pts | 0.241 |
| ResNet-18 (`resnet`) | 69.1% / 0.664 | 60.4% / 0.449 | 8.7 pts | 0.215 |
| ResNet-18, 224px (`resnet_224`) | 70.3% / 0.675 | 61.3% / 0.465 | 9.0 pts | 0.210 |
| ResNet-18, frozen backbone | 34.5% / 0.316 | 36.5% / 0.233 | -2.0 pts* | 0.083* |
| **ResNet-50** (`resnet50`) | 70.1% / **0.687** | 64.2% / **0.493** | 5.9 pts | 0.194 |
| ViT | TODO | TODO | TODO | TODO |

\*The frozen model is weak on both datasets, so its small "drop" is not evidence of good transfer.

Of the two selected configurations, ResNet-50 transfers better than the tuned CNN: 64.2% vs 61.2% on
RAF-DB, macro-F1 0.493 vs 0.443, and a smaller drop in both. Across all rows it has the highest RAF-DB
macro-F1, but the untuned CNN baseline (not selected) scores 65.1% / 0.480 and loses less still.
ResNet-50's advantage over a simple CNN is therefore small (macro-F1 +0.013, one seed), not a clear win.
Accuracy also flatters every
model here, because RAF-DB is 39% happy; the macro-F1 drop is the better guide. The CNN shows a
trade-off: tuning on FER-2013 raised FER-2013 accuracy (67.6% to 70.2%) and lowered RAF-DB accuracy
(65.1% to 61.2%) (`cnn_tuning.md`).

## Subgroup bias audit

RAF-DB's manual annotations give gender, race and a five-bucket age group. For each attribute we
report accuracy per group with a 95% bootstrap confidence interval, and the gap between the best and
worst group. We judge each gap two ways. An asterisk means the best and worst groups' intervals are
**separated** (do not overlap), a simple but conservative check: overlapping intervals do not show the
gap is zero. The p-value is a two-proportion z-test between those two groups. Neither accounts for the
best and worst groups being picked out of several, so both flatter a gap somewhat. Gender is compared
as female vs male (the 199 "unsure" images are left out of the gap).

| Model | Age gap | Gender gap (F-M) | Race gap |
|---|---|---|---|
| CNN (`cnn_best`) | 15.2 * (p = 0.005) | 10.4 * (p < 0.001) | 5.3 (p = 0.11) |
| ResNet-18 | 18.2 * (p < 0.001) | 9.0 * (p < 0.001) | 3.1 (p = 0.42) |
| ResNet-18, 224px | 9.2 (p = 0.09) | 8.2 * (p < 0.001) | 7.1 (p = 0.07) |
| ResNet-50 | 11.7 (p = 0.03) | 8.1 * (p < 0.001) | 4.7 (p = 0.22) |
| ViT | TODO | TODO | TODO |

Gaps are in accuracy points. Full tables are in `results/bias_report/gap_table.csv`.

Per-group accuracy for ResNet-50 (n in brackets): female 67.5% (1,620), male 59.4% (1,249);
age 0-3 66.0% (329), 4-19 66.0% (486), 20-39 63.1% (1,662), 40-69 66.7% (502), 70+ 55.1% (89);
caucasian 63.8% (2,351), asian 67.1% (483), african-american 62.4% (234).

### Findings

1. **Gender is the one consistent bias.** Every model recognises women 8-10 points more accurately
   than men, and the gap is clear in all of them (p < 0.001, intervals separated). Larger or
   higher-resolution models barely change it (9.0 for ResNet-18, 8.2 at 224px, 8.1 for ResNet-50).
2. **Age gaps vary widely between models.** They range from 9.2 (ResNet-18 at 224px) to 21.2 (the untuned
   CNN), with the tuned CNN at 15.2, ResNet-18 at 18.2, ResNet-50 at 11.7 and the frozen model at 19.4.
   In every model except the frozen one, 70+ is the weakest group; it has only 89 images, so its
   interval is wide and its estimate the least certain. We did not test whether the differences between
   models are significant; with one seed they may be noise.
3. **No race gap reaches p < 0.05 for any model**, but the smallest group (african-american, n=234)
   has wide intervals, so this is an absence of evidence rather than evidence of fairness.
4. **Fairness did not track accuracy.** ResNet-50 transfers best of the two selected configurations
   and has a smaller age gap than the tuned CNN (11.7 vs 15.2), but its gender gap is as large as the
   others'. Tuning the CNN widened its gender gap (8.6 to 10.4) while narrowing its age gap (21.2 to
   15.2), so one change moved the two in opposite directions.

### Why is accuracy lower for men?

We checked one explanation on ResNet-50's RAF-DB predictions: that the two genders simply have
different emotion mixes. They do differ (men: 34% happy, 9% angry; women: 44% happy, 3% angry), but
re-weighting men's accuracy to the women's emotion mix raises it only from 59.4% to 61.5%, closing
about a quarter of the 8.1-point gap. The rest comes from lower accuracy *within* emotions, most
clearly neutral (54% for men, 69% for women), and also happy, fear and angry. So for ResNet-50 the
emotion mix is a minor part and the model is less accurate on men's faces even within the same
emotions. We did not test why; possible
causes such as differences in how the datasets show expression across gender are hypotheses. This
check used one model and the test set only; `python -m src.class_mix_check --run results/resnet50`
reproduces it (it needs the local `predictions.csv`, which is not committed because of the RAF-DB
licence).

## Mitigation: subgroup-balanced fine-tuning

We fine-tuned ResNet-50 on RAF-DB's training split for 10 epochs at a low learning rate (1e-5) under
three sampling conditions and re-audited each on the RAF-DB test set. The control (`none`) uses
ordinary shuffling; the gender- and age-balanced runs give each group of that attribute equal total
sampling probability. Every attribute value counts as a group, so in the gender run the small
"unsure" group (about 6% of the data) is sampled roughly 5x more often than its share. The control
matters: fine-tuning on RAF-DB raises RAF-DB accuracy on its own, so only the *difference* between a
balanced run and the control is evidence that balancing helped.

| Run | RAF-DB acc | FER-2013 acc | Gender gap (F-M) | Age gap | Race gap |
|---|---|---|---|---|---|
| ResNet-50 (before) | 64.2% | 70.1% | 8.1 (p < 0.001) | 11.7 (p = 0.03) | 4.7 (p = 0.22) |
| Control (no balancing) | 79.4% | 63.4% | 4.1 (p = 0.007) | 10.9 (p = 0.02) | 1.9 (p = 0.34) |
| Gender-balanced | 78.9% | 62.5% | 3.6 (p = 0.02) | 13.3 (p = 0.003) | 3.1 (p = 0.27) |
| Age-balanced | 78.4% | 62.6% | 4.5 (p = 0.004) | 10.9 (p = 0.02) | 3.6 (p = 0.21) |

(p-values as defined above. Note that the gender-balanced run's 3.6-point gap is still a gap, p = 0.02;
it is not removed.)

**Most of the improvement comes from fine-tuning on RAF-DB, not from balancing.** The control alone
halves the gender gap (8.1 to 4.1) and lifts RAF-DB accuracy by 15 points. Gender-balanced sampling
reaches a gender gap of 3.6, only 0.5 points below the control and well inside the 2-3 point width of
the per-group intervals, at a cost of 0.5 points of accuracy. Age-balanced sampling leaves the age gap
exactly where the control has it (10.9). Gender balancing also widened the age gap to 13.3 (p = 0.003
vs 0.02 for the control), so improving one attribute can disturb another. Every fine-tuned model loses
about 7 points on FER-2013, the price of adapting to RAF-DB. We therefore cannot claim that
subgroup-balanced sampling reduced bias in this setting; what reduced the gaps was training on data
from the target distribution. Loss reweighting, adversarial debiasing and collecting more data for
under-represented groups were not tried.

## Discussion

**Accuracy, transfer and fairness are separate questions.** Within the CNN, tuning raised FER-2013
accuracy (67.6% to 70.2%) but lowered RAF-DB accuracy (65.1% to 61.2%). ResNet-50 transferred best of the two selected configurations (a 5.9-point accuracy drop vs 9.0 for the
tuned CNN), but pretraining alone did not help: ResNet-18 transfers no better than the CNN (8.7 vs 9.0),
and the untuned CNN baseline transferred as well as anything. Full fine-tuning clearly
beat our frozen-backbone run (34.5% vs 69.1%), though that run is one linear head trained one way, not a
ceiling for frozen features. Fairness followed neither pattern cleanly, as the persistent gender gap
shows. The mitigation experiment points the same way: sampling tricks did little, while exposure to
data from the target distribution did most of the work, which suggests that part of the bias comes
from a mismatch between training and target data rather than from how often each group is seen. A
gender gap of about 4 points remains after fine-tuning, so that mismatch is not the whole explanation.

**Limitations.**
- Demographic labels exist only for RAF-DB, so the audit measures bias in cross-dataset transfer. FER-2013
  has no demographic labels, so we cannot say whether the bias is already present in-domain.
- One training seed per configuration; differences under about one point are within noise.
- Subgroups are coarse and unevenly sized (70+: 89 images; african-american: 234), so several
  comparisons have wide intervals. An overlap or a large p-value is not "no bias".
- Labels come from crowd annotation of faces and are themselves a possible source of bias.
- FER-2013 contains duplicated and conflicting-label images, and about 7% of its test images also
  appear in training (see the Data section). The FER-2013 test accuracies are therefore somewhat
  optimistic. So is the validation macro-F1 we used to pick checkpoints and configurations: the
  validation split is a random 10% of a training set that contains many exact duplicates. The
  cross-dataset numbers are not affected by this leakage.
- Gender is a three-way label in RAF-DB (female, male, unsure) and race a three-way label, which
  simplifies a continuous reality.
- ViT results (**TODO**) are not yet included, so conclusions about "all three model families" are
  provisional.
