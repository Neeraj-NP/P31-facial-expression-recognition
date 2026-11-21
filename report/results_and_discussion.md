# Bias results and discussion

*Numbers come from the committed `results/*/eval_*/` files and are reproducible with
`python -m src.bias_report` and `python -m src.cross_dataset`.*

## Cross-dataset performance

All models are trained on FER-2013 only. RAF-DB (3,068 test images) is never used for training or
model selection, so the drop from FER-2013 to RAF-DB measures how well each model transfers. `cnn_best`, `resnet50` and `vit` (ViT-B) are the configurations selected on FER-2013 validation macro-F1
from each family's tuning runs; the other rows are baselines or tuning runs, shown for comparison.

| Model | FER-2013 acc / macro-F1 | RAF-DB acc / macro-F1 | Accuracy drop | Macro-F1 drop | Balanced-acc. drop |
|---|---|---|---|---|---|
| CNN baseline (untuned, `cnn`) | 67.6% / 0.648 | 65.1% / 0.480 | 2.5 pts | 0.167 | 16.5 pts |
| **CNN, tuned** (`cnn_best`) | 70.2% / 0.684 | 61.2% / 0.443 | 9.0 pts | 0.241 | 21.7 pts |
| ResNet-18 (`resnet`) | 69.1% / 0.664 | 60.4% / 0.449 | 8.7 pts | 0.215 | 20.4 pts |
| ResNet-18, 224px (`resnet_224`) | 70.3% / 0.675 | 61.3% / 0.465 | 9.0 pts | 0.210 | 19.6 pts |
| ResNet-18, frozen backbone | 34.5% / 0.316 | 36.5% / 0.233 | -2.0 pts* | 0.083* | 12.9 pts* |
| **ResNet-50** (`resnet50`) | 70.1% / 0.687 | 64.2% / 0.493 | 5.9 pts | 0.194 | 15.2 pts |
| ViT-S, untuned baseline (`vit_small`) | 70.1% / 0.661 | 62.8% / 0.475 | 7.4 pts | 0.186 | 18.4 pts |
| **ViT-B** (`vit`) | 70.7% / 0.679 | 64.7% / 0.490 | 6.0 pts | 0.189 | 17.4 pts |

Bold rows are the three selected configurations.

Balanced accuracy is the mean per-class recall; unlike accuracy it is not inflated by RAF-DB being 39%
happy. Full table, per-emotion figures and confusion matrices: `results/cross_dataset_analysis.md`
(`python -m src.cross_dataset`).

\*The frozen model is weak on both datasets, so its small "drop" is not evidence of good transfer.

In-domain the three selected models are tied (70.1-70.7% accuracy, macro-F1 0.679-0.687, one seed
each). On RAF-DB, ResNet-50 and ViT-B transfer better than the tuned CNN: macro-F1 0.493 and 0.490
against 0.443, and accuracy drops of 5.9 and 6.0 points against 9.0. ResNet-50 and ViT-B are within
noise of each other. The untuned CNN baseline (not selected) scores 65.1% / 0.480 and loses less than
either, so ResNet-50's advantage over a simple CNN is small (macro-F1 +0.013, one seed), not a clear
win. The ViT tuning runs show no consistent link between FER-2013 validation score and RAF-DB
score: RAF-DB macro-F1 spans only 0.473-0.504 across them, and the best (ViT-S at learning rate 3e-5)
was not the best on validation (`results/vit_tuning.md`).
Accuracy also flatters every
model here, because RAF-DB is 39% happy: balanced accuracy drops 15-22 points for every trained
model, against 2.5-9 for plain accuracy, so the macro-F1 and balanced-accuracy drops are the better
guide. The CNN shows a trade-off: tuning on FER-2013 raised FER-2013 accuracy (67.6% to 70.2%) and
lowered RAF-DB accuracy (65.1% to 61.2%) (`cnn_tuning.md`).

**Which emotions transfer.** Happy, sad and neutral keep their F1 on RAF-DB (changes of 0.10 or
less for all three selected models). Disgust collapses (F1 0.66 to 0.13 for the tuned CNN, 0.68 to
0.16 for ResNet-50, 0.58 to 0.14 for ViT-B) and is mostly read as anger, and surprise and fear lose
0.25-0.38 F1 in every selected model. Surprise is under-predicted rather than confused at random: its
RAF-DB recall falls to 0.29 (CNN), 0.38 (ResNet-50) and 0.33 (ViT-B) while precision stays high (0.91,
0.83 and 0.94). The tuned CNN also over-predicts anger on RAF-DB (484 predictions against 162 true
angry faces). The models differ most on angry: ViT-B keeps it (F1 -0.09), ResNet-50 loses 0.19 and
the CNN 0.28; this is where most of ViT-B's advantage over the CNN comes from. Fear is the
exception: ViT-B has the largest fear drop (-0.38; recall 0.62 to 0.39, where ResNet-50 keeps 0.50). The small classes (fear 74, disgust 160, angry 162 images) make these per-class rates
uncertain.

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
| ViT-S (untuned baseline) | 14.6 * (p = 0.007) | 5.4 * (p = 0.003) | 2.5 (p = 0.52) |
| ViT-B (`vit`) | 12.7 (p = 0.02) | 7.3 * (p < 0.001) | 7.6 (p = 0.04) |

Gaps are in accuracy points. Full tables are in `results/bias_report/gap_table.csv`.

Per-group accuracy for ResNet-50 (n in brackets): female 67.5% (1,620), male 59.4% (1,249);
age 0-3 66.0% (329), 4-19 66.0% (486), 20-39 63.1% (1,662), 40-69 66.7% (502), 70+ 55.1% (89);
caucasian 63.8% (2,351), asian 67.1% (483), african-american 62.4% (234).
For ViT-B: female 67.7%, male 60.4%; age 0-3 66.9%, 4-19 72.2%, 20-39 61.1%, 40-69 69.1%, 70+ 59.6%;
caucasian 64.4%, asian 68.3%, african-american 60.7% (same n).

### Findings

1. **Gender is the one consistent bias.** Every CNN and ResNet model recognises women 8-10 points
   more accurately than men (p < 0.001, intervals separated). Larger or higher-resolution ResNets
   barely change it (9.0 for ResNet-18, 8.2 at 224px, 8.1 for ResNet-50). ViT-B, the selected ViT,
   has a gap of 7.3 points, still clear (p < 0.001, intervals separated); its 0.8-point difference
   from ResNet-50 is well inside the width of the intervals (men 60.4% vs 59.4%, women 67.7% vs 67.5%),
   so we do not read it as ViT-B being fairer. The ViT-S baseline's smaller gap (5.4, p = 0.003) came
   from lower accuracy on women (65.0% vs 67.5% for ResNet-50) rather than better accuracy on men;
   by macro-F1 its gender difference even reverses (male 0.473, female 0.458, overlapping intervals),
   whereas ViT-B's does not (female 0.499, male 0.462).
2. **Age gaps vary widely between models.** They range from 9.2 (ResNet-18 at 224px) to 21.2 (the untuned
   CNN), with the tuned CNN at 15.2, ResNet-18 at 18.2, ResNet-50 at 11.7, ViT-B at 12.7, ViT-S at 14.6
   and the frozen model at 19.4.
   In every model except the frozen one, 70+ is the weakest group; it has only 89 images, so its
   interval is wide and its estimate the least certain. We did not test whether the differences between
   models are significant; with one seed they may be noise.
3. **Race gaps are small and mostly unclear, with one borderline case.** No race gap passes the
   interval check, and only ViT-B's reaches p < 0.05: 7.6 points, asian 68.3% vs african-american
   60.7% (p = 0.04). That p-value is not corrected for picking the best and worst of three groups, or
   for the many comparisons across models and attributes, and the african-american group has only
   234 images, so we treat it as unclear rather than as a finding. The other models' race gaps have
   p above 0.06. The wide intervals mean the rest is an absence of evidence, not evidence of
   fairness.
4. **Fairness did not track accuracy.** ResNet-50 and ViT-B are tied on accuracy and transfer, yet
   differ in their gaps (age 11.7 vs 12.7, race 4.7 vs 7.6; we did not test these differences).
   ResNet-50 has a smaller age gap than the tuned CNN (11.7 vs 15.2), but its gender gap is as large
   as the others'. Tuning the CNN widened its gender gap (8.6 to 10.4) while narrowing its age gap
   (21.2 to 15.2), so one change moved the two in opposite directions. ViT-S and ResNet-50 have the
   same FER-2013 accuracy but different gaps (gender 5.4 vs 8.1, age 14.6 vs 11.7).

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
accuracy (67.6% to 70.2%) but lowered RAF-DB accuracy (65.1% to 61.2%). ResNet-50 and ViT-B
transferred better than the tuned CNN (accuracy drops of 5.9 and 6.0 points vs 9.0), but pretraining
alone did not help: ResNet-18 transfers no better than the CNN (8.7 vs 9.0), and the untuned CNN
baseline transferred as well as anything. Full fine-tuning clearly beat our frozen-backbone run
(34.5% vs 69.1%), though that run is one linear head trained one way, not a ceiling for frozen
features. ViT-B and ResNet-50 are tied in-domain and on RAF-DB (64.7% / 0.490 vs 64.2% / 0.493), so
in our runs a transformer backbone brought no clear advantage over a CNN-family one, in either
transfer or fairness. Fairness followed neither pattern cleanly, as the persistent gender gap
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
- Each family's tuning was small (one factor at a time, one seed per run), and the three selected
  models are tied in-domain. ViT-B was chosen over the best ViT-S by a validation margin of only
  0.010 macro-F1, which one seed cannot separate from noise; on the FER-2013 test set the best ViT-S
  is not worse (macro-F1 0.682 vs 0.679). The layer-decay ViT run changed weight decay as well as the
  learning-rate schedule (see Method), and freezing blocks was not run. The bias mitigation was run
  on ResNet-50 only.
- We ran many comparisons (three attributes, several models). The race-gap p-value for ViT-B
  (0.04) in particular is uncorrected and should not be read as established.
