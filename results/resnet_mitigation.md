# Mitigation: subgroup-balanced fine-tuning (issue #12)

Starting point: `resnet50`, the best model from tuning (`resnet_tuning.md`). Each run fine-tunes it
for 10 epochs on RAF-DB's training split (lr 1e-5, cosine decay, 15% of RAF-DB train held out
for checkpoint selection) and is then evaluated on the RAF-DB test set, which is never used for
selection. Runs differ only in how training images are sampled:

- `none`: ordinary shuffling (the **control**)
- `gender`: sampling weights make each gender group equally likely. The three gender values (female,
  male, unsure) each get a third of the draws, so the "unsure" group (about 6% of the data) is
  sampled roughly 5x more often than its share; we did not exclude it
- `age`: sampling weights make each age group equally likely

The control is what makes the comparison meaningful: fine-tuning on RAF-DB raises RAF-DB
accuracy by itself, so only the difference from the control can be credited to balancing.

## Results

| Run | RAF-DB acc / macro-F1 | FER-2013 acc / macro-F1 | Gender gap (F-M) | Age gap | Race gap |
|---|---|---|---|---|---|
| `resnet50` (before) | 64.2% / 0.493 | 70.1% / 0.687 | 8.1 (p < 0.001) | 11.7 (p = 0.03) | 4.7 (p = 0.22) |
| `resnet50_mit_none` (control) | 79.4% / 0.649 | 63.4% / 0.598 | 4.1 (p = 0.007) | 10.9 (p = 0.02) | 1.9 (p = 0.34) |
| `resnet50_mit_gender` | 78.9% / 0.637 | 62.5% / 0.592 | 3.6 (p = 0.02) | 13.3 (p = 0.003) | 3.1 (p = 0.27) |
| `resnet50_mit_age` | 78.4% / 0.630 | 62.6% / 0.593 | 4.5 (p = 0.004) | 10.9 (p = 0.02) | 3.6 (p = 0.21) |

p = two-proportion z-test between the best and worst group, uncorrected (the best and worst of
several groups are picked, so p is optimistic). The 95% bootstrap intervals of the best and worst
group are separated only for the original gender gap, the control's and age-balanced run's gender
gaps, and the gender-balanced run's age gap. Overlapping intervals are a conservative test, so a gap
without that is not shown to be zero: the gender-balanced run's 3.6 gap (p = 0.02) is still a gap.
Per-group numbers are in each run's `eval_rafdb/subgroup_audit.csv`.

## Findings

- **Most of the gap reduction comes from fine-tuning on RAF-DB, not from balancing.** The control alone
  cuts the gender gap from 8.1 to 4.1 points and raises RAF-DB accuracy by 15 points.
- **Balancing adds little beyond the control.** Gender-balanced sampling gives a gender gap of 3.6 vs
  4.1 for the control (0.5 pts, well inside the 2-3 point width of the per-group intervals) and
  costs 0.5 pts of RAF-DB accuracy. Age-balanced sampling leaves the age gap unchanged vs the control
  (10.9 vs 10.9); its one visible gain is the smallest group, 70+ (n=89), at 74.2% vs 73.0%.
- **Gender balancing made the age gap worse** (13.3 vs 10.9 for the control; p = 0.003 vs 0.02), so fixing
  one attribute can disturb another.
- **There is a cost in the original domain.** All three runs fall from 70.1% to 62-63% on FER-2013,
  because the model adapts to RAF-DB and partly forgets FER-2013.
- **The 70+ group improves a lot from fine-tuning alone** (55.1% to 73.0% in the control), which
  suggests that group's low accuracy came from domain mismatch rather than from sampling frequency.

## Caveats

- One seed per condition, so differences of about a point or less should not be read as real.
- Sampling-based balancing is one of several mitigation methods; loss reweighting, adversarial
  debiasing or collecting more data were not tried.
- The subgroup attributes in RAF-DB are coarse; the 70+ group has only 89 test images.
- Per-image predictions and checkpoints are not in git (RAF-DB licence, file size).
