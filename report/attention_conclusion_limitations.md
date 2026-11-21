# Attention results, conclusion and limitations

*Covers #18. Final models are `cnn_best`, `resnet50` and `vit` (ViT-B), each chosen on FER-2013 validation
macro-F1; the untuned ViT-S (`vit_small`) is kept as a comparison. All numbers come from the committed
`results/*/eval_*/` files and `results/final_comparison.csv`. Results and discussion are in
`results_and_discussion.md`, ViT tuning in `results/vit_tuning.md`.*

## Attention results (Grad-CAM)

We computed Grad-CAM for the predicted class on the last convolutional block (CNN, ResNet-50) and on the last
transformer block's `norm1` output reshaped to a 14x14 patch grid (ViT). The grids show 4 random FER-2013 test
images per emotion, and the same 28 images appear in every model's grid:
`results/cnn_best/`, `results/resnet50/`, `results/vit_small/` and `results/vit/` under `explain_fer2013/gradcam_grid.png`.

What the grids show:

1. **CNN and ResNet-50 look at one compact region, usually the lower middle of the face.** ResNet-50 is the most
   regular: nearly every map is a single hotspot over the mouth and nose, whatever the emotion and whether or not the
   prediction is right, and the eyes rarely light up. The CNN is similar but less consistent, with some maps on the
   forehead, chin or the edge of the image.
2. **The ViTs spread their heat much more widely.** ViT-S shows many small patchy hotspots over the whole face. ViT-B
   is more diffuse still, with strong heat on hair, jaw, neck, hands and the image borders in several images.
3. **Some wrong predictions have heat away from the face.** For example ViT-B calls a happy face neutral with most of the
   heat along the bottom edge, and ViT-S calls an angry face neutral with heat on the top edge and corners.
4. **One image defeats all four models.** A stylised cartoon face labelled surprise is predicted as angry
   (ResNet-50), fear (CNN and ViT-B) and disgust (ViT-S).
5. **Where the model looks does not track accuracy here.** ResNet-50 has the most compact maps and ViT-B the most
   diffuse, yet their FER-2013 test accuracy is 70.1% and 70.7%.

How far to trust this:
- It is qualitative and rests on 28 images. We did not measure it (for example overlap with facial landmarks, or
  deletion and insertion scores).
- The ViT maps come from a different layer type and resolution (a 14x14 patch grid) than the convolutional maps, so
  part of the difference in how diffuse they look may come from the method and not from the model.
- Grad-CAM shows where the output is sensitive, not why the model decided. A hotspot on the mouth does not show that
  the model understood the expression.
- We generated mean Grad-CAM maps per RAF-DB gender group but did not analyse them, so we make no claim about
  whether the models look at different places for different groups. RAF-DB images are not stored in the repo.

## Conclusion

| Model | FER-2013 acc / macro-F1 | RAF-DB acc / macro-F1 | Acc. drop | Macro-F1 drop | Gender gap | Age gap | Race gap |
|---|---|---|---|---|---|---|---|
| CNN (tuned) | 70.2% / 0.684 | 61.2% / 0.443 | 9.0 pts | 0.241 | 10.4 | 15.2 | 5.3 |
| ResNet-50 | 70.1% / 0.687 | 64.2% / 0.493 | 5.9 pts | 0.194 | 8.1 | 11.7 | 4.7 |
| ViT-B | 70.7% / 0.679 | 64.7% / 0.490 | 6.0 pts | 0.189 | 7.3 | 12.7 | 7.6 |
| ViT-S, untuned (comparison) | 70.1% / 0.661 | 62.8% / 0.475 | 7.4 pts | 0.186 | 5.4 | 14.6 | 2.5 |

Gaps are best minus worst RAF-DB subgroup accuracy in points (gender: female minus male).

- **In-domain, the models are tied.** The three final models are within 0.6 points of accuracy (70.1-70.7%) and 0.008
  of macro-F1 (0.679-0.687), with one seed each, so we do not rank them.
- **Pretraining helped transfer, but not by much.** ResNet-50 and ViT-B lose about 6 points of accuracy on RAF-DB and
  reach macro-F1 0.49, against a 9-point loss and 0.443 for the tuned CNN. The untuned CNN baseline reached 0.480, so
  tuning on FER-2013 itself did not carry over to RAF-DB. ViT-B was chosen over ViT-S (at its best learning rate) by a
  validation margin of only 0.010, and ViT-S is slightly higher on FER-2013 test macro-F1 (0.682).
- **A gender gap exists in every model.** Women are recognised more accurately than men in all of them, and the gap is
  clear in the three final models (intervals do not overlap, p < 0.001): 10.4, 8.1 and 7.3 points. A larger or
  different architecture did not remove it. The age gap is 11.7 to 15.2 points; the interval check separates the groups
  only for the CNN. No race gap passes the interval check, although ViT-B's, the largest at 7.6 points, has p = 0.043
  before any correction for picking the best and worst of three groups, so we treat it as unclear. We did not test
  whether gaps differ between models.
- **What reduced the gap was target-domain data, not architecture.** For ResNet-50, fine-tuning on RAF-DB roughly halved
  the gender gap (8.1 to 4.1 points) at a cost of about 7 points of FER-2013 accuracy, while balanced sampling did
  little (see Mitigation in `results_and_discussion.md`). Mitigation was run on ResNet-50 only.
- **Attention maps differ strongly between architectures, but not in a way that predicts accuracy** (see above).

In short, accuracy, transfer, fairness and where a model looks are separate properties in our runs. No architecture was
best at all of them, and the largest gains came from pretraining (for transfer) and from seeing target-domain data (for
bias), not from a bigger or different network.

## Limitations

**Data and labels**
- FER-2013 labels come from crowd annotation and include duplicated and conflicting images; about 7% of its test images
  also appear in training. FER-2013 test accuracy, and the validation macro-F1 used to pick checkpoints and configs, are
  therefore somewhat optimistic. The RAF-DB numbers are not affected by this leakage.
- An expression label is not a person's felt emotion, and the label sets of the two datasets only roughly match.
- Demographic labels exist only in RAF-DB, so the audit measures bias in cross-dataset transfer. We cannot say whether
  the bias is already present in-domain.
- RAF-DB's labels are coarse and, as far as we know, assigned by annotators from appearance, not self-reported: gender
  has an "unsure" group (199 images, left out of the gender gap), race has three groups, age has five buckets. Groups
  are uneven (70+: 89 images; african-american: 234), so several intervals are wide, and an overlap or a large p-value
  is not "no bias".

**Training, tuning and compute**
- One seed per configuration, so differences of about a point between models or runs are within noise. We did not
  compute intervals for differences between models.
- Compute was a Kaggle T4. Tuning changed one factor at a time and was small: for ViT the best ViT-S learning rate was
  the highest tried (a higher one may do better), freezing blocks was not run, ViT-B was run with one configuration,
  and a ViT-B run takes about 93 minutes. GPU memory was not recorded.
- All models see greyscale images only. The ViT weights we start from (timm) were trained with mean and std 0.5, while
  every model here uses ImageNet normalisation so that the pipeline is shared; we did not test whether this costs
  accuracy.

**Explanations and scope**
- The attention analysis is qualitative (see above) and does not cover subgroups.
- p-values and interval checks do not account for choosing the best and worst group, and neither compares models.
- The demo and the models are research tools. They are not validated for judging real people's emotions, and photos
  unlike FER-2013 (pose, lighting, age, culture) should be expected to do worse.
