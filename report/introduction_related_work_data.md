# Introduction, related work and data

*Report sections 1-3. Method is in `method.md`; results and discussion in `results_and_discussion.md`.*

## 1. Introduction

Facial expression recognition (FER) maps a face image to an emotion category. It is used, or proposed,
in human-computer interaction, driver monitoring, education, healthcare and market research. In most of
these settings the system meets faces it was not trained on: different cameras, lighting, image quality
and people. Two questions therefore matter as much as headline accuracy. **Does the model still work
on data from a different source?** And **does it work equally well for everyone?** A model that is
accurate on average but systematically worse for one gender or age group can harm the people it
misreads.

Most FER work reports a single accuracy on the test split of the same dataset it was trained on. That
number answers neither question. We train three model families of increasing size and pretraining on
FER-2013 [1] and study them along three axes:

1. **Accuracy.** How do a CNN trained from scratch, an ImageNet-pretrained ResNet [5] and a Vision
   Transformer (ViT) [6] compare on FER-2013, under one shared training and evaluation pipeline?
2. **Transfer.** How much performance survives when the same models, unchanged, are tested on RAF-DB
   [2, 3], a dataset collected and labelled independently? Which emotions transfer worst, and why?
3. **Bias.** Using RAF-DB's gender, race and age annotations, do accuracy gaps between demographic
   groups exist, are they statistically supported, and does subgroup-balanced fine-tuning reduce them?

We also use Grad-CAM [12] to inspect which facial regions each model relies on.

**Contributions.**
- One configuration-driven pipeline (`src/train.py`, `src/evaluate.py`) in which the three model
  families differ only in their YAML config, so the comparisons are not artefacts of different code.
- A cross-dataset evaluation showing that plain accuracy hides most of the transfer loss. Balanced
  accuracy falls by 15-22 points on RAF-DB, against 2.5-9 points for plain accuracy, and the loss is
  concentrated in disgust, surprise, fear and anger. We also find that tuning for in-domain accuracy
  can make transfer worse.
- A subgroup audit with bootstrap confidence intervals and significance tests. It finds a consistent
  8-10 point accuracy gap between women and men across all models. A controlled mitigation
  experiment shows that fine-tuning on target-domain data, not balanced sampling, is what narrows it.

## 2. Related work

**FER datasets and benchmarks.** FER-2013 [1] was released for the ICML 2013 representation learning
challenge. It holds 35,887 48x48 greyscale faces gathered by Google image search and labelled with seven
emotions. The challenge paper estimates human accuracy at about 65 ± 5%, which reflects its label noise.
Well-tuned single CNNs reach about 73% [8], and CNN ensembles about 75% [9]. RAF-DB [2, 3] contains about
30,000 in-the-wild images from the internet, each labelled by around 40 independent annotators. Its
"basic" subset (15,339 images, seven classes) also provides manual gender, race and age annotations,
which makes it one of the few FER datasets that support a demographic audit. AffectNet [10] is larger
but was not used here.

**Architectures.** Deep CNNs replaced hand-crafted features (e.g. LBP, HOG) for FER around 2013-2016;
Li and Deng's survey [4] reviews this shift. ImageNet-pretrained residual networks [5] are a standard
transfer-learning baseline. Vision Transformers [6] split an image into patches and model their
relations with self-attention. Fine-tuned from ImageNet, they match or beat CNNs on many vision tasks.
Transformer-based FER models such as TransFER [7] report over 90% accuracy on RAF-DB when trained on it.
We use plain pretrained backbones rather than FER-specific designs, because our question is how the
families compare under identical training, not which reaches the highest score.

**Cross-dataset generalisation.** Torralba and Efros [13] showed that vision datasets carry a
recognisable "signature", and that models trained on one degrade on another. For FER, the gap comes from
differences in capture conditions, pose and resolution, and also in annotation: Zeng et al. [14] show
that FER datasets label the same expressions inconsistently, and Li and Deng [4] list cross-dataset
evaluation as an open problem. We measure this gap for FER-2013 -> RAF-DB per emotion, and report
balanced accuracy alongside accuracy because the two test sets have very different class mixes.

**Bias and fairness in face analysis.** Buolamwini and Gebru [15] found that commercial gender
classifiers had much higher error rates for darker-skinned women, which started much of the work on
demographic auditing in face analysis. For FER specifically, Xu et al. [11] audited models on RAF-DB and
CelebA. They found accuracy differences across gender, race and age, and compared mitigation strategies
including data augmentation and attribute-aware and disentangled representations. Our audit follows the
same idea on RAF-DB. We add bootstrap confidence intervals and two-proportion z-tests per gap, and a
controlled mitigation experiment that separates the effect of balanced sampling from the effect of
fine-tuning on target-domain data.

**Explanation methods.** Grad-CAM [12] weights a convolutional layer's activations by the gradient of the
class score to produce a coarse map of the image regions that drive a prediction. It is a standard way to
check whether FER models attend to expressive regions (eyes, brows, mouth) or to background and
artefacts.

## 3. Data

### 3.1 FER-2013 (training and in-domain test)

FER-2013 contains 35,887 greyscale 48x48 face images in seven classes. We use the official training set
(28,709 images) and hold out a stratified, seeded 10% of it for validation. The validation split is
used for checkpoint and configuration selection. The official public and private test sets (7,178
images together) are used only for the final in-domain test.

| Emotion | Train | Val | Test | Share of train |
|---|---|---|---|---|
| angry | 3,596 | 399 | 958 | 13.9% |
| disgust | 392 | 44 | 111 | 1.5% |
| fear | 3,687 | 410 | 1,024 | 14.3% |
| happy | 6,493 | 722 | 1,774 | 25.1% |
| sad | 4,347 | 483 | 1,247 | 16.8% |
| surprise | 2,854 | 317 | 831 | 11.0% |
| neutral | 4,469 | 496 | 1,233 | 17.3% |
| **Total** | **25,838** | **2,871** | **7,178** | |

Our exploratory analysis (`notebooks/01_eda.ipynb`, figures in `results/eda/`) found four properties that
shape the rest of the study:

- **Strong class imbalance.** Happy outnumbers disgust 16.6 to 1 in our training split, and disgust has
  only 392 training images. We therefore use an inverse-frequency class-weighted loss and report
  **macro-F1** as the main metric, because accuracy rewards predicting the majority classes.
- **Label noise.** 3,369 images have at least one exact pixel duplicate (1,516 distinct images). In 57 of
  these duplicate groups the copies carry **different labels**, direct evidence of annotation noise.
  14 images are near-blank (pixel standard deviation below 5), mostly in the training set.
- **Train/test leakage.** 531 test images (7.4% of the test set) also appear in the training or
  validation data. FER-2013 test scores, ours and those in the literature, are therefore somewhat
  optimistic. So is our validation macro-F1, since the random validation split shares duplicates with
  training. The RAF-DB results are not affected.
- **Brightness differs by class.** Mean pixel intensity ranges from 121 (sad) to 146 (surprise).
  This is a cue a model could partly exploit without reading the expression, and one reason to check
  attention maps.

### 3.2 RAF-DB (cross-dataset test and bias audit)

We use RAF-DB's basic subset: 15,339 aligned 100x100 colour face images with the same seven emotions,
split into 12,271 training and 3,068 test images. Its labels use a different class order, which we remap
to ours (`src/datasets.py`). **The RAF-DB test set is never used for training or model selection.** It
serves only as the cross-dataset test set and the population for the bias audit. The RAF-DB training
split is used in one place: the mitigation experiment, which fine-tunes on it. To keep the input format
fixed, every image, from either dataset and for every model, is converted to greyscale and resized to the
model's input size (48x48 for the CNN, 112x112 or 224x224 for ResNet and ViT; 3-channel models receive
the grey image replicated). The cross-dataset gap we measure is therefore not caused by colour.

The test set's class mix differs sharply from FER-2013's:

| Emotion | RAF-DB test n | Share | FER-2013 test share |
|---|---|---|---|
| angry | 162 | 5.3% | 13.3% |
| disgust | 160 | 5.2% | 1.5% |
| fear | 74 | 2.4% | 14.3% |
| happy | 1,185 | 38.6% | 24.7% |
| sad | 478 | 15.6% | 17.4% |
| surprise | 329 | 10.7% | 11.6% |
| neutral | 680 | 22.2% | 17.2% |

Because RAF-DB is 39% happy, a model that is good at happy and poor elsewhere keeps a high accuracy. We
therefore report macro-F1 and balanced accuracy (mean per-class recall) as well.

**Demographic annotations.** Each RAF-DB image has manually annotated gender (female, male, unsure),
race (Caucasian, African-American, Asian) and one of five age groups. The test set is far from
balanced:

| Attribute | Groups (test n, share) |
|---|---|
| Gender | female 1,620 (52.8%), male 1,249 (40.7%), unsure 199 (6.5%) |
| Race | Caucasian 2,351 (76.6%), Asian 483 (15.7%), African-American 234 (7.6%) |
| Age | 0-3: 329 (10.7%), 4-19: 486 (15.8%), 20-39: 1,662 (54.2%), 40-69: 502 (16.4%), 70+: 89 (2.9%) |

Small groups (70+ with 89 images, African-American with 234) give wide confidence intervals, which is
why every subgroup result is reported with a 95% bootstrap interval and a significance test rather than
as a bare accuracy. The attribute labels are themselves coarse, perceived categories assigned by
annotators. They simplify gender, race and age, and are a possible source of bias in their own right.

**Licence.** RAF-DB is available for non-commercial research only, and redistribution is not allowed. We
therefore commit only aggregate results (metrics, confusion counts, subgroup tables) and never RAF-DB
images, per-image predictions or RAF-DB Grad-CAM overlays.

## References

1. I. J. Goodfellow et al. Challenges in Representation Learning: A report on three machine learning
   contests. ICONIP 2013 / arXiv:1307.0414.
2. S. Li, W. Deng and J. Du. Reliable Crowdsourcing and Deep Locality-Preserving Learning for Expression
   Recognition in the Wild. CVPR 2017.
3. S. Li and W. Deng. Reliable Crowdsourcing and Deep Locality-Preserving Learning for Unconstrained
   Facial Expression Recognition. IEEE Transactions on Image Processing, 28(1), 2019.
4. S. Li and W. Deng. Deep Facial Expression Recognition: A Survey. IEEE Transactions on Affective
   Computing, 2020 / arXiv:1804.08348.
5. K. He, X. Zhang, S. Ren and J. Sun. Deep Residual Learning for Image Recognition. CVPR 2016.
6. A. Dosovitskiy et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.
   ICLR 2021.
7. F. Xue, Q. Wang and G. Guo. TransFER: Learning Relation-aware Facial Expression Representations with
   Transformers. ICCV 2021.
8. Y. Khaireddin and Z. Chen. Facial Emotion Recognition: State of the Art Performance on FER2013.
   arXiv:2105.03588, 2021.
9. C. Pramerdorfer and M. Kampel. Facial Expression Recognition using Convolutional Neural Networks:
   State of the Art. arXiv:1612.02903, 2016.
10. A. Mollahosseini, B. Hasani and M. H. Mahoor. AffectNet: A Database for Facial Expression, Valence,
    and Arousal Computing in the Wild. IEEE Transactions on Affective Computing, 2017.
11. T. Xu, J. White, S. Kalkan and H. Gunes. Investigating Bias and Fairness in Facial Expression
    Recognition. ECCV 2020 Workshops / arXiv:2007.10075.
12. R. R. Selvaraju et al. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based
    Localization. ICCV 2017.
13. A. Torralba and A. A. Efros. Unbiased Look at Dataset Bias. CVPR 2011.
14. J. Zeng, S. Shan and X. Chen. Facial Expression Recognition with Inconsistently Annotated Datasets.
    ECCV 2018.
15. J. Buolamwini and T. Gebru. Gender Shades: Intersectional Accuracy Disparities in Commercial Gender
    Classification. FAT* 2018.
