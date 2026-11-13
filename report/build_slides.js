// Builds report/slides.pptx. Needs pptxgenjs (npm install pptxgenjs, anywhere on NODE_PATH):
//   node report/build_slides.js [out.pptx]
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const REPO = path.resolve(__dirname, "..");
const OUT = process.argv[2] || path.join(__dirname, "slides.pptx");

// pptxgenjs writes Office's default palette; put the deck's own colors into the theme part.
async function applyTheme(file, theme) {
  const JSZip = require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] }));
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  const slots = ["dk1", "lt1", "dk2", "lt2", "accent1", "accent2", "accent3", "accent4", "accent5", "accent6", "hlink", "folHlink"];
  const scheme = `<a:clrScheme name="${theme.name}">` + slots.map((k) => `<a:${k}><a:srgbClr val="${theme.colors[k].toUpperCase()}"/></a:${k}>`).join("") + "</a:clrScheme>";
  const part = "ppt/theme/theme1.xml";
  const xml = (await zip.file(part).async("string"))
    .replace(/<a:clrScheme\b[\s\S]*?<\/a:clrScheme>/, () => scheme)
    .replace(/(<a:(?:theme|fontScheme)\b[^>]*?\bname=")[^"]*"/g, (_, head) => `${head}${theme.name}"`);
  zip.file(part, xml);
  for (const name of Object.keys(zip.files)) {
    if (!name.endsWith(".xml")) continue;
    const x = await zip.file(name).async("string");
    if (x.includes('<c:crossesAt val="autoZero"/>')) zip.file(name, x.replaceAll('<c:crossesAt val="autoZero"/>', '<c:crosses val="autoZero"/>'));
  }
  for (const name of Object.keys(zip.files)) if (zip.files[name].dir) delete zip.files[name];
  fs.writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}
const THEME = {
  name: "P31 Emotion",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B2A41", lt1: "FFFFFF", dk2: "324A5F", lt2: "EEF3F7",
    accent1: "2A9D8F", accent2: "E76F51", accent3: "E9C46A",
    accent4: "264653", accent5: "8AB17D", accent6: "9AAEC0",
    hlink: "2A9D8F", folHlink: "324A5F",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Facial Expression Recognition: Accuracy, Attention and Bias";
pres.author = "Team P31";
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "TITLE_SLIDE",
  background: { color: HEX.dk1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 1.3, w: 8.6, h: 1.8, fontSize: 38, bold: true, color: C.background1, align: "left", valign: "top", margin: 0 }, text: "" } },
  ],
});

pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: HEX.lt1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9, h: 0.9, fontSize: 28, bold: true, color: C.text1, align: "left", valign: "middle", margin: 0 }, text: "" } },
    { text: { text: "P31  |  Facial Expression Recognition", options: { x: 0.5, y: 5.2, w: 6, h: 0.3, fontSize: 10, color: C.text2, margin: 0, isTextBox: true } } },
  ],
  slideNumber: { x: 9.0, y: 5.2, w: 0.5, h: 0.3, fontSize: 10, color: C.text2, align: "right" },
});

function content(title, notes, section) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  s.addNotes(notes);
  return s;
}

function card(s, x, y, w, h, name) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: C.background2 }, line: { color: C.background2, width: 0 }, rectRadius: 0.08, objectName: name });
}

function dot(s, x, y, n, name) {
  s.addText(String(n), { x, y, w: 0.45, h: 0.45, shape: pres.shapes.OVAL, fill: { color: C.accent1 }, color: C.background1, bold: true, fontSize: 16, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: name });
}

const chartBase = {
  x: 0.5, y: 1.35, w: 5.6, h: 3.75,
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt",
  legendFontFace: "+mn-lt", titleFontFace: "+mn-lt",
  catAxisLabelFontSize: 12, valAxisLabelFontSize: 12, dataLabelFontSize: 12, legendFontSize: 12, titleFontSize: 14,
  catAxisLabelColor: HEX.dk2, valAxisLabelColor: HEX.dk2, dataLabelColor: HEX.dk1, legendColor: HEX.dk2, titleColor: HEX.dk1,
  valGridLine: { color: "DCE4EA", size: 0.5 }, catGridLine: { style: "none" },
  showTitle: true, showValue: true, dataLabelFormatCode: "0.0",
  valAxisMinVal: 0,
};

// ---------- 1. Title ----------
pres.addSection({ title: "Setup" });
{
  const s = pres.addSlide({ masterName: "TITLE_SLIDE", sectionTitle: "Setup" });
  s.addText("Facial Expression Recognition: Accuracy, Attention and Bias", { placeholder: "title" });
  s.addText("Team P31: Neeraj Parmar, Rakshita K Biradar, Pawan Kushwah", { x: 0.7, y: 3.6, w: 8.6, h: 0.45, fontSize: 18, color: C.background1, margin: 0, isTextBox: true });
  s.addText("Deep learning project, November 2026", { x: 0.7, y: 4.1, w: 8.6, h: 0.4, fontSize: 14, color: C.accent6, margin: 0, isTextBox: true });
  s.addNotes("Introduce the project: we classify faces into 7 emotions, but the goal is to also ask whether the models generalise, whether they are fair, and where they look.");
}

// ---------- 2. Problem ----------
{
  const s = content("Accuracy alone is not enough",
    "Task: 7 emotions (angry, disgust, fear, happy, sad, surprise, neutral). We train on FER-2013 only, then ask three questions. Generalise: test on RAF-DB, a different dataset. Fair: break RAF-DB results down by gender, race and age. Attention: Grad-CAM heatmaps.", "Setup");
  s.addText("Task: classify a face photo into one of 7 emotions: angry, disgust, fear, happy, sad, surprise, neutral.", { x: 0.5, y: 1.3, w: 9, h: 0.7, fontSize: 16, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  const items = [
    ["Does it generalise?", "Train on FER-2013, test on RAF-DB: a different dataset the model has never seen."],
    ["Is it fair?", "Accuracy broken down by gender, race and age group, with confidence intervals."],
    ["Where does it look?", "Grad-CAM heatmaps show which face regions drive each prediction."],
  ];
  items.forEach((it, i) => {
    const x = 0.5 + i * 3.1;
    card(s, x, 2.15, 2.8, 2.85, `Card ${i + 1}`);
    dot(s, x + 0.2, 2.35, i + 1, `Number ${i + 1}`);
    s.addText(it[0], { x: x + 0.2, y: 2.95, w: 2.4, h: 0.4, fontSize: 18, bold: true, color: C.text1, margin: 0, isTextBox: true });
    s.addText(it[1], { x: x + 0.2, y: 3.4, w: 2.4, h: 1.5, fontSize: 14, color: C.text2, margin: 0, valign: "top", isTextBox: true });
  });
}

// ---------- 3. Data ----------
{
  const s = content("Two datasets, two jobs",
    "FER-2013: 35,887 images, 48x48 grayscale, used for training, validation (10% of train) and in-domain test. RAF-DB: used only for testing; it has gender, race and age labels, which is what makes the bias audit possible. EDA (PR #21): happy has 16.6x more training images than disgust; 531 FER-2013 test images (7.4%) also appear in train/val.", "Setup");
  const tiles = [
    ["35,887", "FER-2013 images (48×48 grayscale, 7 emotions): training"],
    ["3,068", "RAF-DB test images, in the wild, with gender, race and age labels"],
    ["16.6×", "class imbalance in FER-2013: happy vs disgust"],
    ["7.4%", "of FER-2013 test images also appear in training"],
  ];
  tiles.forEach((t, i) => {
    const x = 0.5 + (i % 2) * 4.6, y = 1.4 + Math.floor(i / 2) * 1.8;
    card(s, x, y, 4.4, 1.6, `Stat tile ${i + 1}`);
    s.addText(t[0], { x: x + 0.2, y: y + 0.2, w: 1.9, h: 1.2, fontSize: 34, bold: true, color: C.accent1, margin: 0, valign: "middle", isTextBox: true, fontFace: "Cambria" });
    s.addText(t[1], { x: x + 2.15, y: y + 0.2, w: 2.05, h: 1.2, fontSize: 14, color: C.text1, margin: 0, valign: "middle", isTextBox: true });
  });
}

// ---------- 4. Models ----------
{
  const s = content("Three models, one pipeline",
    "CNN: about 5M parameters, trained from scratch (Neeraj). ResNet-18/50: ImageNet-pretrained, new 7-class head, 112 px (Rakshita). ViT: ImageNet-pretrained transformer, 224 px (Pawan; results pending). All three use the same training and evaluation scripts, selected by a YAML config.", "Setup");
  const cols = [
    ["CNN from scratch", ["About 5M parameters", "4 conv blocks, 48×48 grayscale", "No pretraining"]],
    ["ResNet-18 / 50", ["ImageNet-pretrained, new 7-class head", "112 px, 3-channel input", "Fully fine-tuned"]],
    ["Vision Transformer", ["ImageNet-pretrained (timm)", "224 px input", "Fine-tuned; results pending"]],
  ];
  cols.forEach((c, i) => {
    const x = 0.5 + i * 3.1;
    card(s, x, 1.4, 2.8, 3.5, `Model card ${i + 1}`);
    s.addText(c[0], { x: x + 0.2, y: 1.6, w: 2.4, h: 0.5, fontSize: 18, bold: true, color: C.text1, margin: 0, isTextBox: true });
    s.addText(c[1].map((t, k) => ({ text: t, options: { bullet: true, breakLine: k < c[1].length - 1 } })),
      { x: x + 0.2, y: 2.25, w: 2.45, h: 2.5, fontSize: 14, color: C.text2, margin: 0, valign: "top", paraSpaceAfter: 8, isTextBox: true });
  });
}

// ---------- 5. Training ----------
{
  const s = content("The same recipe for every model",
    "One training script, one split (10% of FER-2013 train held out as validation, same seed), one selection metric (validation macro-F1). Differences between models therefore come from the model, not from the pipeline.", "Setup");
  const rows = [
    ["Optimiser", "AdamW, short warmup, cosine decay"],
    ["Loss", "Cross-entropy, label smoothing 0.1, class weights"],
    ["Augmentation", "Flip, small affine, colour jitter, random erasing"],
    ["Model selection", "Best validation macro-F1, with early stopping"],
    ["Comparison", "Same pipeline, split and metric for all models"],
  ];
  rows.forEach((r, i) => {
    const y = 1.4 + i * 0.72;
    dot(s, 0.5, y, i + 1, `Step ${i + 1}`);
    s.addText([{ text: r[0] + ": ", options: { bold: true } }, { text: r[1] }],
      { x: 1.2, y: y - 0.02, w: 8.3, h: 0.5, fontSize: 16, color: C.text1, margin: 0, valign: "middle", isTextBox: true });
  });
}

// ---------- 6. Tuning ----------
pres.addSection({ title: "Results" });
{
  const s = content("Fine-tuning matters most; ResNet-50 is best",
    "ResNet tuning (results/resnet_tuning.md). FER-2013 test accuracy / RAF-DB accuracy: ResNet-18 69.1 / 60.4; frozen backbone 34.5 / 36.5; 224 px 70.3 / 61.3; ResNet-50 70.1 / 64.2. ResNet-50 was selected on validation macro-F1 (0.681), not on RAF-DB. One seed per run, so differences under about a point are noise.", "Results");
  s.addChart(pres.charts.BAR, [
    { name: "FER-2013", labels: ["ResNet-18", "Frozen", "224 px", "ResNet-50"], values: [69.1, 34.5, 70.3, 70.1] },
    { name: "RAF-DB", labels: ["ResNet-18", "Frozen", "224 px", "ResNet-50"], values: [60.4, 36.5, 61.3, 64.2] },
  ], { ...chartBase, barDir: "col", barGapWidthPct: 60, chartColors: [HEX.accent1, HEX.accent2], title: "Test accuracy (%)", showLegend: true, legendPos: "b", valAxisMaxVal: 80, dataLabelPosition: "outEnd", objectName: "Tuning chart" });
  s.addText([
    { text: "Frozen backbone: 34.5% vs 69.1% fine-tuned (this head, trained this way).", options: { bullet: true, breakLine: true } },
    { text: "224 px adds 1.2 points on FER-2013: no clear benefit.", options: { bullet: true, breakLine: true } },
    { text: "ResNet-50 is the best ResNet on RAF-DB (64.2%), chosen on validation macro-F1.", options: { bullet: true } },
  ], { x: 6.4, y: 1.5, w: 3.1, h: 3.4, fontSize: 14, color: C.text1, margin: 0, valign: "top", paraSpaceAfter: 10, isTextBox: true });
}

// ---------- 7. Transfer ----------
{
  const s = content("ResNet-50 beats the tuned CNN on RAF-DB",
    "FER-2013 to RAF-DB, accuracy drop / macro-F1 drop: untuned CNN baseline 67.6 to 65.1 (2.5) / 0.648 to 0.480 (0.168); tuned CNN 70.2 to 61.2 (9.0) / 0.684 to 0.443 (0.241); ResNet-18 69.1 to 60.4 (8.7) / 0.664 to 0.449 (0.215); ResNet-18 at 224 px 70.3 to 61.3 (9.0) / 0.675 to 0.465 (0.210); ResNet-50 70.1 to 64.2 (5.9) / 0.687 to 0.493 (0.194). Of the two configurations selected on FER-2013 validation (tuned CNN and ResNet-50), ResNet-50 transfers better, but the untuned CNN baseline (not selected) loses even less and scores 65.1% / 0.480 on RAF-DB, so ResNet-50's advantage over a simple CNN is small (macro-F1 +0.013, one seed). RAF-DB is 39% happy, which flatters accuracy, so the macro-F1 drop is the better guide. ViT row pending Pawan's runs.", "Results");
  s.addChart(pres.charts.BAR, [
    { name: "Accuracy drop", labels: ["CNN untuned", "CNN tuned", "ResNet-18", "ResNet-18 224 px", "ResNet-50"], values: [2.5, 9.0, 8.7, 9.0, 5.9] },
    { name: "Macro-F1 drop (points)", labels: ["CNN untuned", "CNN tuned", "ResNet-18", "ResNet-18 224 px", "ResNet-50"], values: [16.8, 24.1, 21.5, 21.0, 19.4] },
  ], { ...chartBase, barDir: "col", barGapWidthPct: 50, chartColors: [HEX.accent6, HEX.accent1], title: "Lost on RAF-DB vs FER-2013 (points)", showLegend: true, legendPos: "b", valAxisMaxVal: 30, dataLabelPosition: "outEnd", catAxisLabelFontSize: 11, objectName: "Transfer chart" });
  s.addText([
    { text: "Of the two selected configurations, ResNet-50 transfers better than the tuned CNN.", options: { bullet: true, breakLine: true } },
    { text: "The untuned CNN baseline (not selected) transferred as well: 65.1% / 0.480.", options: { bullet: true, breakLine: true } },
    { text: "RAF-DB is 39% happy, so accuracy flatters; macro-F1 is the better guide.", options: { bullet: true, breakLine: true } },
    { text: "ViT: pending.", options: { bullet: true } },
  ], { x: 6.4, y: 1.5, w: 3.1, h: 3.5, fontSize: 14, color: C.text1, margin: 0, valign: "top", paraSpaceAfter: 8, isTextBox: true });
}
// ---------- 8. Gender ----------
pres.addSection({ title: "Bias" });
{
  const s = content("Every model is worse on men",
    "RAF-DB accuracy by gender (female / male): CNN 65.5 / 55.1, ResNet-18 63.8 / 54.8, ResNet-18 224 px 65.0 / 56.8, ResNet-50 67.5 / 59.4. The gap is clear in all four (p < 0.001, intervals separated). We checked ResNet-50 only (python -m src.class_mix_check): men and women have different emotion mixes, but re-weighting men to the women's mix lifts their accuracy only from 59.4 to 61.5, so the mix explains about a quarter of the gap.", "Bias");
  s.addChart(pres.charts.BAR, [
    { name: "Female", labels: ["CNN", "ResNet-18", "ResNet-18 224 px", "ResNet-50"], values: [65.5, 63.8, 65.0, 67.5] },
    { name: "Male", labels: ["CNN", "ResNet-18", "ResNet-18 224 px", "ResNet-50"], values: [55.1, 54.8, 56.8, 59.4] },
  ], { ...chartBase, barDir: "col", barGapWidthPct: 60, chartColors: [HEX.accent1, HEX.accent2], title: "RAF-DB accuracy by gender (%)", showLegend: true, legendPos: "b", valAxisMaxVal: 80, dataLabelPosition: "outEnd", objectName: "Gender chart" });
  s.addText("8–10 points", { x: 6.4, y: 1.45, w: 3.1, h: 0.6, fontSize: 28, bold: true, color: C.accent2, margin: 0, fontFace: "Cambria", isTextBox: true });
  s.addText("gap between women and men, clear in every model", { x: 6.4, y: 2.05, w: 3.1, h: 0.7, fontSize: 14, color: C.text2, margin: 0, valign: "top", isTextBox: true });
  s.addText("In ResNet-50, emotion mix explains only about a quarter of it: the model is less accurate on men even within the same emotions.",
    { x: 6.4, y: 3.0, w: 3.1, h: 1.8, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });
}

// ---------- 9. Gap table ----------
{
  const s = content("Age gaps vary; the gender gap stays",
    "Gaps are best group minus worst group in accuracy points; gender is female minus male. An asterisk = the two groups' 95% bootstrap intervals do not overlap (a conservative test); p-values are two-proportion z-tests between best and worst group, uncorrected. Age gaps range from 9.2 to 21.2 across models (p = 0.03 for ResNet-50, 0.09 for 224 px); we did not test whether the differences between models are real. No race gap reaches p < 0.05, but african-american has n=234 so its intervals are wide. The 70+ group has only 89 images. ViT row pending.", "Bias");
  const head = (t) => ({ text: t, options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk1 }, align: "center", valign: "middle" } });
  const cell = (t, sig) => ({ text: t, options: { bold: !!sig, color: HEX.dk1, align: "center", valign: "middle" } });
  const name = (t) => ({ text: t, options: { color: HEX.dk1, align: "left", valign: "middle" } });
  const rows = [
    [head("Model"), head("Age gap"), head("Gender gap"), head("Race gap")],
    [name("CNN (best)"), cell("15.2 *", true), cell("10.4 *", true), cell("5.3")],
    [name("ResNet-18"), cell("18.2 *", true), cell("9.0 *", true), cell("3.1")],
    [name("ResNet-18, 224 px"), cell("9.2"), cell("8.2 *", true), cell("7.1")],
    [name("ResNet-50"), cell("11.7"), cell("8.1 *", true), cell("4.7")],
    [name("ViT"), cell("pending"), cell("pending"), cell("pending")],
  ];
  s.addTable(rows, { x: 0.5, y: 1.4, w: 6.0, colW: [2.1, 1.25, 1.4, 1.25], rowH: 0.5, fontSize: 14, fontFace: "Calibri", border: { type: "solid", pt: 0.75, color: "C9D4DC" }, fill: { color: HEX.lt1 }, objectName: "Gap table" });
  s.addText("* intervals of best and worst group do not overlap (conservative). Unmarked is not zero: ResNet-50 age gap p = 0.03. Points.", { x: 0.5, y: 4.45, w: 6.0, h: 0.65, fontSize: 12, color: C.text2, margin: 0, valign: "top", isTextBox: true });
  s.addText([
    { text: "Age gaps range from 9 to 21 points across models.", options: { bullet: true, breakLine: true } },
    { text: "Race: no gap reaches p < 0.05, but groups are small.", options: { bullet: true, breakLine: true } },
    { text: "70+ has only 89 images, so it is the least certain.", options: { bullet: true } },
  ], { x: 6.8, y: 1.5, w: 2.7, h: 3.3, fontSize: 14, color: C.text1, margin: 0, valign: "top", paraSpaceAfter: 10, isTextBox: true });
}

// ---------- 10. Mitigation ----------
{
  const s = content("Balanced sampling added little",
    "Mitigation (results/resnet_mitigation.md): fine-tune ResNet-50 on RAF-DB train for 10 epochs, lr 1e-5. Gender gap (F-M): before 8.1; control (no balancing) 4.1; gender-balanced 3.6; age-balanced 4.5. The control alone halves the gap, so fine-tuning on target data did most of the work; balancing added about 0.5 points, within noise. Gender balancing widened the age gap to 13.3 (control 10.9). Cost: FER-2013 accuracy falls from 70.1 to about 63. Loss reweighting and adversarial debiasing were not tried.", "Bias");
  s.addChart(pres.charts.BAR, [
    { name: "Gender gap", labels: ["Before", "Control", "Gender-balanced", "Age-balanced"], values: [8.1, 4.1, 3.6, 4.5] },
  ], { ...chartBase, barDir: "col", barGapWidthPct: 60, chartColors: [HEX.accent6, HEX.accent1, HEX.accent1, HEX.accent1], title: "Gender gap, female minus male (points)", showLegend: false, valAxisMaxVal: 10, dataLabelPosition: "outEnd", objectName: "Mitigation chart" });
  s.addText([
    { text: "Fine-tuning on RAF-DB alone halves the gap (8.1 → 4.1).", options: { bullet: true, breakLine: true } },
    { text: "Gender balancing: 3.6 vs 4.1, within noise.", options: { bullet: true, breakLine: true } },
    { text: "It also widened the age gap (13.3 vs 10.9).", options: { bullet: true, breakLine: true } },
    { text: "Cost: FER-2013 accuracy falls from 70.1% to about 63%.", options: { bullet: true } },
  ], { x: 6.4, y: 1.5, w: 3.1, h: 3.4, fontSize: 14, color: C.text1, margin: 0, valign: "top", paraSpaceAfter: 10, isTextBox: true });
}

// ---------- 11. Attention ----------
pres.addSection({ title: "Wrap-up" });
{
  const s = content("Where does ResNet-50 look?",
    "Grad-CAM grid for ResNet-50 on FER-2013 test images (results/resnet50/explain_fer2013/gradcam_grid.png): rows are true classes, titles read true -> predicted. In these 28 examples the heat sits mostly on the mouth and nose; the eyes are rarely highlighted even for fear and surprise, and wrong predictions look much like right ones. This is a small qualitative sample. From the CNN run (PR #22), Grad-CAM also focused on mouth and eyes, with several fear errors attending to hands or background. ViT attention maps and per-subgroup mean heatmaps are Pawan's (#11) and pending.", "Wrap-up");
  s.addText([
    { text: "ResNet-50 on FER-2013 test images (28 examples shown).", options: { bullet: true, breakLine: true } },
    { text: "Heat sits mostly on the mouth and nose. The eyes are rarely highlighted, even for fear and surprise.", options: { bullet: true, breakLine: true } },
    { text: "Wrong predictions (for example fear → sad) look much like correct ones.", options: { bullet: true, breakLine: true } },
    { text: "ViT attention and per-group heatmaps: pending.", options: { bullet: true } },
  ], { x: 0.5, y: 1.45, w: 6.4, h: 3.5, fontSize: 16, color: C.text1, margin: 0, valign: "top", paraSpaceAfter: 12, isTextBox: true });
  s.addImage({ path: REPO + "/results/resnet50/explain_fer2013/gradcam_grid.png", x: 7.3, y: 1.35, w: 2.14, h: 3.75, altText: "Grad-CAM heatmaps of ResNet-50 on FER-2013 test faces, one row per emotion", objectName: "Grad-CAM grid" });
}

// ---------- 12. Conclusion ----------
{
  const s = content("What we learned",
    "Three messages. Transfer: of the two selected configurations ResNet-50 transfers better (64.2% vs 61.2% on RAF-DB), but the untuned CNN reached 65.1%, so the margin over a simple CNN is small, and pretraining alone (ResNet-18) did not improve transfer. Bias: women are recognised 8-10 points better than men in every model, and age gaps vary between models (9 to 21 points). Mitigation: balanced sampling added about 0.5 points over simply training on target data, so we cannot claim it reduced bias. Limits: one seed per run, small subgroups, audit on RAF-DB only, and the ViT comparison is still pending.", "Wrap-up");
  const items = [
    ["Transfer", "ResNet-50 transfers better than the tuned CNN (64.2% vs 61.2% on RAF-DB); the untuned CNN reached 65.1%."],
    ["Bias persists", "Women beat men by 8–10 points in every model; age gaps vary from 9 to 21 points."],
    ["Mitigation", "Balanced sampling added about 0.5 points over training on target data."],
  ];
  items.forEach((it, i) => {
    const x = 0.5 + i * 3.1;
    card(s, x, 1.4, 2.8, 2.6, `Takeaway ${i + 1}`);
    s.addText(it[0], { x: x + 0.2, y: 1.6, w: 2.4, h: 0.4, fontSize: 18, bold: true, color: C.accent1, margin: 0, isTextBox: true, fontFace: "Cambria" });
    s.addText(it[1], { x: x + 0.2, y: 2.1, w: 2.4, h: 1.8, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  });
  s.addText("Limits: one seed per run, small subgroups (70+ has 89 images), audit on RAF-DB only. ViT comparison pending.",
    { x: 0.5, y: 4.25, w: 9, h: 0.7, fontSize: 14, color: C.text2, margin: 0, valign: "top", isTextBox: true });
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
