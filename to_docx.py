from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_BREAK
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_ROW_HEIGHT_RULE
from docx.enum.text import WD_LINE_SPACING
from pathlib import Path

out = Path("/mnt/data/Galaxy10_DECals_Concise_Research_Report.docx")

doc = Document()
sec = doc.sections[0]
sec.top_margin = Inches(0.62)
sec.bottom_margin = Inches(0.62)
sec.left_margin = Inches(0.72)
sec.right_margin = Inches(0.72)
sec.header_distance = Inches(0.3)
sec.footer_distance = Inches(0.3)

# Global styles
styles = doc.styles
styles["Normal"].font.name = "Arial"
styles["Normal"].font.size = Pt(9.5)
styles["Normal"].paragraph_format.space_after = Pt(4)
styles["Normal"].paragraph_format.line_spacing = 1.03

for style_name, size, color in [
    ("Title", 21, "17365D"),
    ("Heading 1", 15, "17365D"),
    ("Heading 2", 11, "2F5597"),
]:
    st = styles[style_name]
    st.font.name = "Arial"
    st.font.size = Pt(size)
    st.font.bold = True
    st.font.color.rgb = RGBColor.from_string(color)
    st.paragraph_format.space_before = Pt(5)
    st.paragraph_format.space_after = Pt(4)

# Header/footer
header = sec.header.paragraphs[0]
header.text = "GALAXY10 DECALS  |  GALAXY IMAGE CLASSIFICATION"
header.style = styles["Caption"]
header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
for r in header.runs:
    r.font.name = "Arial"
    r.font.size = Pt(8)
    r.font.color.rgb = RGBColor(100, 100, 100)

footer = sec.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = footer.add_run("Project report  •  17 November 2026  •  Page ")
run.font.name = "Arial"
run.font.size = Pt(8)
fld = OxmlElement('w:fldSimple')
fld.set(qn('w:instr'), 'PAGE')
footer._p.append(fld)

def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:fill'), fill)
    tcPr.append(shd)

def set_cell_text(cell, text, bold=False, color=None, size=8.5):
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.space_before = Pt(1)
    r = p.add_run(text)
    r.bold = bold
    r.font.name = "Arial"
    r.font.size = Pt(size)
    if color:
        r.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER

def add_table(headers, rows, widths=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    for i, h in enumerate(headers):
        set_cell_text(table.rows[0].cells[i], h, True, "FFFFFF", 8.5)
        shade(table.rows[0].cells[i], "17365D")
    for row in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            set_cell_text(cells[i], str(val), size=8.2)
    if widths:
        for row in table.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return table

def para(text, boldlead=None):
    p = doc.add_paragraph()
    p.paragraph_format.keep_together = True
    if boldlead and text.startswith(boldlead):
        r = p.add_run(boldlead)
        r.bold = True
        p.add_run(text[len(boldlead):])
    else:
        p.add_run(text)
    return p

def bullet(text):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.left_indent = Inches(0.2)
    p.paragraph_format.first_line_indent = Inches(-0.12)
    p.add_run(text)
    return p

# PAGE 1
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(1)
r = p.add_run("GALAXY IMAGE CLASSIFICATION USING DEEP LEARNING")
r.bold = True
r.font.name = "Arial"
r.font.size = Pt(18)
r.font.color.rgb = RGBColor(23, 54, 93)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(7)
r = p.add_run("A concise technical report based on the Galaxy10 DECals project")
r.italic = True
r.font.size = Pt(10)
r.font.color.rgb = RGBColor(90, 90, 90)

doc.add_heading("1. Introduction and project aim", level=1)
para("This project develops a machine-learning system for classifying astronomical galaxy images using the Galaxy10 DECals dataset. Its main aim is to learn visual patterns from labelled galaxy images and use those patterns to predict the class of an image that the model has not previously seen. The project combines data preparation, feature processing, deep-learning training, model evaluation and visual analysis.")
para("The directory structure shows the original dataset in data/raw/Galaxy10_DECals.h5, processed numerical arrays in data/processed/, PyTorch model code in src/cnn/, saved model checkpoints in models/checkpoints/, and evaluation outputs in results/. The figures and metric files indicate that the work extends beyond model fitting to include comparisons, per-class evaluation and inspection of incorrect predictions.")
doc.add_heading("2. Methodology", level=1)
doc.add_heading("2.1 Dataset and preparation", level=2)
para("The raw input is the Galaxy10 DECals HDF5 dataset. The project stores prepared features and labels as galaxy_features.npy and galaxy_labels.npy. This separation allows the training scripts to load numerical arrays without repeatedly processing the complete source dataset. The saved session/checkpoint artefacts also indicate that sample indices and experiment state were retained for reproducibility.")
para("The accompanying model-evaluation notebook documents a 32 × 32 image representation, with RGB values normalised to the range [0, 1]. It also describes a grayscale representation derived from the same resized RGB data for a separate Random Forest comparison. These details apply to the notebook's baseline/export workflow; the precise input tensor transformations used by the CNN should be confirmed directly in src/cnn/dataset.py and src/processing/process_image.py.")
doc.add_heading("2.2 Model development and training", level=2)
para("The principal deep-learning model is a Convolutional Neural Network (CNN), implemented with PyTorch in src/cnn/model.py and trained through src/cnn/train.py. CNNs are appropriate for image classification because convolutional filters can learn spatial features such as edges, shapes, textures and larger visual structures. The project log records a two-stage training strategy: an initial stage with the backbone frozen, followed by a fine-tuning stage with the backbone unfrozen.")
para("The first stage updates the trainable classification component while retaining the pretrained backbone weights. The second stage allows the backbone to adapt to the galaxy images. The best checkpoint is saved as best_cnn.pt, while checkpoint_latest.pt preserves the latest training state.")
doc.add_heading("2.3 Evaluation design", level=2)
para("The project includes training curves, a confusion matrix, per-class metrics, a classifier comparison and examples of misclassified images. The evaluation notebook describes train, validation and test indices, checks for duplicate or overlapping split indices, selects RGB versus grayscale representation using validation macro-F1, and reserves the test set for final evaluation. Its Random Forest export is a separate baseline workflow and must not be confused with the PyTorch CNN.")
doc.add_page_break()

# PAGE 2
doc.add_heading("3. Results", level=1)
doc.add_heading("3.1 CNN training progress", level=2)
para("The available training log records one frozen-backbone epoch followed by three unfrozen fine-tuning epochs. The frozen stage achieved 37.3% training accuracy and 30.6% validation accuracy. During fine-tuning, the recorded training and validation accuracies increased across the three epochs, reaching 78.5% and 81.5%, respectively.")
add_table(
    ["Training stage", "Epoch", "Training accuracy", "Validation accuracy"],
    [
        ["Frozen backbone", "1", "37.3%", "30.6%"],
        ["Unfrozen fine-tuning", "1", "60.1%", "72.2%"],
        ["Unfrozen fine-tuning", "2", "74.6%", "79.2%"],
        ["Unfrozen fine-tuning", "3", "78.5%", "81.5%"],
    ],
    [2.0, 0.7, 1.6, 1.7]
)
para("The best recorded validation accuracy is therefore 81.5%, associated with the final reported fine-tuning epoch and the best checkpoint selection. Validation accuracy estimates performance on data used for model selection; it is not the same as final test accuracy. The test result must be taken from results/metrics/cnn_test_summary.json or the corresponding test summary after confirming that the file reflects the completed run.")
doc.add_heading("3.2 Training behaviour", level=2)
para("The increase from 30.6% validation accuracy in the frozen stage to 81.5% after fine-tuning suggests that adapting the feature extractor substantially improved the model's ability to represent the target galaxy images. Training accuracy also increased steadily. At the final reported epoch, validation accuracy was 3.0 percentage points higher than training accuracy. This difference does not, by itself, demonstrate a problem; it can occur because of augmentation, regularisation or differences in the difficulty of the training and validation samples. The training configuration should be checked before assigning a cause.")
para("The log also reports that the first frozen epoch took approximately 2,603 seconds on CPU. This indicates a significant computational cost for local CPU training and supports using a suitable GPU for future experiments where available.")
doc.add_heading("3.3 Supporting results and stored outputs", level=2)
add_table(
    ["Artefact", "Role in the results"],
    [
        ["cnn_history.csv", "Epoch-level training and validation history."],
        ["cnn_training_curves.png", "Visual trend in loss and accuracy over training."],
        ["cnn_test_summary.json", "Intended summary of final CNN test performance."],
        ["test_per_class_metrics.csv", "Class-level precision, recall, F1 and support."],
        ["baseline_comparison.csv", "Comparison of the CNN with baseline classifier(s), subject to file contents."],
        ["classifier_comparison.csv", "Reported classifier comparison output."],
    ],
    [2.5, 4.2]
)
para("The directory also contains test_summary.json, original_class_distribution.csv and representative_galaxies.png. These artefacts can support a complete account of the experiment, but their numerical contents were not available in the reviewed material. Consequently, no test accuracy, macro-F1 score, sample count, or winning baseline result is asserted in this report.")
doc.add_page_break()

# PAGE 3
doc.add_heading("4. Figures and interpretation", level=1)
para("The following figures are already present in results/figures/. They should be inserted into the final submitted version at the points indicated below. The filenames are retained exactly as they appear in the project directory.")
doc.add_heading("Figure 1. Representative galaxy images", level=2)
para("File: results/figures/representative_galaxies.png. This figure should illustrate the visual variety of the dataset and provide context for the classification task. It can be used to show that galaxies differ in shape, orientation, structure and visible features.")
doc.add_heading("Figure 2. CNN training and validation curves", level=2)
para("File: results/figures/cnn_training_curves.png. This figure should be used to examine whether training and validation loss decrease and whether accuracy improves over successive epochs. The logged accuracy values show a clear improvement during fine-tuning. The plotted loss curves should be checked for divergence or unstable behaviour before making a stronger convergence claim.")
doc.add_heading("Figure 3. Confusion matrix", level=2)
para("File: results/figures/confusion_matrix.png. The confusion matrix compares true class labels with predicted labels. Strong values along the main diagonal indicate correct classification; off-diagonal values identify classes that the model confuses. The exact class pairs and error counts must be read from the actual image or its underlying predictions.")
doc.add_heading("Figure 4. Per-class performance", level=2)
para("File: results/figures/per_class_metrics.png. This figure is useful for identifying whether performance is consistent across all galaxy categories. A lower recall for a class means that more of its true examples are being missed, while lower precision means that predictions assigned to that class are more frequently incorrect.")
doc.add_heading("Figure 5. Misclassified examples", level=2)
para("File: results/figures/misclassified_examples.png. These examples provide a qualitative view of model errors. They should be discussed alongside the confusion matrix and per-class metrics, rather than treated as proof of a particular cause.")
doc.add_heading("Figure 6. Classifier comparison", level=2)
para("File: results/figures/classifier_comparison.png. This figure provides a visual comparison between the CNN and the baseline model(s), if the plotted series and metrics are confirmed from classifier_comparison.csv or baseline_comparison.csv. The comparison should use the same evaluation split and clearly identify whether the reported score is validation or test performance.")
doc.add_heading("5. Error analysis", level=1)
para("The project includes three useful sources for error analysis: the confusion matrix, per-class metrics and misclassified image examples. Together, they allow errors to be examined quantitatively and visually. However, the actual image pixels and CSV rows were not available for direct inspection during preparation of this report. It would therefore be misleading to name a specific pair of confused galaxy classes or claim that one class has the lowest F1-score without checking those artefacts.")
doc.add_page_break()

# PAGE 4
doc.add_heading("5. Error analysis (continued)", level=1)
para("Potential explanations to investigate include visual similarity between galaxy categories, class imbalance, faint or low-resolution structures, image orientation, overlapping morphological features, and label ambiguity. These are plausible sources of difficulty in galaxy morphology classification, not confirmed findings from this particular run.")
doc.add_heading("5.1 Recommended interpretation of errors", level=2)
bullet("Use the confusion matrix to identify the largest off-diagonal counts and name the true/predicted class pairs involved.")
bullet("Use test_per_class_metrics.csv to identify classes with comparatively low recall, precision or F1-score; include the support column so that small class counts are not overlooked.")
bullet("Inspect misclassified_examples.png to determine whether errors involve genuinely similar appearances, weak visual detail, unusual orientations or possible data/label problems.")
bullet("Compare original_class_distribution.csv with per-class results to assess whether uneven class representation may contribute to weaker performance.")
bullet("Check whether preprocessing is identical during training and prediction. A mismatch in resizing, channel order, normalisation or tensor layout can reduce real-world prediction reliability.")
doc.add_heading("5.2 Limitations and validity", level=2)
para("The available record is sufficient to report CNN validation progress, but not to make a verified claim about final test performance or the best-performing classifier. The project contains both a PyTorch CNN and a separate Random Forest comparison/export workflow. These should be reported as distinct models, with their evaluation datasets and metrics clearly labelled. The evaluation notebook includes split-integrity checks and a validation-based RGB/grayscale selection procedure; it also warns that historical experiments must be audited to establish whether the test set influenced earlier model choices.")
para("The directory listing shows data/splits/ as empty. If split indices are stored only inside session.pkl, that file should be preserved and documented, or a split manifest should be exported into data/splits/ so that the experiment can be reproduced. The final report should also record the exact dataset version, sample counts, class mapping, software versions and random seed.")
doc.add_heading("6. Conclusion", level=1)
para("The Galaxy10 DECals project implements an end-to-end framework for astronomical image classification, including data storage and processing, a PyTorch CNN, checkpoint management, prediction scripts, model evaluation and visual reporting. The recorded training history demonstrates substantial improvement after unfreezing and fine-tuning the backbone. Validation accuracy increased from 30.6% after the frozen stage to 81.5% after the third fine-tuning epoch, indicating that the training process learned useful image representations.")
para("The existing figures and metric files provide the foundation for a deeper assessment of class-specific strengths and weaknesses. Nevertheless, the final test accuracy and detailed error patterns must be confirmed from the saved metric files and figures before the results can be treated as final. Overall, the project demonstrates a promising approach to automated galaxy morphology classification, while further work should prioritise verified test evaluation, reproducible data splits, per-class error reduction and consistent preprocessing between training and deployment.")
doc.add_heading("Project files consulted", level=1)
for item in [
    "Project structure supplied for this report: data/, models/checkpoints/, src/, results/ and project-level scripts.",
    "Training record: results/logs/run_20261112_095526.log (training progress also reported in the project conversation).",
    "Metrics and reports: results/metrics/cnn_history.csv, cnn_test_summary.json, test_summary.json, test_per_class_metrics.csv, baseline_comparison.csv; results/reports/classifier_comparison.csv and original_class_distribution.csv.",
    "Figures: results/figures/representative_galaxies.png, cnn_training_curves.png, confusion_matrix.png, per_class_metrics.png, misclassified_examples.png and classifier_comparison.png.",
    "Model and code references: src/cnn/model.py, src/cnn/dataset.py, src/cnn/train.py, src/processing/process_image.py, predict.py and export_model.py.",
    "Evaluation/export reference: Galaxy10_Model_Training_Evaluation_Export.ipynb."
]:
    bullet(item)

# Compact style adjustments for lists/tables
for p in doc.paragraphs:
    if p.style.name.startswith("Heading"):
        p.paragraph_format.keep_with_next = True
for table in doc.tables:
    for row in table.rows:
        row.height_rule = WD_ROW_HEIGHT_RULE.AT_LEAST

doc.save(out)
print(f"Created: {out}")
