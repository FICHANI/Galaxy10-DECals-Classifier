<div align="center">

# `GALAXY10  DECaLS CLASSIFIER`

### A reproducible machine-learning research pipeline for galaxy morphology

[![Python](https://img.shields.io/badge/Python-ML%20pipeline-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/CNN-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Galaxy10%20DECaLS-6C63FF)](https://zenodo.org/records/10845026)
[![Status](https://img.shields.io/badge/Release-Alpha-orange)](https://github.com/FICHANI/Galaxy10-DECals-Classifier)

**OBSERVE → PREPROCESS → TRAIN → EVALUATE → CLASSIFY**

*A research codebase for exploring galaxy morphology with classical machine learning and convolutional neural networks.*

[Repository](https://github.com/FICHANI/Galaxy10-DECals-Classifier) · [Dataset](https://zenodo.org/records/10845026) · [Report a bug](https://github.com/FICHANI/Galaxy10-DECals-Classifier/issues)

</div>

---

## `01  SYSTEM OVERVIEW`

**Galaxy10-DECals-Classifier** is a Python research project for classifying galaxy images into ten morphological categories. It combines a pixel-based feature pipeline, machine-learning experiments, a ResNet18-based CNN workflow, evaluation utilities, and visual research outputs.

The project is organised as an interactive research pipeline:

```text
RAW GALAXY IMAGES
       │
       ▼
[ A ] DATA EXPLORATION
       │  inspect images, labels and class distribution
       ▼
[ B ] IMAGE PROCESSING
       │  RGB conversion, resize, normalization, feature extraction
       ▼
[ C ] EXPERIMENT DESIGN
       │  reproducible stratified train / validation / test split
       ▼
[ D ] MODEL DEVELOPMENT
       │  classical ML experiments and CNN training workflow
       ▼
[ E ] MODEL SELECTION
       │  validation-based configuration selection
       ▼
[ F ] FINAL EVALUATION
          metrics, confusion matrix and diagnostic figures
```

> **Release status:** Alpha. This repository is a research and educational codebase. Check the issues and project outputs before relying on a result for scientific or operational decisions.

## `02  CAPABILITIES`

- **Galaxy image exploration** — inspect the HDF5 dataset, image dimensions and class distribution.
- **Preprocessing** — convert images to RGB, resize, normalize and flatten images into numerical feature vectors.
- **Classical ML experiments** — compare configured machine-learning classifiers using the processed features.
- **CNN workflow** — train and evaluate a ResNet18-based image classifier using PyTorch.
- **Reproducibility controls** — central configuration, fixed random seed and stratified data splits.
- **Evaluation and diagnostics** — classification metrics, per-class results, confusion matrices and training curves where the required artifacts are available.
- **Interactive tools** — a main research terminal and a post-pipeline utility menu.

## `03  THE TEN MORPHOLOGY CLASSES`

The project configuration defines these ten labels:

| Class ID | Morphology label |
|---:|---|
| 0 | Disturbed |
| 1 | Merging |
| 2 | Round Smooth |
| 3 | In-between Round Smooth |
| 4 | Cigar Shaped Smooth |
| 5 | Barred Spiral |
| 6 | Unbarred Tight Spiral |
| 7 | Unbarred Loose Spiral |
| 8 | Edge-on without Bulge |
| 9 | Edge-on with Bulge |

The pipeline's balanced-subset setting is configured for a maximum of 1,000 samples per class. The actual number selected depends on the available class counts and the code's selection logic.

## `04  DATASET`

This project uses **Galaxy10 DECaLS**, an astronomical image dataset containing 17,736 colour galaxy images at 256 × 256 pixels across ten morphology classes. The HDF5 file used by this project is expected to contain:

- `images` — galaxy image array
- `ans` — class labels

The dataset is not stored in this Git repository. It is approximately 2.54 GB and must be downloaded separately.

### Download

Official dataset record: <https://zenodo.org/records/10845026>

Direct file: <https://zenodo.org/records/10845026/files/Galaxy10_DECals.h5>

From the project root on Linux, macOS or WSL:

```bash
mkdir -p data/raw
wget -O data/raw/Galaxy10_DECals.h5 \
  https://zenodo.org/records/10845026/files/Galaxy10_DECals.h5
```

Verify the downloaded file:

```bash
echo "19AEFC477C41BB7F77FF07599A6B82A038DC042F889A111B0D4D98BB755C1571  data/raw/Galaxy10_DECals.h5" \
  | sha256sum -c -
```

Expected location:

```text
data/raw/Galaxy10_DECals.h5
```

The path is defined in `config.py` as `ProjectConfig.DATA_PATH`. If you store the dataset elsewhere, update that setting or provide the requested path when a tool prompts for it.

## `05  QUICK START`

### Requirements

- Python 3
- Git
- The packages listed in `requirements.txt`
- Sufficient free disk space for the dataset, generated arrays, model checkpoints and outputs
- A CUDA-capable GPU is recommended for CNN training; CPU execution may be considerably slower

### 1. Clone the repository

```bash
git clone https://github.com/FICHANI/Galaxy10-DECals-Classifier.git
cd Galaxy10-DECals-Classifier
```

### 2. Create and activate a virtual environment

Linux / macOS / WSL:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PyTorch installation fails, use the official selector to choose a build compatible with your operating system and CUDA setup: <https://pytorch.org/get-started/locally/>. Then install any remaining packages listed in `requirements.txt`.

### 4. Download the dataset

Follow the download instructions in [Section 04](#04--dataset). Confirm the HDF5 file is at `data/raw/Galaxy10_DECals.h5` before starting dataset-based tasks.

### 5. Launch the main research pipeline

```bash
python main.py
```

Follow the interactive terminal prompts. For a fresh run, begin with **Data Exploration** and then complete **Image Processing**. The processing stage writes the generated feature and label arrays into `data/processed/`.

### 6. Launch the post-pipeline tools

```bash
python run_it.py
```

The secondary menu provides analysis, diagnostics, CNN training/evaluation and prediction-related actions. Some actions require artifacts produced by the main pipeline; check the artifact-status screen and complete the relevant earlier task if an item is missing.

## `06  PIPELINE CONFIGURATION`

Most project settings are centralised in `config.py` under `ProjectConfig`.

| Setting | Current configured value | Purpose |
|---|---:|---|
| Random seed | `42` | Reproducibility |
| Train split | `0.70` | Training partition |
| Validation split | `0.15` | Model/configuration selection |
| Test split | `0.15` | Final evaluation |
| Maximum subset | `1,000` per class | Balanced-subset cap |
| Classical feature resize | `32 × 32` | Pixel feature preparation |
| Classical feature length | `3,072` values/image | `32 × 32 × 3` flattened RGB image |
| CNN image size | `64` | CNN input configuration |
| CNN batch size | `16` | CNN training batch |
| CNN head epochs | `1` | Initial training phase |
| CNN fine-tuning epochs | `3` | Fine-tuning phase |

These are the defaults currently present in `config.py`; changing them changes the experiment and should be recorded when reporting results.

## `07  GENERATED FILES & REPRODUCIBILITY`

The following files are generated locally and are intentionally not required to be committed to Git:

| Artifact | Role | How it is obtained |
|---|---|---|
| `data/processed/galaxy_features.npy` | Flattened, processed image features | Generated during Image Processing (Task B) |
| `data/processed/galaxy_labels.npy` | Labels aligned with the feature rows | Generated during Image Processing (Task B) |
| `models/checkpoints/session.pkl` | Saved interactive pipeline state | Written by `main.py` after successful tasks |

**Important:** A fresh clone does not include these generated artifacts. To recreate the feature arrays, download the original HDF5 dataset and run the main pipeline through Data Exploration and Image Processing.

`session.pkl` is a local Python pickle used by the current session save/restore implementation. It is not a portable interchange format. Only load a pickle you created yourself or otherwise fully trust. Some session-dependent analysis, including the current learning-curve implementation, may be unavailable without a compatible session file.

The trained CNN checkpoint, if included in the repository, is separate from the generated feature cache. If it is absent, train the CNN before using checkpoint-dependent inference or evaluation actions.

## `08  PROJECT STRUCTURE`

```text
Galaxy10-DECals-Classifier/
├── config.py                  # Central paths and experiment settings
├── main.py                    # Interactive research pipeline
├── run_it.py                  # Post-pipeline analysis and utility menu
├── requirements.txt           # Python dependencies
│
├── data/
│   ├── raw/                   # External HDF5 dataset (not committed)
│   ├── processed/              # Generated feature and label arrays
│   └── splits/                 # Experiment split artifacts, if generated
│
├── src/
│   ├── exploration.py          # Dataset exploration
│   ├── processing/             # Image processing utilities
│   ├── experimentation.py      # Experiment and split design
│   ├── training.py             # Classical model training
│   ├── evaluation.py           # Evaluation utilities
│   ├── analysis.py              # Research analysis and figures
│   ├── interpretation.py       # Interpretation and explanations
│   └── cnn/                    # CNN dataset, model, training and prediction
│
├── models/checkpoints/         # Model weights and local session state
└── results/
    ├── figures/                # Plots and visual diagnostics
    ├── metrics/                # Evaluation metrics
    └── reports/                # Generated reports and comparisons
```

This is a guide to the main project areas; files may be added as the project evolves.

## `09  EVALUATION & REPORTING`

Use the evaluation outputs produced by the project rather than relying on training accuracy alone. Review the available artifacts in `results/`, including:

- Confusion matrix
- Per-class precision, recall and F1 scores
- Overall test metrics
- Classifier comparison tables
- CNN training history and learning curves, where generated
- Misclassified-example and representative-galaxy figures

A model should not be described as successful solely because its training accuracy is high. Examine held-out test performance, per-class behaviour, class imbalance, leakage between splits, and differences between training and validation performance. Report the configuration, random seed, dataset version and split procedure alongside headline metrics.

## `10  TROUBLESHOOTING`

**`Galaxy10_DECals.h5` not found**  
Download the dataset and place it at `data/raw/Galaxy10_DECals.h5`, or update `ProjectConfig.DATA_PATH` in `config.py`.

**`galaxy_features.npy` or `galaxy_labels.npy` not found**  
Run `python main.py`, complete Data Exploration and Image Processing, then launch `python run_it.py` again.

**CNN training is very slow**  
The configured training can run on CPU, but it may take substantially longer than GPU training. Check that your installed PyTorch build detects CUDA if you expect GPU acceleration.

**A session cannot be restored**  
Start a fresh pipeline run. A session pickle is a local convenience artifact and may not be compatible across code or dependency changes.

**A prediction/evaluation action reports a missing checkpoint**  
Check whether `models/checkpoints/best_cnn.pt` exists. Train the CNN or obtain a compatible checkpoint before running checkpoint-dependent actions.

## `11  RESPONSIBLE USE`

This project is intended for education, experimentation and research into astronomical image classification. Model predictions are automated estimates and should not be treated as authoritative scientific classifications without domain review. Document data provenance, preprocessing, model configuration and evaluation limitations when sharing results.

Do not commit personal credentials, access tokens, private datasets or untrusted pickle files. Large source datasets and generated artifacts should be distributed through appropriate data hosting rather than ordinary Git commits.

## `12  DATASET CREDIT & REFERENCES`

Galaxy10 DECaLS is a dataset created from Galaxy Zoo classifications and images from the DESI Legacy Imaging Surveys. Please consult the dataset record and upstream project for its provenance, terms and recommended scholarly citation.

- Leung & Bovy — [Galaxy10 DECaLS dataset record, Zenodo](https://doi.org/10.5281/zenodo.10845026)
- Upstream dataset project — [henrysky/Galaxy10](https://github.com/henrysky/Galaxy10)

If you use this project or its results in academic work, cite the dataset and the relevant original research according to the dataset providers' guidance.

## `13  LICENSE & PROJECT STATUS`

**Project source-code license:** check the repository for a `LICENSE` file before reusing or redistributing this code. If no license file is present, the project should not be described as released under a particular open-source license until the owner adds one. The external dataset has its own provenance and usage terms; do not assume that a project-code license also applies to the dataset.

**Current release:** Alpha — interfaces, reproducibility features and workflows may change. Please report reproducible bugs through [GitHub Issues](https://github.com/FICHANI/Galaxy10-DECals-Classifier/issues).

---

<div align="center">

```text
[ GALAXY10 // DECaLS ]
DATA IN. SIGNAL OUT. MORPHOLOGY CLASSIFIED.
```

**Built for exploration. Structured for reproducibility.**

</div>
