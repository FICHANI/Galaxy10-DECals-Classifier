from pathlib import Path

class ProjectConfig:
    """Galaxy10 — single source of truth for paths, seeds, hyper-params."""

    # ---------- paths ----------
    ROOT             = Path(__file__).resolve().parent
    DATA_PATH        = ROOT / "data/raw/Galaxy10_DECals.h5"
    #DATA_PATH = Path("/content/Galaxy10_DECals.h5")
    PROCESSED_PATH   = ROOT / "data/processed"
    SPLIT_PATH       = ROOT / "data/splits"
    RESULTS_PATH     = ROOT / "results"
    FIGURES_PATH     = RESULTS_PATH / "figures"
    REPORTS_PATH     = RESULTS_PATH / "reports"
    METRICS_PATH     = RESULTS_PATH / "metrics"
    CHECKPOINT_PATH  = ROOT / "models/checkpoints"

    # ---- CNN (reduced for CPU) ----
    CNN_IMAGE_SIZE       = 64      # was 128 — 4× fewer pixels, ~3× faster
    CNN_BATCH_SIZE       = 16      # was 32 — smaller = safer on WSL memory
    CNN_NUM_WORKERS      = 2       # keep 2 for HDF5 read parallelism
    CNN_EPOCHS_HEAD      = 1       # was 3  — 1 warmup epoch is enough
    CNN_EPOCHS_FINETUNE  = 3       # was 15 — 3 fine-tune epochs
    CNN_LR_HEAD          = 1e-3
    CNN_LR_FINETUNE      = 1e-4
    CNN_WEIGHT_DECAY     = 1e-4
    CNN_LABEL_SMOOTHING  = 0.1
    CNN_MODEL_PATH       = CHECKPOINT_PATH / "best_cnn.pt"
    CNN_HISTORY_PATH     = METRICS_PATH    / "cnn_history.csv"

    # ---------- HDF5 layout (Galaxy10 DECals) ----------
    HDF5_KEY_IMAGES  = "images"
    HDF5_KEY_LABELS  = "ans"      # labels/targets are stored under 'ans'
    HDF5_CHUNK       = 512        # rows read per HDF5 access (memory guard)

    # ---------- reproducibility ----------
    RANDOM_SEED      = 42         # used everywhere: numpy / random / sklearn

    # ---------- dataset ----------
    NUM_CLASSES             = 10
    CLASS_NAMES             = [
        "Disturbed",
        "Merging",
        "Round Smooth",
        "In-between Round Smooth",
        "Cigar Shaped Smooth",
        "Barred Spiral",
        "Unbarred Tight Spiral",
        "Unbarred Loose Spiral",
        "Edge-on without Bulge",
        "Edge-on with Bulge",
    ]
    MAX_SAMPLES_PER_CLASS   = 1000    # balance cap (memory + speed)

    # ---------- split ratios ----------
    TRAIN_SIZE        = 0.70
    VALIDATION_SIZE   = 0.15
    TEST_SIZE         = 0.15

    # ---------- pixel-based preprocessing ----------
    RESIZE_TO         = (32, 32)      # 32x32x3 = 3072 pixel features per image
    PROCESS_BATCH     = 256           # images per resize/normalize batch
    NORMALIZE_SCALE   = 255.0

    # ---------- training ----------
    BATCH_SIZE        = 16
    NUM_WORKERS       = 0
    CV_FOLDS          = 5
    N_JOBS            = -1            # sklearn parallel jobs

    # ---------- memory ----------
    MEMORY_LIMIT_GB   = None          # e.g. 4.0 — hard-cap the process (Linux/macOS)

    # ---------- filenames ----------
    FEATURES_FILE     = PROCESSED_PATH / "galaxy_features.npy"
    FEATURE_LABELS    = PROCESSED_PATH / "galaxy_labels.npy"

    @classmethod
    def create_directories(cls):
        """Called once at startup by main.py."""
        for d in (
            cls.PROCESSED_PATH,
            cls.SPLIT_PATH,
            cls.FIGURES_PATH,
            cls.REPORTS_PATH,
            cls.METRICS_PATH,
            cls.CHECKPOINT_PATH,
        ):
            d.mkdir(parents=True, exist_ok=True)