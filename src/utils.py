# src/utils.py — add at the very top, BEFORE importing numpy/sklearn
import os

def configure_cpu_only_environment():
    """
    Prevent BLAS/OpenMP from spawning a thread per core.
    This is the single biggest CPU-only memory lever.
    """
    cores = os.cpu_count() or 4
    threads = max(1, cores // 2)     # or 1 for strict, reproducible runtime
    for var in ("OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, str(threads))
        
import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import ConfusionMatrixDisplay

from config import ProjectConfig


# ----------------------------------------------------------------------
# Reproducibility & memory
# ----------------------------------------------------------------------

def set_global_seed(seed: int = ProjectConfig.RANDOM_SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # scikit-learn: no global seed exists; every estimator in this
    # pipeline receives random_state=ProjectConfig.RANDOM_SEED explicitly.
    # This call documents the convention and protects any future code
    # that forgets to pass it.
    try:
        from sklearn.utils import check_random_state
        check_random_state(seed)
    except ImportError:
        pass

    # PyTorch (only if installed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark     = False
    except ImportError:
        pass

    # TensorFlow (only if installed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
    except ImportError:
        pass

    # Note: PYTHONHASHSEED only takes effect if set BEFORE the interpreter
    # starts — setting it at runtime here is a no-op for the current process.
    # Reproducibility for hash-ordered structures (sets, dict iteration)
    # must come from a fixed environment variable at launch, or by avoiding
    # iteration order dependence in the pipeline (which we do).


def apply_memory_limit(gb: float | None) -> None:
    """
    Hard-cap process memory on Linux/macOS so runaway allocations fail fast
    instead of swapping. No-op on Windows or when gb is None.
    """
    if gb is None:
        return
    try:
        import resource
        limit = int(gb * 1024**3)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        print(f"[memory] RLIMIT_AS = {gb} GB")
    except (ImportError, ValueError, OSError):
        print("[memory] resource module unavailable — skipping cap")


# ----------------------------------------------------------------------
# Visualisation helpers
# ----------------------------------------------------------------------

class GalaxyVisualizer:

    @staticmethod
    def PlotClassifierComparison(results: dict) -> None:
        if not results:
            raise ValueError("Empty results dict — nothing to plot.")

        # ------------------------------------------------------------------
        # 1. tabulate
        # ------------------------------------------------------------------
        rows = []
        for name, r in results.items():
            rows.append({
                "config":       name,
                "val_accuracy": float(r["val_accuracy"]),
                "model":        r.get("params", {}).get("model", "?"),
            })
        df = pd.DataFrame(rows).sort_values("val_accuracy", ascending=True)

        # ------------------------------------------------------------------
        # 2. plot
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(11, max(4, 0.5 * len(df) + 2)))

        bars = ax.barh(df["config"], df["val_accuracy"], color="steelblue")

        # highlight the best configuration
        best_idx = df["val_accuracy"].idxmax()
        best_name = df.loc[best_idx, "config"]
        for bar, name in zip(bars, df["config"]):
            if name == best_name:
                bar.set_color("darkorange")

        # annotate each bar with its accuracy
        for bar, acc in zip(bars, df["val_accuracy"]):
            ax.text(
                acc + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{acc:.3f}",
                va="center", fontsize=9,
            )

        ax.set_xlabel("Validation accuracy")
        ax.set_title(
            f"Classifier comparison on validation set "
            f"(best = {best_name}, highlighted)"
        )
        ax.set_xlim(0.0, 1.0)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        out = ProjectConfig.FIGURES_PATH / "classifier_comparison.png"
        plt.savefig(out, dpi=140)
        plt.close(fig)
        print(f"Saved: {out}")

    @staticmethod
    def PlotMetricComparison(per_class_metrics: dict) -> None:
        for key in ("precision", "recall", "f1"):
            if key not in per_class_metrics:
                raise KeyError(
                    f"per_class_metrics missing '{key}'. "
                    f"Keys present: {list(per_class_metrics.keys())}"
                )

        precision = np.asarray(per_class_metrics["precision"], dtype=float)
        recall    = np.asarray(per_class_metrics["recall"],    dtype=float)
        f1        = np.asarray(per_class_metrics["f1"],        dtype=float)

        n_classes = ProjectConfig.NUM_CLASSES
        if not (len(precision) == len(recall) == len(f1) == n_classes):
            raise ValueError(
                f"Expected {n_classes} values per metric; got "
                f"precision={len(precision)}, recall={len(recall)}, "
                f"f1={len(f1)}."
            )

        labels = ProjectConfig.CLASS_NAMES
        x      = np.arange(n_classes)
        width  = 0.27

        fig, ax = plt.subplots(figsize=(13, 6))
        ax.bar(x - width, precision, width, label="precision")
        ax.bar(x,         recall,    width, label="recall")
        ax.bar(x + width, f1,        width, label="F1")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.0)
        ax.set_title("Per-class precision / recall / F1 — Test Set")
        ax.legend(loc="lower right")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        out = ProjectConfig.FIGURES_PATH / "per_class_metrics.png"
        plt.savefig(out, dpi=140)
        plt.close(fig)
        print(f"Saved: {out}")

    @staticmethod
    def PlotLabeledConfusionMatrix(cm, labels=ProjectConfig.CLASS_NAMES):
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)

        # annotate each cell with its count
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, str(int(cm[i, j])),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8,
                )

        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title("Confusion Matrix — Test Set")
        plt.tight_layout()

        out = ProjectConfig.FIGURES_PATH / "confusion_matrix.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved: {out}")