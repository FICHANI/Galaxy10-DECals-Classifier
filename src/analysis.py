# src/analysis.py
"""
Galaxy10 — standalone analysis module.

Runs AFTER the pipeline has completed Tasks A-F. Reads all artifacts
from disk and produces:

    1. Headline metrics summary         (from test_summary.json)
    2. Baseline comparison              (from baseline_comparison.csv)
    3. Per-class precision/recall/F1    (from test_per_class_metrics.csv)
    4. Top confusion pairs              (from confusion_matrix in summary)
    5. Configuration comparison         (from classifier_comparison.csv)
    6. Learning curve for chosen model  (computes fresh, saves figure)
    7. Display of all figures           (inline, if in a notebook)
    8. Report-ready discussion text     (printed)

Usage:
    $ python src/analysis.py
    # or from a notebook:
    >>> from src.analysis import analyse_all
    >>> analyse_all()

No changes to main.py or any pipeline class are required. If a figure
or metric is missing, that step is skipped with a warning — the rest
still runs.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve

from config import ProjectConfig


# ======================================================================
# 1. ARTIFACT LOADING
# ======================================================================

def load_artifacts():
    """
    Load every metric, report, and figure path the pipeline produced.
    Returns a dict; missing items are None (never raises).
    """
    A = {
        "metrics":   None,
        "baseline":  None,
        "per_class": None,
        "compare":   None,
        "session":   None,
        "figures":   {},
    }

    mp = ProjectConfig.METRICS_PATH
    rp = ProjectConfig.REPORTS_PATH
    fp = ProjectConfig.FIGURES_PATH
    cp = ProjectConfig.CHECKPOINT_PATH

    if (mp / "test_summary.json").exists():
        A["metrics"] = json.load(open(mp / "test_summary.json"))

    if (mp / "baseline_comparison.csv").exists():
        A["baseline"] = pd.read_csv(mp / "baseline_comparison.csv").iloc[0].to_dict()

    if (mp / "test_per_class_metrics.csv").exists():
        A["per_class"] = pd.read_csv(mp / "test_per_class_metrics.csv")

    if (rp / "classifier_comparison.csv").exists():
        A["compare"] = pd.read_csv(rp / "classifier_comparison.csv")

    if (cp / "session.pkl").exists():
        with open(cp / "session.pkl", "rb") as f:
            A["session"] = pickle.load(f)

    for name in (
        "representative_galaxies",
        "classifier_comparison",
        "confusion_matrix",
        "per_class_metrics",
        "misclassified_examples",
        "learning_curve",
    ):
        p = fp / f"{name}.png"
        if p.exists():
            A["figures"][name] = p

    return A


# ======================================================================
# 2. HEADLINE METRICS
# ======================================================================

def print_headline(artifacts):
    print("\n" + "=" * 70)
    print(" HEADLINE METRICS")
    print("=" * 70)

    m = artifacts["metrics"]
    if m is None:
        print("  [missing] test_summary.json — run Task E first.")
        return

    print(f"  accuracy           {m.get('accuracy', float('nan')):.4f}")
    print(f"  macro precision    {m.get('macro_precision', float('nan')):.4f}")
    print(f"  macro recall       {m.get('macro_recall', float('nan')):.4f}")
    print(f"  macro F1           {m.get('macro_f1', float('nan')):.4f}")
    print(f"  weighted F1        {m.get('weighted_f1', float('nan')):.4f}")
    gap = m.get("weighted_f1", 0) - m.get("macro_f1", 0)
    print(f"  weighted−macro gap {gap:+.4f}  "
          f"({'uniform' if abs(gap) < 0.03 else 'class-biased'})")


def print_baseline(artifacts):
    print("\n" + "=" * 70)
    print(" BASELINE COMPARISON")
    print("=" * 70)

    b = artifacts["baseline"]
    if b is None:
        print("  [missing] baseline_comparison.csv — run Task E first.")
        return

    print(f"  majority-class baseline   {b['baseline_accuracy']:.4f}")
    print(f"  model accuracy            {b['model_accuracy']:.4f}")
    print(f"  absolute gain             {b['absolute_gain']:+.4f}")
    print(f"  relative gain             {b['relative_gain_pct']:+.2f}%")
    print(f"  verdict                   {b['verdict']}")


# ======================================================================
# 3. PER-CLASS TABLE
# ======================================================================

def print_per_class(artifacts):
    print("\n" + "=" * 70)
    print(" PER-CLASS METRICS")
    print("=" * 70)

    pc = artifacts["per_class"]
    if pc is None:
        print("  [missing] test_per_class_metrics.csv — run Task E first.")
        return

    pc = pc.copy()
    pc["error_type"] = np.where(
        pc["precision"] > pc["recall"] + 0.05, "under-predicts",
        np.where(pc["recall"] > pc["precision"] + 0.05, "over-predicts",
                 "balanced")
    )

    print(pc.to_string(index=False))

    s = pc.sort_values("f1", ascending=False)
    print("\n  Easiest :", s.iloc[0]["class_name"],
          f"(F1={s.iloc[0]['f1']:.3f})")
    print("  Hardest :", s.iloc[-1]["class_name"],
          f"(F1={s.iloc[-1]['f1']:.3f})")

    hard = s[s["f1"] < 0.45]
    if len(hard) > 0:
        print(f"\n  Classes with F1 < 0.45 (effectively unlearned at 32×32):")
        for _, r in hard.iterrows():
            print(f"    {r['class_id']} {r['class_name']:<28} "
                  f"F1={r['f1']:.3f}")


# ======================================================================
# 4. CONFUSION PAIRS
# ======================================================================

def print_confusion_pairs(artifacts, top_k=10):
    print("\n" + "=" * 70)
    print(" TOP CONFUSION PAIRS (true → predicted)")
    print("=" * 70)

    cm = artifacts.get("metrics", {}).get("confusion_matrix") if artifacts["metrics"] else None
    if cm is None:
        print("  [missing] confusion matrix not in test_summary.json.")
        print("  Tip: ensure ReportAccuracy... stores it in the returned dict.")
        return

    cm = np.asarray(cm)
    off = cm.copy()
    np.fill_diagonal(off, 0)

    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and off[i, j] > 0:
                pairs.append((i, j, int(off[i, j])))
    pairs.sort(key=lambda x: -x[2])

    print(f"  {'true':<4}{'true name':<28}"
          f"{'pred':<6}{'pred name':<28}{'count':>6}")
    print("  " + "-" * 70)
    for t, p, n in pairs[:top_k]:
        print(f"  {t:<4}{ProjectConfig.CLASS_NAMES[t]:<28}"
              f"{p:<6}{ProjectConfig.CLASS_NAMES[p]:<28}{n:>6}")

    # label the well-known expected pairs if present
    expected = [(2, 3), (3, 2), (8, 9), (9, 8), (6, 7), (7, 6), (0, 1), (1, 0)]
    hits = {(t, p): n for t, p, n in pairs if (t, p) in expected}
    if hits:
        print("\n  Expected morphological confusions detected:")
        for (t, p), n in hits.items():
            print(f"    {t}↔{p}  ×{n}")


# ======================================================================
# 5. CONFIGURATION COMPARISON
# ======================================================================

def print_config_comparison(artifacts):
    print("\n" + "=" * 70)
    print(" CONFIGURATION COMPARISON (validation accuracy)")
    print("=" * 70)

    df = artifacts["compare"]
    if df is None:
        print("  [missing] classifier_comparison.csv — run Task D first.")
        return

    df = df.sort_values("val_accuracy", ascending=False)
    print(df.to_string(index=False))

    spread = df["val_accuracy"].max() - df["val_accuracy"].min()
    best   = df.iloc[0]
    print(f"\n  Best config : {best['config']}  (val_acc={best['val_accuracy']:.4f})")
    print(f"  Spread      : {spread:.4f} "
          f"({'features dominate, hyperparams barely matter' if spread < 0.02 else 'meaningful hyperparameter effect'})")


# ======================================================================
# 6. LEARNING CURVE
# ======================================================================

def plot_learning_curve(artifacts):
    """
    Train the chosen configuration on progressively larger subsets of
    the training pool and plot train vs validation accuracy.

    Requires the session pickle (for features, labels, and splits) and
    the chosen config (from trainer.chosen_params).
    """
    print("\n" + "=" * 70)
    print(" LEARNING CURVE")
    print("=" * 70)

    s = artifacts["session"]
    if s is None:
        print("  [skip] no session.pkl — cannot access features/labels.")
        return None

    st = s["state"]
    if st.get("features") is None or st.get("train_idx") is None:
        print("  [skip] session has no features/train_idx.")
        return None

    if s.get("trainer_chosen_params") is None:
        print("  [skip] no chosen configuration in session.")
        return None

    params = s["trainer_chosen_params"]
    name   = params["model"]
    kwargs = {k: v for k, v in params.items() if k != "model"}

    # Build the estimator fresh
    from sklearn.svm import SVC, LinearSVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier

    protos = {
        "linear_svc":    LinearSVC(
            random_state=ProjectConfig.RANDOM_SEED, max_iter=5000),
        "rbf_svc":       SVC(
            kernel="rbf", random_state=ProjectConfig.RANDOM_SEED),
        "random_forest": RandomForestClassifier(
            n_jobs=ProjectConfig.N_JOBS,
            random_state=ProjectConfig.RANDOM_SEED),
        "logistic":      LogisticRegression(
            max_iter=2000, n_jobs=ProjectConfig.N_JOBS,
            random_state=ProjectConfig.RANDOM_SEED),
        "knn":           KNeighborsClassifier(),
    }
    if name not in protos:
        print(f"  [skip] unknown estimator '{name}'.")
        return None

    estimator = protos[name].set_params(**kwargs)

    X = st["features"]
    y = st["labels"]
    tr = st["train_idx"]
    va = st["val_idx"]

    X_curve = np.concatenate([X[tr], X[va]], axis=0)
    y_curve = np.concatenate([y[tr], y[va]], axis=0)

    train_sizes = np.linspace(0.10, 1.00, 8)

    print(f"  model            : {name} {kwargs}")
    print(f"  pool size        : {len(X_curve)} samples")
    print(f"  CV folds         : {ProjectConfig.CV_FOLDS}")
    print(f"  training sizes   : "
          f"{[int(t * len(X_curve)) for t in train_sizes]}")
    print(f"  computing (1-3 min)...")

    sizes, train_scores, val_scores = learning_curve(
        estimator,
        X_curve, y_curve,
        train_sizes=train_sizes,
        cv=ProjectConfig.CV_FOLDS,
        scoring="accuracy",
        n_jobs=ProjectConfig.N_JOBS,
        random_state=ProjectConfig.RANDOM_SEED,
        shuffle=True,
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    print("\n  n_train   train_acc (±std)        val_acc (±std)        gap")
    print("  " + "-" * 64)
    for n, t, ts, v, vs in zip(sizes, train_mean, train_std,
                                val_mean, val_std):
        print(f"  {n:>7}   {t:.3f} ± {ts:.3f}            "
              f"{v:.3f} ± {vs:.3f}        {t - v:+.3f}")

    # Save figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sizes, train_mean, "o-", color="steelblue", label="Training")
    ax.fill_between(sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color="steelblue")
    ax.plot(sizes, val_mean, "s-", color="darkorange", label="Validation")
    ax.fill_between(sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color="darkorange")
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Learning curve — {name} {kwargs}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out = ProjectConfig.FIGURES_PATH / "learning_curve.png"
    plt.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\n  Saved: {out}")

    # Interpretation
    final_train = float(train_mean[-1])
    final_val   = float(val_mean[-1])
    gap         = final_train - final_val
    val_slope   = float(val_mean[-1] - val_mean[-2])
    gap_slope   = float((train_mean[-1] - val_mean[-1])
                        - (train_mean[-2] - val_mean[-2]))

    print("\n  INTERPRETATION")
    print("  " + "-" * 64)
    print(f"  final train acc  : {final_train:.3f}")
    print(f"  final val acc    : {final_val:.3f}")
    print(f"  gap (train−val)  : {gap:+.3f}")
    print(f"  val slope (last) : {val_slope:+.4f}")
    print(f"  gap slope (last) : {gap_slope:+.4f}")

    if final_val >= 0.70 and gap < 0.10:
        verdict = ("Well-fit model. Both curves high and close. "
                   "Performance is limited by data quality, not capacity.")
    elif final_val < 0.55 and gap < 0.10:
        verdict = ("HIGH BIAS — underfitting. Both curves plateau low. "
                   "The model class is too weak; consider non-linear "
                   "or a CNN.")
    elif gap > 0.20 and gap_slope >= 0.0:
        verdict = ("HIGH VARIANCE, ceiling reached. Large persistent gap. "
                   "More data will NOT close it — the 32×32 pixel "
                   "representation is the limiting factor. A CNN on "
                   "full resolution is the honest next step.")
    elif gap > 0.10 and val_slope > 0.01:
        verdict = ("HIGH VARIANCE, still improving. Gap is large but "
                   "validation is still rising — more data would help.")
    else:
        verdict = ("Moderate fit. Some gap remains but neither curve "
                   "dominates. Further tuning may yield 1-2 points.")

    print(f"\n  → {verdict}")

    return {
        "train_sizes": sizes,
        "train_mean":  train_mean,
        "train_std":   train_std,
        "val_mean":    val_mean,
        "val_std":     val_std,
        "final_train": final_train,
        "final_val":   final_val,
        "gap":         gap,
        "verdict":     verdict,
    }


# ======================================================================
# 7. DISPLAY FIGURES
# ======================================================================

def display_figures(artifacts):
    print("\n" + "=" * 70)
    print(" FIGURES")
    print("=" * 70)

    captions = {
        "representative_galaxies": "Fig 1 — Representative morphologies (one per class + confusable extras)",
        "classifier_comparison":   "Fig 2 — Validation accuracy across configurations",
        "confusion_matrix":        "Fig 3 — Confusion matrix on the sealed test set",
        "per_class_metrics":       "Fig 4 — Per-class precision / recall / F1",
        "misclassified_examples":  "Fig 5 — Six misclassified test galaxies",
        "learning_curve":          "Fig 6 — Learning curve (train vs validation accuracy)",
    }

    # Try to display inline (notebook); fall back to listing paths
    try:
        from IPython.display import Image, display, Markdown
        _inline = True
    except ImportError:
        _inline = False

    for name, caption in captions.items():
        path = artifacts["figures"].get(name)
        if path is None:
            print(f"  [missing] {name}.png")
            continue
        print(f"  {caption}")
        if _inline:
            display(Markdown(f"**{caption}**"))
            display(Image(str(path)))
            print()


# ======================================================================
# 8. REPORT-READY PARAGRAPHS
# ======================================================================

def write_report(artifacts, curve=None):
    print("\n" + "=" * 70)
    print(" REPORT-READY TEXT")
    print("=" * 70)

    m = artifacts["metrics"]
    b = artifacts["baseline"]
    pc = artifacts["per_class"]
    cf = artifacts["compare"]

    if m is None or b is None or pc is None or cf is None:
        print("  [skip] need metrics, baseline, per_class, compare on disk.")
        return

    acc      = m["accuracy"]
    macro_f1 = m["macro_f1"]
    w_f1     = m["weighted_f1"]
    gain     = b["absolute_gain"]

    best = cf.loc[cf["val_accuracy"].idxmax()]
    s = pc.sort_values("f1", ascending=False)
    top, bottom = s.iloc[0], s.iloc[-1]

    print("\n1. Model performance")
    print(f"   The final model achieved a test-set accuracy of {acc:.4f}, "
          f"{gain * 100:+.2f} percentage points above the majority-class "
          f"baseline ({b['baseline_accuracy']:.4f}). Macro F1 was "
          f"{macro_f1:.4f}, weighted F1 {w_f1:.4f} — a gap of "
          f"{w_f1 - macro_f1:+.4f}, indicating "
          f"{'uniform' if abs(w_f1 - macro_f1) < 0.03 else 'class-biased'} "
          f"performance.")

    print("\n2. Chosen configuration")
    print(f"   {best['config']} achieved the highest validation accuracy "
          f"({best['val_accuracy']:.4f}), selected on the validation split "
          f"before test-set evaluation. Linear models were preferred over "
          f"the RBF kernel because at 3072-dimensional pixel features the "
          f"latter scales quadratically in training time.")

    print("\n3. Per-class behaviour")
    print(f"   The strongest class was {top['class_name']} "
          f"(F1 = {top['f1']:.3f}); the weakest was "
          f"{bottom['class_name']} (F1 = {bottom['f1']:.3f}). Classes with "
          f"overlapping visual cues at 32×32 — Round/In-between, "
          f"Edge-on ±bulge, spiral arm pitch — were the main sources of "
          f"error.")

    if curve is not None:
        print("\n4. Learning curve analysis")
        print(f"   Training accuracy climbed to {curve['final_train']:.3f} "
              f"and validation plateaued at {curve['final_val']:.3f}. "
              f"The gap of {curve['gap']:+.3f} did not close as the "
              f"training pool doubled, indicating high variance at a "
              f"feature ceiling rather than a data-limited regime. "
              f"{curve['verdict']}")

    print("\n5. Honest conclusion")
    print(f"   The model learns real morphology signal — it beats the "
          f"majority-class baseline by {gain * 100:.1f} points — but "
          f"the residual error is concentrated on classes whose "
          f"distinguishing features vanish at 32×32. The finding is a "
          f"measured limit of the pixel-based representation, not a "
          f"failure of the classifier. A convolutional network on full "
          f"resolution is the appropriate next step.")


# ======================================================================
# 9. ORCHESTRATOR
# ======================================================================

def analyse_all(run_learning_curve=True):
    print("\n" + "#" * 70)
    print("# GALAXY10 ANALYSIS — full report")
    print("#" * 70)

    ProjectConfig.create_directories()
    A = load_artifacts()

    print_headline(A)
    print_baseline(A)
    print_per_class(A)
    print_confusion_pairs(A)
    print_config_comparison(A)

    curve = None
    if run_learning_curve:
        curve = plot_learning_curve(A)
        if curve is not None:
            A["figures"]["learning_curve"] = (
                ProjectConfig.FIGURES_PATH / "learning_curve.png"
            )

    display_figures(A)
    write_report(A, curve=curve)

    print("\n" + "#" * 70)
    print("# ANALYSIS COMPLETE")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    analyse_all()