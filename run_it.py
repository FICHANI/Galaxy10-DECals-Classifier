# run_it.py
"""
Galaxy10 — secondary interactive entrance.

main.py       → train the classical pipeline (Tasks A–F)
run_it.py     → analyse, diagnose, run the CNN, predict

This file is post-pipeline: it does not touch state dicts, session
pickles, or the trainer/evaluator objects. It reads everything it
needs from disk.

Run:
    $ python run_it.py

No imports of numpy / sklearn / torch happen at module load — each
action imports what it needs lazily, so the menu opens instantly and
works even if torch isn't installed.
"""

import os
import sys
import traceback


# ======================================================================
# TERMINAL HELPERS
# ======================================================================

def clear_terminal():
    os.system("cls" if os.name == "nt" else "clear")


def terminal_pause():
    input("\nPress ENTER to return to the main menu...")


def terminal_header():
    clear_terminal()
    print("=" * 66)
    print("              GALAXY10 — POST-PIPELINE TOOL")
    print("=" * 66)
    print("    Analysis · Diagnostics · CNN · Predictions")
    print("=" * 66)


def _artifact_status():
    """Return a quick dict of what exists on disk."""
    try:
        from config import ProjectConfig
    except ImportError:
        return {"config": False}

    checks = {
        "features":        ProjectConfig.FEATURES_FILE.exists(),
        "labels":          ProjectConfig.FEATURE_LABELS.exists(),
        "session":         (ProjectConfig.CHECKPOINT_PATH / "session.pkl").exists(),
        "test_metrics":    (ProjectConfig.METRICS_PATH / "test_summary.json").exists(),
        "baseline":        (ProjectConfig.METRICS_PATH / "baseline_comparison.csv").exists(),
        "per_class":       (ProjectConfig.METRICS_PATH / "test_per_class_metrics.csv").exists(),
        "config_compare":  (ProjectConfig.REPORTS_PATH / "classifier_comparison.csv").exists(),
        "confusion_png":   (ProjectConfig.FIGURES_PATH / "confusion_matrix.png").exists(),
        "learning_curve":  (ProjectConfig.FIGURES_PATH / "learning_curve.png").exists(),
        "cnn_model":       ProjectConfig.CNN_MODEL_PATH.exists(),
        "cnn_metrics":     (ProjectConfig.METRICS_PATH / "cnn_test_summary.json").exists(),
    }
    return checks


def terminal_status():
    s = _artifact_status()
    if not s.get("config"):
        print("\n[!] Cannot import config — are you in the project root?")
        return

    print("\nARTIFACT STATUS")
    print("-" * 45)
    for key in ("features", "labels", "session", "test_metrics",
                "baseline", "per_class", "config_compare",
                "confusion_png", "learning_curve",
                "cnn_model", "cnn_metrics"):
        tag = "✓" if s.get(key) else "·"
        print(f"  [{tag}]  {key}")
    print("-" * 45)


# ======================================================================
# ACTIONS
# ======================================================================

def action_run_analysis():
    """Full analysis report — headline, per-class, confusion, learning curve."""
    from src.analysis import analyse_all
    analyse_all(run_learning_curve=True)


def action_analysis_fast():
    """Analysis without the learning curve (2 seconds)."""
    from src.analysis import analyse_all
    analyse_all(run_learning_curve=False)


def action_learning_curve_only():
    """Just compute and plot the learning curve."""
    from src.analysis import load_artifacts, plot_learning_curve
    A = load_artifacts()
    plot_learning_curve(A)


def action_diagnostics():
    """Sanity check: are the features healthy? Do they beat 10%?"""
    import numpy as np
    from config import ProjectConfig

    if not ProjectConfig.FEATURES_FILE.exists():
        print("\n[skip] no features.npy — run Task B in main.py first.")
        return

    X = np.load(ProjectConfig.FEATURES_FILE)
    y = np.load(ProjectConfig.FEATURE_LABELS)

    print("\n=== FEATURES ===")
    print(f"  shape     : {X.shape}")
    print(f"  dtype     : {X.dtype}")
    print(f"  min/max   : {X.min():.4f} / {X.max():.4f}")
    print(f"  mean/std  : {X.mean():.4f} / {X.std():.4f}")
    print(f"  NaN/Inf   : {np.isnan(X).any()} / {np.isinf(X).any()}")
    print(f"  zero-var columns: {(X.std(axis=0) < 1e-8).sum()}")

    print("\n=== LABELS ===")
    counts = np.bincount(y.astype(int), minlength=10)
    for cid in range(10):
        bar = "█" * (counts[cid] // 50)
        print(f"  {cid} {ProjectConfig.CLASS_NAMES[cid]:<28} {counts[cid]:>5}  {bar}")

    # Quick classifier
    print("\n=== QUICK SANITY CLASSIFIER ===")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(Xtr, ytr)
    acc = accuracy_score(yte, clf.predict(Xte))
    print(f"  LogisticRegression accuracy : {acc:.4f}")

    if acc < 0.12:
        print("\n  ⚠️  FEATURES ARE BROKEN")
        print("     Accuracy ≈ 10% = random. Check:")
        print("     - FlattenImages double-normalises? (float input /255 again)")
        print("     - Features and labels misaligned?")
        print("     - Resize collapsed the image?")
    elif acc < 0.30:
        print("\n  ⚠️  Features are weak but carry some signal.")
        print("     Check class 4 support and training configuration.")
    else:
        print("\n  ✅ Features carry real signal. Pipeline is healthy.")


def action_compare_models():
    """Classical vs CNN — same test split, side by side."""
    import json
    from config import ProjectConfig

    cl_path = ProjectConfig.METRICS_PATH / "test_summary.json"
    cn_path = ProjectConfig.METRICS_PATH / "cnn_test_summary.json"

    if not cl_path.exists():
        print("\n[skip] no classical metrics. Run Task E in main.py first.")
        return
    if not cn_path.exists():
        print("\n[skip] no CNN metrics. Train the CNN first (option 5).")
        return

    classical = json.load(open(cl_path))
    cnn       = json.load(open(cn_path))

    print("\nCLASSICAL vs CNN — same test split")
    print("-" * 60)
    print(f"{'metric':<16}{'classical':>14}{'CNN':>14}{'Δ (pt)':>12}")
    print("-" * 60)
    for key in ("accuracy", "macro_f1", "weighted_f1"):
        c = classical.get(key)
        n = cnn.get(key)
        if c is None or n is None:
            continue
        print(f"{key:<16}{c:>14.4f}{n:>14.4f}{(n - c) * 100:>+11.1f}")
    print("-" * 60)


def action_train_cnn():
    """Train ResNet18 on the same split as the classical pipeline."""
    import numpy as np
    from config import ProjectConfig

    if not ProjectConfig.FEATURE_LABELS.exists():
        print("\n[skip] no galaxy_labels.npy — run Task B in main.py first.")
        return

    print("\n[CNN] checking GPU…")
    try:
        import torch
        print(f"  torch : {torch.__version__}")
        print(f"  cuda  : {torch.cuda.is_available()}")
        if not torch.cuda.is_available():
            print("\n  ⚠️  Training on CPU will take hours.")
            if input("  Continue anyway? (yes/no): ").strip().lower() != "yes":
                return
    except ImportError:
        print("  torch not installed. Run: pip install torch torchvision")
        return

    labels = np.load(ProjectConfig.FEATURE_LABELS)

    from src.experimentation import DesignExperiment
    designer = DesignExperiment()
    tr, va, te = designer.CreateStratifiedTrainingValidationAndTestSetWithAFixedRandomSeed(labels)

    #h5_default = str(ProjectConfig.DATA_PATH)
    h5_default =  os.path.expanduser("~/data/Galaxy10_DECals.h5")
    h5 = input(f"\n  HDF5 path [{h5_default}]: ").strip() or h5_default

    from src.cnn.train import train_cnn, plot_training_history

    result = train_cnn(tr, va, h5_path=h5, verbose=True)
    plot_training_history(result["history"])

    print(f"\n[CNN] best val accuracy: {result['best_val_accuracy']:.4f}")
    print(f"[CNN] saved to          : {ProjectConfig.CNN_MODEL_PATH}")


def action_evaluate_cnn():
    """Evaluate the trained CNN on the sealed test set."""
    import json
    import numpy as np
    from config import ProjectConfig
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix,
    )

    if not ProjectConfig.CNN_MODEL_PATH.exists():
        print("\n[skip] no CNN checkpoint. Train first (option 5).")
        return

    labels = np.load(ProjectConfig.FEATURE_LABELS)

    from src.experimentation import DesignExperiment
    designer = DesignExperiment()
    _, _, test_idx = designer.CreateStratifiedTrainingValidationAndTestSetWithAFixedRandomSeed(labels)

    h5_default = str(ProjectConfig.DATA_PATH)
    h5 = input(f"\n  HDF5 path [{h5_default}]: ").strip() or h5_default

    try:
        import torch
        from torch.utils.data import DataLoader
        from src.cnn.dataset import Galaxy10Dataset
        from src.cnn.predict import load_cnn
    except ImportError as e:
        print(f"\n[skip] missing dependency: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_cnn().to(device)

    ds = Galaxy10Dataset(h5, test_idx, ProjectConfig.CNN_IMAGE_SIZE, augment=False)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

    preds, truths = [], []
    with torch.no_grad():
        for imgs, lbls in dl:
            logits = model(imgs.to(device))
            preds.extend(logits.argmax(1).cpu().numpy())
            truths.extend(lbls.numpy())

    preds  = np.array(preds)
    truths = np.array(truths)

    summary = {
        "accuracy":         float(accuracy_score(truths, preds)),
        "macro_f1":         float(f1_score(truths, preds, average="macro")),
        "weighted_f1":      float(f1_score(truths, preds, average="weighted")),
        "macro_precision":  float(precision_score(truths, preds, average="macro", zero_division=0)),
        "macro_recall":     float(recall_score(truths, preds, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(truths, preds).tolist(),
    }

    out = ProjectConfig.METRICS_PATH / "cnn_test_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nCNN TEST METRICS")
    print("-" * 55)
    print(f"  accuracy         : {summary['accuracy']:.4f}")
    print(f"  macro F1         : {summary['macro_f1']:.4f}")
    print(f"  weighted F1      : {summary['weighted_f1']:.4f}")
    print(f"\n  saved: {out}")


def action_predict_one():
    """Classify a single image file with the trained CNN."""
    from config import ProjectConfig

    if not ProjectConfig.CNN_MODEL_PATH.exists():
        print("\n[skip] no CNN checkpoint. Train first (option 5).")
        return

    path = input("\n  image path (jpg / png / npy): ").strip()
    if not path or not os.path.exists(path):
        print(f"  [skip] file not found: {path}")
        return

    try:
        import numpy as np
        from skimage import io
        if path.lower().endswith(".npy"):
            img = np.load(path)
        else:
            img = io.imread(path)
    except ImportError as e:
        print(f"\n[skip] missing dependency: {e}")
        return

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    try:
        import torch
        from src.cnn.predict import load_cnn, predict_cnn
    except ImportError as e:
        print(f"\n[skip] {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_cnn().to(device)
    result = predict_cnn(model, img, device=device)

    print("\nPREDICTION")
    print("-" * 55)
    print(f"  input        : {path}")
    print(f"  shape        : {img.shape}")
    print(f"  class        : {result['class_id']} — {result['class_name']}")
    print(f"  confidence   : {result['confidence']:.4f}")

    print("\n  top-5 probabilities:")
    probs = result["probabilities"]
    order = probs.argsort()[::-1][:5]
    for i in order:
        bar = "█" * int(probs[i] * 40)
        print(f"    {i} {ProjectConfig.CLASS_NAMES[i]:<28} "
              f"{probs[i]:.4f}  {bar}")

    if result["confidence"] < 0.50:
        print("\n  ⚠️  Low confidence — model may not have seen this morphology.")


def action_predict_folder():
    """Classify every image in a folder."""
    from config import ProjectConfig

    if not ProjectConfig.CNN_MODEL_PATH.exists():
        print("\n[skip] no CNN checkpoint. Train first (option 5).")
        return

    folder = input("\n  folder path: ").strip()
    if not folder or not os.path.isdir(folder):
        print(f"  [skip] not a directory: {folder}")
        return

    exts = (".jpg", ".jpeg", ".png", ".npy")
    files = sorted(
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(exts)
    )
    if not files:
        print("  [skip] no images found.")
        return

    print(f"\n  found {len(files)} images — loading model…")

    import numpy as np
    from skimage import io
    import torch
    from src.cnn.predict import load_cnn, predict_cnn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_cnn().to(device)

    print(f"\n  {'file':<40}{'class':<28}{'confidence':>12}")
    print("  " + "-" * 82)
    for path in files:
        try:
            img = np.load(path) if path.lower().endswith(".npy") else io.imread(path)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            r = predict_cnn(model, img, device=device)
            print(f"  {os.path.basename(path):<40}"
                  f"{r['class_name']:<28}"
                  f"{r['confidence']:>11.4f}")
        except Exception as e:
            print(f"  {os.path.basename(path):<40}  [error] {e}")


# ======================================================================
# MENU
# ======================================================================

def run_it():
    actions = {
        "1": ("Full analysis (headline + metrics + learning curve)", action_run_analysis),
        "2": ("Fast analysis (skip learning curve)",                 action_analysis_fast),
        "3": ("Learning curve only",                                 action_learning_curve_only),
        "4": ("Diagnostics — is the pipeline producing real signal?", action_diagnostics),
        "5": ("Compare classical vs CNN",                            action_compare_models),
        "6": ("Train the CNN (ResNet18)",                            action_train_cnn),
        "7": ("Evaluate CNN on the test set",                        action_evaluate_cnn),
        "8": ("Predict a single image (CNN)",                        action_predict_one),
        "9": ("Predict a folder of images (CNN)",                    action_predict_folder),
    }

    while True:
        terminal_header()
        terminal_status()

        print("\nMENU")
        print("-" * 45)
        for k, (label, _) in actions.items():
            print(f"{k}. {label}")
        print("\n0. Exit")
        print("-" * 45)

        choice = input("\nSelect an option: ").strip()

        if choice == "0":
            print("\nBye.")
            break

        if choice not in actions:
            print("\nInvalid selection.")
            terminal_pause()
            continue

        label, fn = actions[choice]
        print(f"\n>>> {label}")
        print("-" * 55)
        try:
            fn()
        except Exception as e:
            print(f"\n[ERROR] {e}")
            traceback.print_exc()
        terminal_pause()


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    try:
        run_it()
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        sys.exit(0)