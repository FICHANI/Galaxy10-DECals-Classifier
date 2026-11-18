# export_model.py
"""
Extract the trained model from session.pkl, refit on train+val, save as
a standalone joblib artifact ready for submission.
"""

import json
import pickle
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier

from config import ProjectConfig


def main():
    session_path = ProjectConfig.CHECKPOINT_PATH / "session.pkl"
    if not session_path.exists():
        raise FileNotFoundError(session_path)

    with open(session_path, "rb") as f:
        snap = pickle.load(f)

    st          = snap["state"]
    X           = st["features"]
    y           = st["labels"]
    train_idx   = st["train_idx"]
    val_idx     = st["val_idx"]

    X_fit = np.concatenate([X[train_idx], X[val_idx]], axis=0)
    y_fit = np.concatenate([y[train_idx], y[val_idx]], axis=0)

    print(f"refitting RF(200) on {len(X_fit)} samples "
          f"× {X.shape[1]} features …")

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=ProjectConfig.RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_fit, y_fit)

    out_dir = ProjectConfig.RESULTS_PATH / "model"
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib_path = out_dir / "final_model.joblib"
    joblib.dump(model, joblib_path)
    print(f"✅ model saved : {joblib_path} "
          f"({joblib_path.stat().st_size / 1e6:.1f} MB)")

    # Save metadata the grader needs
    meta = {
        "model_type":       "RandomForestClassifier",
        "sklearn_version":  __import__("sklearn").__version__,
        "n_estimators":     200,
        "random_state":     ProjectConfig.RANDOM_SEED,
        "n_features":       int(X.shape[1]),
        "feature_layout":   "flattened (32, 32, 3) float32, RGB-normalized to [0,1]",
        "image_preprocessing": "resize 256x256 -> 32x32 (preserve_range, anti_aliasing), then /255, then reshape(-1)",
        "training_samples": int(len(X_fit)),
        "test_samples":     int(len(st["test_idx"])),
        "test_accuracy":    float(snap["evaluator_metrics"]["accuracy"]),
        "test_macro_f1":    float(snap["evaluator_metrics"]["macro_f1"]),
        "class_names":      ProjectConfig.CLASS_NAMES,
    }
    meta_path = out_dir / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✅ metadata    : {meta_path}")


if __name__ == "__main__":
    main()