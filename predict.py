# predict.py
"""
Standalone inference for the submitted Galaxy10 model.

Usage:
    from predict import load_model, predict, predict_batch

    model = load_model()
    result = predict(model, image_path="some_galaxy.png")
    # → {"class_id": 5, "class_name": "Barred Spiral", "confidence": 0.87}

    # Or from a numpy array:
    result = predict(model, image_array=arr)   # arr shape (H, W, 3) uint8
"""

import json
import numpy as np
import joblib
from pathlib import Path
from skimage.transform import resize
from skimage import io

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "results/model/final_model.joblib"
META_PATH  = HERE / "results/model/model_metadata.json"

CLASS_NAMES = [
    "Disturbed", "Merging", "Round Smooth", "In-between Round Smooth",
    "Cigar Shaped Smooth", "Barred Spiral", "Unbarred Tight Spiral",
    "Unbarred Loose Spiral", "Edge-on without Bulge", "Edge-on with Bulge",
]

IMAGE_SIZE = 32
NORMALIZE_SCALE = 255.0


def load_model(path=None):
    """Load the trained RandomForest classifier."""
    path = Path(path) if path else MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def _preprocess(img):
    """
    (H, W, 3) uint8 or (H, W, 3) float -> (3072,) float32.

    MUST match the pipeline's preprocessing:
      resize -> 32x32 (preserve_range, anti_aliasing)
      normalize -> /255
      flatten   -> 3072
    """
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Expected (H, W, 3); got {img.shape}")

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / NORMALIZE_SCALE
    else:
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / NORMALIZE_SCALE

    img = resize(img, (IMAGE_SIZE, IMAGE_SIZE),
                 preserve_range=True, anti_aliasing=True)
    return img.reshape(-1).astype(np.float32)


def predict(model, image_path=None, image_array=None):
    """
    Classify one image.
    Returns: {class_id, class_name, confidence, probabilities}
    """
    if image_path is not None:
        img = io.imread(image_path)
    elif image_array is not None:
        img = image_array
    else:
        raise ValueError("Pass image_path= or image_array=")

    features = _preprocess(img).reshape(1, -1)
    probs = model.predict_proba(features)[0]
    pred = int(np.argmax(probs))

    return {
        "class_id":      pred,
        "class_name":    CLASS_NAMES[pred],
        "confidence":    float(probs[pred]),
        "probabilities": probs,
    }


def predict_batch(model, image_arrays):
    """Classify N images at once. Returns arrays of class_ids and confidences."""
    feats = np.stack([_preprocess(im) for im in image_arrays])
    probs = model.predict_proba(feats)
    preds = probs.argmax(axis=1)
    return {
        "class_ids":     preds,
        "class_names":   [CLASS_NAMES[i] for i in preds],
        "confidences":   probs[np.arange(len(preds)), preds],
        "probabilities": probs,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    model = load_model()
    result = predict(model, image_path=sys.argv[1])
    print(f"\nPredicted : {result['class_name']} (class {result['class_id']})")
    print(f"Confidence: {result['confidence']:.4f}")
    print("\nTop probabilities:")
    for i in np.argsort(result["probabilities"])[::-1][:5]:
        print(f"  {CLASS_NAMES[i]:<28} {result['probabilities'][i]:.4f}")