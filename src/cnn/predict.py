import numpy as np
import torch
import torch.nn.functional as F
from skimage.transform import resize

from config import ProjectConfig
from src.cnn.model import build_resnet18


def load_cnn(device="cpu"):
    """
    Load the saved checkpoint into a fresh ResNet18.
    Raises FileNotFoundError if no model has been trained yet.
    """
    if not ProjectConfig.CNN_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No CNN checkpoint at {ProjectConfig.CNN_MODEL_PATH}. "
            f"Run train_cnn() first."
        )
    model = build_resnet18(pretrained=False)
    state = torch.load(
        ProjectConfig.CNN_MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def _preprocess_one(img):
    """(H, W, 3) uint8 or float in [0,1] → (3, S, S) float tensor."""
    img = np.asarray(img)
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Expected (H, W, 3); got {img.shape}.")
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
    img = resize(
        img,
        (ProjectConfig.CNN_IMAGE_SIZE, ProjectConfig.CNN_IMAGE_SIZE),
        preserve_range=True,
        anti_aliasing=True,
    )
    return torch.from_numpy(img).permute(2, 0, 1)


def predict_cnn(model, image_array, device="cpu"):
    """
    Predict a single image.

    image_array : (H, W, 3) OR (1, H, W, 3)

    Returns dict with class_id, class_name, confidence, probabilities.
    """
    img = np.asarray(image_array)
    if img.ndim == 4:
        if img.shape[0] != 1:
            raise ValueError(
                f"predict_cnn expects one image; got {img.shape[0]}. "
                f"Use predict_batch for multi-image input."
            )
        img = img[0]

    tensor = _preprocess_one(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).cpu().numpy()[0]

    pred = int(np.argmax(probs))
    return {
        "class_id":      pred,
        "class_name":    ProjectConfig.CLASS_NAMES[pred],
        "confidence":    float(probs[pred]),
        "probabilities": probs,
    }


def predict_batch(model, images, device="cpu", batch_size=64):
    """
    Predict a batch of images.

    images : (N, H, W, 3) uint8 or float

    Returns dict with numpy arrays: class_ids, class_names, confidences,
    probabilities (N, 10).
    """
    images = np.asarray(images)
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Expected (N, H, W, 3); got {images.shape}.")

    n = len(images)
    if n == 0:
        return {
            "class_ids":     np.array([], dtype=int),
            "class_names":   [],
            "confidences":   np.array([], dtype=float),
            "probabilities": np.zeros((0, ProjectConfig.NUM_CLASSES)),
        }

    all_probs = np.zeros((n, ProjectConfig.NUM_CLASSES), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_tensors = torch.stack(
            [_preprocess_one(img) for img in images[start:end]]
        ).to(device)

        with torch.no_grad():
            logits = model(batch_tensors)
            probs  = F.softmax(logits, dim=1).cpu().numpy()

        all_probs[start:end] = probs

    class_ids     = all_probs.argmax(axis=1)
    class_names   = [ProjectConfig.CLASS_NAMES[i] for i in class_ids]
    confidences   = all_probs[np.arange(n), class_ids]

    return {
        "class_ids":     class_ids,
        "class_names":   class_names,
        "confidences":   confidences,
        "probabilities": all_probs,
    }