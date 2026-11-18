# Galaxy10 DECals — Morphology Classification Model

## What this is
A RandomForestClassifier trained on pixel-based features from Galaxy10 DECals
to classify 10 galaxy morphology types.

## Performance
| Metric | Value |
|---|---|
| **Test accuracy** | **0.5396** |
| Majority-class baseline | 0.1071 |
| **Improvement** | **+0.4325 (404% lift)** |
| Macro F1 | 0.5001 |
| Weighted F1 | 0.5235 |

## Model
- **Type:** RandomForestClassifier (sklearn 1.5.2)
- **Hyperparameters:** `n_estimators=200, random_state=42`
- **Input:** 3072-dim flattened features from 32×32×3 pixel arrays
- **Training samples:** 7933 (train + validation split)
- **Test samples:** 1401 (sealed, evaluated once)

## How to use

```python
from predict import load_model, predict

model = load_model()
result = predict(model, image_path="galaxy.png")
print(result["class_name"], result["confidence"])