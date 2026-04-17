import os
import re
import numpy as np
import json
from pathlib import Path

# Lazy imports for heavy deps
_tf = None
_model = None
_class_names = None
_class_meta = None

IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.6


def _get_tf():
    global _tf
    if _tf is None:
        import tensorflow as tf
        _tf = tf
    return _tf


def parse_label(label_name: str):
    """Parse a class folder name into (crop, disease) tuple."""
    raw = os.path.splitext(str(label_name).strip())[0].lower()
    raw = raw.replace('-', '_').replace(' ', '_')

    if '___' in raw:
        crop, disease = raw.split('___', 1)
    elif '__' in raw:
        crop, disease = raw.split('__', 1)
    elif '_' in raw:
        crop, disease = raw.split('_', 1)
    else:
        crop, disease = raw, 'healthy'

    crop = re.sub(r'_+', '_', crop).strip('_')
    disease = re.sub(r'_+', '_', disease).strip('_') or 'healthy'
    return crop, disease


def format_label(label: str) -> str:
    """Convert snake_case label to human-readable Title Case."""
    return label.replace('_', ' ').title()


def load_model(model_path: str, class_names_path: str):
    """Load the saved Keras model and class names JSON."""
    global _model, _class_names, _class_meta
    tf = _get_tf()

    _model = tf.keras.models.load_model(model_path)

    with open(class_names_path, 'r') as f:
        _class_names = json.load(f)

    _class_meta = {name: parse_label(name) for name in _class_names}
    return _model, _class_names


def get_available_crops() -> list[str]:
    """Return sorted unique list of crop names from loaded class metadata."""
    if _class_meta is None:
        return []
    crops = sorted(set(meta[0] for meta in _class_meta.values()))
    return crops


def predict_disease_for_crop(crop_name: str, img_bytes: bytes) -> dict:
    """
    Predict crop disease from image bytes.
    Returns a dict with crop, disease, confidence, all_scores, status.
    """
    if _model is None or _class_names is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    tf = _get_tf()
    from tensorflow.keras.preprocessing.image import img_to_array, load_img
    import io
    from PIL import Image

    # Load and preprocess image
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize(IMG_SIZE)
    img_array = img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Raw predictions
    preds = _model.predict(img_array, verbose=0)[0]

    crop_name_clean = crop_name.strip().lower()

    # Filter to candidate classes for this crop
    candidate_indices = [
        i for i, name in enumerate(_class_names)
        if _class_meta[name][0] == crop_name_clean
    ]

    if candidate_indices:
        crop_probs = preds[candidate_indices]
        best_local_idx = int(np.argmax(crop_probs))
        best_idx = candidate_indices[best_local_idx]
        total = float(crop_probs.sum())
        confidence = float(crop_probs[best_local_idx] / total) if total > 0 else float(preds[best_idx])

        # All scores for this crop (normalized)
        all_scores = {
            _class_names[i]: float(preds[i] / total) if total > 0 else float(preds[i])
            for i in candidate_indices
        }
    else:
        # Fallback: use global argmax
        best_idx = int(np.argmax(preds))
        confidence = float(preds[best_idx])
        all_scores = {name: float(preds[i]) for i, name in enumerate(_class_names)}

    class_name = _class_names[best_idx]
    predicted_crop, predicted_disease = _class_meta[class_name]

    return {
        "crop": predicted_crop,
        "predicted_disease": predicted_disease,
        "class_name": class_name,
        "confidence": confidence,
        "all_scores": dict(sorted(all_scores.items(), key=lambda x: -x[1])),
        "status": "confirmed" if confidence >= CONFIDENCE_THRESHOLD else "review",
    }
