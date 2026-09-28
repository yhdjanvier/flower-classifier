"""
Inference: validate an uploaded image, pre-process it and return a friendly result.

This module never shows Python tracebacks to users: it raises small, well-named
exceptions (`InvalidImageError`, `ModelNotFoundError`, `PredictionError`) whose
messages are safe to display.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow import keras

from src import config as C
from src.preprocessing import preprocess_for_inference
from src.utils import load_json

ALLOWED_FORMATS = {"JPEG", "PNG", "BMP", "WEBP"}
MIN_SIDE = 64                 # smaller than this cannot show a recognisable flower
MAX_ASPECT_RATIO = 3.0


class InvalidImageError(ValueError):
    """The upload is not a usable image (message is user-friendly)."""


class ModelNotFoundError(FileNotFoundError):
    """Model or class-name file is missing (message is user-friendly)."""


class PredictionError(RuntimeError):
    """The model produced something unusable."""


@dataclass
class Prediction:
    label: str
    confidence: float                       # probability of the top class, 0..1
    top_k: list[tuple[str, float]]          # [(class, probability), ...] best first
    is_uncertain: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "top_k": [{"label": n, "probability": p} for n, p in self.top_k],
            "is_uncertain": self.is_uncertain,
            "warnings": self.warnings,
        }


# ----------------------------------------------------------------------------- image validation
def load_image(source: bytes | bytearray | BinaryIO) -> Image.Image:
    """Turn uploaded bytes / a file object into a fully-decoded PIL image, or raise
    InvalidImageError with a helpful message."""
    try:
        data = bytes(source) if isinstance(source, (bytes, bytearray)) else source.read()
    except Exception as exc:  # unreadable upload object
        raise InvalidImageError("The uploaded file could not be read. Please upload it again.") from exc
    if not data:
        raise InvalidImageError("The uploaded file is empty.")

    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()                              # detects many corrupted files
        image = Image.open(io.BytesIO(data))
        image.load()                                    # forces a full decode
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise InvalidImageError(
            "This file is not a valid or readable image. Please upload a JPG, PNG, BMP or WEBP photo."
        ) from exc

    if (image.format or "").upper() not in ALLOWED_FORMATS:
        raise InvalidImageError("Unsupported image type. Please upload a JPG, PNG, BMP or WEBP photo.")
    width, height = image.size
    if min(width, height) < MIN_SIDE:
        raise InvalidImageError(f"The image is too small ({width}x{height}px). Please use one at least {MIN_SIDE}px wide and tall.")
    return image


def assess_quality(image: Image.Image) -> list[str]:
    """Non-fatal warnings about inputs the model may handle badly."""
    notes: list[str] = []
    width, height = image.size
    if min(width, height) < 224:
        notes.append("Low resolution: the image will be enlarged, which can reduce accuracy.")
    if max(width, height) / max(1, min(width, height)) > MAX_ASPECT_RATIO:
        notes.append("Very wide/tall image: it will be squeezed to a square, which distorts the flower.")
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    if float(gray.std()) < 8.0:
        notes.append("The image is almost uniform (blank, very dark or very bright).")
    return notes


# ----------------------------------------------------------------------------- predictor
class FlowerPredictor:
    """Wraps the trained Keras model plus its class names."""

    def __init__(self, model: keras.Model, class_names: list[str], img_size: tuple[int, int] = (224, 224),
                 confidence_threshold: float = C.Config().confidence_threshold):
        if not class_names:
            raise ModelNotFoundError("The class-label file is empty or missing.")
        self.model = model
        self.class_names = list(class_names)
        self.img_size = tuple(img_size)
        self.confidence_threshold = confidence_threshold

    @classmethod
    def from_directory(cls, models_dir: Path = C.MODELS_DIR) -> "FlowerPredictor":
        models_dir = Path(models_dir)
        model_path = models_dir / C.MODEL_FILE.name
        names_path = models_dir / C.CLASS_NAMES_FILE.name
        config_path = models_dir / C.CONFIG_FILE.name
        if not model_path.exists():
            raise ModelNotFoundError("The trained model file was not found. Train the model first (see README).")
        if not names_path.exists():
            raise ModelNotFoundError("The class-label file (class_names.json) was not found.")

        class_names = load_json(names_path)
        cfg = load_json(config_path) if config_path.exists() else {}
        model = keras.models.load_model(str(model_path), compile=False)   # compile not needed to predict

        n_out = int(model.output_shape[-1])
        if n_out != len(class_names):
            raise ModelNotFoundError(f"Model has {n_out} outputs but {len(class_names)} class names were found.")
        return cls(
            model,
            class_names,
            img_size=(cfg.get("img_height", 224), cfg.get("img_width", 224)),
            confidence_threshold=cfg.get("confidence_threshold", C.Config().confidence_threshold),
        )

    def predict(self, image: Image.Image, top_k: int = 3, confidence_threshold: float | None = None) -> Prediction:
        threshold = self.confidence_threshold if confidence_threshold is None else confidence_threshold
        try:
            batch = preprocess_for_inference(image, self.img_size)
            probs = np.asarray(self.model(batch, training=False).numpy()[0], dtype=np.float64)
        except Exception as exc:
            raise PredictionError("The model could not analyse this image. Please try another one.") from exc

        if probs.shape != (len(self.class_names),) or not np.all(np.isfinite(probs)):
            raise PredictionError("The model returned an unexpected output. Please try another image.")

        order = np.argsort(probs)[::-1]
        k = max(1, min(top_k, len(self.class_names)))
        ranked = [(self.class_names[i], float(probs[i])) for i in order[:k]]
        best_label, best_conf = ranked[0]
        return Prediction(
            label=best_label,
            confidence=best_conf,
            top_k=ranked,
            is_uncertain=best_conf < threshold,
            warnings=assess_quality(image),
        )
