"""Prediction output format, class labels, confidence, model loading (needs TensorFlow)."""

import json

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402

from src import config as C  # noqa: E402
from src.predict import FlowerPredictor, ModelNotFoundError, load_image  # noqa: E402
from tests.conftest import make_image_bytes  # noqa: E402

CLASSES = ["daisy", "dandelion", "rose", "sunflower", "tulip"]


def tiny_model(n=len(CLASSES)):
    inp = keras.Input(shape=(224, 224, 3))
    x = layers.GlobalAveragePooling2D()(inp)
    return keras.Model(inp, layers.Dense(n, activation="softmax")(x))


def test_prediction_format_and_confidence():
    predictor = FlowerPredictor(tiny_model(), CLASSES)
    result = predictor.predict(load_image(make_image_bytes((256, 256))), top_k=3)
    assert result.label in CLASSES
    assert len(result.top_k) == 3
    probs = [p for _, p in result.top_k]
    assert probs == sorted(probs, reverse=True)                 # best first
    assert result.confidence == probs[0]
    assert 0.0 <= result.confidence <= 1.0
    assert {"label", "confidence", "top_k", "is_uncertain", "warnings"} <= set(result.to_dict())


def test_all_probabilities_sum_to_one():
    predictor = FlowerPredictor(tiny_model(), CLASSES)
    result = predictor.predict(load_image(make_image_bytes()), top_k=5)
    assert abs(sum(p for _, p in result.top_k) - 1.0) < 1e-4


def test_uncertainty_threshold():
    predictor = FlowerPredictor(tiny_model(), CLASSES)           # untrained -> probabilities near 0.2 each
    img = load_image(make_image_bytes())
    assert predictor.predict(img, confidence_threshold=0.9).is_uncertain is True
    assert predictor.predict(img, confidence_threshold=0.0).is_uncertain is False


def test_missing_model_raises_friendly_error(tmp_path):
    with pytest.raises(ModelNotFoundError):
        FlowerPredictor.from_directory(tmp_path)


def test_model_loading_roundtrip(tmp_path):
    tiny_model().save(tmp_path / C.MODEL_FILE.name)
    (tmp_path / C.CLASS_NAMES_FILE.name).write_text(json.dumps(CLASSES))
    predictor = FlowerPredictor.from_directory(tmp_path)
    assert predictor.class_names == CLASSES
    assert predictor.predict(load_image(make_image_bytes())).label in CLASSES


def test_class_count_mismatch_detected(tmp_path):
    tiny_model().save(tmp_path / C.MODEL_FILE.name)
    (tmp_path / C.CLASS_NAMES_FILE.name).write_text(json.dumps(["only", "two"]))
    with pytest.raises(ModelNotFoundError):
        FlowerPredictor.from_directory(tmp_path)
