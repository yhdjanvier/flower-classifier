"""Architecture checks (needs TensorFlow; uses weights=None so nothing is downloaded)."""

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from src.config import Config  # noqa: E402
from src.model import build_model, get_backbone, unfreeze_top_layers  # noqa: E402


def test_shapes_and_frozen_backbone():
    model = build_model(5, Config(), pretrained=False)
    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, 5)
    assert get_backbone(model).trainable is False
    out = model(np.zeros((2, 224, 224, 3), dtype="float32"), training=False).numpy()
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-4)          # softmax


def test_augmentation_only_active_in_training():
    model = build_model(5, Config(), pretrained=False)
    x = np.random.default_rng(0).uniform(0, 255, (2, 224, 224, 3)).astype("float32")
    a = model(x, training=False).numpy()
    b = model(x, training=False).numpy()
    assert np.allclose(a, b)                                     # deterministic at inference


def test_unfreeze_top_layers_only():
    model = build_model(5, Config(), pretrained=False)
    n = unfreeze_top_layers(model, fine_tune_at=100)
    backbone = get_backbone(model)
    assert n > 0
    assert all(not layer.trainable for layer in backbone.layers[:100])
