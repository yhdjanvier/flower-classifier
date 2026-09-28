"""Image loading / validation / preprocessing (needs TensorFlow)."""

import io

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("tensorflow")

from src.predict import InvalidImageError, assess_quality, load_image  # noqa: E402
from src.preprocessing import preprocess_for_inference  # noqa: E402
from tests.conftest import make_image_bytes  # noqa: E402


def test_preprocess_shape_dtype_range():
    arr = preprocess_for_inference(load_image(make_image_bytes((300, 200))), (224, 224))
    assert arr.shape == (1, 224, 224, 3)
    assert arr.dtype == np.float32
    assert 0.0 <= arr.min() and arr.max() <= 255.0          # scaling to [-1,1] happens INSIDE the model


def test_rgba_and_grayscale_are_converted_to_rgb():
    for mode in ("RGBA", "L"):
        buf = io.BytesIO()
        Image.new(mode, (100, 100)).save(buf, format="PNG")
        assert preprocess_for_inference(load_image(buf.getvalue()), (224, 224)).shape == (1, 224, 224, 3)


def test_invalid_inputs_raise_friendly_error():
    for bad in (b"", b"not an image at all", b"\x89PNG\r\n\x1a\n" + b"corrupt"):
        with pytest.raises(InvalidImageError):
            load_image(bad)


def test_too_small_image_rejected():
    with pytest.raises(InvalidImageError):
        load_image(make_image_bytes((20, 20)))


def test_quality_warnings():
    assert any("Low resolution" in w for w in assess_quality(load_image(make_image_bytes((100, 100)))))
    flat = io.BytesIO()
    Image.new("RGB", (300, 300), (10, 10, 10)).save(flat, format="PNG")
    assert any("uniform" in w for w in assess_quality(load_image(flat.getvalue())))
