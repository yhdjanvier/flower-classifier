"""Shared test helpers: tiny synthetic images and a tiny model (no dataset, no downloads)."""

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def make_image_bytes(size=(128, 96), fmt="JPEG", seed=0) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def jpeg_bytes():
    return make_image_bytes()
