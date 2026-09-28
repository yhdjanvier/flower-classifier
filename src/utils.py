"""Small helpers shared by training, evaluation and the app."""

from __future__ import annotations

import json
import logging
import os
import random
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np


def get_logger(name: str = "flower") -> logging.Logger:
    """Console logger with a compact format (created once per name)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


class _JsonEncoder(json.JSONEncoder):
    """Lets us save NumPy numbers/arrays and Paths without manual conversion."""

    def default(self, o: Any):  # noqa: D401
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return o.as_posix()
        return super().default(o)


def save_json(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, cls=_JsonEncoder), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and TensorFlow. (Results on multi-threaded CPUs can still differ
    very slightly between runs; the seed makes them repeatable in practice, not bit-exact.)"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
    except ImportError:  # data/evaluation scripts can run without TensorFlow
        pass


def download_file(url: str, destination: Path, timeout: int = 120, max_mb: int = 300) -> None:
    """Download a public file (used only as an optional fallback for hosting the model
    outside GitHub). HTTPS only, size-limited, written atomically."""
    if not url.lower().startswith("https://"):
        raise ValueError("Only https:// URLs are allowed.")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as response, open(tmp, "wb") as out:  # noqa: S310
        total = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_mb * 1024 * 1024:
                tmp.unlink(missing_ok=True)
                raise ValueError(f"Download exceeds {max_mb} MB limit.")
            out.write(chunk)
    tmp.replace(destination)
