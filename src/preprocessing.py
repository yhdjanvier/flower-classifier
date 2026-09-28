"""
Image pre-processing shared by TRAINING and INFERENCE.

The single most common deployment bug is "the app pre-processes differently from
training".  We avoid it in three ways:
  1. resizing is done by ONE function (`tf.image.resize`) in both places;
  2. pixel scaling to [-1, 1] (what MobileNetV2 expects) is a layer INSIDE the saved
     model (see model.py), so it travels with the model file;
  3. augmentation is also inside the model and is only active while training.

Images therefore enter the model as float32 pixels in the range 0..255.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageOps

from src.config import Config


# ----------------------------------------------------------------------------- training side (tf.data)
def decode_and_resize(path: tf.Tensor, img_size: tuple[int, int]) -> tf.Tensor:
    """Read a file, decode to 3-channel RGB, resize to img_size, return float32 0..255."""
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    return tf.image.resize(image, img_size)  # bilinear, returns float32


def make_dataset(paths, labels, cfg: Config, training: bool) -> tf.data.Dataset:
    """Build a batched, prefetched tf.data pipeline.

    training=True  -> shuffled every epoch (seeded).
    training=False -> fixed order, which evaluation code relies on to line predictions up
                      with the true labels.
    No augmentation happens here: it lives in the model and is off for val/test data.
    """
    paths = np.asarray(paths, dtype=str)
    labels = np.asarray(labels, dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(paths), seed=cfg.seed, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda p, y: (decode_and_resize(p, cfg.img_size), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)


def dataset_from_split(df: pd.DataFrame, split: str, class_names: list[str], cfg: Config) -> tuple[tf.data.Dataset, np.ndarray]:
    """Dataset + the true label indices for one split ('train' / 'val' / 'test')."""
    part = df[df["split"] == split]
    index = {name: i for i, name in enumerate(class_names)}
    y = part["label"].map(index).to_numpy(dtype=np.int32)
    return make_dataset(part["abs_path"].to_numpy(), y, cfg, training=(split == "train")), y


# ----------------------------------------------------------------------------- inference side (PIL image)
def preprocess_for_inference(image: Image.Image, img_size: tuple[int, int]) -> np.ndarray:
    """PIL image -> array of shape (1, H, W, 3), float32, 0..255 (same convention as training)."""
    rgb = ImageOps.exif_transpose(image).convert("RGB")     # honour phone-camera rotation
    array = np.asarray(rgb, dtype=np.float32)
    resized = tf.image.resize(array, img_size).numpy()      # SAME resize op as training
    return resized[np.newaxis, ...].astype(np.float32)
