"""
CNN model: MobileNetV2 transfer learning.

Architecture (top to bottom):

  Input image (224x224x3, pixels 0..255)
    -> Data augmentation            (random flip/rotate/zoom/shift/brightness/contrast; TRAINING ONLY)
    -> Rescaling to [-1, 1]         (the exact pre-processing MobileNetV2 was trained with)
    -> MobileNetV2 backbone         (ImageNet-pretrained convolutional feature extractor)
    -> GlobalAveragePooling2D       (1280 feature maps -> one 1280-number feature vector)
    -> Dropout                      (randomly ignores neurons while training -> less over-fitting)
    -> Dense(num_classes, softmax)  (probabilities that sum to 1)

Why MobileNetV2?  ~2.3 M parameters and a ~10 MB saved model: fast on a CPU, small
enough to commit to GitHub, light enough for Streamlit Cloud, yet strong on ImageNet-type
photos (which include flowers).  Reference: Sandler et al., "MobileNetV2: Inverted
Residuals and Linear Bottlenecks", CVPR 2018.
"""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import Config


def build_augmentation(cfg: Config) -> keras.Sequential:
    """Realistic photo variations only. NO vertical flip / heavy colour shifts: they could
    change what a flower looks like. These layers do nothing at inference time."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(cfg.rotation_factor, fill_mode="reflect"),
            layers.RandomZoom(cfg.zoom_factor, fill_mode="reflect"),
            layers.RandomTranslation(cfg.translation_factor, cfg.translation_factor, fill_mode="reflect"),
            layers.RandomBrightness(cfg.brightness_factor, value_range=(0, 255)),
            layers.RandomContrast(cfg.contrast_factor),
        ],
        name="augmentation",
    )


def build_model(num_classes: int, cfg: Config, pretrained: bool = True) -> keras.Model:
    """Create the (frozen-backbone) classifier. `pretrained=False` is used only by unit
    tests so they do not need to download the ImageNet weights."""
    inputs = keras.Input(shape=(*cfg.img_size, 3), name="image")
    x = build_augmentation(cfg)(inputs)
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="mobilenet_preprocess")(x)

    backbone = keras.applications.MobileNetV2(
        input_shape=(*cfg.img_size, 3),
        include_top=False,                       # drop ImageNet's 1000-class head
        weights="imagenet" if pretrained else None,
    )
    backbone.trainable = False                   # PHASE 1: keep the learned features fixed
    x = backbone(x, training=False)              # training=False keeps BatchNorm statistics fixed

    x = layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = layers.Dropout(cfg.dropout, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name="flower_mobilenetv2")


def get_backbone(model: keras.Model) -> keras.Model:
    """Return the nested MobileNetV2 sub-model."""
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            # the augmentation Sequential is also a Model; the backbone is the big one
            if len(layer.layers) > 50:
                return layer
    raise ValueError("Backbone not found in model.")


def compile_model(model: keras.Model, learning_rate: float) -> None:
    """Adam optimiser + sparse categorical cross-entropy (labels are integers 0..K-1)."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def unfreeze_top_layers(model: keras.Model, fine_tune_at: int) -> int:
    """PHASE 2 set-up: make the upper backbone layers trainable.

    * Early layers detect generic edges/colours/textures -> keep frozen.
    * Later layers detect object parts and are the ones worth adapting to flowers.
    * BatchNormalization layers stay frozen: with a small dataset, updating their running
      statistics usually hurts more than it helps.
    Returns the number of backbone layers that are now trainable. Re-compile afterwards!"""
    backbone = get_backbone(model)
    backbone.trainable = True
    for layer in backbone.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in backbone.layers[fine_tune_at:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    return sum(1 for layer in backbone.layers if layer.trainable and layer.weights)


def summary_text(model: keras.Model) -> str:
    lines: list[str] = []
    model.summary(print_fn=lambda line, *a, **k: lines.append(str(line)))
    return "\n".join(lines)


def export_model(model: keras.Model, path: Path) -> None:
    """Save the deployable `.keras` file.

    We re-compile with plain SGD first: it has no per-weight state, so the file does not
    contain Adam's extra variables and stays small (good for GitHub / Streamlit Cloud)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.compile(optimizer=keras.optimizers.SGD(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.save(str(path))
