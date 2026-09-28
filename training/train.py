"""
Two-phase transfer-learning training.

    python -m training.train --quick     # 1-2 minute smoke test (tiny subset, 1 epoch per phase)
    python -m training.train             # the real training run

PHASE 1  backbone frozen, train only the new classification head (lr 1e-3)
PHASE 2  unfreeze the upper backbone layers and fine-tune with a tiny lr (1e-5)

Model selection uses the VALIDATION set only.  After phase 2 we keep whichever of
"phase-1 weights" / "phase-2 weights" scored the higher validation accuracy.  The TEST set
is not touched here - it is used once, by `python -m training.evaluate`.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from src import config as C
from src.data import load_splits
from src.evaluation import plot_history
from src.model import build_model, compile_model, export_model, summary_text, unfreeze_top_layers
from src.preprocessing import dataset_from_split
from src.utils import ensure_dirs, get_logger, save_json, set_seed

log = get_logger("train")


def _callbacks(cfg: C.Config, checkpoint_path: Path | None) -> list[keras.callbacks.Callback]:
    cbs: list[keras.callbacks.Callback] = [
        # stop when validation loss stops improving, and go back to the best epoch
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.early_stopping_patience, restore_best_weights=True),
        # halve the learning rate when progress stalls
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=cfg.reduce_lr_factor, patience=cfg.reduce_lr_patience, min_lr=1e-7),
    ]
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        cbs.append(keras.callbacks.ModelCheckpoint(str(checkpoint_path), monitor="val_loss", save_best_only=True))
    return cbs


def _score(model: keras.Model, ds: tf.data.Dataset, y_true: np.ndarray) -> dict:
    """Validation loss / accuracy / macro-F1."""
    loss, acc = model.evaluate(ds, verbose=0)
    y_pred = np.argmax(model.predict(ds, verbose=0), axis=1)
    return {"loss": float(loss), "accuracy": float(acc), "macro_f1": float(f1_score(y_true, y_pred, average="macro"))}


def _class_weights(y_train: np.ndarray, n_classes: int, threshold: float) -> dict[int, float] | None:
    counts = np.bincount(y_train, minlength=n_classes)
    ratio = counts.max() / max(1, counts.min())
    if ratio <= threshold:
        log.info("Class imbalance ratio %.2f <= %.2f -> class weights NOT used.", ratio, threshold)
        return None
    weights = compute_class_weight("balanced", classes=np.arange(n_classes), y=y_train)
    log.info("Class imbalance ratio %.2f > %.2f -> using class weights %s", ratio, threshold, np.round(weights, 2))
    return {i: float(w) for i, w in enumerate(weights)}


def run_training(
    cfg: C.Config,
    run_name: str = "final",
    fine_tune: bool = True,
    export_dir: Path | None = C.MODELS_DIR,
    reports_dir: Path | None = C.REPORTS_DIR,
    max_per_class: int | None = None,
) -> dict:
    """Train, select, optionally export. Returns a summary dict (validation scores only)."""
    started = time.time()
    set_seed(cfg.seed)

    df = load_splits()
    if max_per_class:                                    # used by --quick
        df = df.groupby(["split", "label"]).head(max_per_class)
    class_names = sorted(df["label"].unique().tolist())
    train_ds, y_train = dataset_from_split(df, "train", class_names, cfg)
    val_ds, y_val = dataset_from_split(df, "val", class_names, cfg)
    log.info("[%s] classes=%s | train=%d val=%d", run_name, class_names, len(y_train), len(y_val))

    class_weight = _class_weights(y_train, len(class_names), cfg.class_weight_ratio_threshold)
    model = build_model(len(class_names), cfg)
    ckpt_dir = C.CHECKPOINT_DIR / run_name

    # ---------------------------------------------------------------- phase 1
    compile_model(model, cfg.phase1_lr)
    log.info("[%s] PHASE 1: training the classification head (backbone frozen), lr=%g", run_name, cfg.phase1_lr)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=cfg.phase1_epochs, class_weight=class_weight,
                   callbacks=_callbacks(cfg, ckpt_dir / "phase1.keras"), verbose=2).history
    val1 = _score(model, val_ds, y_val)
    log.info("[%s] phase-1 validation: %s", run_name, val1)
    weights_phase1 = model.get_weights()

    histories = [("phase 1 (head only)", h1)]
    val2, chosen = None, "phase1"

    # ---------------------------------------------------------------- phase 2
    if fine_tune:
        n_trainable = unfreeze_top_layers(model, cfg.fine_tune_at)
        compile_model(model, cfg.phase2_lr)               # must re-compile after changing trainable flags
        log.info("[%s] PHASE 2: fine-tuning %d backbone layers from index %d, lr=%g",
                 run_name, n_trainable, cfg.fine_tune_at, cfg.phase2_lr)
        h2 = model.fit(train_ds, validation_data=val_ds, epochs=cfg.phase2_epochs, class_weight=class_weight,
                       callbacks=_callbacks(cfg, ckpt_dir / "phase2.keras"), verbose=2).history
        histories.append(("phase 2 (fine-tune)", h2))
        val2 = _score(model, val_ds, y_val)
        log.info("[%s] phase-2 validation: %s", run_name, val2)
        if val2["accuracy"] >= val1["accuracy"]:
            chosen = "phase2"
        else:
            model.set_weights(weights_phase1)             # fine-tuning did not help -> go back
            log.info("[%s] Fine-tuning did not improve validation accuracy; restored phase-1 weights.", run_name)
    final_val = val2 if chosen == "phase2" else val1

    summary = {
        "run_name": run_name,
        "chosen": chosen,
        "fine_tune": fine_tune,
        "val_phase1": val1,
        "val_phase2": val2,
        "val_final": final_val,
        "config": asdict(cfg),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "class_weights_used": class_weight is not None,
        "epochs_run": {name: len(h["loss"]) for name, h in histories},
        "minutes": round((time.time() - started) / 60, 2),
    }

    # ---------------------------------------------------------------- save artefacts
    if reports_dir is not None:
        ensure_dirs(reports_dir / "figures", reports_dir / "evaluation")
        save_json({name: hist for name, hist in histories}, reports_dir / "evaluation" / "training_history.json")
        save_json(summary, reports_dir / "evaluation" / "model_selection.json")
        (reports_dir / "evaluation" / "model_summary.txt").write_text(summary_text(model), encoding="utf-8")
        plot_history(histories, reports_dir / "figures" / "training_curves.png")
    if export_dir is not None:
        export_model(model, export_dir / C.MODEL_FILE.name)
        save_json(class_names, export_dir / C.CLASS_NAMES_FILE.name)
        save_json(asdict(cfg), export_dir / C.CONFIG_FILE.name)
        log.info("[%s] Saved model + class names + config to %s", run_name, export_dir)
    log.info("[%s] finished in %.1f min. Final validation: %s", run_name, summary["minutes"], final_val)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="tiny smoke test; writes to models/_quick_test, not models/")
    parser.add_argument("--no-fine-tune", action="store_true", help="skip phase 2 (baseline: frozen backbone only)")
    args = parser.parse_args()

    cfg = C.Config()
    if args.quick:
        cfg = replace(cfg, phase1_epochs=1, phase2_epochs=1, batch_size=16)
        run_training(cfg, run_name="quick", fine_tune=not args.no_fine_tune,
                     export_dir=C.MODELS_DIR / "_quick_test", reports_dir=C.REPORTS_DIR / "_quick_test",
                     max_per_class=20)
        log.info("Smoke test OK. Now run the real training:  python -m training.train")
    else:
        run_training(cfg, run_name="final", fine_tune=not args.no_fine_tune)
        log.info("Next: python -m training.evaluate")


if __name__ == "__main__":
    main()
