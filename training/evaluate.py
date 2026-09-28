"""
Evaluate the saved model on the held-out TEST set (images it never saw in training).

    python -m training.evaluate                    # test set, real model
    python -m training.evaluate --split val        # validation set

Smoke-test variant (after `train --quick`):
    python -m training.evaluate --model-dir models/_quick_test --reports-dir reports/_quick_test --split val

Outputs (in reports/evaluation and reports/figures):
    <split>_metrics.json, <split>_classification_report.txt, <split>_misclassified.csv,
    <split>_confusion_matrix.png, <split>_confusion_matrix_normalized.png

IMPORTANT: do not adjust hyper-parameters after looking at TEST results - that would turn
the test set into a second validation set. Tune with validation results only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras

from src import config as C
from src.data import load_splits
from src.evaluation import compute_metrics, plot_confusion_matrix, text_report
from src.preprocessing import dataset_from_split
from src.utils import ensure_dirs, get_logger, load_json, save_json

log = get_logger("evaluate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--model-dir", default=str(C.MODELS_DIR))
    parser.add_argument("--reports-dir", default=str(C.REPORTS_DIR))
    args = parser.parse_args()

    model_dir, reports_dir = Path(args.model_dir), Path(args.reports_dir)
    model_path = model_dir / C.MODEL_FILE.name
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}. Run `python -m training.train` first.")
    ensure_dirs(reports_dir / "figures", reports_dir / "evaluation")

    class_names = load_json(model_dir / C.CLASS_NAMES_FILE.name)
    cfg = C.Config()
    model = keras.models.load_model(str(model_path), compile=False)

    df = load_splits()
    df = df[df["label"].isin(class_names)]
    ds, y_true = dataset_from_split(df, args.split, class_names, cfg)
    log.info("Evaluating on the %s split: %d images", args.split, len(y_true))

    probs = model.predict(ds, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    metrics = compute_metrics(y_true, y_pred, class_names)
    metrics["split"] = args.split

    ev, fig = reports_dir / "evaluation", reports_dir / "figures"
    save_json(metrics, ev / f"{args.split}_metrics.json")
    (ev / f"{args.split}_classification_report.txt").write_text(text_report(y_true, y_pred, class_names), encoding="utf-8")
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, fig / f"{args.split}_confusion_matrix.png")
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, fig / f"{args.split}_confusion_matrix_normalized.png", normalize=True)

    part = df[df["split"] == args.split].reset_index(drop=True)
    wrong = part.assign(
        predicted=[class_names[i] for i in y_pred],
        confidence=np.max(probs, axis=1).round(4),
    )
    wrong = wrong[wrong["label"] != wrong["predicted"]][["path", "label", "predicted", "confidence"]]
    wrong.to_csv(ev / f"{args.split}_misclassified.csv", index=False)

    print("\n" + text_report(y_true, y_pred, class_names))
    print(f"Accuracy: {metrics['accuracy']:.4f} | macro precision {metrics['macro_precision']:.4f} | "
          f"macro recall {metrics['macro_recall']:.4f} | macro F1 {metrics['macro_f1']:.4f}")
    print("Most frequent mistakes:", metrics["top_confusions"])
    log.info("Saved results to %s and %s", ev, fig)


if __name__ == "__main__":
    main()
