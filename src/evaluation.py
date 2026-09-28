"""
Evaluation metrics and plots  (NO TensorFlow in this file).

Metric cheat-sheet (for the viva):
  accuracy   = correct predictions / all predictions
  precision  = of the images predicted as class X, how many really were X
  recall     = of the images that really are X, how many did we find
  F1-score   = harmonic mean of precision and recall (punishes imbalance between them)
  confusion matrix: rows = true class, columns = predicted class; the diagonal is correct.
  macro average  = plain mean over classes (every class counts equally)
  weighted avg   = mean weighted by class size
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed (works on servers too)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_metrics(y_true, y_pred, class_names: list[str]) -> dict:
    """All numbers we report, computed from true and predicted class indices."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(len(class_names)))

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    macro = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "per_class": {
            name: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, name in enumerate(class_names)
        },
        "class_names": list(class_names),
        "confusion_matrix": cm.tolist(),
        "top_confusions": top_confusions(cm, class_names),
    }


def text_report(y_true, y_pred, class_names: list[str]) -> str:
    return classification_report(
        y_true, y_pred, labels=list(range(len(class_names))), target_names=class_names, digits=4, zero_division=0
    )


def top_confusions(cm, class_names: list[str], k: int = 5) -> list[dict]:
    """The k most frequent mistakes (true class -> predicted class) - useful for interpretation."""
    cm = np.asarray(cm)
    errors = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                errors.append({"true": class_names[i], "predicted": class_names[j], "count": int(cm[i, j])})
    return sorted(errors, key=lambda e: e["count"], reverse=True)[:k]


# ----------------------------------------------------------------------------- plots
def plot_confusion_matrix(cm, class_names: list[str], path: Path, normalize: bool = False, title: str = "") -> None:
    cm = np.asarray(cm, dtype=float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=35, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title or ("Confusion matrix (row-normalised)" if normalize else "Confusion matrix (counts)"))
    threshold = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, text, ha="center", va="center", color="white" if cm[i, j] > threshold else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(phase_histories: list[tuple[str, dict]], path: Path) -> None:
    """Accuracy and loss curves, phases drawn one after another with a dashed divider."""
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 4.2))
    offset = 0
    for name, hist in phase_histories:
        n = len(hist.get("loss", []))
        epochs = np.arange(offset + 1, offset + n + 1)
        ax_acc.plot(epochs, hist.get("accuracy", []), "b-o", ms=3, label="train" if offset == 0 else None)
        ax_acc.plot(epochs, hist.get("val_accuracy", []), "r-o", ms=3, label="validation" if offset == 0 else None)
        ax_loss.plot(epochs, hist.get("loss", []), "b-o", ms=3, label="train" if offset == 0 else None)
        ax_loss.plot(epochs, hist.get("val_loss", []), "r-o", ms=3, label="validation" if offset == 0 else None)
        if offset > 0:
            for ax in (ax_acc, ax_loss):
                ax.axvline(offset + 0.5, color="gray", ls="--")
                ax.text(offset + 0.6, ax.get_ylim()[0], f" {name}", va="bottom", fontsize=8, color="gray")
        offset += n
    for ax, ttl in ((ax_acc, "Accuracy"), (ax_loss, "Loss")):
        ax.set_title(f"{ttl}: training vs validation")
        ax.set_xlabel("Epoch (dashed line = start of fine-tuning)")
        ax.set_ylabel(ttl.lower())
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_class_distribution(per_class: dict[str, dict[str, int]], path: Path) -> None:
    """Stacked bars: how many train/val/test images each class has."""
    classes = sorted(per_class)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bottom = np.zeros(len(classes))
    for split, color in (("train", "#1f77b4"), ("val", "#ff7f0e"), ("test", "#2ca02c")):
        values = np.array([per_class[c].get(split, 0) for c in classes])
        ax.bar(classes, values, bottom=bottom, label=split, color=color)
        bottom += values
    ax.set_ylabel("Number of images")
    ax.set_title("Images per class and split")
    ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
