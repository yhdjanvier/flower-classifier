"""Metric calculations checked against hand-computed values (no TensorFlow needed)."""

import numpy as np

from src.evaluation import compute_metrics, plot_confusion_matrix, plot_history, top_confusions


def test_metrics_hand_computed():
    #        true: a a a b b c        pred: a a b b c c
    y_true = [0, 0, 0, 1, 1, 2]
    y_pred = [0, 0, 1, 1, 2, 2]
    m = compute_metrics(y_true, y_pred, ["a", "b", "c"])
    assert m["accuracy"] == 4 / 6
    assert m["per_class"]["a"]["precision"] == 1.0          # 2 predicted a, both right
    assert m["per_class"]["a"]["recall"] == 2 / 3           # 3 real a, found 2
    assert m["per_class"]["b"]["precision"] == 0.5
    assert m["per_class"]["b"]["recall"] == 0.5
    assert m["per_class"]["c"]["precision"] == 0.5
    assert m["per_class"]["c"]["recall"] == 1.0
    assert m["confusion_matrix"] == [[2, 1, 0], [0, 1, 1], [0, 0, 1]]
    assert abs(m["macro_recall"] - np.mean([2 / 3, 0.5, 1.0])) < 1e-9


def test_top_confusions_ordering():
    cm = np.array([[5, 3, 0], [1, 5, 0], [0, 0, 6]])
    top = top_confusions(cm, ["a", "b", "c"])
    assert top[0] == {"true": "a", "predicted": "b", "count": 3}


def test_plots_are_written(tmp_path):
    plot_confusion_matrix([[2, 1], [0, 3]], ["a", "b"], tmp_path / "cm.png")
    plot_confusion_matrix([[2, 1], [0, 3]], ["a", "b"], tmp_path / "cmn.png", normalize=True)
    hist = {"loss": [1, .5], "val_loss": [1.1, .6], "accuracy": [.5, .8], "val_accuracy": [.4, .7]}
    plot_history([("p1", hist), ("p2", hist)], tmp_path / "h.png")
    for f in ("cm.png", "cmn.png", "h.png"):
        assert (tmp_path / f).stat().st_size > 1000
