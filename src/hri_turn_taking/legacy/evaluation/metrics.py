import numpy as np
from numpy.typing import ArrayLike


def binary_accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    truth = np.asarray(y_true, dtype=bool)
    pred = np.asarray(y_pred, dtype=bool)
    _validate_same_shape(truth, pred)
    return float(np.mean(truth == pred))


def f1_score_binary(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    truth = np.asarray(y_true, dtype=bool)
    pred = np.asarray(y_pred, dtype=bool)
    _validate_same_shape(truth, pred)

    tp = np.logical_and(truth, pred).sum()
    fp = np.logical_and(~truth, pred).sum()
    fn = np.logical_and(truth, ~pred).sum()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def _validate_same_shape(left: np.ndarray, right: np.ndarray) -> None:
    if left.shape != right.shape:
        raise ValueError("arrays must have the same shape")
    if left.size == 0:
        raise ValueError("arrays must not be empty")
