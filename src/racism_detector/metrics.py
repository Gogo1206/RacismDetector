import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def best_threshold(y_true, probs) -> float:
    """Decision threshold that maximises F1. Pick it on validation, apply it to test."""
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    # The last precision/recall point has no threshold.
    return float(thresholds[np.argmax(f1[:-1])])


def evaluate(y_true, probs, threshold: float, race_supportive) -> dict[str, float]:
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    pred = probs >= threshold
    supportive = np.asarray(race_supportive, dtype=bool)
    return {
        "pr_auc": float(average_precision_score(y_true, probs)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "f1": float(f1_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred)),
        "fpr": float(pred[y_true == 0].mean()),
        # Share of supportive comments about race wrongly flagged as racist.
        "fpr_race_supportive": float(pred[supportive].mean()),
        "threshold": threshold,
    }
