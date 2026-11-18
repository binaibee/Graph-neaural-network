"""Stage 6 - Evaluation: measure how good a model's risk scores are."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """The decision line that gives the highest F1-score.

    Always called on *validation* data, so the test data plays no part in
    choosing it.
    """
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[int(np.argmax(f1[:-1]))]) if len(thresholds) else 0.5


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    labels = np.asarray(labels).astype(int)
    predicted = np.asarray(scores) >= threshold
    tp = int((predicted & (labels == 1)).sum())
    fp = int((predicted & (labels == 0)).sum())
    fn = int((~predicted & (labels == 1)).sum())
    tn = int((~predicted & (labels == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    both_classes = 0 < labels.sum() < len(labels)
    return {
        "accuracy": (tp + tn) / max(len(labels), 1),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "roc_auc": float(roc_auc_score(labels, scores)) if both_classes else float("nan"),
        "pr_auc": float(average_precision_score(labels, scores)) if both_classes else float("nan"),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def evaluate_scores(events: pd.DataFrame, scores: np.ndarray) -> dict:
    """Threshold from the validation split, metrics on the test split (overall and per channel)."""
    labels = events["label"].to_numpy()
    validation = (events["split"] == "validation").to_numpy()
    test = (events["split"] == "test").to_numpy()
    threshold = best_threshold(labels[validation], scores[validation])
    result = {"threshold": threshold, "overall": binary_metrics(labels[test], scores[test], threshold)}
    for channel in sorted(events["channel"].unique()):
        mask = test & (events["channel"] == channel).to_numpy()
        result[channel] = binary_metrics(labels[mask], scores[mask], threshold)
    return result


def mean_and_std(runs: list[dict]) -> dict:
    """Average the same metrics over several training runs (seeds)."""
    summary: dict = {"threshold": {"mean": float(np.mean([run["threshold"] for run in runs])), "std": float(np.std([run["threshold"] for run in runs]))}}
    for scope in (key for key in runs[0] if key != "threshold"):
        summary[scope] = {
            name: {"mean": float(np.mean([run[scope][name] for run in runs])), "std": float(np.std([run[scope][name] for run in runs]))}
            for name in METRIC_NAMES
        }
    return summary
