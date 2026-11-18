from __future__ import annotations

import math

import numpy as np
import pandas as pd


def precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]
    tp = np.cumsum(y_true == 1)
    fp = np.cumsum(y_true == 0)
    precision = tp / np.maximum(tp + fp, 1)
    positives = max(int((y_true == 1).sum()), 1)
    recall = tp / positives
    return precision, recall


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    precision, recall = precision_recall_curve(y_true, y_score)
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return float(np.trapezoid(precision, recall))


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = y_score[y_true == 1]
    negatives = y_score[y_true == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return math.nan
    comparisons = (positives[:, None] > negatives[None, :]).mean()
    ties = (positives[:, None] == negatives[None, :]).mean() * 0.5
    return float(comparisons + ties)


def recall_at_precision(y_true: np.ndarray, y_score: np.ndarray, target_precision: float) -> float:
    precision, recall = precision_recall_curve(y_true, y_score)
    valid = recall[precision >= target_precision]
    return float(valid.max()) if len(valid) else 0.0


def threshold_for_target_precision(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_precision: float,
    default: float = 0.5,
) -> float:
    order = np.argsort(-y_score, kind="mergesort")
    sorted_true = y_true[order]
    sorted_score = y_score[order]
    tp = np.cumsum(sorted_true == 1)
    fp = np.cumsum(sorted_true == 0)
    precision = tp / np.maximum(tp + fp, 1)
    valid = np.where(precision >= target_precision)[0]
    if len(valid) == 0:
        return default
    return float(sorted_score[valid[-1]])


def binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float = 0.5,
    target_precision: float = 0.9,
) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "roc_auc": roc_auc(y_true, y_score),
        "pr_auc": pr_auc(y_true, y_score),
        "recall_at_precision": recall_at_precision(y_true, y_score, target_precision),
    }


def per_channel_metrics(events: pd.DataFrame, scores: np.ndarray, target_precision: float) -> dict[str, dict[str, float]]:
    frame = events.drop_duplicates("message_id").copy()
    frame["score"] = scores
    payload: dict[str, dict[str, float]] = {}
    for channel, channel_frame in frame.groupby("channel"):
        payload[channel] = binary_metrics(
            channel_frame["label"].to_numpy(dtype=int),
            channel_frame["score"].to_numpy(dtype=float),
            target_precision=target_precision,
        )
    return payload
