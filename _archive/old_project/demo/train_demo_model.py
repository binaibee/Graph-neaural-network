"""Train the demo's GNN with the project's EXISTING training loop and save it.

The research pipeline trains its models and keeps only their scores, so there is
no checkpoint to load.  This script runs the same steps as
``pipeline.build_scored_snapshot`` for the heterogeneous GraphSAGE model and
intercepts the model object so its weights can be written to ``demo/model``.
Nothing about the data, features, model or training procedure is changed.

Usage (from the repository root):

    python demo\\train_demo_model.py
"""
from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone

import numpy as np

import engine  # sets sys.path for scam_gnn
from scam_gnn import pipeline
from scam_gnn.config import load_config
from scam_gnn.features import build_feature_bundle
from scam_gnn.graph import build_pyg_heterodata


def _threshold_counts(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Confusion matrix, accuracy and F1 at the 0.5 threshold.

    These are not produced by the research code (which reports precision,
    recall, ROC-AUC, PR-AUC and recall@precision); they are derived here from
    the same test-set predictions so the demo can display them.
    """
    predicted = scores >= engine.THRESHOLD
    tp = int((predicted & (labels == 1)).sum())
    fp = int((predicted & (labels == 0)).sum())
    fn = int((~predicted & (labels == 1)).sum())
    tn = int((~predicted & (labels == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": (tp + tn) / max(len(labels), 1),
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }


def main() -> None:
    import os

    import torch
    import torch_geometric

    os.chdir(engine.ROOT)
    config = load_config(engine.CONFIG_PATH)
    seed = engine.SEED
    print(f"Config: {engine.CONFIG_PATH.relative_to(engine.ROOT)}  |  model: {engine.MODEL_NAME}  |  seed: {seed}")

    print("1/4 Loading data and building the graph…")
    split_events, tables = engine.prepare_base(config, seed)
    print("2/4 Building features…")
    feature_bundle = build_feature_bundle(tables, config.features)
    messages, _, _ = pipeline._prepare_matrices(feature_bundle, tables)
    hetero_data = build_pyg_heterodata(tables, feature_bundle.entity_features)

    captured: dict[str, object] = {}
    original_builder = pipeline.build_heterogeneous_graphsage

    def capturing_builder(metadata, *args, **kwargs):
        captured["model"] = original_builder(metadata, *args, **kwargs)
        return captured["model"]

    print("3/4 Training with pipeline._train_heterogeneous_gnn (unchanged research code)…")
    pipeline.build_heterogeneous_graphsage = capturing_builder
    started = time.perf_counter()
    try:
        node_scores = pipeline._train_heterogeneous_gnn(hetero_data, seed)
    finally:
        pipeline.build_heterogeneous_graphsage = original_builder
    train_seconds = time.perf_counter() - started
    model = captured["model"]

    scores = pipeline._align_message_scores(node_scores, pipeline._message_node_order(tables), messages["message_id"])
    metrics = pipeline._evaluate_scores(messages, scores, config)

    print("4/4 Saving model, metadata and predictions…")
    engine.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), engine.WEIGHTS_PATH)

    predictions = messages[["event_id", "message_id", "channel", "split", "label"]].copy()
    predictions["score"] = scores
    predictions.to_csv(engine.PREDICTIONS_PATH, index=False)

    test = predictions[predictions["split"] == "test"]
    derived = {"overall": _threshold_counts(test["label"].to_numpy(), test["score"].to_numpy())}
    for channel, frame in test.groupby("channel"):
        derived[channel] = _threshold_counts(frame["label"].to_numpy(), frame["score"].to_numpy())

    node_types, edge_types = hetero_data.metadata()
    meta = {
        "model_name": engine.MODEL_NAME,
        "seed": seed,
        "config": str(engine.CONFIG_PATH.relative_to(engine.ROOT)),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_seconds": round(train_seconds, 1),
        "embedding_backend": engine.embedding_backend(config),
        "threshold": engine.THRESHOLD,
        "feature_columns": list(feature_bundle.message_features.columns),
        "metadata": {"node_types": list(node_types), "edge_types": [list(edge_type) for edge_type in edge_types]},
        "dataset": {
            "num_messages": int(len(messages)),
            "num_nodes": int(len(tables.entities)),
            "num_edges": int(len(tables.relations)),
            "nodes_by_type": tables.entities["entity_type"].value_counts().to_dict(),
            "edges_by_type": tables.relations["edge_type"].value_counts().to_dict(),
            "split_counts": {
                f"{split}/{channel}": {"messages": int(len(frame)), "scam": int(frame["label"].sum())}
                for (split, channel), frame in messages.groupby(["split", "channel"])
            },
        },
        "test_metrics": pipeline._json_safe(metrics),
        "test_metrics_derived_by_demo": derived,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
        },
    }
    engine.META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nTrained in {train_seconds:.0f}s. Test-set metrics (threshold 0.5):")
    for name in ["precision", "recall", "roc_auc", "pr_auc", "recall_at_precision"]:
        print(f"  {name:20s} {metrics[name]:.4f}")
    for channel, channel_metrics in metrics["per_channel"].items():
        print(f"  [{channel}] precision {channel_metrics['precision']:.4f}  recall {channel_metrics['recall']:.4f}  pr_auc {channel_metrics['pr_auc']:.4f}")
    print(f"Saved to {engine.MODEL_DIR.relative_to(engine.ROOT)}")


if __name__ == "__main__":
    main()
