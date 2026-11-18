"""Stage 5 - Training and evaluation of every model.

Run from the repository root:

    python -m scam_gnn.train

It executes the whole pipeline (data -> graph -> features -> models), evaluates
the baselines, the GNN and the graph ablation on the test split, and writes to
``results/``:

    results.json     every number shown in the report and in the demo
    model.pt         the trained GNN used by the demo
    prepared.pkl     the graph and features, so the demo starts quickly
    predictions.csv  the risk score of every message
"""
from __future__ import annotations

import argparse
import json
import pickle
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from scam_gnn.config import Config, load_config
from scam_gnn.data import load_events
from scam_gnn.evaluate import evaluate_scores, mean_and_std
from scam_gnn.features import Scaler, TextEncoder, entity_features, message_features
from scam_gnn.graph import GraphTables, build_graph, graph_summary
from scam_gnn.models import BASELINES, GNN_NAME, GRAPH_VARIANTS, apply_variant, balanced_sample_weights, predict_gnn, to_tensors, train_baseline, train_gnn

MODEL_FILE = "model.pt"
PREPARED_FILE = "prepared.pkl"
RESULTS_FILE = "results.json"
PREDICTIONS_FILE = "predictions.csv"


@dataclass
class Prepared:
    """Everything the models need, produced by stages 1-3."""

    tables: GraphTables
    features: dict[str, pd.DataFrame]  # unscaled features per node type
    scaler: Scaler
    data_report: dict
    text_embedding: str
    text_description: str


def prepare(config: Config, encoder: TextEncoder | None = None, log=print) -> Prepared:
    log("1/5 Loading and cleaning the three datasets")
    events, data_report = load_events(config)
    log("2/5 Building the graph")
    tables = build_graph(events, config)
    log("3/5 Computing features")
    encoder = encoder or TextEncoder(config)
    vectors = encoder.encode_cached(tables.events["text"].tolist(), config.results_path / "cache")
    features = entity_features(tables)
    features["message"] = message_features(tables.events, vectors)
    scaler = Scaler()
    train_ids = tables.events.loc[tables.events["split"] == "train", "message_id"]
    scaler.fit("message", features["message"].loc[train_ids])
    for node_type, frame in features.items():
        if node_type != "message":
            scaler.fit(node_type, frame)
    return Prepared(tables, features, scaler, data_report, encoder.backend, encoder.description)


def aligned_events(tables: GraphTables, message_ids: list[str]) -> pd.DataFrame:
    """The events table in the same order as the message nodes of the GNN."""
    return tables.events.set_index("message_id").loc[message_ids].reset_index()


def run(config: Config, log=print) -> dict:
    started = time.perf_counter()
    prepared = prepare(config, log=log)
    x_dict, edge_index_dict, node_ids = to_tensors(prepared.tables, prepared.features, prepared.scaler)
    events = aligned_events(prepared.tables, node_ids["message"])
    labels = events["label"].to_numpy()
    train = (events["split"] == "train").to_numpy()
    weights = balanced_sample_weights(events)
    x_message = x_dict["message"].numpy()

    log("4/5 Training and evaluating the models")
    models: dict[str, dict] = {}
    scores_by_model: dict[str, np.ndarray] = {}
    for name, label in BASELINES.items():
        runs = []
        for seed in config.seeds:
            model = train_baseline(name, x_message[train], labels[train], weights[train], seed)
            scores = model.predict_proba(x_message)[:, 1]
            runs.append(evaluate_scores(events, scores))
            scores_by_model.setdefault(name, scores)
        models[name] = {"label": label, "uses_graph": False, "summary": mean_and_std(runs), "runs": runs}
        log(f"    {label}: test PR-AUC {models[name]['summary']['overall']['pr_auc']['mean']:.4f}")

    ablation: dict[str, dict] = {}
    saved_model, saved_info = None, None
    for variant, description in GRAPH_VARIANTS.items():
        edges = apply_variant(edge_index_dict, variant)
        runs, infos = [], []
        for seed in config.seeds:
            model, info = train_gnn(x_dict, edges, events, config, seed)
            scores = predict_gnn(model, x_dict, edges)
            runs.append(evaluate_scores(events, scores))
            infos.append({"seed": seed, **info})
            if variant == "full" and saved_model is None:
                saved_model, saved_info = model, {"seed": seed, **info}
                scores_by_model["gnn"] = scores
        ablation[variant] = {"label": f"{GNN_NAME}, {description}", "summary": mean_and_std(runs), "runs": runs, "training": infos}
        log(f"    GNN ({description}): test PR-AUC {ablation[variant]['summary']['overall']['pr_auc']['mean']:.4f}")
    models["gnn"] = {"label": GNN_NAME, "uses_graph": True, "summary": ablation["full"]["summary"], "runs": ablation["full"]["runs"]}

    log("5/5 Saving the model and the results")
    demo_metrics = ablation["full"]["runs"][0]
    results_dir = config.results_path
    results_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": saved_model.state_dict(),
            "input_dims": {node_type: int(x.shape[1]) for node_type, x in x_dict.items()},
            "hidden_dim": config.hidden_dim,
            "threshold": demo_metrics["threshold"],
            "scaler": prepared.scaler.to_dict(),
            "text_embedding": prepared.text_embedding,
            "hashed_dim": config.hashed_dim,
            "feature_columns": {node_type: list(frame.columns) for node_type, frame in prepared.features.items()},
        },
        results_dir / MODEL_FILE,
    )
    with (results_dir / PREPARED_FILE).open("wb") as handle:
        pickle.dump(
            {
                "events": prepared.tables.events,
                "entities": prepared.tables.entities,
                "relations": prepared.tables.relations,
                "features": prepared.features,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    predictions = events[["event_id", "message_id", "channel", "split", "label"]].copy()
    for name, scores in scores_by_model.items():
        predictions[f"score_{name}"] = scores
    predictions.to_csv(results_dir / PREDICTIONS_FILE, index=False)

    results = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime_seconds": round(time.perf_counter() - started, 1),
        "config": asdict(config),
        "text_representation": prepared.text_description,
        "data": prepared.data_report,
        "splits": {
            f"{channel}/{split}": {"messages": int(len(frame)), "scam": int(frame["label"].sum())}
            for (channel, split), frame in events.groupby(["channel", "split"])
        },
        "graph": graph_summary(prepared.tables),
        "feature_counts": {node_type: int(frame.shape[1]) for node_type, frame in prepared.features.items()},
        "models": models,
        "ablation": ablation,
        "demo_model": {"name": GNN_NAME, **saved_info, "threshold": demo_metrics["threshold"], "test_metrics": demo_metrics},
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "pandas": pd.__version__},
    }
    (results_dir / RESULTS_FILE).write_text(json.dumps(results, indent=2), encoding="utf-8")
    log(f"Done in {results['runtime_seconds']:.0f} s. Results are in {results_dir}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate all models, and save the GNN for the demo.")
    parser.add_argument("--config", default=None, help="Path to a config JSON file (default: config.json).")
    args = parser.parse_args()
    run(load_config(args.config) if args.config else load_config())


if __name__ == "__main__":
    main()
