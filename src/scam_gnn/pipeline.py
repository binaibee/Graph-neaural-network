from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from scam_gnn.analysis import derive_entity_risk_scores, summarize_error_patterns
from scam_gnn.config import ExperimentConfig, load_config
from scam_gnn.data.splits import assign_time_based_splits
from scam_gnn.data.standardize import (
    assemble_multichannel_events,
    standardize_email_dataframe,
    standardize_sms_dataframe,
    standardize_url_dataframe,
)
from scam_gnn.data.public_datasets import prepare_phiusiil_url_dataset, prepare_uci_sms_spam_dataset
from scam_gnn.data.synthetic import generate_synthetic_multichannel_events
from scam_gnn.dependencies import MissingOptionalDependency, optional_import
from scam_gnn.features import build_feature_bundle
from scam_gnn.graph import build_graph_tables, build_pyg_heterodata, build_pyg_homogeneous_data
from scam_gnn.metrics import binary_metrics, per_channel_metrics, pr_auc
from scam_gnn.models.baselines import NumpyLogisticRegression, NumpyMLPClassifier
from scam_gnn.models.gnn import build_heterogeneous_graphsage, build_heterogeneous_hgt, build_homogeneous_model

GNN_EPOCHS = 40
GNN_EARLY_STOPPING_PATIENCE = 8
HETERO_HGT_EPOCHS = 10
HETERO_HGT_EARLY_STOPPING_PATIENCE = 3


def _dependency_status() -> dict[str, bool]:
    return {
        "numpy": True,
        "pandas": True,
        "sklearn": optional_import("sklearn") is not None,
        "sentence_transformers": optional_import("sentence_transformers") is not None,
        "torch": optional_import("torch") is not None,
        "torch_geometric": optional_import("torch_geometric") is not None,
    }


def _limit_rows(frame: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None or limit <= 0 or len(frame) <= limit:
        return frame
    return frame.head(limit).reset_index(drop=True)


def _interleave_synthetic_timestamps(frames: list[pd.DataFrame]) -> list[pd.DataFrame]:
    if len(frames) <= 1:
        return frames
    has_native_flags = []
    for frame in frames:
        if "has_native_timestamp" not in frame.columns:
            return frames
        has_native_flags.extend(frame["has_native_timestamp"].fillna(False).astype(bool).tolist())
    if any(has_native_flags):
        return frames

    working_frames = [frame.reset_index(drop=True).copy() for frame in frames]
    ranked_rows: list[tuple[float, int, dict[str, object]]] = []
    for frame_index, frame in enumerate(working_frames):
        size = len(frame)
        for row_index, (_, row) in enumerate(frame.iterrows()):
            normalized_rank = (row_index + 0.5) / max(size, 1)
            ranked_rows.append((normalized_rank, frame_index, row.to_dict()))
    ranked_rows.sort(key=lambda item: (item[0], item[1]))
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    reassigned_rows: list[dict[str, object]] = []
    for index, (_, _, payload) in enumerate(ranked_rows):
        payload = dict(payload)
        payload["timestamp"] = base_time + timedelta(seconds=index)
        reassigned_rows.append(payload)
    combined = pd.DataFrame.from_records(reassigned_rows)
    return [combined]


def _load_custom_events(config: ExperimentConfig, seed: int) -> pd.DataFrame:
    if config.dataset.kind == "synthetic":
        return generate_synthetic_multichannel_events(seed=seed)
    if config.dataset.kind == "uci_sms":
        sms_path = config.dataset.sms_path or str(config.output_dir / "uci_sms_spam_collection.csv")
        prepared_path = prepare_uci_sms_spam_dataset(sms_path)
        return _limit_rows(standardize_sms_dataframe(pd.read_csv(prepared_path)), config.dataset.max_sms_rows)
    if config.dataset.kind == "phiusiil_url":
        url_path = config.dataset.url_path or str(config.output_dir / "phiusiil_url_collection.csv")
        prepared_path = prepare_phiusiil_url_dataset(url_path)
        return _limit_rows(standardize_url_dataframe(pd.read_csv(prepared_path)), config.dataset.max_url_rows)
    frames: list[pd.DataFrame] = []
    if config.dataset.sms_path:
        frames.append(_limit_rows(standardize_sms_dataframe(pd.read_csv(config.dataset.sms_path)), config.dataset.max_sms_rows))
    if config.dataset.email_path:
        frames.append(_limit_rows(standardize_email_dataframe(pd.read_csv(config.dataset.email_path)), config.dataset.max_email_rows))
    if config.dataset.url_path:
        frames.append(_limit_rows(standardize_url_dataframe(pd.read_csv(config.dataset.url_path)), config.dataset.max_url_rows))
    if not frames:
        if config.track == "benchmark":
            raise ValueError(
                "Benchmark track requires local CSV exports or standardized benchmark tables. "
                "Synthetic fallback is only enabled for the custom track."
            )
        return generate_synthetic_multichannel_events(seed=seed)
    frames = _interleave_synthetic_timestamps(frames)
    return assemble_multichannel_events(frames)


def _prepare_matrices(feature_bundle, tables):
    messages = tables.events.drop_duplicates("message_id").copy()
    ordered_ids = list(messages["message_id"])
    x = feature_bundle.message_features.reindex(ordered_ids).fillna(0.0).to_numpy(dtype=np.float32)
    y = messages["label"].to_numpy(dtype=np.float32)
    return messages, x, y


def _evaluate_scores(messages: pd.DataFrame, scores: np.ndarray, config: ExperimentConfig) -> dict[str, object]:
    test_frame = messages[messages["split"] == "test"].copy()
    test_scores = scores[test_frame.index]
    y_test = test_frame["label"].to_numpy(dtype=int)
    metrics = binary_metrics(y_test, test_scores, target_precision=config.target_precision)
    metrics["per_channel"] = per_channel_metrics(test_frame, test_scores, config.target_precision)
    return metrics


def _fit_numpy_baselines(
    messages: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    config: ExperimentConfig | None = None,
):
    train_mask = messages["split"] == "train"
    x_train = x[train_mask.to_numpy()]
    y_train = y[train_mask.to_numpy()]
    logistic = NumpyLogisticRegression().fit(x_train, y_train)
    mlp = NumpyMLPClassifier().fit(x_train, y_train, seed=seed)
    outputs = {
        "logistic_regression": logistic.predict_proba(x),
        "mlp": mlp.predict_proba(x),
    }
    sklearn_ensemble = optional_import("sklearn.ensemble")
    if sklearn_ensemble is not None:
        # Use the serial gradient boosting implementation to avoid Windows sandbox
        # failures triggered by histogram boosting's multiprocessing internals.
        tree_model = sklearn_ensemble.GradientBoostingClassifier(random_state=seed)
        tree_model.fit(x_train, y_train)
        outputs["tree_ensemble"] = tree_model.predict_proba(x)[:, 1]
    return outputs


def _set_torch_seed(torch, seed: int) -> None:
    torch.manual_seed(seed)
    if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _weighted_cross_entropy(functional, logits, targets, mask, sample_weight):
    losses = functional.cross_entropy(logits[mask], targets[mask], reduction="none")
    if sample_weight is None:
        return losses.mean()
    weights = sample_weight[mask]
    return (losses * weights).sum() / weights.sum().clamp_min(1e-6)


def _masked_pr_auc(targets, scores, mask) -> float:
    y_true = targets[mask].detach().cpu().numpy().astype(int)
    y_score = scores[mask].detach().cpu().numpy().astype(float)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("-inf")
    return pr_auc(y_true, y_score)


def _train_homogeneous_gnn(data, model_name: str, seed: int) -> np.ndarray:
    torch = optional_import("torch")
    functional = optional_import("torch.nn.functional")
    if torch is None or functional is None:
        raise MissingOptionalDependency("Homogeneous GNN training requires torch.")
    _set_torch_seed(torch, seed)
    model = build_homogeneous_model(model_name, input_dim=data.x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best_state = None
    best_val_pr_auc = float("-inf")
    epochs_without_improvement = 0
    for _ in range(GNN_EPOCHS):
        model.train()
        optimizer.zero_grad()
        logits = model(data)
        loss = _weighted_cross_entropy(
            functional,
            logits,
            data.y,
            data.train_mask,
            getattr(data, "sample_weight", None),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(data)
            val_scores = torch.softmax(val_logits, dim=-1)[:, 1]
            val_pr_auc = _masked_pr_auc(data.y, val_scores, data.val_mask)
            if val_pr_auc >= best_val_pr_auc:
                best_val_pr_auc = val_pr_auc
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= GNN_EARLY_STOPPING_PATIENCE:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(data)
        return torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()


def _train_heterogeneous_gnn(data, seed: int) -> np.ndarray:
    return _train_heterogeneous_model(data, seed, build_heterogeneous_graphsage)


def _train_heterogeneous_hgt(data, seed: int) -> np.ndarray:
    return _train_heterogeneous_model(
        data,
        seed,
        build_heterogeneous_hgt,
        epochs=HETERO_HGT_EPOCHS,
        patience=HETERO_HGT_EARLY_STOPPING_PATIENCE,
    )


def _train_heterogeneous_model(
    data,
    seed: int,
    model_builder,
    *,
    epochs: int = GNN_EPOCHS,
    patience: int = GNN_EARLY_STOPPING_PATIENCE,
) -> np.ndarray:
    torch = optional_import("torch")
    functional = optional_import("torch.nn.functional")
    if torch is None or functional is None:
        raise MissingOptionalDependency("Heterogeneous GNN training requires torch.")
    _set_torch_seed(torch, seed)
    model = model_builder(data.metadata())
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best_state = None
    best_val_pr_auc = float("-inf")
    epochs_without_improvement = 0
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x_dict, data.edge_index_dict)
        loss = _weighted_cross_entropy(
            functional,
            logits,
            data["message"].y,
            data["message"].train_mask,
            getattr(data["message"], "sample_weight", None),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(data.x_dict, data.edge_index_dict)
            val_scores = torch.softmax(val_logits, dim=-1)[:, 1]
            val_pr_auc = _masked_pr_auc(data["message"].y, val_scores, data["message"].val_mask)
            if val_pr_auc >= best_val_pr_auc:
                best_val_pr_auc = val_pr_auc
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(data.x_dict, data.edge_index_dict)
        return torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()


def _score_model_by_name(
    model_name: str,
    messages: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    feature_bundle,
    tables,
    seed: int,
    dependency_status: dict[str, bool] | None = None,
) -> np.ndarray:
    if model_name in {"logistic_regression", "mlp", "tree_ensemble"}:
        baseline_scores = _fit_numpy_baselines(messages, x, y, seed, config=None)
        if model_name not in baseline_scores:
            raise MissingOptionalDependency(f"Model '{model_name}' is unavailable because optional dependencies are missing.")
        return baseline_scores[model_name]
    if model_name in {"gcn", "graphsage"}:
        if dependency_status is None:
            dependency_status = _dependency_status()
        if not (dependency_status["torch"] and dependency_status["torch_geometric"]):
            raise MissingOptionalDependency(f"Model '{model_name}' requires torch and torch-geometric.")
        homogeneous_data = build_pyg_homogeneous_data(feature_bundle.message_features, tables)
        return _train_homogeneous_gnn(homogeneous_data, model_name, seed)
    if model_name in {"hetero_graphsage", "hetero_hgt"}:
        if dependency_status is None:
            dependency_status = _dependency_status()
        if not (dependency_status["torch"] and dependency_status["torch_geometric"]):
            raise MissingOptionalDependency(f"Model '{model_name}' requires torch and torch-geometric.")
        hetero_data = build_pyg_heterodata(tables, feature_bundle.entity_features)
        if model_name == "hetero_hgt":
            return _train_heterogeneous_hgt(hetero_data, seed)
        return _train_heterogeneous_gnn(hetero_data, seed)
    raise ValueError(f"Unsupported model name '{model_name}'.")


def _aggregate_results(result_payload: dict[str, list[dict[str, object]]]) -> dict[str, dict[str, object]]:
    aggregated: dict[str, dict[str, object]] = {}
    for model_name, runs in result_payload.items():
        metrics_by_name: dict[str, list[float]] = {}
        per_channel: dict[str, dict[str, list[float]]] = {}
        for run in runs:
            for key, value in run["metrics"].items():
                if key == "per_channel":
                    for channel, channel_metrics in value.items():
                        per_channel.setdefault(channel, {})
                        for metric_name, metric_value in channel_metrics.items():
                            per_channel[channel].setdefault(metric_name, []).append(float(metric_value))
                else:
                    metrics_by_name.setdefault(key, []).append(float(value))
        aggregated[model_name] = {
            "runs": runs,
            "metrics": {
                key: {
                    "mean": float(np.nanmean(values)),
                    "std": float(np.nanstd(values)),
                }
                for key, values in metrics_by_name.items()
            },
            "per_channel": {
                channel: {
                    metric_name: {
                        "mean": float(np.nanmean(values)),
                        "std": float(np.nanstd(values)),
                    }
                    for metric_name, values in metrics.items()
                }
                for channel, metrics in per_channel.items()
            },
        }
    return aggregated


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    dependency_status = _dependency_status()
    all_results: dict[str, list[dict[str, object]]] = {}
    best_scores = None
    best_events = None
    best_tables = None
    best_metric = float("-inf")

    for seed in config.seeds:
        events = _load_custom_events(config, seed)
        split_events = assign_time_based_splits(events, config.splits)
        tables = build_graph_tables(
            split_events,
            include_cross_channel_edges=config.ablation.include_cross_channel_edges,
            include_infrastructure_nodes=config.ablation.include_infrastructure_nodes,
        )
        feature_bundle = build_feature_bundle(tables, config.features)
        messages, x, y = _prepare_matrices(feature_bundle, tables)
        baseline_scores = _fit_numpy_baselines(messages, x, y, seed, config)
        for model_name, scores in baseline_scores.items():
            metrics = _evaluate_scores(messages, scores, config)
            all_results.setdefault(model_name, []).append({"seed": seed, "metrics": metrics})
            score_value = float(metrics.get(config.primary_metric, float("-inf")))
            if score_value > best_metric:
                best_metric = score_value
                best_scores = scores
                best_events = messages
                best_tables = tables

        if dependency_status["torch"] and dependency_status["torch_geometric"]:
            homogeneous_data = build_pyg_homogeneous_data(feature_bundle.message_features, tables)
            hetero_data = build_pyg_heterodata(tables, feature_bundle.entity_features)
            for model_name in ["gcn", "graphsage"]:
                scores = _train_homogeneous_gnn(homogeneous_data, model_name, seed)
                metrics = _evaluate_scores(messages, scores, config)
                all_results.setdefault(model_name, []).append({"seed": seed, "metrics": metrics})
            hetero_scores = _train_heterogeneous_gnn(hetero_data, seed)
            metrics = _evaluate_scores(messages, hetero_scores, config)
            all_results.setdefault("hetero_graphsage", []).append({"seed": seed, "metrics": metrics})
            score_value = float(metrics.get(config.primary_metric, float("-inf")))
            if score_value > best_metric:
                best_metric = score_value
                best_scores = hetero_scores
                best_events = messages
                best_tables = tables
            hetero_hgt_scores = _train_heterogeneous_hgt(hetero_data, seed)
            metrics = _evaluate_scores(messages, hetero_hgt_scores, config)
            all_results.setdefault("hetero_hgt", []).append({"seed": seed, "metrics": metrics})
            score_value = float(metrics.get(config.primary_metric, float("-inf")))
            if score_value > best_metric:
                best_metric = score_value
                best_scores = hetero_hgt_scores
                best_events = messages
                best_tables = tables

    summary = {
        "name": config.name,
        "track": config.track,
        "dependency_status": dependency_status,
        "results": _aggregate_results(all_results),
        "notes": [
            "Tree ensembles run only when scikit-learn is installed.",
            "GNNs run only when torch and torch-geometric are installed.",
            "GNN training uses channel-label-balanced message weights with validation checkpointing.",
            "The heterogeneous model is a message-centric, relation-pruned GraphSAGE encoder.",
            "HGTConv is available as an additional heterogeneous backend.",
            "MiniLM embeddings fall back to deterministic hashed text embeddings when sentence-transformers is unavailable.",
        ],
    }

    if best_scores is not None and best_events is not None and best_tables is not None:
        summary["error_analysis"] = summarize_error_patterns(best_events, best_scores)
        summary["entity_risk_scores"] = {
            key: value.head(10).reset_index().to_dict(orient="records")
            for key, value in derive_entity_risk_scores(best_tables, best_scores).items()
        }
        summary["dataset"] = {
            "num_events": int(len(best_tables.events.drop_duplicates("message_id"))),
            "num_entities": int(len(best_tables.entities)),
            "num_relations": int(len(best_tables.relations)),
        }
    return summary


def run_experiment_from_config(config_path: str | Path) -> dict[str, object]:
    config = load_config(config_path)
    summary = run_experiment(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = config.output_dir / f"{config.name}_summary.json"
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    return summary


def build_scored_snapshot(
    config: ExperimentConfig,
    model_name: str,
    *,
    seed: int | None = None,
) -> dict[str, object]:
    selected_seed = config.seeds[0] if seed is None else seed
    dependency_status = _dependency_status()
    events = _load_custom_events(config, selected_seed)
    split_events = assign_time_based_splits(events, config.splits)
    tables = build_graph_tables(
        split_events,
        include_cross_channel_edges=config.ablation.include_cross_channel_edges,
        include_infrastructure_nodes=config.ablation.include_infrastructure_nodes,
    )
    feature_bundle = build_feature_bundle(tables, config.features)
    messages, x, y = _prepare_matrices(feature_bundle, tables)
    scores = _score_model_by_name(model_name, messages, x, y, feature_bundle, tables, selected_seed, dependency_status)
    return {
        "seed": selected_seed,
        "model_name": model_name,
        "messages": messages,
        "tables": tables,
        "scores": scores,
        "metrics": _evaluate_scores(messages, scores, config),
        "error_analysis": summarize_error_patterns(messages, scores),
        "entity_risk_scores": derive_entity_risk_scores(tables, scores),
    }
