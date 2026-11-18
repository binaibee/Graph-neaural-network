from __future__ import annotations

import json
import sqlite3
import threading
import time as _time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from scam_gnn.config import (
    AblationConfig,
    DatasetConfig,
    ExperimentConfig,
    FeatureConfig,
    SplitConfig,
    load_config,
)
from scam_gnn.data.public_datasets import prepare_phiusiil_url_dataset, prepare_uci_sms_spam_dataset
from scam_gnn.data.synthetic import generate_synthetic_multichannel_events
from scam_gnn.pipeline import build_scored_snapshot, run_experiment_from_config


# ---------------------------------------------------------------------------
# Command runner – background thread execution for CLI commands
# ---------------------------------------------------------------------------

_command_lock = threading.Lock()
_command_status: dict[str, dict[str, object]] = {}


def _set_command_status(name: str, status: str, **kwargs: object) -> None:
    with _command_lock:
        _command_status[name] = {"status": status, "updated_at": datetime.now(timezone.utc).isoformat(), **kwargs}


def _get_command_status(name: str) -> dict[str, object]:
    with _command_lock:
        return dict(_command_status.get(name, {"status": "idle"}))


def _run_command_thread(name: str, target, args=(), kwargs=None) -> None:
    kwargs = kwargs or {}

    def wrapper():
        _set_command_status(name, "running")
        try:
            result = target(*args, **kwargs)
            _set_command_status(name, "completed", result=str(result) if result is not None else None)
        except Exception as exc:
            _set_command_status(name, "error", error=str(exc))

    thread = threading.Thread(target=wrapper, daemon=True)
    thread.start()


def _cmd_generate_synthetic(output: str, seed: int, num_events: int) -> str:
    frame = generate_synthetic_multichannel_events(seed=seed, num_events=num_events)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return str(output_path)


def _cmd_prepare_uci_sms(output: str, force: bool) -> str:
    return str(prepare_uci_sms_spam_dataset(output, force=force))


def _cmd_prepare_phiusiil_url(output: str, force: bool) -> str:
    return str(prepare_phiusiil_url_dataset(output, force=force))


def _cmd_run_research(config_path: str) -> str:
    summary = run_experiment_from_config(config_path)
    return json.dumps({"name": summary.get("name"), "models": list(summary.get("results", {}).keys())})


def _cmd_build_ui_db(config_path: str, db_path: str, model_name: str, seed: int | None) -> str:
    return str(materialize_ui_database(config_path, db_path, model_name=model_name, seed=seed))


# ---------------------------------------------------------------------------
# Config management helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Feature & model info helpers
# ---------------------------------------------------------------------------

FEATURE_CATEGORIES = [
    {
        "id": "text_embeddings",
        "name": "Text Embeddings",
        "description": "Dense vector representations of message text. Uses MiniLM-L6-v2 sentence-transformers (384 dims) when available, otherwise falls back to deterministic SHA-256 hash-based embeddings (24 dims).",
        "features": [
            {"name": "text_emb_000 .. text_emb_N", "type": "float", "description": "Each dimension of the text embedding vector. Captures semantic meaning of the message content."},
        ],
    },
    {
        "id": "surface_stats",
        "name": "Surface Statistics",
        "description": "Simple numeric properties extracted directly from the raw message text without any ML model.",
        "features": [
            {"name": "length", "type": "int", "description": "Total character count of the message text."},
            {"name": "digit_ratio", "type": "float", "description": "Fraction of characters that are digits (0-9). Scam messages often contain phone numbers or codes."},
            {"name": "uppercase_ratio", "type": "float", "description": "Fraction of characters that are uppercase. Scam messages often use CAPS for urgency."},
            {"name": "url_count", "type": "int", "description": "Number of URLs found in the message text."},
            {"name": "urgent_keyword_count", "type": "int", "description": "Count of urgent/scam keywords: urgent, verify, warning, refund, blocked, suspend, reset, failed, confirm."},
        ],
    },
    {
        "id": "channel_encoding",
        "name": "Channel One-Hot Encoding",
        "description": "Binary flags indicating which communication channel the message came from.",
        "features": [
            {"name": "channel_sms", "type": "binary", "description": "1 if the message is an SMS, 0 otherwise."},
            {"name": "channel_email", "type": "binary", "description": "1 if the message is an email, 0 otherwise."},
            {"name": "channel_url", "type": "binary", "description": "1 if the event is a URL record, 0 otherwise."},
        ],
    },
    {
        "id": "graph_statistics",
        "name": "Graph Statistics",
        "description": "Structural features derived from the message's position in the relationship graph. Toggled by include_graph_statistics config.",
        "features": [
            {"name": "out_degree", "type": "int", "description": "Number of outgoing edges from this message node (sent_by, sent_to, contains_url, shares_*)."},
            {"name": "in_degree", "type": "int", "description": "Number of incoming edges to this message node."},
            {"name": "infrastructure_degree", "type": "int", "description": "Number of URL/infrastructure-related edges (contains_url + shares_infrastructure)."},
        ],
    },
    {
        "id": "entity_features",
        "name": "Entity Features",
        "description": "Features computed for non-message nodes (senders, recipients, URLs, domains). Used by heterogeneous GNN models.",
        "features": [
            {"name": "out_degree", "type": "int", "description": "Number of outgoing relations from this entity."},
            {"name": "in_degree", "type": "int", "description": "Number of incoming relations to this entity."},
            {"name": "value_length", "type": "int", "description": "Character length of the entity value (e.g., length of a domain name)."},
            {"name": "dot_count", "type": "int", "description": "(Domain only) Number of dots in the domain name. More subdomains can indicate phishing."},
            {"name": "slash_count", "type": "int", "description": "(URL only) Number of slashes in the URL path."},
            {"name": "domain_length", "type": "int", "description": "(URL only) Character length of the extracted domain."},
        ],
    },
    {
        "id": "extra_numeric",
        "name": "Extra Numeric Features",
        "description": "Any additional numeric columns from the original dataset that are not part of the standard schema. For example, PhiUSIIL URL features like URLLength, NumParams, PathEntropy, etc.",
        "features": [
            {"name": "extra_*", "type": "float", "description": "Passthrough numeric columns from the source CSV, prefixed with extra_."},
        ],
    },
]

MODEL_DEFINITIONS = [
    {
        "id": "logistic_regression",
        "name": "Logistic Regression",
        "tier": 1,
        "tier_label": "Baseline",
        "requires": "numpy",
        "description": "The simplest classifier. Learns one weight per feature plus a bias, applies sigmoid to output a scam probability.",
        "architecture": ["Standardize features (subtract mean, divide by std)", "Linear: weighted sum of all features + bias", "Sigmoid activation \u2192 probability"],
        "params": {"learning_rate": 0.1, "max_iter": 400, "l2_regularization": 1e-4},
        "implementation": "Pure numpy (no ML library needed)",
    },
    {
        "id": "mlp",
        "name": "Multi-Layer Perceptron",
        "tier": 1,
        "tier_label": "Baseline",
        "requires": "numpy",
        "description": "A 2-layer feed-forward neural network. Adds a hidden layer with non-linearity so it can learn feature interactions that logistic regression cannot.",
        "architecture": ["Standardize features", "Dense layer: input \u2192 32 hidden units", "ReLU activation", "Dense layer: 32 \u2192 1 output", "Sigmoid activation \u2192 probability"],
        "params": {"hidden_dim": 32, "learning_rate": 0.05, "max_iter": 300, "l2_regularization": 1e-4},
        "implementation": "Pure numpy with manual backpropagation",
    },
    {
        "id": "tree_ensemble",
        "name": "Tree Ensemble",
        "tier": 2,
        "tier_label": "Traditional ML",
        "requires": "scikit-learn",
        "description": "Gradient Boosted Decision Trees. Builds an ensemble of shallow trees, each correcting the errors of the previous one. Often strong on tabular data.",
        "architecture": ["GradientBoostingClassifier (serial, not histogram)", "Sequential tree boosting with residual learning", "Ensemble prediction via weighted voting"],
        "params": {"n_estimators": 100, "random_state": "per-seed"},
        "implementation": "scikit-learn GradientBoostingClassifier",
    },
    {
        "id": "gcn",
        "name": "GCN (Graph Convolutional Network)",
        "tier": 3,
        "tier_label": "Homogeneous GNN",
        "requires": "torch + torch-geometric",
        "description": "Treats the message graph as homogeneous (all nodes same type). Each layer aggregates neighbor features via spectral convolution. 2 layers = 2-hop neighborhood.",
        "architecture": ["Input projection: features \u2192 64-dim", "GCNConv layer 1: 64-dim \u2192 64-dim", "ReLU + Dropout(0.2)", "GCNConv layer 2: 64-dim \u2192 64-dim", "Residual connection from input projection", "Classifier MLP: 128 \u2192 64 \u2192 2 (scam/legit)"],
        "params": {"hidden_dim": 64, "epochs": 40, "lr": 0.002, "early_stopping_patience": 8},
        "implementation": "PyTorch Geometric GCNConv",
    },
    {
        "id": "graphsage",
        "name": "GraphSAGE",
        "tier": 3,
        "tier_label": "Homogeneous GNN",
        "requires": "torch + torch-geometric",
        "description": "Like GCN but uses sampling-based neighborhood aggregation (mean aggregator). More scalable and often better on large graphs.",
        "architecture": ["Input projection: features \u2192 64-dim", "SAGEConv layer 1: 64-dim \u2192 64-dim", "ReLU + Dropout(0.2)", "SAGEConv layer 2: 64-dim \u2192 64-dim", "Residual connection from input projection", "Classifier MLP: 128 \u2192 64 \u2192 2"],
        "params": {"hidden_dim": 64, "epochs": 40, "lr": 0.002, "early_stopping_patience": 8},
        "implementation": "PyTorch Geometric SAGEConv",
    },
    {
        "id": "hetero_graphsage",
        "name": "Heterogeneous GraphSAGE",
        "tier": 4,
        "tier_label": "Heterogeneous GNN",
        "requires": "torch + torch-geometric",
        "description": "Message-centric model that understands different node and edge types. Information flows in 4 stages: phone/email \u2192 sender/recipient, domain \u2192 URL, then all \u2192 message.",
        "architecture": [
            "Per-type encoder: LazyLinear \u2192 64-dim for each node type",
            "Stage 1: Sender context (phone/email \u2192 sender via HeteroConv+SAGEConv)",
            "Stage 2: Recipient context (phone/email \u2192 recipient)",
            "Stage 3: URL context (domain \u2192 url)",
            "Stage 4: Message aggregation (2 layers of SAGEConv over message edges)",
            "LayerNorm + residual at each message layer",
            "Classifier MLP: concat(base, enriched) \u2192 64 \u2192 2",
        ],
        "params": {"hidden_dim": 64, "layers": 2, "epochs": 40, "lr": 0.002, "early_stopping_patience": 8},
        "implementation": "PyTorch Geometric HeteroConv + SAGEConv",
    },
    {
        "id": "hetero_hgt",
        "name": "Heterogeneous Graph Transformer (HGT)",
        "tier": 5,
        "tier_label": "Transformer GNN",
        "requires": "torch + torch-geometric",
        "description": "The most advanced model. Uses multi-head attention over typed edges to learn which relationships matter most. Attention weights are conditioned on node and edge types.",
        "architecture": [
            "Per-type encoder: LazyLinear \u2192 32-dim for each node type",
            "HGTConv layer: 2-head attention over pruned edge types",
            "Per-type LayerNorm + residual connection",
            "Classifier MLP: concat(base, enriched) \u2192 32 \u2192 2",
        ],
        "params": {"hidden_dim": 32, "layers": 1, "heads": 2, "epochs": 10, "lr": 0.002, "early_stopping_patience": 3},
        "implementation": "PyTorch Geometric HGTConv",
    },
]


def _fetch_feature_stats(connection: sqlite3.Connection) -> dict[str, object]:
    """Compute live feature-related statistics from the database."""
    entity_types = [
        dict(row)
        for row in connection.execute(
            "SELECT entity_type, COUNT(*) as count FROM entities GROUP BY entity_type ORDER BY count DESC"
        ).fetchall()
    ]
    edge_types = [
        dict(row)
        for row in connection.execute(
            "SELECT edge_type, COUNT(*) as count FROM relations GROUP BY edge_type ORDER BY count DESC"
        ).fetchall()
    ]
    channels = [
        dict(row)
        for row in connection.execute("SELECT * FROM dataset_channels ORDER BY artifact_count DESC").fetchall()
    ]
    # Score distribution buckets
    score_buckets = [
        dict(row)
        for row in connection.execute(
            """
            SELECT
                CASE
                    WHEN predicted_score >= 0.9 THEN '0.9-1.0'
                    WHEN predicted_score >= 0.7 THEN '0.7-0.9'
                    WHEN predicted_score >= 0.5 THEN '0.5-0.7'
                    WHEN predicted_score >= 0.3 THEN '0.3-0.5'
                    WHEN predicted_score >= 0.1 THEN '0.1-0.3'
                    ELSE '0.0-0.1'
                END as bucket,
                COUNT(*) as count
            FROM alerts
            GROUP BY bucket
            ORDER BY bucket DESC
            """
        ).fetchall()
    ]
    # High-risk entity type breakdown
    risky_entity_types = [
        dict(row)
        for row in connection.execute(
            """
            SELECT entity_type, COUNT(*) as count, AVG(risk_mean) as avg_risk
            FROM entities
            WHERE risk_mean IS NOT NULL AND risk_mean > 0.5
            GROUP BY entity_type
            ORDER BY count DESC
            """
        ).fetchall()
    ]
    return {
        "entity_types": entity_types,
        "edge_types": edge_types,
        "channels": channels,
        "score_distribution": score_buckets,
        "high_risk_entity_types": risky_entity_types,
    }


def _fetch_model_availability() -> dict[str, bool]:
    from scam_gnn.dependencies import optional_import
    return {
        "numpy": True,
        "pandas": True,
        "scikit-learn": optional_import("sklearn") is not None,
        "sentence-transformers": optional_import("sentence_transformers") is not None,
        "torch": optional_import("torch") is not None,
        "torch-geometric": optional_import("torch_geometric") is not None,
    }


CONFIGS_DIR = Path("examples/configs")


def _list_configs() -> list[dict[str, str]]:
    configs = []
    search_dir = CONFIGS_DIR if CONFIGS_DIR.exists() else Path(".")
    for path in sorted(search_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            configs.append({"file": str(path), "name": data.get("name", path.stem)})
        except Exception:
            continue
    return configs


def _load_config_as_dict(path: str) -> dict[str, object]:
    config = load_config(path)
    return asdict(config)


def _save_config_dict(path: str, data: dict[str, object]) -> str:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(output_path)

DEFAULT_CONFIG_PATH = Path("examples/configs/uci_sms_phiusiil_multichannel.json")
DEFAULT_DB_PATH = Path("artifacts/ui/scam_intelligence.db")
STATIC_DIR = Path(__file__).resolve().parent / "ui_static"
MESSAGE_LINK_EDGE_TYPES = {"shares_sender", "shares_recipient", "shares_infrastructure"}


def _summary_path_for_config(config_path: str | Path) -> Path:
    config = load_config(config_path)
    return config.output_dir / f"{config.name}_summary.json"


def _load_summary(config_path: str | Path) -> tuple[dict[str, object], object]:
    config = load_config(config_path)
    summary_path = _summary_path_for_config(config_path)
    if not summary_path.exists():
        run_experiment_from_config(config_path)
    return json.loads(summary_path.read_text(encoding="utf-8")), config


def _choose_model(summary: dict[str, object], primary_metric: str, requested_model: str) -> str:
    if requested_model != "auto":
        return requested_model
    best_model = None
    best_value = float("-inf")
    for model_name, payload in summary["results"].items():
        value = payload.get("metrics", {}).get(primary_metric, {}).get("mean", float("-inf"))
        if value > best_value:
            best_value = value
            best_model = model_name
    if best_model is None:
        raise ValueError("Could not choose a model from the experiment summary.")
    return best_model


def _connect(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    return connection


def _iso_timestamp(value) -> str:
    return pd.Timestamp(value).isoformat()


def _json_dump(value) -> str:
    return json.dumps(value, ensure_ascii=True)


def _init_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS alerts (
            alert_id TEXT PRIMARY KEY,
            message_id TEXT NOT NULL,
            event_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            channel TEXT NOT NULL,
            text_or_url TEXT NOT NULL,
            sender_key TEXT,
            recipient_key TEXT,
            label INTEGER NOT NULL,
            predicted_score REAL NOT NULL,
            predicted_label INTEGER NOT NULL,
            split TEXT NOT NULL,
            status TEXT NOT NULL,
            model_name TEXT NOT NULL,
            campaign_id TEXT
        );
        CREATE TABLE IF NOT EXISTS campaigns (
            campaign_id TEXT PRIMARY KEY,
            size INTEGER NOT NULL,
            risk_level REAL NOT NULL,
            mean_score REAL NOT NULL,
            channels_json TEXT NOT NULL,
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            representative_alert_id TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS entities (
            entity_id TEXT PRIMARY KEY,
            entity_type TEXT NOT NULL,
            entity_value TEXT NOT NULL,
            risk_mean REAL,
            risk_max REAL,
            linked_messages INTEGER NOT NULL,
            first_seen TEXT,
            last_seen TEXT
        );
        CREATE TABLE IF NOT EXISTS relations (
            relation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            src_id TEXT NOT NULL,
            src_type TEXT NOT NULL,
            dst_id TEXT NOT NULL,
            dst_type TEXT NOT NULL,
            edge_type TEXT NOT NULL,
            timestamp TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS experiment_metrics (
            model_name TEXT NOT NULL,
            scope_type TEXT NOT NULL,
            scope_value TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            mean REAL NOT NULL,
            std REAL NOT NULL,
            PRIMARY KEY (model_name, scope_type, scope_value, metric_name)
        );
        CREATE TABLE IF NOT EXISTS experiment_errors (
            category TEXT PRIMARY KEY,
            count INTEGER NOT NULL,
            sample_event_ids_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS dataset_channels (
            channel TEXT PRIMARY KEY,
            artifact_count INTEGER NOT NULL,
            positive_rate REAL NOT NULL,
            mean_score REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_alerts_channel_score ON alerts(channel, predicted_score DESC);
        CREATE INDEX IF NOT EXISTS idx_alerts_campaign ON alerts(campaign_id);
        CREATE INDEX IF NOT EXISTS idx_relations_src ON relations(src_id, edge_type);
        CREATE INDEX IF NOT EXISTS idx_relations_dst ON relations(dst_id, edge_type);
        """
    )


def _clear_tables(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        DELETE FROM metadata;
        DELETE FROM alerts;
        DELETE FROM campaigns;
        DELETE FROM entities;
        DELETE FROM relations;
        DELETE FROM experiment_metrics;
        DELETE FROM experiment_errors;
        DELETE FROM dataset_channels;
        """
    )


def _component_map(messages: pd.DataFrame, relations: pd.DataFrame, scores) -> tuple[dict[str, str | None], list[dict[str, object]]]:
    alerts = messages.copy()
    alerts["predicted_score"] = scores
    alerts["predicted_label"] = (alerts["predicted_score"] >= 0.5).astype(int)
    suspicious = set(alerts.loc[alerts["predicted_label"] == 1, "message_id"])
    parent = {message_id: message_id for message_id in suspicious}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    link_rows = relations[
        relations["edge_type"].isin(MESSAGE_LINK_EDGE_TYPES)
        & (relations["src_type"] == "message")
        & (relations["dst_type"] == "message")
    ]
    for row in link_rows.itertuples(index=False):
        if row.src_id in suspicious and row.dst_id in suspicious:
            union(row.src_id, row.dst_id)

    components: dict[str, list[str]] = defaultdict(list)
    for message_id in suspicious:
        components[find(message_id)].append(message_id)

    membership: dict[str, str | None] = {message_id: None for message_id in alerts["message_id"]}
    campaigns: list[dict[str, object]] = []
    for index, component in enumerate(
        sorted((members for members in components.values() if len(members) > 1), key=len, reverse=True),
        start=1,
    ):
        component_frame = alerts[alerts["message_id"].isin(component)].copy()
        campaign_id = f"campaign-{index:04d}"
        for message_id in component:
            membership[message_id] = campaign_id
        representative = component_frame.sort_values("predicted_score", ascending=False).iloc[0]
        campaigns.append(
            {
                "campaign_id": campaign_id,
                "size": int(len(component_frame)),
                "risk_level": float(component_frame["predicted_score"].max()),
                "mean_score": float(component_frame["predicted_score"].mean()),
                "channels_json": _json_dump(sorted(component_frame["channel"].unique().tolist())),
                "first_seen": _iso_timestamp(component_frame["timestamp"].min()),
                "last_seen": _iso_timestamp(component_frame["timestamp"].max()),
                "representative_alert_id": str(representative["event_id"]),
            }
        )
    return membership, campaigns


def _entity_statistics(tables, alerts: pd.DataFrame) -> pd.DataFrame:
    direct_links = tables.relations[
        (tables.relations["src_type"] == "message") & (tables.relations["dst_type"] != "message")
    ][["src_id", "dst_id"]].rename(columns={"src_id": "message_id", "dst_id": "entity_id"})
    message_scores = alerts[["message_id", "predicted_score"]].copy()
    direct_scores = direct_links.merge(message_scores, on="message_id", how="left")

    secondary_links = tables.relations[
        (tables.relations["src_type"] != "message") & (tables.relations["dst_type"] != "message")
    ][["src_id", "dst_id"]]
    secondary_scores = (
        secondary_links.merge(direct_scores, left_on="src_id", right_on="entity_id", how="inner")
        [["message_id", "dst_id", "predicted_score"]]
        .rename(columns={"dst_id": "entity_id"})
    )

    all_scores = pd.concat(
        [
            direct_scores[["message_id", "entity_id", "predicted_score"]],
            secondary_scores[["message_id", "entity_id", "predicted_score"]],
        ],
        ignore_index=True,
    ).drop_duplicates()

    score_stats = (
        all_scores.groupby("entity_id")
        .agg(
            risk_mean=("predicted_score", "mean"),
            risk_max=("predicted_score", "max"),
            linked_messages=("message_id", "nunique"),
        )
        .reset_index()
    )

    entity_timestamps = pd.concat(
        [
            tables.relations[tables.relations["src_type"] != "message"][["src_id", "timestamp"]].rename(columns={"src_id": "entity_id"}),
            tables.relations[tables.relations["dst_type"] != "message"][["dst_id", "timestamp"]].rename(columns={"dst_id": "entity_id"}),
        ],
        ignore_index=True,
    )
    time_stats = entity_timestamps.groupby("entity_id")["timestamp"].agg(first_seen="min", last_seen="max").reset_index()
    entity_frame = tables.entities.merge(score_stats, on="entity_id", how="left").merge(time_stats, on="entity_id", how="left")
    entity_frame["linked_messages"] = entity_frame["linked_messages"].fillna(0).astype(int)
    return entity_frame


def materialize_ui_database(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    model_name: str = "auto",
    seed: int | None = None,
) -> Path:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    summary, config = _load_summary(config_path)
    selected_model = _choose_model(summary, config.primary_metric, model_name)
    snapshot = build_scored_snapshot(config, selected_model, seed=seed)

    messages = snapshot["messages"].copy()
    scores = snapshot["scores"]
    tables = snapshot["tables"]
    messages["predicted_score"] = scores
    messages["predicted_label"] = (messages["predicted_score"] >= 0.5).astype(int)

    campaign_membership, campaigns = _component_map(messages, tables.relations, scores)
    messages["campaign_id"] = messages["message_id"].map(campaign_membership)
    entities = _entity_statistics(tables, messages)

    connection = _connect(db_path)
    try:
        _init_schema(connection)
        _clear_tables(connection)

        metadata_rows = {
            "config_name": config.name,
            "track": config.track,
            "selected_model": selected_model,
            "selected_seed": str(snapshot["seed"]),
            "primary_metric": config.primary_metric,
            "summary_path": str(_summary_path_for_config(config_path)),
            "num_events": str(summary.get("dataset", {}).get("num_events", len(messages))),
            "num_entities": str(summary.get("dataset", {}).get("num_entities", len(entities))),
            "num_relations": str(summary.get("dataset", {}).get("num_relations", len(tables.relations))),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        connection.executemany("INSERT INTO metadata(key, value) VALUES(?, ?)", metadata_rows.items())

        alert_rows = [
            (
                str(row.event_id),
                str(row.message_id),
                str(row.event_id),
                _iso_timestamp(row.timestamp),
                str(row.channel),
                str(row.text_or_url),
                str(row.sender_key or ""),
                str(row.recipient_key or ""),
                int(row.label),
                float(row.predicted_score),
                int(row.predicted_label),
                str(row.split),
                "new",
                selected_model,
                row.campaign_id,
            )
            for row in messages.itertuples(index=False)
        ]
        connection.executemany(
            """
            INSERT INTO alerts(
                alert_id, message_id, event_id, timestamp, channel, text_or_url, sender_key, recipient_key,
                label, predicted_score, predicted_label, split, status, model_name, campaign_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            alert_rows,
        )

        connection.executemany(
            """
            INSERT INTO campaigns(
                campaign_id, size, risk_level, mean_score, channels_json, first_seen, last_seen, representative_alert_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    campaign["campaign_id"],
                    campaign["size"],
                    campaign["risk_level"],
                    campaign["mean_score"],
                    campaign["channels_json"],
                    campaign["first_seen"],
                    campaign["last_seen"],
                    campaign["representative_alert_id"],
                )
                for campaign in campaigns
            ],
        )

        entity_rows = [
            (
                str(row.entity_id),
                str(row.entity_type),
                str(row.entity_value),
                None if pd.isna(row.risk_mean) else float(row.risk_mean),
                None if pd.isna(row.risk_max) else float(row.risk_max),
                int(row.linked_messages),
                None if pd.isna(row.first_seen) else _iso_timestamp(row.first_seen),
                None if pd.isna(row.last_seen) else _iso_timestamp(row.last_seen),
            )
            for row in entities.itertuples(index=False)
        ]
        connection.executemany(
            """
            INSERT INTO entities(
                entity_id, entity_type, entity_value, risk_mean, risk_max, linked_messages, first_seen, last_seen
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            entity_rows,
        )

        relation_rows = [
            (
                str(row.src_id),
                str(row.src_type),
                str(row.dst_id),
                str(row.dst_type),
                str(row.edge_type),
                _iso_timestamp(row.timestamp),
            )
            for row in tables.relations.itertuples(index=False)
        ]
        connection.executemany(
            """
            INSERT INTO relations(src_id, src_type, dst_id, dst_type, edge_type, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            relation_rows,
        )

        metric_rows = _summary_metric_rows(summary)
        if selected_model not in summary["results"]:
            metric_rows.extend(_snapshot_metric_rows(selected_model, snapshot["metrics"]))
        connection.executemany(
            """
            INSERT INTO experiment_metrics(model_name, scope_type, scope_value, metric_name, mean, std)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            metric_rows,
        )

        error_rows = [
            (category, int(payload["count"]), _json_dump(payload["sample_event_ids"]))
            for category, payload in snapshot["error_analysis"].items()
        ]
        connection.executemany(
            "INSERT INTO experiment_errors(category, count, sample_event_ids_json) VALUES (?, ?, ?)",
            error_rows,
        )

        dataset_rows = (
            messages.groupby("channel")
            .agg(artifact_count=("message_id", "count"), positive_rate=("label", "mean"), mean_score=("predicted_score", "mean"))
            .reset_index()
        )
        connection.executemany(
            "INSERT INTO dataset_channels(channel, artifact_count, positive_rate, mean_score) VALUES (?, ?, ?, ?)",
            [(str(row.channel), int(row.artifact_count), float(row.positive_rate), float(row.mean_score)) for row in dataset_rows.itertuples(index=False)],
        )
        connection.commit()
    finally:
        connection.close()
    return db_path


def _fetch_metadata(connection: sqlite3.Connection) -> dict[str, str]:
    return {row["key"]: row["value"] for row in connection.execute("SELECT key, value FROM metadata")}


def _summary_metric_rows(summary: dict[str, object]) -> list[tuple[str, str, str, str, float, float]]:
    metric_rows: list[tuple[str, str, str, str, float, float]] = []
    for experiment_model, payload in summary["results"].items():
        for metric_name, stats in payload.get("metrics", {}).items():
            metric_rows.append((experiment_model, "overall", "overall", metric_name, float(stats["mean"]), float(stats["std"])))
        for channel, channel_metrics in payload.get("per_channel", {}).items():
            for metric_name, stats in channel_metrics.items():
                metric_rows.append((experiment_model, "channel", channel, metric_name, float(stats["mean"]), float(stats["std"])))
    return metric_rows


def _snapshot_metric_rows(model_name: str, metrics: dict[str, object]) -> list[tuple[str, str, str, str, float, float]]:
    metric_rows: list[tuple[str, str, str, str, float, float]] = []
    for metric_name, value in metrics.items():
        if metric_name == "per_channel":
            continue
        metric_rows.append((model_name, "overall", "overall", metric_name, float(value), 0.0))
    for channel, channel_metrics in metrics.get("per_channel", {}).items():
        for metric_name, value in channel_metrics.items():
            metric_rows.append((model_name, "channel", channel, metric_name, float(value), 0.0))
    return metric_rows


def _fetch_experiments(connection: sqlite3.Connection) -> list[dict[str, object]]:
    rows = connection.execute(
        """
        SELECT model_name, scope_type, scope_value, metric_name, mean, std
        FROM experiment_metrics
        ORDER BY model_name, scope_type, scope_value, metric_name
        """
    ).fetchall()
    experiments: dict[str, dict[str, object]] = {}
    for row in rows:
        payload = experiments.setdefault(row["model_name"], {"model_name": row["model_name"], "metrics": {}, "channels": {}})
        target = payload["metrics"] if row["scope_type"] == "overall" else payload["channels"].setdefault(row["scope_value"], {})
        target[row["metric_name"]] = {"mean": row["mean"], "std": row["std"]}
    return list(experiments.values())


def _fetch_overview(connection: sqlite3.Connection) -> dict[str, object]:
    metadata = _fetch_metadata(connection)
    totals = connection.execute(
        """
        SELECT
            COUNT(*) AS total_artifacts,
            SUM(CASE WHEN predicted_score >= 0.9 THEN 1 ELSE 0 END) AS high_risk_artifacts,
            SUM(CASE WHEN split = 'test' THEN 1 ELSE 0 END) AS test_artifacts
        FROM alerts
        """
    ).fetchone()
    campaign_count = connection.execute("SELECT COUNT(*) AS total FROM campaigns").fetchone()["total"]
    entity_count = connection.execute("SELECT COUNT(*) AS total FROM entities").fetchone()["total"]
    channels = [dict(row) for row in connection.execute("SELECT channel, artifact_count, positive_rate, mean_score FROM dataset_channels ORDER BY artifact_count DESC").fetchall()]
    top_entities = [
        dict(row)
        for row in connection.execute(
            """
            SELECT entity_id, entity_type, entity_value, risk_mean, risk_max, linked_messages
            FROM entities
            WHERE risk_mean IS NOT NULL
            ORDER BY risk_mean DESC, linked_messages DESC
            LIMIT 6
            """
        ).fetchall()
    ]
    top_campaigns = [
        {**dict(row), "channels": json.loads(row["channels_json"])}
        for row in connection.execute(
            """
            SELECT campaign_id, size, risk_level, mean_score, channels_json, representative_alert_id
            FROM campaigns
            ORDER BY risk_level DESC, size DESC
            LIMIT 6
            """
        ).fetchall()
    ]
    trends = [dict(row) for row in connection.execute(
        """
        SELECT substr(timestamp, 1, 10) AS day, channel, COUNT(*) AS count
        FROM alerts
        GROUP BY substr(timestamp, 1, 10), channel
        ORDER BY day ASC, channel ASC
        """
    ).fetchall()]
    errors = [
        {"category": row["category"], "count": row["count"], "sample_event_ids": json.loads(row["sample_event_ids_json"])}
        for row in connection.execute("SELECT category, count, sample_event_ids_json FROM experiment_errors ORDER BY count DESC").fetchall()
    ]
    best_model = connection.execute(
        """
        SELECT model_name, mean
        FROM experiment_metrics
        WHERE scope_type = 'overall' AND metric_name = 'pr_auc'
        ORDER BY mean DESC
        LIMIT 1
        """
    ).fetchone()
    return {
        "dataset_name": metadata.get("config_name", ""),
        "selected_model": metadata.get("selected_model", ""),
        "primary_metric": metadata.get("primary_metric", ""),
        "generated_at": metadata.get("generated_at", ""),
        "totals": {
            "artifacts": totals["total_artifacts"],
            "high_risk_artifacts": totals["high_risk_artifacts"] or 0,
            "campaigns": campaign_count,
            "entities": entity_count,
            "test_artifacts": totals["test_artifacts"] or 0,
        },
        "best_model": None if best_model is None else {"model_name": best_model["model_name"], "pr_auc": best_model["mean"]},
        "channels": channels,
        "trends": trends,
        "top_entities": top_entities,
        "top_campaigns": top_campaigns,
        "error_analysis": errors,
    }


def _fetch_alerts(connection: sqlite3.Connection, params: dict[str, list[str]]) -> dict[str, object]:
    conditions = []
    query_params: list[object] = []
    if params.get("channel"):
        conditions.append("channel = ?")
        query_params.append(params["channel"][0])
    if params.get("split"):
        conditions.append("split = ?")
        query_params.append(params["split"][0])
    if params.get("status"):
        conditions.append("status = ?")
        query_params.append(params["status"][0])
    if params.get("min_score"):
        conditions.append("predicted_score >= ?")
        query_params.append(float(params["min_score"][0]))
    if params.get("q"):
        conditions.append("(event_id LIKE ? OR sender_key LIKE ? OR recipient_key LIKE ? OR text_or_url LIKE ?)")
        pattern = f"%{params['q'][0]}%"
        query_params.extend([pattern, pattern, pattern, pattern])
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    limit = int(params.get("limit", ["50"])[0])
    offset = int(params.get("offset", ["0"])[0])
    count = connection.execute(f"SELECT COUNT(*) AS total FROM alerts {where_clause}", query_params).fetchone()["total"]
    rows = connection.execute(
        f"""
        SELECT alert_id, event_id, channel, timestamp, sender_key, recipient_key, predicted_score, predicted_label,
               status, split, campaign_id, substr(text_or_url, 1, 160) AS preview
        FROM alerts
        {where_clause}
        ORDER BY predicted_score DESC, timestamp DESC
        LIMIT ? OFFSET ?
        """,
        [*query_params, limit, offset],
    ).fetchall()
    return {"total": count, "items": [dict(row) for row in rows]}


def _build_reasons(connection: sqlite3.Connection, message_id: str) -> list[str]:
    counts = {
        row["edge_type"]: row["count"]
        for row in connection.execute(
            """
            SELECT edge_type, COUNT(*) AS count
            FROM relations
            WHERE src_id = ?
            GROUP BY edge_type
            """,
            [message_id],
        ).fetchall()
    }
    reasons = []
    if counts.get("contains_url", 0):
        reasons.append(f"Contains {counts['contains_url']} linked URL(s)")
    if counts.get("shares_sender", 0):
        reasons.append(f"Shares sender with {counts['shares_sender']} related messages")
    if counts.get("shares_recipient", 0):
        reasons.append(f"Shares recipient with {counts['shares_recipient']} related messages")
    if counts.get("shares_infrastructure", 0):
        reasons.append(f"Shares infrastructure with {counts['shares_infrastructure']} related messages")
    if not reasons:
        reasons.append("High model score based on text, metadata, and graph features")
    return reasons


def _direct_entities(connection: sqlite3.Connection, message_id: str) -> list[dict[str, object]]:
    direct = connection.execute(
        """
        SELECT e.entity_id, e.entity_type, e.entity_value, e.risk_mean, e.risk_max, e.linked_messages, r.edge_type
        FROM relations r
        JOIN entities e ON e.entity_id = r.dst_id
        WHERE r.src_id = ? AND r.dst_type != 'message'
        ORDER BY (e.risk_mean IS NULL), e.risk_mean DESC, e.linked_messages DESC
        """,
        [message_id],
    ).fetchall()
    seen = set()
    entities = []
    for row in direct:
        row_dict = dict(row)
        if row_dict["entity_id"] in seen:
            continue
        seen.add(row_dict["entity_id"])
        entities.append(row_dict)
    return entities


def _related_alerts(connection: sqlite3.Connection, message_id: str) -> list[dict[str, object]]:
    rows = connection.execute(
        """
        SELECT a.alert_id, a.channel, a.timestamp, a.predicted_score, a.campaign_id, r.edge_type, substr(a.text_or_url, 1, 120) AS preview
        FROM relations r
        JOIN alerts a ON a.message_id = r.dst_id
        WHERE r.src_id = ? AND r.dst_type = 'message'
        ORDER BY a.predicted_score DESC, a.timestamp DESC
        LIMIT 12
        """,
        [message_id],
    ).fetchall()
    return [dict(row) for row in rows]


def _graph_payload(connection: sqlite3.Connection, message_id: str) -> dict[str, object]:
    nodes: dict[str, dict[str, object]] = {}
    edges: list[dict[str, object]] = []
    center = connection.execute(
        """
        SELECT message_id, event_id, channel, predicted_score, predicted_label, text_or_url
        FROM alerts
        WHERE message_id = ?
        """,
        [message_id],
    ).fetchone()
    if center is None:
        return {"nodes": [], "edges": []}
    nodes[center["message_id"]] = {
        "id": center["message_id"],
        "label": center["event_id"],
        "type": "message",
        "score": center["predicted_score"],
        "subtitle": center["channel"],
    }

    first_hop = connection.execute(
        """
        SELECT src_id, dst_id, src_type, dst_type, edge_type
        FROM relations
        WHERE src_id = ?
        ORDER BY edge_type, dst_id
        LIMIT 64
        """,
        [message_id],
    ).fetchall()
    neighbor_entities = set()
    for row in first_hop:
        edges.append({"source": row["src_id"], "target": row["dst_id"], "type": row["edge_type"]})
        if row["dst_type"] == "message":
            alert = connection.execute("SELECT event_id, channel, predicted_score FROM alerts WHERE message_id = ?", [row["dst_id"]]).fetchone()
            if alert is not None:
                nodes[row["dst_id"]] = {
                    "id": row["dst_id"],
                    "label": alert["event_id"],
                    "type": "message",
                    "score": alert["predicted_score"],
                    "subtitle": alert["channel"],
                }
        else:
            neighbor_entities.add(row["dst_id"])
            entity = connection.execute("SELECT entity_type, entity_value, risk_mean FROM entities WHERE entity_id = ?", [row["dst_id"]]).fetchone()
            if entity is not None:
                nodes[row["dst_id"]] = {
                    "id": row["dst_id"],
                    "label": entity["entity_value"],
                    "type": entity["entity_type"],
                    "score": entity["risk_mean"],
                    "subtitle": row["edge_type"],
                }

    if neighbor_entities:
        placeholders = ",".join("?" for _ in neighbor_entities)
        second_hop = connection.execute(
            f"""
            SELECT src_id, dst_id, src_type, dst_type, edge_type
            FROM relations
            WHERE src_id IN ({placeholders}) AND dst_type != 'message'
            ORDER BY edge_type, dst_id
            LIMIT 64
            """,
            list(neighbor_entities),
        ).fetchall()
        for row in second_hop:
            edges.append({"source": row["src_id"], "target": row["dst_id"], "type": row["edge_type"]})
            entity = connection.execute("SELECT entity_type, entity_value, risk_mean FROM entities WHERE entity_id = ?", [row["dst_id"]]).fetchone()
            if entity is not None:
                nodes[row["dst_id"]] = {
                    "id": row["dst_id"],
                    "label": entity["entity_value"],
                    "type": entity["entity_type"],
                    "score": entity["risk_mean"],
                    "subtitle": row["edge_type"],
                }
    return {"nodes": list(nodes.values()), "edges": edges}


def _fetch_alert_detail(connection: sqlite3.Connection, alert_id: str) -> dict[str, object] | None:
    alert = connection.execute("SELECT * FROM alerts WHERE alert_id = ?", [alert_id]).fetchone()
    if alert is None:
        return None
    message_id = alert["message_id"]
    return {
        "alert": dict(alert),
        "reasons": _build_reasons(connection, message_id),
        "entities": _direct_entities(connection, message_id),
        "related_alerts": _related_alerts(connection, message_id),
        "graph": _graph_payload(connection, message_id),
    }


def _fetch_entity_detail(connection: sqlite3.Connection, entity_id: str) -> dict[str, object] | None:
    entity = connection.execute("SELECT * FROM entities WHERE entity_id = ?", [entity_id]).fetchone()
    if entity is None:
        return None
    rows = connection.execute(
        """
        SELECT DISTINCT a.alert_id, a.event_id, a.channel, a.timestamp, a.predicted_score, a.predicted_label, a.campaign_id, substr(a.text_or_url, 1, 160) AS preview
        FROM alerts a
        JOIN relations r ON ((r.src_id = a.message_id AND r.dst_id = ?) OR (r.dst_id = a.message_id AND r.src_id = ?))
        ORDER BY a.predicted_score DESC, a.timestamp DESC
        LIMIT 25
        """,
        [entity_id, entity_id],
    ).fetchall()
    return {"entity": dict(entity), "linked_alerts": [dict(row) for row in rows]}


def _fetch_campaigns(connection: sqlite3.Connection) -> list[dict[str, object]]:
    rows = connection.execute(
        """
        SELECT campaign_id, size, risk_level, mean_score, channels_json, first_seen, last_seen, representative_alert_id
        FROM campaigns
        ORDER BY risk_level DESC, size DESC
        LIMIT 20
        """
    ).fetchall()
    return [{**dict(row), "channels": json.loads(row["channels_json"])} for row in rows]


def _fetch_campaign_detail(connection: sqlite3.Connection, campaign_id: str) -> dict[str, object] | None:
    campaign = connection.execute("SELECT * FROM campaigns WHERE campaign_id = ?", [campaign_id]).fetchone()
    if campaign is None:
        return None
    alerts = connection.execute(
        """
        SELECT alert_id, event_id, channel, timestamp, predicted_score, predicted_label, substr(text_or_url, 1, 160) AS preview
        FROM alerts
        WHERE campaign_id = ?
        ORDER BY predicted_score DESC, timestamp DESC
        """,
        [campaign_id],
    ).fetchall()
    return {**dict(campaign), "channels": json.loads(campaign["channels_json"]), "alerts": [dict(row) for row in alerts]}


def _fetch_datasets(connection: sqlite3.Connection) -> dict[str, object]:
    metadata = _fetch_metadata(connection)
    channels = [dict(row) for row in connection.execute("SELECT * FROM dataset_channels ORDER BY artifact_count DESC").fetchall()]
    return {
        "config_name": metadata.get("config_name", ""),
        "track": metadata.get("track", ""),
        "num_events": int(metadata.get("num_events", "0")),
        "num_entities": int(metadata.get("num_entities", "0")),
        "num_relations": int(metadata.get("num_relations", "0")),
        "channels": channels,
    }


def _fetch_search(connection: sqlite3.Connection, query: str) -> dict[str, object]:
    pattern = f"%{query}%"
    alerts = [
        dict(row)
        for row in connection.execute(
            """
            SELECT alert_id, event_id, channel, predicted_score, substr(text_or_url, 1, 120) AS preview
            FROM alerts
            WHERE event_id LIKE ? OR sender_key LIKE ? OR recipient_key LIKE ? OR text_or_url LIKE ?
            ORDER BY predicted_score DESC
            LIMIT 8
            """,
            [pattern, pattern, pattern, pattern],
        ).fetchall()
    ]
    entities = [
        dict(row)
        for row in connection.execute(
            """
            SELECT entity_id, entity_type, entity_value, risk_mean, linked_messages
            FROM entities
            WHERE entity_value LIKE ?
            ORDER BY (risk_mean IS NULL), risk_mean DESC, linked_messages DESC
            LIMIT 8
            """,
            [pattern],
        ).fetchall()
    ]
    return {"alerts": alerts, "entities": entities}


class _UIRequestHandler(SimpleHTTPRequestHandler):
    db_path: Path = DEFAULT_DB_PATH
    static_dir: Path = STATIC_DIR

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(self.static_dir), **kwargs)

    def log_message(self, fmt, *args):
        # Suppress default stderr logging for cleaner output.
        pass

    def _send_json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self._handle_api(parsed)
            return
        asset_path = self.static_dir / parsed.path.lstrip("/")
        if parsed.path in {"", "/"} or not asset_path.exists():
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self._handle_post_api(parsed)
            return
        self._send_json({"error": "Not found"}, status=404)

    def _handle_post_api(self, parsed) -> None:
        path = parsed.path
        body = self._read_body()

        # --- Command endpoints ---
        if path == "/api/commands/generate-synthetic":
            output = body.get("output", "artifacts/synthetic_events.csv")
            seed = int(body.get("seed", 7))
            num_events = int(body.get("num_events", 96))
            _run_command_thread("generate-synthetic", _cmd_generate_synthetic, args=(output, seed, num_events))
            self._send_json({"status": "started", "command": "generate-synthetic"})
            return

        if path == "/api/commands/prepare-uci-sms":
            output = body.get("output", "data/public/uci_sms_spam_collection.csv")
            force = bool(body.get("force", False))
            _run_command_thread("prepare-uci-sms", _cmd_prepare_uci_sms, args=(output, force))
            self._send_json({"status": "started", "command": "prepare-uci-sms"})
            return

        if path == "/api/commands/prepare-phiusiil-url":
            output = body.get("output", "data/public/phiusiil_url_collection.csv")
            force = bool(body.get("force", False))
            _run_command_thread("prepare-phiusiil-url", _cmd_prepare_phiusiil_url, args=(output, force))
            self._send_json({"status": "started", "command": "prepare-phiusiil-url"})
            return

        if path == "/api/commands/run-research":
            config_path = body.get("config", str(DEFAULT_CONFIG_PATH))
            _run_command_thread("run-research", _cmd_run_research, args=(config_path,))
            self._send_json({"status": "started", "command": "run-research"})
            return

        if path == "/api/commands/build-ui-db":
            config_path = body.get("config", str(DEFAULT_CONFIG_PATH))
            db = body.get("db", str(DEFAULT_DB_PATH))
            model = body.get("model", "auto")
            seed = body.get("seed", None)
            if seed is not None:
                seed = int(seed)
            _run_command_thread("build-ui-db", _cmd_build_ui_db, args=(config_path, db, model, seed))
            self._send_json({"status": "started", "command": "build-ui-db"})
            return

        # --- Config endpoints ---
        if path == "/api/config/save":
            file_path = body.get("file")
            config_data = body.get("config")
            if not file_path or not config_data:
                self._send_json({"error": "file and config are required"}, status=400)
                return
            try:
                saved = _save_config_dict(file_path, config_data)
                self._send_json({"status": "saved", "file": saved})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return

        self._send_json({"error": "Endpoint not found"}, status=404)

    def _handle_api(self, parsed) -> None:
        params = parse_qs(parsed.query)
        path = parsed.path

        # --- Non-DB endpoints (commands, config) ---
        if path == "/api/commands/status":
            name = params.get("name", [""])[0]
            if name:
                self._send_json(_get_command_status(name))
            else:
                with _command_lock:
                    self._send_json({k: dict(v) for k, v in _command_status.items()})
            return

        if path == "/api/configs":
            self._send_json(_list_configs())
            return

        if path == "/api/config/load":
            file_path = params.get("file", [""])[0]
            if not file_path:
                self._send_json({"error": "file parameter required"}, status=400)
                return
            try:
                self._send_json(_load_config_as_dict(file_path))
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return

        if path == "/api/config/defaults":
            self._send_json(asdict(ExperimentConfig(name="new_experiment")))
            return

        if path == "/api/features":
            with _connect(self.db_path) as connection:
                stats = _fetch_feature_stats(connection)
            self._send_json({"categories": FEATURE_CATEGORIES, "stats": stats})
            return

        if path == "/api/models/info":
            availability = _fetch_model_availability()
            # Attach experiment metrics if available
            model_metrics = {}
            try:
                with _connect(self.db_path) as connection:
                    for row in connection.execute(
                        "SELECT model_name, metric_name, mean, std FROM experiment_metrics WHERE scope_type='overall'"
                    ).fetchall():
                        model_metrics.setdefault(row["model_name"], {})[row["metric_name"]] = {
                            "mean": row["mean"], "std": row["std"]
                        }
            except Exception:
                pass
            self._send_json({
                "models": MODEL_DEFINITIONS,
                "availability": availability,
                "metrics": model_metrics,
            })
            return

        # --- DB-backed endpoints ---
        with _connect(self.db_path) as connection:
            if path == "/api/overview":
                self._send_json(_fetch_overview(connection))
                return
            if path == "/api/alerts":
                self._send_json(_fetch_alerts(connection, params))
                return
            if path.startswith("/api/alerts/"):
                payload = _fetch_alert_detail(connection, path.removeprefix("/api/alerts/"))
                self._send_json({"error": "Alert not found"} if payload is None else payload, status=404 if payload is None else 200)
                return
            if path == "/api/entities/top":
                limit = int(params.get("limit", ["12"])[0])
                entity_type = params.get("type", [""])[0]
                query = "SELECT entity_id, entity_type, entity_value, risk_mean, risk_max, linked_messages FROM entities WHERE risk_mean IS NOT NULL"
                query_params: list[object] = []
                if entity_type:
                    query += " AND entity_type = ?"
                    query_params.append(entity_type)
                query += " ORDER BY risk_mean DESC, linked_messages DESC LIMIT ?"
                query_params.append(limit)
                self._send_json([dict(row) for row in connection.execute(query, query_params).fetchall()])
                return
            if path.startswith("/api/entities/"):
                payload = _fetch_entity_detail(connection, path.removeprefix("/api/entities/"))
                self._send_json({"error": "Entity not found"} if payload is None else payload, status=404 if payload is None else 200)
                return
            if path == "/api/campaigns":
                self._send_json(_fetch_campaigns(connection))
                return
            if path.startswith("/api/campaigns/"):
                payload = _fetch_campaign_detail(connection, path.removeprefix("/api/campaigns/"))
                self._send_json({"error": "Campaign not found"} if payload is None else payload, status=404 if payload is None else 200)
                return
            if path == "/api/experiments":
                self._send_json(_fetch_experiments(connection))
                return
            if path == "/api/datasets":
                self._send_json(_fetch_datasets(connection))
                return
            if path == "/api/search":
                query = params.get("q", [""])[0].strip()
                self._send_json(_fetch_search(connection, query) if query else {"alerts": [], "entities": []})
                return
        self._send_json({"error": "Endpoint not found"}, status=404)


def serve_ui(db_path: str | Path = DEFAULT_DB_PATH, *, host: str = "127.0.0.1", port: int = 8000) -> None:
    handler = type("ScamWorkbenchHandler", (_UIRequestHandler,), {"db_path": Path(db_path), "static_dir": STATIC_DIR})
    with ThreadingHTTPServer((host, port), handler) as server:
        print(f"http://{host}:{port}")
        server.serve_forever()
