"""Inference layer for the live demonstration.

This module adds NO new model and NO new methodology.  It reuses the research
code in ``src/scam_gnn`` to (1) rebuild the same graph the GNN was trained on,
(2) reload the GNN weights saved by ``train_demo_model.py`` and (3) score one
additional message by inserting it into that graph.

How a typed message reaches the GNN:

    text -> event row (project's UCI SMS rule) -> build_graph_tables
         -> message/entity features -> tensors -> trained GNN forward pass
         -> softmax probability of the "scam" class for the new message node
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DEMO_DIR = Path(__file__).resolve().parent
ROOT = DEMO_DIR.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from scam_gnn import pipeline  # noqa: E402
from scam_gnn.config import ExperimentConfig, load_config  # noqa: E402
from scam_gnn.data.normalize import extract_domain_from_url, extract_urls  # noqa: E402
from scam_gnn.data.public_datasets import parse_uci_sms_spam_text  # noqa: E402
from scam_gnn.data.splits import assign_time_based_splits  # noqa: E402
from scam_gnn.data.standardize import standardize_sms_dataframe  # noqa: E402
from scam_gnn.features import (  # noqa: E402
    URGENT_KEYWORDS,
    FeatureBundle,
    _entity_feature_frame,
    _load_sentence_transformer,
    _message_graph_stats,
    _tokenize,
    build_feature_bundle,
    build_message_feature_frame,
)
from scam_gnn.graph import GraphTables, build_graph_tables, build_pyg_heterodata  # noqa: E402
from scam_gnn.models.gnn import build_heterogeneous_graphsage  # noqa: E402

CONFIG_PATH = ROOT / "examples" / "configs" / "uci_sms_phiusiil_multichannel.json"
MODEL_DIR = DEMO_DIR / "model"
WEIGHTS_PATH = MODEL_DIR / "hetero_graphsage.pt"
META_PATH = MODEL_DIR / "meta.json"
PREDICTIONS_PATH = MODEL_DIR / "predictions.csv"
CACHE_PATH = MODEL_DIR / "prepared_graph.pkl"

MODEL_NAME = "hetero_graphsage"
SEED = 7
# Same decision threshold the research code uses (metrics.binary_metrics, ui.py).
THRESHOLD = 0.5
MAX_MESSAGE_CHARS = 1000
LIVE_EVENT_ID = "live-message"
SHARE_EDGE_TYPES = ["shares_sender", "shares_recipient", "shares_infrastructure"]
# graph.py skips pairwise edges for groups above 128 members; near that size we
# use the full rebuild so the shortcut can never disagree with the research code.
FAST_PATH_MAX_RELATED = 100
BASIC_FEATURES = [
    "length",
    "digit_ratio",
    "uppercase_ratio",
    "url_count",
    "urgent_keyword_count",
    "out_degree",
    "in_degree",
    "infrastructure_degree",
]


class DemoSetupError(RuntimeError):
    """The demo cannot start (for example, the model has not been trained yet)."""


class InvalidMessage(ValueError):
    """User input that cannot be analysed."""


def embedding_backend(config: ExperimentConfig) -> str:
    """Which text representation the feature pipeline is actually using right now."""
    if _load_sentence_transformer(config.features.embedding_model) is not None:
        return f"sentence-transformers ({config.features.embedding_model})"
    return f"hashed bag-of-words fallback ({config.features.fallback_embedding_dim} dimensions)"


def prepare_base(config: ExperimentConfig, seed: int):
    """Events -> splits -> graph tables, exactly as ``pipeline.build_scored_snapshot`` does."""
    events = pipeline._load_custom_events(config, seed)
    split_events = assign_time_based_splits(events, config.splits)
    tables = build_graph_tables(
        split_events,
        include_cross_channel_edges=config.ablation.include_cross_channel_edges,
        include_infrastructure_nodes=config.ablation.include_infrastructure_nodes,
    )
    return split_events, tables


def bundle_from_text_features(tables: GraphTables, text_features: pd.DataFrame, config: ExperimentConfig) -> FeatureBundle:
    """Same steps as ``features.build_feature_bundle`` but with the text part supplied.

    The text features of the 20,574 historical messages never change, so they are
    computed once; only the graph-degree features are recomputed per analysis.
    """
    message_features = text_features
    if config.features.include_graph_statistics:
        message_features = message_features.join(_message_graph_stats(tables), how="left").fillna(0.0)
    entity_features = {
        entity_type: _entity_feature_frame(entity_type, tables)
        for entity_type in tables.entities["entity_type"].drop_duplicates().tolist()
    }
    entity_features["message"] = message_features
    return FeatureBundle(message_features=message_features, entity_features=entity_features)


def build_inference_tensors(tables: GraphTables, entity_features: dict[str, pd.DataFrame]):
    """Node features and edge lists in the layout of ``graph.build_pyg_heterodata``.

    Labels, masks and sample weights are left out because they are only needed
    for training.  ``tests/test_engine.py`` checks the tensors are identical.
    """
    import torch

    x_dict = {}
    index_maps: dict[str, pd.Series] = {}
    node_ids: dict[str, list[str]] = {}
    for entity_type, frame in tables.entities.groupby("entity_type", sort=False):
        ids = list(frame["entity_id"])
        node_ids[entity_type] = ids
        index_maps[entity_type] = pd.Series(np.arange(len(ids), dtype=np.int64), index=ids)
        features = entity_features.get(entity_type)
        if features is None or features.empty:
            x_dict[entity_type] = torch.zeros((len(ids), 1), dtype=torch.float32)
        else:
            aligned = features.reindex(ids).fillna(0.0)
            x_dict[entity_type] = torch.tensor(aligned.to_numpy(dtype=np.float32), dtype=torch.float32)

    edge_index_dict = {}
    for (src_type, edge_type, dst_type), frame in tables.relations.groupby(["src_type", "edge_type", "dst_type"]):
        src = index_maps[src_type].reindex(frame["src_id"]).to_numpy(dtype=np.int64)
        dst = index_maps[dst_type].reindex(frame["dst_id"]).to_numpy(dtype=np.int64)
        edge_index_dict[(src_type, edge_type, dst_type)] = torch.tensor(np.stack([src, dst]), dtype=torch.long)
        edge_index_dict[(dst_type, f"rev_{edge_type}", src_type)] = torch.tensor(np.stack([dst, src]), dtype=torch.long)
    return x_dict, edge_index_dict, node_ids


def confidence_band(score: float) -> str:
    """Display convention only: distance of the probability from the 0.5 threshold."""
    margin = abs(score - THRESHOLD)
    if margin >= 0.35:
        return "High"
    if margin >= 0.15:
        return "Medium"
    return "Low"


@dataclass
class Analysis:
    source: str  # "live" or "held-out"
    event: dict
    message_id: str
    score: float
    prediction: int
    confidence: str
    score_without_links: float | None
    features: dict[str, float]
    urgent_keywords: list[str]
    relations: pd.DataFrame  # edges of the local subgraph around the message
    entities: pd.DataFrame  # nodes of the local subgraph (non-message)
    linked_messages: pd.DataFrame  # other messages reached through shares_* edges
    timings: dict[str, float] = field(default_factory=dict)
    path: str = ""
    true_label: int | None = None

    @property
    def label_text(self) -> str:
        return "SCAM" if self.prediction == 1 else "LEGITIMATE"


class DemoEngine:
    def __init__(self, progress=None):
        import torch

        self._torch = torch
        say = progress or (lambda message: None)
        os.chdir(ROOT)  # config paths (data/public/...) are relative to the repo root
        for path in (WEIGHTS_PATH, META_PATH, PREDICTIONS_PATH):
            if not path.exists():
                raise DemoSetupError(
                    f"Missing {path.relative_to(ROOT)}. Train the demo model first:  python demo\\train_demo_model.py"
                )
        self.meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        self.config = load_config(CONFIG_PATH)
        self.seed = int(self.meta["seed"])
        self.load_timings: dict[str, float] = {}

        self.embedding_backend = embedding_backend(self.config)
        if self.embedding_backend != self.meta["embedding_backend"]:
            raise DemoSetupError(
                "The text-embedding backend changed since the model was trained "
                f"(trained with: {self.meta['embedding_backend']}; now: {self.embedding_backend}). "
                "Re-run  python demo\\train_demo_model.py"
            )
        self._load_prepared(say, use_cache=True)
        scores, node_ids = self._load_model_and_score(say)
        self.predictions = pd.read_csv(PREDICTIONS_PATH)
        saved = self.predictions.set_index("message_id")["score"]

        def difference() -> float:
            return float((pd.Series(scores, index=node_ids["message"]).reindex(saved.index) - saved).abs().max())

        # Integrity check: the reloaded model must reproduce the scores saved at training time.
        self.max_score_difference = difference()
        if self.prepared_from_cache and not self.max_score_difference < 1e-4:
            # The saved graph does not match the model: discard it and rebuild from the raw data.
            self._load_prepared(say, use_cache=False)
            scores, node_ids = self._load_model_and_score(say)
            self.max_score_difference = difference()
        self.base_scores = pd.Series(scores, index=node_ids["message"])

        events = self.tables.events
        self.events_by_id = events.set_index("message_id")
        self._sender_norm = events["sender_key"].fillna("").astype(str).str.strip().str.lower()
        self._recipient_norm = events["recipient_key"].fillna("").astype(str).str.strip().str.lower()
        relations = self.tables.relations
        contains = relations[relations["edge_type"] == "contains_url"][["src_id", "dst_id"]]
        resolves = relations[relations["edge_type"] == "resolves_to_domain"][["src_id", "dst_id"]]
        joined = contains.merge(resolves, left_on="dst_id", right_on="src_id", suffixes=("_message", "_domain"))
        self._messages_by_domain = joined.groupby("dst_id_domain")["src_id_message"].agg(lambda ids: sorted(set(ids))).to_dict()
        self._max_timestamp = events["timestamp"].max()

        # Training-split averages per channel and label, shown next to a message's own feature values.
        train = events[events["split"] == "train"]
        columns = [name for name in BASIC_FEATURES if name in self.base_features.columns]
        reference = self.base_features.loc[train["message_id"], columns].copy()
        reference["channel"] = train["channel"].to_numpy()
        reference["label"] = train["label"].to_numpy()
        self.feature_reference = reference.groupby(["channel", "label"]).mean()

    # ---------------------------------------------------------------- loading
    def _fingerprint(self) -> str:
        """Changes whenever anything that determines the prepared graph changes."""
        digest = hashlib.sha256()
        digest.update(f"{self.seed}|{self.embedding_backend}|{pd.__version__}|{np.__version__}".encode())
        digest.update(CONFIG_PATH.read_bytes())
        for path in sorted((SRC / "scam_gnn").rglob("*.py")) + [Path(__file__)]:
            if path.name not in {"ui.py", "cli.py"}:
                digest.update(path.read_bytes())
        for data_path in (self.config.dataset.sms_path, self.config.dataset.email_path, self.config.dataset.url_path):
            if data_path and Path(data_path).exists():
                stat = Path(data_path).stat()
                digest.update(f"{data_path}|{stat.st_size}|{stat.st_mtime_ns}".encode())
        return digest.hexdigest()

    def _load_prepared(self, say, *, use_cache: bool) -> None:
        """Graph tables and features for the historical data.

        Building them from the raw CSV files takes about 30 seconds, so the result
        is saved to ``model/prepared_graph.pkl`` and reused while the data, config
        and code are unchanged.  The file only stores what the research code produced.
        """
        started = time.perf_counter()
        fingerprint = self._fingerprint()
        prepared = None
        if use_cache and CACHE_PATH.exists():
            say("Loading the prepared graph...")
            try:
                with CACHE_PATH.open("rb") as handle:
                    candidate = pickle.load(handle)
                if candidate.get("fingerprint") == fingerprint:
                    prepared = candidate
            except Exception:  # noqa: BLE001 - an unreadable cache is simply rebuilt
                prepared = None
        self.prepared_from_cache = prepared is not None
        if prepared is None:
            say("Reading the datasets and building the graph (about 30 seconds, first time only)...")
            split_events, tables = prepare_base(self.config, self.seed)
            say("Computing features...")
            text_features = build_message_feature_frame(tables.events, self.config.features)
            bundle = bundle_from_text_features(tables, text_features, self.config)
            prepared = {
                "fingerprint": fingerprint,
                "split_events": split_events,
                "events": tables.events,
                "entities": tables.entities,
                "relations": tables.relations,
                "text_features": text_features,
                "entity_features": bundle.entity_features,
            }
            try:
                temporary = CACHE_PATH.with_suffix(".tmp")
                with temporary.open("wb") as handle:
                    pickle.dump(prepared, handle, protocol=pickle.HIGHEST_PROTOCOL)
                temporary.replace(CACHE_PATH)
            except OSError:
                pass  # the demo still works without the cache, it just starts slowly
        self.split_events = prepared["split_events"]
        self.tables = GraphTables(events=prepared["events"], entities=prepared["entities"], relations=prepared["relations"])
        self.text_features = prepared["text_features"]
        self.base_entity_features = prepared["entity_features"]
        self.base_features = self.base_entity_features["message"]
        if list(self.base_features.columns) != self.meta["feature_columns"]:
            raise DemoSetupError("Feature columns differ from training. Re-run  python demo\\train_demo_model.py")
        self._base_entity_ids = set(self.tables.entities["entity_id"])
        self.load_timings["prepared graph (saved file)" if self.prepared_from_cache else "build graph + features"] = time.perf_counter() - started

    def _load_model_and_score(self, say):
        torch = self._torch
        started = time.perf_counter()
        say("Loading the trained GNN...")
        x_dict, edge_index_dict, node_ids = build_inference_tensors(self.tables, self.base_entity_features)
        metadata = (
            list(self.meta["metadata"]["node_types"]),
            [tuple(edge_type) for edge_type in self.meta["metadata"]["edge_types"]],
        )
        self.model = build_heterogeneous_graphsage(metadata)
        self.model.eval()
        with torch.no_grad():
            self.model(x_dict, edge_index_dict)  # materialise lazily-sized layers before loading weights
        self.model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
        self.model.eval()
        scores = self._forward(x_dict, edge_index_dict)
        self.load_timings["load model + score graph"] = time.perf_counter() - started
        return scores, node_ids

    # ------------------------------------------------------------------ model
    def _forward(self, x_dict, edge_index_dict) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            logits = self.model(x_dict, edge_index_dict)
            return torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

    def _updated_bundle(self, tables: GraphTables, text_features: pd.DataFrame, touched_ids: set[str]) -> FeatureBundle:
        """Features for a graph that differs from the cached one only around ``touched_ids``.

        Degree features change only for nodes that gained or lost an edge, so the
        research feature functions are run on just those nodes (with all of their
        edges) and every other node keeps its cached features.
        """
        relations = tables.relations
        sub_tables = GraphTables(
            events=tables.events[tables.events["message_id"].isin(touched_ids)],
            entities=tables.entities[tables.entities["entity_id"].isin(touched_ids)],
            relations=relations[relations["src_id"].isin(touched_ids) | relations["dst_id"].isin(touched_ids)],
        )
        touched_messages = text_features.loc[sub_tables.events["message_id"]]
        if self.config.features.include_graph_statistics:
            touched_messages = touched_messages.join(_message_graph_stats(sub_tables), how="left").fillna(0.0)
        message_features = pd.concat(
            [self.base_features.drop(index=touched_messages.index, errors="ignore"), touched_messages]
        ).reindex(text_features.index)
        entity_features = dict(self.base_entity_features)
        for entity_type in sub_tables.entities["entity_type"].drop_duplicates().tolist():
            updated = _entity_feature_frame(entity_type, sub_tables)
            entity_features[entity_type] = pd.concat(
                [self.base_entity_features[entity_type].drop(index=updated.index, errors="ignore"), updated]
            )
        entity_features["message"] = message_features
        return FeatureBundle(message_features=message_features, entity_features=entity_features)

    def _score_tables(self, tables: GraphTables, text_features: pd.DataFrame, touched_ids: set[str], *, mode: str) -> tuple[pd.Series, pd.DataFrame]:
        if mode == "full":
            # Reference path: research code only, nothing cached.
            bundle = build_feature_bundle(tables, self.config.features)
            data = build_pyg_heterodata(tables, bundle.entity_features)
            scores = self._forward(data.x_dict, data.edge_index_dict)
            return pd.Series(scores, index=pipeline._message_node_order(tables)), bundle.message_features
        bundle = self._updated_bundle(tables, text_features, touched_ids)
        x_dict, edge_index_dict, node_ids = build_inference_tensors(tables, bundle.entity_features)
        scores = self._forward(x_dict, edge_index_dict)
        return pd.Series(scores, index=node_ids["message"]), bundle.message_features

    # ------------------------------------------------------------ live events
    def make_sms_event(self, text: str, sender: str | None = None) -> pd.DataFrame:
        """Turn typed text into one event row using the project's own UCI SMS rule."""
        cleaned = " ".join(str(text or "").split())
        if not cleaned:
            raise InvalidMessage("Please enter a message to analyse.")
        if len(cleaned) > MAX_MESSAGE_CHARS:
            raise InvalidMessage(f"Message is too long ({len(cleaned)} characters). The limit is {MAX_MESSAGE_CHARS}.")
        if not _tokenize(cleaned):
            raise InvalidMessage("The message has no letters or digits, so no features can be extracted from it.")
        # Reuse the dataset parser so the sender is derived exactly as for training
        # data: first phone number in the text, else first URL, else "unknown".
        parsed = parse_uci_sms_spam_text("ham\t" + cleaned)
        event = standardize_sms_dataframe(parsed).iloc[[0]].copy()
        sender = (sender or "").strip()
        if sender:
            event["sender_key"] = sender
        elif str(event["sender_key"].iloc[0]).startswith("sms_sender_unknown"):
            event["sender_key"] = "sms_sender_unknown_live"
        event["recipient_key"] = "sms_recipient_unknown_live"
        event["event_id"] = LIVE_EVENT_ID
        event["timestamp"] = self._max_timestamp + pd.Timedelta(seconds=1)
        event["source_dataset"] = "live_demo_input"
        # Placeholder only: the label is never a model input and "live" is in no train/val/test mask.
        event["label"] = 0
        event["split"] = "live"
        return event

    def _related_base_events(self, event: pd.DataFrame) -> pd.DataFrame:
        """Historical events that share a sender, recipient or domain with the new one."""
        row = event.iloc[0]
        mask = (self._sender_norm == str(row["sender_key"]).strip().lower()) | (
            self._recipient_norm == str(row["recipient_key"]).strip().lower()
        )
        related_ids: set[str] = set()
        if self.config.ablation.include_infrastructure_nodes:
            for url in filter(None, extract_urls(row["text_or_url"])):
                domain = extract_domain_from_url(url)
                if domain:
                    related_ids.update(self._messages_by_domain.get(f"domain:{domain}", []))
        events = self.tables.events
        if related_ids:
            mask = mask | events["message_id"].isin(related_ids)
        return events[mask.to_numpy()].drop(columns=["message_id"])

    def _tables_with_event(self, event: pd.DataFrame, *, mode: str) -> tuple[GraphTables, str]:
        ablation = self.config.ablation
        kwargs = {
            "include_cross_channel_edges": ablation.include_cross_channel_edges,
            "include_infrastructure_nodes": ablation.include_infrastructure_nodes,
        }
        related = self._related_base_events(event)
        if mode == "full" or len(related) >= FAST_PATH_MAX_RELATED:
            combined = pd.concat([self.split_events, event], ignore_index=True)
            return build_graph_tables(combined, **kwargs), "full graph rebuild"
        # Fast path: run the research graph builder on the new event plus every
        # historical event it can connect to, then merge the result into the cached graph.
        local = build_graph_tables(pd.concat([related, event], ignore_index=True), **kwargs)
        base = self.tables
        events = pd.concat([base.events, local.events[local.events["event_id"] == LIVE_EVENT_ID]], ignore_index=True)
        new_entities = local.entities[~local.entities["entity_id"].isin(base.entities["entity_id"])]
        entities = (
            pd.concat([base.entities, new_entities], ignore_index=True)
            .sort_values(["entity_type", "entity_id"])
            .reset_index(drop=True)
        )
        relations = (
            pd.concat([base.relations, local.relations], ignore_index=True)
            .drop_duplicates(subset=["src_id", "dst_id", "edge_type"])
            .sort_values(["timestamp", "edge_type", "src_id", "dst_id"])
            .reset_index(drop=True)
        )
        return GraphTables(events=events, entities=entities, relations=relations), "incremental graph update"

    def _text_features_with_event(self, tables: GraphTables) -> pd.DataFrame:
        live_rows = tables.events[tables.events["event_id"] == LIVE_EVENT_ID]
        live_features = build_message_feature_frame(live_rows, self.config.features)
        live_features = live_features.reindex(columns=self.text_features.columns, fill_value=0.0).astype(np.float32)
        return pd.concat([self.text_features, live_features])

    # --------------------------------------------------------------- analysis
    def analyze_sms(self, text: str, sender: str | None = None, *, mode: str = "fast") -> Analysis:
        timings: dict[str, float] = {}
        started = time.perf_counter()
        event = self.make_sms_event(text, sender)
        timings["1. preprocessing"] = time.perf_counter() - started

        started = time.perf_counter()
        tables, path = self._tables_with_event(event, mode=mode)
        timings["2. graph construction"] = time.perf_counter() - started

        started = time.perf_counter()
        text_features = self._text_features_with_event(tables)
        message_id = f"message:{LIVE_EVENT_ID}"
        relations = tables.relations
        # Every node that has an edge the cached graph did not have.
        new_entities = set(tables.entities["entity_id"]) - self._base_entity_ids
        added = relations[relations["src_id"].isin(new_entities) | relations["dst_id"].isin(new_entities)]
        touched_ids = new_entities | set(added["src_id"]) | set(added["dst_id"])
        scores, message_features = self._score_tables(tables, text_features, touched_ids, mode=mode)
        timings["3. features + GNN forward pass"] = time.perf_counter() - started
        score = float(scores[message_id])

        # Counterfactual: the same GNN on the same message with its links to other
        # messages deleted (degree features recomputed by the research code).
        score_without_links = None
        touches = relations["edge_type"].isin(SHARE_EDGE_TYPES) & (
            (relations["src_id"] == message_id) | (relations["dst_id"] == message_id)
        )
        if touches.any():
            started = time.perf_counter()
            isolated = GraphTables(events=tables.events, entities=tables.entities, relations=relations[~touches].reset_index(drop=True))
            isolated_scores, _ = self._score_tables(isolated, text_features, touched_ids, mode=mode)
            score_without_links = float(isolated_scores[message_id])
            timings["4. re-score with links removed"] = time.perf_counter() - started

        return self._build_analysis(
            "live", tables, message_id, score, scores, message_features, score_without_links, timings, path, None
        )

    def lookup(self, event_id: str) -> Analysis:
        """A message already in the graph (e.g. a held-out test record) with its GNN score."""
        message_id = f"message:{event_id}"
        if message_id not in self.base_scores.index:
            raise InvalidMessage(f"Unknown record '{event_id}'.")
        true_label = int(self.events_by_id.loc[message_id, "label"])
        return self._build_analysis(
            "held-out",
            self.tables,
            message_id,
            float(self.base_scores[message_id]),
            self.base_scores,
            self.base_features,
            None,
            {},
            "already in graph",
            true_label,
        )

    def _build_analysis(self, source, tables, message_id, score, scores, message_features, score_without_links, timings, path, true_label) -> Analysis:
        relations = tables.relations
        direct = relations[relations["src_id"] == message_id]
        entity_ids = direct[direct["dst_type"] != "message"]["dst_id"].tolist()
        second_hop = relations[
            relations["src_id"].isin(entity_ids)
            & relations["edge_type"].isin(["uses_phone", "uses_email", "resolves_to_domain"])
        ]
        subgraph = pd.concat([direct, second_hop], ignore_index=True)
        node_ids = set(entity_ids) | set(second_hop["dst_id"])
        entities = tables.entities[tables.entities["entity_id"].isin(node_ids)].reset_index(drop=True)

        links = direct[direct["dst_type"] == "message"]
        links = links[links["dst_id"] != message_id]
        events = tables.events.set_index("message_id")
        linked_rows = []
        for other_id, group in links.groupby("dst_id", sort=False):
            other = events.loc[other_id]
            linked_rows.append(
                {
                    "message_id": other_id,
                    "linked_by": ", ".join(sorted(group["edge_type"].unique())),
                    "channel": other["channel"],
                    "split": other["split"],
                    "true_label": int(other["label"]),
                    "gnn_score": float(scores[other_id]),
                    "text": str(other["text_or_url"]),
                }
            )
        linked_messages = pd.DataFrame(
            linked_rows, columns=["message_id", "linked_by", "channel", "split", "true_label", "gnn_score", "text"]
        )

        event = events.loc[message_id].to_dict()
        event["message_id"] = message_id
        feature_row = message_features.loc[message_id]
        tokens = _tokenize(str(event["text_or_url"]))
        return Analysis(
            source=source,
            event=event,
            message_id=message_id,
            score=score,
            prediction=int(score >= THRESHOLD),
            confidence=confidence_band(score),
            score_without_links=score_without_links,
            features={name: float(feature_row[name]) for name in BASIC_FEATURES if name in feature_row.index},
            urgent_keywords=sorted({token for token in tokens if token in URGENT_KEYWORDS}),
            relations=subgraph,
            entities=entities,
            linked_messages=linked_messages,
            timings=timings,
            path=path,
            true_label=true_label,
        )

    # ------------------------------------------------------------ test sample
    def sample_held_out(self, per_group: int = 3, random_state: int = 0) -> pd.DataFrame:
        """Random stratified draw from the held-out TEST split (never used for training)."""
        test = self.predictions[self.predictions["split"] == "test"]
        parts = [
            group.sample(n=min(per_group, len(group)), random_state=random_state)
            for _, group in test.groupby(["channel", "label"], sort=True)
        ]
        sample = pd.concat(parts)
        sample["text"] = sample["message_id"].map(self.events_by_id["text_or_url"])
        return sample.reset_index(drop=True)


if __name__ == "__main__":
    # Used by run_demo.bat to prepare the saved graph before the browser opens.
    demo_engine = DemoEngine(progress=print)
    for stage, seconds in demo_engine.load_timings.items():
        print(f"  {stage}: {seconds:.1f} s")
    print(f"Ready. Model check: largest score difference {demo_engine.max_score_difference:.1e}")
