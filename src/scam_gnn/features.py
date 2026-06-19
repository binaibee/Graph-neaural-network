from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scam_gnn.config import FeatureConfig
from scam_gnn.data.normalize import extract_domain_from_url, extract_urls
from scam_gnn.dependencies import optional_import
from scam_gnn.graph import GraphTables

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
URGENT_KEYWORDS = {"urgent", "verify", "warning", "refund", "blocked", "suspend", "reset", "failed", "confirm"}
_SENTENCE_TRANSFORMER_CACHE: dict[tuple[str, bool], object | None] = {}


@dataclass(slots=True)
class FeatureBundle:
    message_features: pd.DataFrame
    entity_features: dict[str, pd.DataFrame]


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall((text or "").lower())


def _hash_embedding(texts: list[str], dimension: int) -> np.ndarray:
    matrix = np.zeros((len(texts), dimension), dtype=np.float32)
    for row_index, text in enumerate(texts):
        tokens = _tokenize(text)
        if not tokens:
            continue
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % dimension
            sign = -1.0 if digest[4] % 2 else 1.0
            matrix[row_index, bucket] += sign
        norm = np.linalg.norm(matrix[row_index])
        if norm > 0:
            matrix[row_index] /= norm
    return matrix


def _load_sentence_transformer(model_name: str):
    sentence_transformers = optional_import("sentence_transformers")
    if sentence_transformers is None:
        return None
    offline = os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    allow_remote = os.environ.get("SCAM_GNN_ALLOW_REMOTE_EMBEDDINGS") == "1"
    local_only = offline or not allow_remote
    cache_key = (model_name, local_only)
    if cache_key in _SENTENCE_TRANSFORMER_CACHE:
        return _SENTENCE_TRANSFORMER_CACHE[cache_key]
    if local_only and not Path(model_name).exists():
        _SENTENCE_TRANSFORMER_CACHE[cache_key] = None
        return None
    try:
        kwargs = {"local_files_only": local_only}
        model = sentence_transformers.SentenceTransformer(model_name, **kwargs)
    except Exception:
        model = None
    _SENTENCE_TRANSFORMER_CACHE[cache_key] = model
    return model


def encode_texts(texts: list[str], feature_config: FeatureConfig) -> np.ndarray:
    model = _load_sentence_transformer(feature_config.embedding_model)
    if model is not None:
        try:
            return np.asarray(model.encode(texts, convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32)
        except Exception:
            pass
    if not feature_config.include_text_hashing_fallback:
        raise RuntimeError(
            "Sentence-transformers embeddings were requested, but the configured model could not be loaded."
        )
    return _hash_embedding(texts, feature_config.fallback_embedding_dim)


def build_message_feature_frame(events: pd.DataFrame, feature_config: FeatureConfig) -> pd.DataFrame:
    messages = events.drop_duplicates("message_id").copy()
    texts = messages["text_or_url"].fillna("").astype(str).tolist()
    embeddings = encode_texts(texts, feature_config)
    lengths = np.array([len(text) for text in texts], dtype=np.float32)
    digit_counts = np.array([sum(char.isdigit() for char in text) for text in texts], dtype=np.float32)
    upper_counts = np.array([sum(char.isupper() for char in text) for text in texts], dtype=np.float32)
    url_counts = np.array([len(extract_urls(text)) for text in texts], dtype=np.float32)
    urgent_counts = np.array(
        [sum(token in URGENT_KEYWORDS for token in _tokenize(text)) for text in texts],
        dtype=np.float32,
    )
    frame = pd.DataFrame(index=messages["message_id"])
    frame["length"] = lengths
    frame["digit_ratio"] = np.divide(digit_counts, np.maximum(lengths, 1.0))
    frame["uppercase_ratio"] = np.divide(upper_counts, np.maximum(lengths, 1.0))
    frame["url_count"] = url_counts
    frame["urgent_keyword_count"] = urgent_counts
    for channel in sorted(messages["channel"].unique()):
        frame[f"channel_{channel}"] = (messages["channel"] == channel).astype(np.float32).to_numpy()
    excluded_columns = {
        "event_id",
        "timestamp",
        "channel",
        "label",
        "text_or_url",
        "sender_key",
        "recipient_key",
        "split",
        "message_id",
        "source_dataset",
        "has_native_timestamp",
    }
    extra_numeric = messages.drop(columns=[column for column in excluded_columns if column in messages.columns], errors="ignore")
    extra_numeric = extra_numeric.select_dtypes(include=["number", "bool"])
    if not extra_numeric.empty:
        extra_numeric = extra_numeric.fillna(0.0).astype(np.float32)
        for column in extra_numeric.columns:
            frame[f"extra_{column}"] = extra_numeric[column].to_numpy(dtype=np.float32)
    for index in range(embeddings.shape[1]):
        frame[f"text_emb_{index:03d}"] = embeddings[:, index]
    return frame


def _message_graph_stats(tables: GraphTables) -> pd.DataFrame:
    message_ids = tables.events["message_id"].drop_duplicates().tolist()
    stats = pd.DataFrame(index=message_ids)
    outgoing = tables.relations.groupby("src_id").size()
    incoming = tables.relations.groupby("dst_id").size()
    stats["out_degree"] = pd.Series(message_ids, index=message_ids).map(outgoing).fillna(0.0)
    stats["in_degree"] = pd.Series(message_ids, index=message_ids).map(incoming).fillna(0.0)
    infrastructure = tables.relations[tables.relations["edge_type"].isin(["contains_url", "shares_infrastructure"])]
    infra_counts = infrastructure.groupby("src_id").size()
    stats["infrastructure_degree"] = pd.Series(message_ids, index=message_ids).map(infra_counts).fillna(0.0)
    return stats.astype(np.float32)


def _entity_feature_frame(entity_type: str, tables: GraphTables) -> pd.DataFrame:
    entities = tables.entities[tables.entities["entity_type"] == entity_type].copy()
    frame = pd.DataFrame(index=entities["entity_id"])
    out_counts = tables.relations.groupby("src_id").size()
    in_counts = tables.relations.groupby("dst_id").size()
    frame["out_degree"] = entities["entity_id"].map(out_counts).fillna(0.0).to_numpy(dtype=np.float32)
    frame["in_degree"] = entities["entity_id"].map(in_counts).fillna(0.0).to_numpy(dtype=np.float32)
    frame["value_length"] = entities["entity_value"].astype(str).map(len).to_numpy(dtype=np.float32)
    if entity_type == "domain":
        frame["dot_count"] = entities["entity_value"].astype(str).map(lambda value: value.count(".")).to_numpy(dtype=np.float32)
    elif entity_type == "url":
        frame["slash_count"] = entities["entity_value"].astype(str).map(lambda value: value.count("/")).to_numpy(dtype=np.float32)
        frame["domain_length"] = entities["entity_value"].astype(str).map(
            lambda value: len(extract_domain_from_url(value) or "")
        ).to_numpy(dtype=np.float32)
    return frame


def build_feature_bundle(tables: GraphTables, feature_config: FeatureConfig) -> FeatureBundle:
    message_features = build_message_feature_frame(tables.events, feature_config)
    if feature_config.include_graph_statistics:
        message_features = message_features.join(_message_graph_stats(tables), how="left").fillna(0.0)
    entity_features = {
        entity_type: _entity_feature_frame(entity_type, tables)
        for entity_type in tables.entities["entity_type"].drop_duplicates().tolist()
    }
    entity_features["message"] = message_features
    return FeatureBundle(message_features=message_features, entity_features=entity_features)
