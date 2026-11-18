"""Stage 3 - Features: describe every node with numbers.

Message nodes get *content* features (what the text looks like).  All other
node types get small *structural* features (how connected they are and what
their value looks like).  The same features are used for every channel.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from scam_gnn.config import CHANNELS, Config
from scam_gnn.graph import GraphTables
from scam_gnn.normalize import extract_phone_numbers, extract_urls

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
URGENT_KEYWORDS = {"urgent", "verify", "warning", "refund", "blocked", "suspend", "reset", "failed", "confirm"}
BASIC_FEATURES = ["length", "digit_ratio", "uppercase_ratio", "url_count", "phone_count", "urgent_keyword_count"]
DEGREE_FEATURES = ["out_degree", "in_degree", "value_length"]
ENTITY_FEATURES = {
    "sender": DEGREE_FEATURES,
    "recipient": DEGREE_FEATURES,
    "phone": DEGREE_FEATURES,
    "url": DEGREE_FEATURES + ["slash_count", "digit_count", "query_count", "is_https"],
    "domain": DEGREE_FEATURES + ["dot_count", "digit_count", "hyphen_count"],
}
# MiniLM reads at most 256 word pieces, so longer text is cut before encoding.
MAX_ENCODED_CHARS = 1000


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall((text or "").lower())


# ---------------------------------------------------------------- text vector
class TextEncoder:
    """Turns text into a fixed-length vector.

    ``minilm``  sentence embeddings from the all-MiniLM-L6-v2 model (384 numbers
                that capture the meaning of the text); loaded from a local folder.
    ``hashed``  a bag of words: every word is counted into one of ``hashed_dim`` slots.
    """

    def __init__(self, config: Config, backend: str | None = None):
        self.backend = backend or config.text_embedding
        self.hashed_dim = config.hashed_dim
        self._model = None
        if self.backend == "minilm":
            model_dir = config.path(config.minilm_path)
            if not model_dir.exists():
                raise FileNotFoundError(f"MiniLM model folder not found: {model_dir}. Run  python -m scam_gnn.download_model")
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(str(model_dir), device="cpu")
        elif self.backend != "hashed":
            raise ValueError(f"Unknown text_embedding '{self.backend}' (use 'minilm' or 'hashed').")

    @property
    def description(self) -> str:
        if self.backend == "minilm":
            return "MiniLM sentence embedding (384 numbers)"
        return f"hashed bag of words ({self.hashed_dim} numbers)"

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.backend == "minilm":
            clipped = [text[:MAX_ENCODED_CHARS] for text in texts]
            vectors = self._model.encode(clipped, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(vectors, dtype=np.float32)
        matrix = np.zeros((len(texts), self.hashed_dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                matrix[row, int.from_bytes(digest[:4], "big") % self.hashed_dim] += -1.0 if digest[4] % 2 else 1.0
            norm = np.linalg.norm(matrix[row])
            if norm > 0:
                matrix[row] /= norm
        return matrix

    def encode_cached(self, texts: list[str], cache_dir: Path) -> np.ndarray:
        """Encoding 30,000 texts with MiniLM takes minutes, so the result is kept on disk."""
        digest = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()[:16]
        cache_file = cache_dir / f"text_vectors_{self.backend}_{digest}.npy"
        if cache_file.exists():
            return np.load(cache_file)
        vectors = self.encode(texts)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, vectors)
        return vectors


# ------------------------------------------------------------ message features
def message_features(events: pd.DataFrame, vectors: np.ndarray) -> pd.DataFrame:
    """Content features of each message; ``vectors`` are the text vectors of the same rows."""
    texts = events["text"].fillna("").astype(str).tolist()
    lengths = np.array([len(text) for text in texts], dtype=np.float32)
    safe = np.maximum(lengths, 1.0)
    frame = pd.DataFrame(index=pd.Index(events["message_id"], name=None))
    frame["length"] = lengths
    frame["digit_ratio"] = np.array([sum(c.isdigit() for c in text) for text in texts], dtype=np.float32) / safe
    frame["uppercase_ratio"] = np.array([sum(c.isupper() for c in text) for text in texts], dtype=np.float32) / safe
    is_url = (events["channel"] == "url").to_numpy()
    frame["url_count"] = np.where(is_url, 1.0, [len(extract_urls(text)) for text in texts]).astype(np.float32)
    frame["phone_count"] = np.where(is_url, 0.0, [len(extract_phone_numbers(text)) for text in texts]).astype(np.float32)
    frame["urgent_keyword_count"] = np.array([sum(token in URGENT_KEYWORDS for token in tokenize(text)) for text in texts], dtype=np.float32)
    for channel in CHANNELS:
        frame[f"channel_{channel}"] = (events["channel"] == channel).to_numpy(dtype=np.float32)
    text_columns = pd.DataFrame(vectors, index=frame.index, columns=[f"text_{index:03d}" for index in range(vectors.shape[1])])
    return pd.concat([frame, text_columns], axis=1).astype(np.float32)


# ------------------------------------------------------------- entity features
def entity_features(tables: GraphTables, only_ids: set[str] | None = None) -> dict[str, pd.DataFrame]:
    """Structural features of sender, recipient, url, domain and phone nodes.

    With ``only_ids`` the features are computed just for those nodes (used when a
    single new message is added and only a handful of nodes change).
    """
    entities = tables.entities[tables.entities["entity_type"] != "message"]
    relations = tables.relations
    if only_ids is not None:
        entities = entities[entities["entity_id"].isin(only_ids)]
        relations = relations[relations["src_id"].isin(only_ids) | relations["dst_id"].isin(only_ids)]
    out_degree = relations.groupby("src_id").size()
    in_degree = relations.groupby("dst_id").size()
    result: dict[str, pd.DataFrame] = {}
    for entity_type in ENTITY_FEATURES:
        group = entities[entities["entity_type"] == entity_type]
        if only_ids is not None and group.empty:
            continue
        values = group["entity_value"].astype(str)
        frame = pd.DataFrame(index=pd.Index(group["entity_id"], name=None))
        frame["out_degree"] = group["entity_id"].map(out_degree).fillna(0.0).to_numpy()
        frame["in_degree"] = group["entity_id"].map(in_degree).fillna(0.0).to_numpy()
        frame["value_length"] = values.str.len().to_numpy()
        if entity_type == "url":
            frame["slash_count"] = values.str.count("/").to_numpy()
            frame["digit_count"] = values.str.count(r"\d").to_numpy()
            frame["query_count"] = values.str.count(r"[?=&]").to_numpy()
            frame["is_https"] = values.str.startswith("https://").to_numpy(dtype=np.float32)
        elif entity_type == "domain":
            frame["dot_count"] = values.str.count(r"\.").to_numpy()
            frame["digit_count"] = values.str.count(r"\d").to_numpy()
            frame["hyphen_count"] = values.str.count("-").to_numpy()
        result[entity_type] = frame[ENTITY_FEATURES[entity_type]].astype(np.float32)
    return result


# -------------------------------------------------------------------- scaling
@dataclass
class Scaler:
    """Puts every feature on a comparable scale (mean 0, spread 1).

    Message features are scaled with statistics of the *training* messages only.
    Structural features are counts with a few very large values, so they are
    log-transformed first.
    """

    mean: dict[str, np.ndarray] = field(default_factory=dict)
    std: dict[str, np.ndarray] = field(default_factory=dict)

    @staticmethod
    def _prepare(node_type: str, values: np.ndarray) -> np.ndarray:
        return values if node_type == "message" else np.log1p(values)

    def fit(self, node_type: str, frame: pd.DataFrame) -> None:
        values = self._prepare(node_type, frame.to_numpy(dtype=np.float32))
        if not len(values):  # a node type that does not occur in the data
            self.mean[node_type] = np.zeros(frame.shape[1], dtype=np.float32)
            self.std[node_type] = np.ones(frame.shape[1], dtype=np.float32)
            return
        self.mean[node_type] = values.mean(axis=0)
        self.std[node_type] = values.std(axis=0) + 1e-6

    def transform(self, node_type: str, frame: pd.DataFrame) -> np.ndarray:
        values = self._prepare(node_type, frame.to_numpy(dtype=np.float32))
        return ((values - self.mean[node_type]) / self.std[node_type]).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": {key: value.tolist() for key, value in self.mean.items()}, "std": {key: value.tolist() for key, value in self.std.items()}}

    @classmethod
    def from_dict(cls, payload: dict) -> "Scaler":
        return cls(
            mean={key: np.asarray(value, dtype=np.float32) for key, value in payload["mean"].items()},
            std={key: np.asarray(value, dtype=np.float32) for key, value in payload["std"].items()},
        )
