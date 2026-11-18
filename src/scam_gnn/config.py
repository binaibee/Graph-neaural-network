"""All settings of the project in one place.

Values come from ``config.json`` in the repository root; anything missing there
falls back to the defaults below.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.json"

CHANNELS = ["sms", "email", "url"]


@dataclass
class Config:
    # Data
    sms_path: str = "data/public/uci_sms_spam_collection.csv"
    email_path: str = "data/public/spamassassin_email.csv"
    url_path: str = "data/public/phiusiil_url_collection.csv"
    max_email_rows: int = 10000
    max_url_rows: int = 15000
    data_seed: int = 7
    train_ratio: float = 0.6
    validation_ratio: float = 0.2

    # Text representation: "hashed" (bag of words) or "minilm" (sentence embeddings, needs a download)
    text_embedding: str = "hashed"
    minilm_path: str = "models/all-MiniLM-L6-v2"
    hashed_dim: int = 24

    # Graph
    max_group_size: int = 128
    max_urls_per_message: int = 10

    # GNN training
    seeds: list[int] = field(default_factory=lambda: [7, 11, 42])
    hidden_dim: int = 64
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    max_epochs: int = 100
    patience: int = 15

    results_dir: str = "results"

    def path(self, relative: str) -> Path:
        return ROOT / relative

    @property
    def results_path(self) -> Path:
        return ROOT / self.results_dir


def load_config(path: str | Path = CONFIG_PATH) -> Config:
    path = Path(path)
    if not path.exists():
        return Config()
    payload = json.loads(path.read_text(encoding="utf-8"))
    known = {item.name for item in fields(Config)}
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ValueError(f"Unknown setting(s) in {path.name}: {unknown}")
    return Config(**payload)
