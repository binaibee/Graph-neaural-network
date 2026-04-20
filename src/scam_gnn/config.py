from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class DatasetConfig:
    kind: str = "synthetic"
    output_dir: str = "artifacts"
    sms_path: str | None = None
    email_path: str | None = None
    url_path: str | None = None
    fdb_key: str | None = None
    max_sms_rows: int | None = None
    max_email_rows: int | None = None
    max_url_rows: int | None = None


@dataclass(slots=True)
class FeatureConfig:
    embedding_model: str = "all-MiniLM-L6-v2"
    fallback_embedding_dim: int = 24
    include_graph_statistics: bool = True
    include_text_hashing_fallback: bool = True


@dataclass(slots=True)
class SplitConfig:
    train_ratio: float = 0.6
    validation_ratio: float = 0.2
    test_ratio: float = 0.2


@dataclass(slots=True)
class AblationConfig:
    include_cross_channel_edges: bool = True
    include_infrastructure_nodes: bool = True


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    track: str = "custom"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    seeds: list[int] = field(default_factory=lambda: [7, 11, 42])
    primary_metric: str = "pr_auc"
    target_precision: float = 0.9

    @property
    def output_dir(self) -> Path:
        return Path(self.dataset.output_dir)


def load_config(path: str | Path) -> ExperimentConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ExperimentConfig(
        name=payload["name"],
        track=payload.get("track", "custom"),
        dataset=DatasetConfig(**payload.get("dataset", {})),
        features=FeatureConfig(**payload.get("features", {})),
        splits=SplitConfig(**payload.get("splits", {})),
        ablation=AblationConfig(**payload.get("ablation", {})),
        seeds=list(payload.get("seeds", [7, 11, 42])),
        primary_metric=payload.get("primary_metric", "pr_auc"),
        target_precision=float(payload.get("target_precision", 0.9)),
    )
