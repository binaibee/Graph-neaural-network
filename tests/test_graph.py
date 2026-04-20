from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scam_gnn.config import FeatureConfig, SplitConfig  # noqa: E402
from scam_gnn.data.splits import assign_time_based_splits  # noqa: E402
from scam_gnn.data.synthetic import generate_synthetic_multichannel_events  # noqa: E402
from scam_gnn.dependencies import optional_import  # noqa: E402
from scam_gnn.features import build_feature_bundle  # noqa: E402
from scam_gnn.graph import build_graph_tables, build_pyg_heterodata, build_pyg_homogeneous_data  # noqa: E402


class GraphTests(unittest.TestCase):
    def test_graph_tables_include_expected_relation_types(self) -> None:
        events = generate_synthetic_multichannel_events(seed=5, num_events=24)
        split_events = assign_time_based_splits(events, SplitConfig())
        tables = build_graph_tables(split_events)
        edge_types = set(tables.relations["edge_type"])
        self.assertTrue({"sent_by", "sent_to", "contains_url", "resolves_to_domain"}.issubset(edge_types))
        self.assertIn("message", set(tables.entities["entity_type"]))
        self.assertIn("domain", set(tables.entities["entity_type"]))

    def test_train_subset_does_not_include_future_timestamps(self) -> None:
        events = generate_synthetic_multichannel_events(seed=9, num_events=24)
        split_events = assign_time_based_splits(events, SplitConfig())
        train_events = split_events[split_events["split"] == "train"].copy()
        train_tables = build_graph_tables(train_events)
        self.assertLessEqual(train_tables.relations["timestamp"].max(), train_events["timestamp"].max())
        self.assertLess(train_events["timestamp"].max(), split_events[split_events["split"] == "test"]["timestamp"].min())

    def test_pyg_data_includes_message_sample_weights(self) -> None:
        if optional_import("torch") is None or optional_import("torch_geometric") is None:
            self.skipTest("PyG optional dependencies are not installed.")
        import os

        previous_offline = os.environ.get("HF_HUB_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            events = generate_synthetic_multichannel_events(seed=3, num_events=36)
            split_events = assign_time_based_splits(events, SplitConfig())
            tables = build_graph_tables(split_events)
            feature_bundle = build_feature_bundle(tables, FeatureConfig())
            homogeneous_data = build_pyg_homogeneous_data(feature_bundle.message_features, tables)
            hetero_data = build_pyg_heterodata(tables, feature_bundle.entity_features)
        finally:
            if previous_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = previous_offline
        self.assertEqual(int(homogeneous_data.sample_weight.shape[0]), int(homogeneous_data.y.shape[0]))
        self.assertEqual(int(hetero_data["message"].sample_weight.shape[0]), int(hetero_data["message"].y.shape[0]))
        self.assertTrue(bool((homogeneous_data.sample_weight > 0).all().item()))
        self.assertTrue(bool((hetero_data["message"].sample_weight > 0).all().item()))


if __name__ == "__main__":
    unittest.main()
