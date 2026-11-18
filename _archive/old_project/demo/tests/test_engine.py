"""Checks that the demo layer agrees with the research pipeline.

Run from the repository root (takes several minutes, because each case is also
scored through a full rebuild of the graph with the original research code):

    python -m unittest discover -s demo\\tests -v
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

DEMO_DIR = Path(__file__).resolve().parents[1]
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import engine  # noqa: E402
from scam_gnn.features import build_feature_bundle  # noqa: E402
from scam_gnn.graph import build_pyg_heterodata  # noqa: E402


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = engine.DemoEngine()
        cls.cases = json.loads((DEMO_DIR / "sample_data" / "test_cases.json").read_text(encoding="utf-8"))["cases"]

    def test_reloaded_model_reproduces_training_scores(self) -> None:
        self.assertLess(self.engine.max_score_difference, 1e-5)

    def test_features_and_tensors_match_research_code(self) -> None:
        eng = self.engine
        reference = build_feature_bundle(eng.tables, eng.config.features)
        pd.testing.assert_frame_equal(reference.message_features, eng.base_features)
        data = build_pyg_heterodata(eng.tables, reference.entity_features)
        x_dict, edge_index_dict, _ = engine.build_inference_tensors(eng.tables, reference.entity_features)
        self.assertEqual(set(data.x_dict), set(x_dict))
        for node_type, x in data.x_dict.items():
            self.assertTrue(bool((x == x_dict[node_type]).all()), node_type)
        self.assertEqual(set(data.edge_index_dict), set(edge_index_dict))
        for edge_type, edge_index in data.edge_index_dict.items():
            self.assertTrue(bool((edge_index == edge_index_dict[edge_type]).all()), edge_type)

    def test_incremental_update_equals_full_rebuild(self) -> None:
        eng = self.engine
        # No links, shared phone number, shared link, and a cross-channel domain.
        for case in [case for case in self.cases if case["id"] in {"T1", "T3", "T4", "T9"}]:
            with self.subTest(case=case["id"]):
                event = eng.make_sms_event(case["text"])
                fast_tables, fast_path = eng._tables_with_event(event, mode="fast")
                full_tables, _ = eng._tables_with_event(event, mode="full")
                self.assertEqual(fast_path, "incremental graph update")
                pd.testing.assert_frame_equal(fast_tables.entities, full_tables.entities)
                pd.testing.assert_frame_equal(fast_tables.relations, full_tables.relations)
                fast = eng.analyze_sms(case["text"], mode="fast")
                full = eng.analyze_sms(case["text"], mode="full")
                self.assertAlmostEqual(fast.score, full.score, places=5)

    def test_analysis_leaves_base_graph_untouched(self) -> None:
        eng = self.engine
        before = (len(eng.tables.events), len(eng.tables.entities), len(eng.tables.relations))
        eng.analyze_sms("Call 08000839402 now to claim your prize")
        self.assertEqual(before, (len(eng.tables.events), len(eng.tables.entities), len(eng.tables.relations)))

    def test_invalid_input_is_rejected(self) -> None:
        for bad in ["", "   ", "!!! ???", "x" * (engine.MAX_MESSAGE_CHARS + 1)]:
            with self.assertRaises(engine.InvalidMessage):
                self.engine.analyze_sms(bad)

    def test_held_out_lookup_uses_saved_score(self) -> None:
        eng = self.engine
        row = eng.predictions[eng.predictions["split"] == "test"].iloc[0]
        analysis = eng.lookup(row["event_id"])
        self.assertTrue(np.isclose(analysis.score, row["score"], atol=1e-5))
        self.assertEqual(analysis.true_label, int(row["label"]))


if __name__ == "__main__":
    unittest.main()
