from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scam_gnn.config import ExperimentConfig  # noqa: E402
from scam_gnn.data.standardize import validate_event_table  # noqa: E402
from scam_gnn.graph import build_graph_tables  # noqa: E402
from scam_gnn.pipeline import (  # noqa: E402
    _align_message_scores,
    _evaluate_scores,
    _message_node_order,
)


def _build_tables_with_divergent_ordering():
    """Event ids whose alphabetical order differs from timestamp order.

    Timestamp order:    m-2, m-10, m-1
    Alphabetical order: m-1, m-10, m-2  (the order PyG message nodes use)
    """
    base = pd.Timestamp("2024-01-01")
    frame = validate_event_table(
        pd.DataFrame(
            [
                {"event_id": "m-2", "timestamp": base, "channel": "sms", "label": 1, "text_or_url": "a", "sender_key": "s1", "recipient_key": "r1", "split": "test"},
                {"event_id": "m-10", "timestamp": base + pd.Timedelta(minutes=1), "channel": "sms", "label": 1, "text_or_url": "b", "sender_key": "s2", "recipient_key": "r2", "split": "test"},
                {"event_id": "m-1", "timestamp": base + pd.Timedelta(minutes=2), "channel": "sms", "label": 0, "text_or_url": "c", "sender_key": "s3", "recipient_key": "r3", "split": "test"},
            ]
        )
    )
    frame["split"] = "test"
    return build_graph_tables(frame)


class ScoreAlignmentTests(unittest.TestCase):
    def test_message_node_order_differs_from_evaluation_order(self) -> None:
        tables = _build_tables_with_divergent_ordering()
        messages = tables.events.drop_duplicates("message_id").reset_index(drop=True)
        eval_order = list(messages["message_id"])
        node_order = _message_node_order(tables)
        # Precondition: the two orderings genuinely diverge, otherwise the bug
        # this test guards against could not occur.
        self.assertNotEqual(eval_order, node_order)

    def test_alignment_reorders_scores_back_to_message_frame(self) -> None:
        tables = _build_tables_with_divergent_ordering()
        messages = tables.events.drop_duplicates("message_id").reset_index(drop=True)
        node_order = _message_node_order(tables)

        # Simulate a "perfect" GNN: each message node scored with its own label,
        # emitted in PyG node order (alphabetical).
        label_by_id = dict(zip(messages["message_id"], messages["label"]))
        node_scores = np.array([float(label_by_id[mid]) for mid in node_order])

        aligned = _align_message_scores(node_scores, node_order, messages["message_id"])
        expected = messages["label"].to_numpy(dtype=float)
        np.testing.assert_array_equal(aligned, expected)

        # And without alignment the naive positional match is wrong, proving the
        # alignment step is load-bearing.
        self.assertFalse(np.array_equal(node_scores, expected))

    def test_evaluate_scores_perfect_after_alignment(self) -> None:
        tables = _build_tables_with_divergent_ordering()
        messages = tables.events.drop_duplicates("message_id").reset_index(drop=True)
        node_order = _message_node_order(tables)
        label_by_id = dict(zip(messages["message_id"], messages["label"]))
        node_scores = np.array([float(label_by_id[mid]) for mid in node_order])

        aligned = _align_message_scores(node_scores, node_order, messages["message_id"])
        config = ExperimentConfig(name="t")
        metrics = _evaluate_scores(messages, aligned, config)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["pr_auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
