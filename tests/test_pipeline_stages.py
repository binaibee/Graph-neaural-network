"""Fast checks of the individual stages (no trained model needed).

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scam_gnn.config import Config  # noqa: E402
from scam_gnn.data import assign_splits, remove_duplicates  # noqa: E402
from scam_gnn.evaluate import best_threshold, binary_metrics  # noqa: E402
from scam_gnn.features import Scaler, TextEncoder, entity_features, message_features  # noqa: E402
from scam_gnn.graph import build_graph  # noqa: E402
from scam_gnn.models import MODEL_EDGES, NODE_TYPES, ScamGNN, apply_variant, predict_gnn, to_tensors  # noqa: E402
from scam_gnn.normalize import extract_email_address, extract_phone_numbers, extract_urls, remove_phone_numbers, remove_urls  # noqa: E402


def sample_events() -> pd.DataFrame:
    rows = [
        ("sms-0", "sms", 1, "WIN a prize! Call 08001234567 now", "", ""),
        ("sms-1", "sms", 1, "Claim your cash, call 08001234567 today", "", ""),
        ("sms-2", "sms", 0, "See you at lunch", "", ""),
        ("email-0", "email", 1, "Verify your account at http://bad.example.net/login", "a@spam.example.net", "bob@uni.example.edu"),
        ("email-1", "email", 0, "Minutes of the meeting attached", "carol@uni.example.edu", "bob@uni.example.edu"),
        ("url-0", "url", 1, "http://bad.example.net/other", "", ""),
        ("url-1", "url", 0, "https://www.good.example.org/", "", ""),
    ]
    frame = pd.DataFrame(rows, columns=["event_id", "channel", "label", "text", "sender", "recipient"])
    frame["split"] = ["train", "validation", "train", "train", "test", "test", "train"]
    return frame


class NormalizeTests(unittest.TestCase):
    def test_links_phones_and_addresses_are_found(self) -> None:
        text = "Call +44 7700 900123 or visit https://Example.com/Offer. Code 12345."
        self.assertEqual(extract_phone_numbers(text), ["+447700900123"])
        self.assertEqual(extract_urls(text), ["https://example.com/Offer"])
        self.assertEqual(extract_email_address("Jane Doe <Jane@Example.org>"), "jane@example.org")

    def test_removal_keeps_the_rest_of_the_text(self) -> None:
        self.assertEqual(remove_phone_numbers("Call 08001234567 now, code 12345"), "Call now, code 12345")
        self.assertEqual(remove_urls("Go to http://x.example.com/a now"), "Go to now")


class DataTests(unittest.TestCase):
    def test_duplicates_are_removed_per_channel(self) -> None:
        frame = pd.DataFrame({"channel": ["sms", "sms", "url"], "text": ["Hello", "hello", "hello"], "label": [0, 0, 1]})
        self.assertEqual(len(remove_duplicates(frame)), 2)

    def test_split_keeps_every_group_in_every_part(self) -> None:
        frame = pd.DataFrame({"channel": ["sms"] * 100 + ["url"] * 100, "label": ([0] * 80 + [1] * 20) * 2, "text": [str(i) for i in range(200)]})
        split = assign_splits(frame, Config())
        counts = split.groupby(["channel", "label", "split"]).size()
        self.assertEqual(counts[("sms", 1, "train")], 12)
        self.assertEqual(counts[("sms", 1, "validation")], 4)
        self.assertEqual(counts[("sms", 1, "test")], 4)
        self.assertTrue(assign_splits(frame, Config())["split"].equals(split["split"]))  # reproducible


class GraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tables = build_graph(sample_events(), Config())

    def test_nodes_and_edges(self) -> None:
        types = self.tables.entities["entity_type"].value_counts().to_dict()
        self.assertEqual(types["message"], 7)
        self.assertEqual(types["phone"], 1)
        self.assertEqual(types["sender"], 2)
        edges = set(self.tables.relations["edge_type"])
        self.assertTrue({"sent_by", "sent_to", "contains_url", "contains_phone", "resolves_to_domain", "uses_domain"} <= edges)

    def test_messages_sharing_a_phone_or_domain_are_linked(self) -> None:
        relations = self.tables.relations
        pairs = set(zip(relations["src_id"], relations["dst_id"], relations["edge_type"]))
        self.assertIn(("message:sms-0", "message:sms-1", "shares_infrastructure"), pairs)
        # An email and a URL record on the same website: a cross-channel link.
        self.assertIn(("message:email-0", "message:url-0", "shares_infrastructure"), pairs)
        self.assertIn(("message:email-0", "message:email-1", "shares_recipient"), pairs)
        self.assertNotIn(("message:sms-2", "message:sms-0", "shares_infrastructure"), pairs)


class ModelTests(unittest.TestCase):
    def test_gnn_runs_on_every_ablation_variant(self) -> None:
        config = Config()
        tables = build_graph(sample_events(), config)
        features = entity_features(tables)
        features["message"] = message_features(tables.events, TextEncoder(config, "hashed").encode(tables.events["text"].tolist()))
        scaler = Scaler()
        for node_type, frame in features.items():
            scaler.fit(node_type, frame)
        x_dict, edge_index_dict, node_ids = to_tensors(tables, features, scaler)
        self.assertEqual(set(x_dict), set(NODE_TYPES))
        self.assertEqual(set(edge_index_dict), set(MODEL_EDGES))
        model = ScamGNN({node_type: x.shape[1] for node_type, x in x_dict.items()})
        for variant in ("full", "no_message_links", "no_edges"):
            scores = predict_gnn(model, x_dict, apply_variant(edge_index_dict, variant))
            self.assertEqual(len(scores), len(node_ids["message"]))
            self.assertTrue(np.all((scores >= 0) & (scores <= 1)))


class EvaluateTests(unittest.TestCase):
    def test_metrics_on_a_known_example(self) -> None:
        labels = np.array([1, 1, 1, 0, 0, 0])
        scores = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.1])
        metrics = binary_metrics(labels, scores, threshold=0.5)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]), (2, 1, 1, 2))
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 2 / 3)
        self.assertAlmostEqual(metrics["accuracy"], 4 / 6)

    def test_threshold_is_the_best_f1_cut(self) -> None:
        labels = np.array([1, 1, 0, 0])
        scores = np.array([0.9, 0.6, 0.4, 0.1])
        self.assertAlmostEqual(best_threshold(labels, scores), 0.6)


if __name__ == "__main__":
    unittest.main()
