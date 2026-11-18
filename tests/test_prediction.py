"""Checks of the trained system (needs  python -m scam_gnn.train  to have been run).

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scam_gnn.explain import explain  # noqa: E402
from scam_gnn.predict import InvalidInput, Predictor, SetupError  # noqa: E402


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.predictor = Predictor()
        except SetupError as error:
            raise unittest.SkipTest(str(error))
        cases = json.loads((ROOT / "app" / "test_cases.json").read_text(encoding="utf-8"))["cases"]
        cls.cases = [case for case in cases if case["channel"] in cls.predictor.channels]

    def test_reloaded_model_reproduces_training_scores(self) -> None:
        self.assertLess(self.predictor.max_score_difference, 1e-4)

    def test_fast_scoring_equals_full_graph_rebuild(self) -> None:
        """The demo adds a message to the stored graph incrementally; a complete rebuild must give the same score."""
        seen_channels = set()
        for case in self.cases:
            # one linked and one unlinked input per channel keeps this test to about a minute
            if case["id"] not in {"S1", "S3", "S4", "U1", "U3", "E1", "E2"}:
                continue
            seen_channels.add(case["channel"])
            with self.subTest(case=case["id"]):
                event = self.predictor.make_event(case["channel"], case["text"], case.get("sender", ""))
                for use_links in (True, False):
                    fast = self.predictor.score(event, use_links=use_links)
                    full = self.predictor.score(event, use_links=use_links, full_rebuild=True)
                    self.assertEqual(fast["path"], "incremental update")
                    self.assertAlmostEqual(fast["score"], full["score"], places=4)
        self.assertEqual(seen_channels, set(self.predictor.channels))

    def test_prediction_does_not_change_the_stored_graph(self) -> None:
        before = (len(self.predictor.tables.entities), len(self.predictor.tables.relations), float(self.predictor.x_dict["domain"].sum()))
        first = self.predictor.predict("sms", "WINNER!! Claim your free prize now at http://www.urawinner.com")
        second = self.predictor.predict("sms", "WINNER!! Claim your free prize now at http://www.urawinner.com")
        after = (len(self.predictor.tables.entities), len(self.predictor.tables.relations), float(self.predictor.x_dict["domain"].sum()))
        self.assertEqual(before, after)
        self.assertEqual(first.score, second.score)

    def test_explanation_removes_one_factor_at_a_time(self) -> None:
        prediction = self.predictor.predict("sms", "URGENT! You have won a prize. Call 08000839402 now or visit http://www.urawinner.com")
        names = {factor.name for factor in explain(self.predictor, prediction)}
        self.assertTrue({"Phone number in the text", "Link in the text", "Urgent words"} <= names)

    def test_invalid_input_is_rejected(self) -> None:
        for channel, text in [("sms", ""), ("sms", "   "), ("sms", "!!! ???"), ("sms", "x" * 5000), ("url", "not a link at all !!"), ("fax", "hello")]:
            with self.subTest(channel=channel, text=text[:12]):
                with self.assertRaises(InvalidInput):
                    self.predictor.predict(channel, text)

    def test_lookup_returns_the_saved_score(self) -> None:
        row = self.predictor.predictions[self.predictor.predictions["split"] == "test"].iloc[0]
        prediction = self.predictor.lookup(row["event_id"])
        self.assertAlmostEqual(prediction.score, row["score_gnn"], places=4)
        self.assertEqual(prediction.true_label, int(row["label"]))


if __name__ == "__main__":
    unittest.main()
