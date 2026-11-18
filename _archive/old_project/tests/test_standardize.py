from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scam_gnn.data.standardize import standardize_sms_dataframe, standardize_url_dataframe  # noqa: E402


class StandardizeTests(unittest.TestCase):
    def test_preserves_already_standardized_sms_frame(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "event_id": "sms-1",
                    "timestamp": "2024-01-01T00:00:00",
                    "channel": "sms",
                    "label": 1,
                    "text_or_url": "verify now",
                    "sender_key": "sender-a",
                    "recipient_key": "recipient-a",
                    "source_dataset": "demo",
                }
            ]
        )
        standardized = standardize_sms_dataframe(frame)
        self.assertEqual(standardized.loc[0, "text_or_url"], "verify now")
        self.assertEqual(standardized.loc[0, "sender_key"], "sender-a")
        self.assertEqual(standardized.loc[0, "source_dataset"], "demo")

    def test_preserves_already_standardized_url_frame(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "event_id": "url-1",
                    "timestamp": "2024-06-01T00:00:00",
                    "channel": "url",
                    "label": 1,
                    "text_or_url": "https://phish.example/login",
                    "sender_key": "url_source_unknown_1",
                    "recipient_key": "phish.example",
                    "URLLength": 33,
                }
            ]
        )
        standardized = standardize_url_dataframe(frame)
        self.assertEqual(standardized.loc[0, "text_or_url"], "https://phish.example/login")
        self.assertEqual(int(standardized.loc[0, "URLLength"]), 33)


if __name__ == "__main__":
    unittest.main()
