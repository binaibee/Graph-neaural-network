from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402

from scam_gnn.data.public_datasets import parse_phiusiil_dataframe, parse_uci_sms_spam_text  # noqa: E402


class PublicDatasetTests(unittest.TestCase):
    def test_parse_uci_sms_text(self) -> None:
        raw_text = (
            "ham\tHey, are we still meeting at 6?\n"
            "spam\tURGENT! Call +234 801 234 5678 or visit https://secure-login-alert.net/reset\n"
        )
        frame = parse_uci_sms_spam_text(raw_text)
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.loc[0, "label"], 0)
        self.assertEqual(frame.loc[1, "label"], 1)
        self.assertEqual(frame.loc[1, "sender_key"], "+2348012345678")
        self.assertEqual(frame.loc[0, "recipient_key"], "sms_recipient_unknown_0")
        self.assertEqual(frame.loc[0, "sender_key"], "sms_sender_unknown_0")
        self.assertFalse(bool(frame.loc[0, "has_native_timestamp"]))

    def test_parse_phiusiil_frame(self) -> None:
        raw_df = pd.DataFrame(
            [
                {"URL": "http://safe.example.org/path", "label": 1, "URLLength": 25, "Domain": "safe.example.org"},
                {"URL": "http://phish-login.co/reset", "label": 0, "URLLength": 22, "Domain": "phish-login.co"},
            ]
        )
        frame = parse_phiusiil_dataframe(raw_df)
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.loc[0, "channel"], "url")
        self.assertEqual(frame.loc[0, "label"], 0)
        self.assertEqual(frame.loc[1, "label"], 1)
        self.assertEqual(frame.loc[1, "recipient_key"], "phish-login.co")
        self.assertIn("URLLength", frame.columns)


if __name__ == "__main__":
    unittest.main()
