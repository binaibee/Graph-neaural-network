from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scam_gnn.pipeline import _interleave_synthetic_timestamps  # noqa: E402


class MultichannelTimestampTests(unittest.TestCase):
    def test_interleaves_public_frames_without_native_timestamps(self) -> None:
        sms = pd.DataFrame(
            [
                {"event_id": "sms-0", "timestamp": "2024-01-01", "channel": "sms", "label": 0, "text_or_url": "a", "sender_key": "s0", "recipient_key": "r0", "has_native_timestamp": False},
                {"event_id": "sms-1", "timestamp": "2024-01-01", "channel": "sms", "label": 1, "text_or_url": "b", "sender_key": "s1", "recipient_key": "r1", "has_native_timestamp": False},
            ]
        )
        url = pd.DataFrame(
            [
                {"event_id": "url-0", "timestamp": "2024-06-01", "channel": "url", "label": 1, "text_or_url": "https://x", "sender_key": "u0", "recipient_key": "d0", "has_native_timestamp": False},
                {"event_id": "url-1", "timestamp": "2024-06-01", "channel": "url", "label": 0, "text_or_url": "https://y", "sender_key": "u1", "recipient_key": "d1", "has_native_timestamp": False},
            ]
        )
        frames = _interleave_synthetic_timestamps([sms, url])
        self.assertEqual(len(frames), 1)
        combined = frames[0]
        self.assertEqual(combined["channel"].tolist(), ["sms", "url", "sms", "url"])

    def test_imbalanced_channels_remain_distributed(self) -> None:
        sms = pd.DataFrame(
            [
                {"event_id": f"sms-{index}", "timestamp": "2024-01-01", "channel": "sms", "label": 0, "text_or_url": "a", "sender_key": f"s{index}", "recipient_key": f"r{index}", "has_native_timestamp": False}
                for index in range(3)
            ]
        )
        url = pd.DataFrame(
            [
                {"event_id": f"url-{index}", "timestamp": "2024-06-01", "channel": "url", "label": 1, "text_or_url": "https://x", "sender_key": f"u{index}", "recipient_key": f"d{index}", "has_native_timestamp": False}
                for index in range(9)
            ]
        )
        combined = _interleave_synthetic_timestamps([sms, url])[0]
        tail_channels = combined.tail(4)["channel"].tolist()
        self.assertIn("sms", tail_channels)


if __name__ == "__main__":
    unittest.main()
