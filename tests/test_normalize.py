from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scam_gnn.data.normalize import (  # noqa: E402
    canonicalize_domain,
    canonicalize_email,
    canonicalize_phone,
    canonicalize_url,
    extract_domain_from_url,
    extract_phone_numbers,
    extract_urls,
)


class NormalizeTests(unittest.TestCase):
    def test_phone_email_and_domain_canonicalization(self) -> None:
        self.assertEqual(canonicalize_phone("+234 801-234-5678"), "+2348012345678")
        self.assertEqual(canonicalize_email(" User@Example.ORG "), "user@example.org")
        self.assertEqual(canonicalize_domain("https://WWW.Example.ORG/path"), "www.example.org")

    def test_url_extraction_and_domain_parsing(self) -> None:
        text = "Visit https://secure-login-alert.net/session/1234 and ignore noise."
        urls = extract_urls(text)
        self.assertEqual(urls, ["https://secure-login-alert.net/session/1234"])
        self.assertEqual(canonicalize_url("secure-login-alert.net/session/1234"), "https://secure-login-alert.net/session/1234")
        self.assertEqual(extract_domain_from_url(urls[0]), "secure-login-alert.net")

    def test_phone_extraction(self) -> None:
        text = "Call +234 801 234 5678 or 0803-111-2222 now."
        self.assertEqual(extract_phone_numbers(text), ["+2348012345678", "08031112222"])


if __name__ == "__main__":
    unittest.main()
