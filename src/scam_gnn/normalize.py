"""Small text helpers: cleaning, and finding links, phone numbers and email addresses."""
from __future__ import annotations

import re
from urllib.parse import urlparse, urlunparse

URL_PATTERN = re.compile(r"https?://[^\s<>()\"']+", re.IGNORECASE)
PHONE_PATTERN = re.compile(r"\+?\d[\d\-\s().]{6,}\d")
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
NON_DIGIT_PATTERN = re.compile(r"\D+")
WHITESPACE_PATTERN = re.compile(r"\s+")
# A phone number has between 9 and 15 digits; shorter digit runs are codes, prices or dates.
MIN_PHONE_DIGITS = 9
MAX_PHONE_DIGITS = 15


def normalize_whitespace(value: str | None) -> str:
    if not value:
        return ""
    return WHITESPACE_PATTERN.sub(" ", str(value)).strip()


def canonicalize_phone(value: str | None) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    digits = NON_DIGIT_PATTERN.sub("", raw)
    if not digits:
        return None
    return f"+{digits}" if raw.startswith("+") else digits


def canonicalize_domain(value: str | None) -> str | None:
    if value is None:
        return None
    domain = normalize_whitespace(value).lower()
    domain = domain.removeprefix("http://").removeprefix("https://")
    domain = domain.split("/")[0].strip(".")
    return domain or None


def canonicalize_url(value: str | None) -> str | None:
    if value is None:
        return None
    raw = normalize_whitespace(value)
    if not raw:
        return None
    if "://" not in raw:
        raw = f"https://{raw}"
    try:
        parsed = urlparse(raw)
    except ValueError:
        return None
    cleaned = parsed._replace(
        scheme=(parsed.scheme or "https").lower(),
        netloc=parsed.netloc.lower(),
        path=(parsed.path or "/").rstrip("/") or "/",
        params="",
        fragment="",
    )
    return urlunparse(cleaned)


def extract_domain_from_url(value: str | None) -> str | None:
    canonical_url = canonicalize_url(value)
    if canonical_url is None:
        return None
    try:
        return canonicalize_domain(urlparse(canonical_url).netloc)
    except ValueError:
        return None


def extract_urls(text: str | None) -> list[str]:
    if not text:
        return []
    urls = (canonicalize_url(match.group(0).rstrip(".,;:!?")) for match in URL_PATTERN.finditer(text))
    return [url for url in urls if url]


def extract_phone_numbers(text: str | None) -> list[str]:
    if not text:
        return []
    numbers: list[str] = []
    for match in PHONE_PATTERN.finditer(text):
        normalized = canonicalize_phone(match.group(0))
        if normalized and MIN_PHONE_DIGITS <= len(normalized.lstrip("+")) <= MAX_PHONE_DIGITS:
            numbers.append(normalized)
    return numbers


def extract_email_address(value: str | None) -> str:
    """'Jane Doe <jane@example.org>' -> 'jane@example.org' ('' when there is none)."""
    match = EMAIL_PATTERN.search(str(value or ""))
    return match.group(0).lower() if match else ""


def remove_urls(text: str) -> str:
    return normalize_whitespace(URL_PATTERN.sub(" ", text or ""))


def remove_phone_numbers(text: str) -> str:
    def _replace(match: re.Match) -> str:
        digits = NON_DIGIT_PATTERN.sub("", match.group(0))
        return " " if MIN_PHONE_DIGITS <= len(digits) <= MAX_PHONE_DIGITS else match.group(0)

    return normalize_whitespace(PHONE_PATTERN.sub(_replace, text or ""))
