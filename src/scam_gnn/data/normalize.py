from __future__ import annotations

import re
from urllib.parse import urlparse, urlunparse

URL_PATTERN = re.compile(r"https?://[^\s<>()]+", re.IGNORECASE)
NON_DIGIT_PATTERN = re.compile(r"\D+")
WHITESPACE_PATTERN = re.compile(r"\s+")
PHONE_PATTERN = re.compile(r"(?:(?:\+?\d[\d\-\s().]{6,}\d))")


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
    if raw.startswith("+"):
        return f"+{digits}"
    return digits


def canonicalize_email(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = normalize_whitespace(value).lower()
    return cleaned or None


def canonicalize_domain(value: str | None) -> str | None:
    if value is None:
        return None
    domain = normalize_whitespace(value).lower()
    domain = domain.removeprefix("http://").removeprefix("https://")
    domain = domain.split("/")[0]
    domain = domain.strip(".")
    return domain or None


def canonicalize_url(value: str | None) -> str | None:
    if value is None:
        return None
    raw = normalize_whitespace(value)
    if not raw:
        return None
    if "://" not in raw:
        raw = f"https://{raw}"
    parsed = urlparse(raw)
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
    return canonicalize_domain(urlparse(canonical_url).netloc)


def extract_urls(text: str | None) -> list[str]:
    if not text:
        return []
    return [canonicalize_url(match.group(0)) for match in URL_PATTERN.finditer(text)]


def extract_phone_numbers(text: str | None) -> list[str]:
    if not text:
        return []
    numbers: list[str] = []
    for match in PHONE_PATTERN.finditer(text):
        normalized = canonicalize_phone(match.group(0))
        if normalized:
            numbers.append(normalized)
    return numbers


def infer_address_type(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = normalize_whitespace(value)
    if not cleaned:
        return None
    if "@" in cleaned:
        return "email_address"
    digits = NON_DIGIT_PATTERN.sub("", cleaned)
    if digits and len(digits) >= 7:
        return "phone"
    return None
