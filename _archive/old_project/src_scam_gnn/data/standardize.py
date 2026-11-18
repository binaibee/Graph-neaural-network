from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable

import pandas as pd

from scam_gnn.data.normalize import normalize_whitespace

REQUIRED_EVENT_COLUMNS = [
    "event_id",
    "timestamp",
    "channel",
    "label",
    "text_or_url",
    "sender_key",
    "recipient_key",
]


def normalize_binary_label(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value > 0)
    cleaned = normalize_whitespace(str(value)).lower()
    positives = {"1", "true", "spam", "phishing", "fraud", "scam", "malicious"}
    return int(cleaned in positives)


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _series_or_default(df: pd.DataFrame, column: str | None, default_value: str) -> pd.Series:
    if column and column in df.columns:
        return df[column].fillna(default_value).astype(str)
    return pd.Series([default_value] * len(df), index=df.index, dtype="object")


def _timestamp_series(df: pd.DataFrame, column: str | None) -> pd.Series:
    if column and column in df.columns:
        return pd.to_datetime(df[column], errors="coerce").fillna(pd.Timestamp("2024-01-01"))
    base = datetime(2024, 1, 1)
    return pd.Series(
        [base + timedelta(minutes=index) for index in range(len(df))],
        index=df.index,
        dtype="datetime64[ns]",
    )


def standardize_records(
    df: pd.DataFrame,
    channel: str,
    *,
    text_candidates: list[str],
    label_candidates: list[str],
    sender_candidates: list[str],
    recipient_candidates: list[str],
    timestamp_candidates: list[str],
) -> pd.DataFrame:
    if set(REQUIRED_EVENT_COLUMNS).issubset(df.columns):
        preserved = df.copy()
        if "channel" in preserved.columns:
            preserved["channel"] = preserved["channel"].fillna(channel).astype(str)
        else:
            preserved["channel"] = channel
        return validate_event_table(preserved)

    text_col = _first_existing(df, text_candidates)
    label_col = _first_existing(df, label_candidates)
    sender_col = _first_existing(df, sender_candidates)
    recipient_col = _first_existing(df, recipient_candidates)
    timestamp_col = _first_existing(df, timestamp_candidates)
    standardized = pd.DataFrame(
        {
            "event_id": [f"{channel}-{index}" for index in range(len(df))],
            "timestamp": _timestamp_series(df, timestamp_col),
            "channel": channel,
            "label": df[label_col].map(normalize_binary_label) if label_col else 0,
            "text_or_url": _series_or_default(df, text_col, ""),
            "sender_key": _series_or_default(df, sender_col, f"{channel}_sender_unknown"),
            "recipient_key": _series_or_default(df, recipient_col, f"{channel}_recipient_unknown"),
        }
    )
    standardized["text_or_url"] = standardized["text_or_url"].map(normalize_whitespace)
    return standardized


def standardize_sms_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return standardize_records(
        df,
        "sms",
        text_candidates=["text_or_url", "text", "message", "body", "sms"],
        label_candidates=["label", "target", "class"],
        sender_candidates=["sender", "from", "phone", "sender_key"],
        recipient_candidates=["recipient", "to", "recipient_key"],
        timestamp_candidates=["timestamp", "date", "created_at"],
    )


def standardize_email_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return standardize_records(
        df,
        "email",
        text_candidates=["text_or_url", "text", "body", "content", "message"],
        label_candidates=["label", "target", "class"],
        sender_candidates=["sender", "from", "sender_email"],
        recipient_candidates=["recipient", "to", "recipient_email"],
        timestamp_candidates=["timestamp", "date", "created_at", "sent_at"],
    )


def standardize_url_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    standardized = standardize_records(
        df,
        "url",
        text_candidates=["text_or_url", "url", "link"],
        label_candidates=["label", "target", "class"],
        sender_candidates=["sender", "source"],
        recipient_candidates=["recipient", "target_entity"],
        timestamp_candidates=["timestamp", "date", "created_at"],
    )
    standardized.loc[:, "sender_key"] = standardized["sender_key"].replace("url_sender_unknown", "url_source_unknown")
    standardized.loc[:, "recipient_key"] = standardized["recipient_key"].replace(
        "url_recipient_unknown",
        "url_target_unknown",
    )
    return standardized


def validate_event_table(events: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_EVENT_COLUMNS if column not in events.columns]
    if missing:
        raise ValueError(f"Events table is missing required columns: {missing}")
    validated = events.copy()
    validated["timestamp"] = pd.to_datetime(validated["timestamp"], errors="raise")
    validated["label"] = validated["label"].map(normalize_binary_label)
    validated["text_or_url"] = validated["text_or_url"].map(normalize_whitespace)
    validated["sender_key"] = validated["sender_key"].fillna("").astype(str)
    validated["recipient_key"] = validated["recipient_key"].fillna("").astype(str)
    return validated.sort_values("timestamp").reset_index(drop=True)


def assemble_multichannel_events(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    events = pd.concat(list(frames), ignore_index=True)
    return validate_event_table(events)
