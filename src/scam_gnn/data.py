"""Stage 1 - Data: load the three datasets, clean them, remove duplicates, split.

Output of this stage is one table of *events*, one row per communication:

    event_id | channel | label | text | sender | recipient | split

``label`` is 1 for scam/spam/phishing and 0 for legitimate.  ``sender`` and
``recipient`` are empty when the dataset does not provide them (SMS and URL).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scam_gnn.config import Config
from scam_gnn.normalize import canonicalize_url, extract_email_address, normalize_whitespace

EVENT_COLUMNS = ["event_id", "channel", "label", "text", "sender", "recipient", "split"]
# Email bodies can be very long; the first part carries the content the model reads.
MAX_TEXT_CHARS = 2000


def _stratified_sample(frame: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    """At most ``limit`` rows, keeping the scam/legitimate proportion."""
    if not limit or len(frame) <= limit:
        return frame
    fraction = limit / len(frame)
    parts = [group.sample(n=int(round(len(group) * fraction)), random_state=seed) for _, group in frame.groupby("label")]
    return pd.concat(parts).sort_index()


def load_sms(config: Config) -> pd.DataFrame:
    """UCI SMS Spam Collection: text and label only (no sender, recipient or time)."""
    raw = pd.read_csv(config.path(config.sms_path), usecols=["text_or_url", "label"])
    return pd.DataFrame(
        {
            "channel": "sms",
            "label": raw["label"].astype(int),
            "text": raw["text_or_url"].fillna("").astype(str),
            "sender": "",
            "recipient": "",
        }
    )


def load_url(config: Config) -> pd.DataFrame:
    """PhiUSIIL phishing URL dataset: only the URL and its label are used."""
    raw = pd.read_csv(config.path(config.url_path), usecols=["text_or_url", "label"])
    frame = pd.DataFrame(
        {
            "channel": "url",
            "label": raw["label"].astype(int),
            "text": raw["text_or_url"].fillna("").astype(str).map(lambda value: canonicalize_url(value) or ""),
            "sender": "",
            "recipient": "",
        }
    )
    return _stratified_sample(frame, config.max_url_rows, config.data_seed)


def load_email(config: Config) -> pd.DataFrame:
    """CEAS 2008 email corpus: sender, receiver, subject, body and label."""
    raw = pd.read_csv(config.path(config.email_path), usecols=["sender", "receiver", "subject", "body", "label"])
    subject = raw["subject"].fillna("").astype(str)
    body = raw["body"].fillna("").astype(str)
    frame = pd.DataFrame(
        {
            "channel": "email",
            "label": raw["label"].astype(int),
            "text": (subject + ". " + body).str.slice(0, MAX_TEXT_CHARS),
            "sender": raw["sender"].map(extract_email_address),
            "recipient": raw["receiver"].map(extract_email_address),
        }
    )
    return _stratified_sample(frame, config.max_email_rows, config.data_seed)


def remove_duplicates(events: pd.DataFrame) -> pd.DataFrame:
    """Drop repeated messages so the same text cannot be in both training and test data."""
    key = events["channel"] + "|" + events["text"].str.lower()
    return events[~key.duplicated()].reset_index(drop=True)


def assign_splits(events: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Random 60/20/20 split, done separately for every (channel, label) group.

    None of the three datasets has usable timestamps for all rows, so a random
    split with a fixed seed is used.  Splitting within each group keeps the
    scam rate of every channel the same in training, validation and test data.
    """
    rng = np.random.default_rng(config.data_seed)
    split = pd.Series("train", index=events.index, dtype=object)
    for _, group in events.groupby(["channel", "label"]):
        order = rng.permutation(group.index.to_numpy())
        train_end = int(len(order) * config.train_ratio)
        validation_end = int(len(order) * (config.train_ratio + config.validation_ratio))
        split.loc[order[train_end:validation_end]] = "validation"
        split.loc[order[validation_end:]] = "test"
    result = events.copy()
    result["split"] = split
    return result


def load_events(config: Config) -> tuple[pd.DataFrame, dict]:
    """All three channels as one events table, plus a report of what cleaning did."""
    report: dict[str, dict[str, int]] = {}
    frames = []
    for channel, loader in (("sms", load_sms), ("email", load_email), ("url", load_url)):
        if not getattr(config, f"{channel}_path"):
            continue  # a channel can be switched off by leaving its path empty
        frame = loader(config)
        frame["text"] = frame["text"].map(normalize_whitespace)
        loaded = len(frame)
        frame = frame[frame["text"] != ""]
        frame = remove_duplicates(frame)
        frame.insert(0, "event_id", [f"{channel}-{index}" for index in range(len(frame))])
        report[channel] = {
            "rows_loaded": loaded,
            "duplicates_or_empty_removed": loaded - len(frame),
            "rows_used": len(frame),
            "scam": int(frame["label"].sum()),
            "legitimate": int((frame["label"] == 0).sum()),
        }
        frames.append(frame)
    events = assign_splits(pd.concat(frames, ignore_index=True), config)
    return events[EVENT_COLUMNS], report


def make_event(channel: str, text: str, sender: str = "", recipient: str = "", event_id: str = "new-message") -> pd.DataFrame:
    """One new, unlabelled communication in the same format as the datasets."""
    text = normalize_whitespace(text)
    if channel == "url":
        text = canonicalize_url(text) or ""
    return pd.DataFrame(
        [
            {
                "event_id": event_id,
                "channel": channel,
                "label": 0,  # placeholder: never used as a model input
                "text": text[:MAX_TEXT_CHARS],
                "sender": extract_email_address(sender),
                "recipient": extract_email_address(recipient),
                "split": "new",
            }
        ]
    )
