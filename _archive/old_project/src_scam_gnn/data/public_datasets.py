from __future__ import annotations

import io
import urllib.request
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from scam_gnn.data.normalize import canonicalize_domain, canonicalize_url, extract_domain_from_url, extract_phone_numbers, extract_urls

UCI_SMS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
UCI_PHIUSIIL_URL = "https://archive.ics.uci.edu/static/public/967/phiusiil%2Bphishing%2Burl%2Bdataset.zip"


def parse_uci_sms_spam_text(raw_text: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    for index, line in enumerate(raw_text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split("\t", 1)
        if len(parts) != 2:
            continue
        raw_label, message = parts
        label = 1 if raw_label.strip().lower() == "spam" else 0
        phones = extract_phone_numbers(message)
        urls = extract_urls(message)
        sender_key = phones[0] if phones else (urls[0] if urls else f"sms_sender_unknown_{index}")
        recipient_key = f"sms_recipient_unknown_{index}"
        records.append(
            {
                "event_id": f"sms-{index}",
                "timestamp": base_time + timedelta(minutes=index),
                "channel": "sms",
                "label": label,
                "text_or_url": message,
                "sender_key": sender_key,
                "recipient_key": recipient_key,
                "source_dataset": "uci_sms_spam_collection",
                "has_native_timestamp": False,
            }
        )
    return pd.DataFrame.from_records(records)


def prepare_uci_sms_spam_dataset(output_csv: str | Path, *, force: bool = False) -> Path:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        return output_path

    with urllib.request.urlopen(UCI_SMS_URL) as response:
        payload = response.read()
    archive = zipfile.ZipFile(io.BytesIO(payload))
    raw_text = archive.read("SMSSpamCollection").decode("utf-8", errors="replace")
    frame = parse_uci_sms_spam_text(raw_text)
    frame.to_csv(output_path, index=False)
    return output_path


def parse_phiusiil_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    url_column = next((column for column in ["URL", "url", "url_text"] if column in raw_df.columns), None)
    label_column = next((column for column in ["label", "Label", "Class", "class"] if column in raw_df.columns), None)
    domain_column = next((column for column in ["Domain", "domain"] if column in raw_df.columns), None)
    if url_column is None or label_column is None:
        raise ValueError("PhiUSIIL frame must contain URL and label columns.")

    base_time = datetime(2024, 6, 1, 0, 0, 0)
    records: list[dict[str, object]] = []
    for index, row_dict in enumerate(raw_df.to_dict(orient="records")):
        raw_url = row_dict[url_column]
        canonical_url_value = canonicalize_url(str(raw_url))
        raw_label = row_dict[label_column]
        # PhiUSIIL uses 1 for legitimate and 0 for phishing; invert for this project.
        label = 0 if int(raw_label) == 1 else 1
        domain = row_dict.get(domain_column) if domain_column else None
        domain = canonicalize_domain(str(domain)) if domain else extract_domain_from_url(canonical_url_value)
        record = {
            "event_id": f"url-{index}",
            "timestamp": base_time + timedelta(seconds=index),
            "channel": "url",
            "label": label,
            "text_or_url": canonical_url_value or str(raw_url),
            "sender_key": f"url_source_unknown_{index}",
            "recipient_key": domain or f"url_target_unknown_{index}",
            "source_dataset": "uci_phiusiil_url",
            "has_native_timestamp": False,
        }
        for key, value in row_dict.items():
            if key in {url_column, label_column, domain_column}:
                continue
            record[key] = value
        records.append(record)
    return pd.DataFrame.from_records(records)


def prepare_phiusiil_url_dataset(output_csv: str | Path, *, force: bool = False) -> Path:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        return output_path

    with urllib.request.urlopen(UCI_PHIUSIIL_URL) as response:
        payload = response.read()
    archive = zipfile.ZipFile(io.BytesIO(payload))
    csv_name = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
    raw_df = pd.read_csv(io.BytesIO(archive.read(csv_name)))
    frame = parse_phiusiil_dataframe(raw_df)
    frame.to_csv(output_path, index=False)
    return output_path
