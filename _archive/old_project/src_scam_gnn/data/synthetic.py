from __future__ import annotations

import random
from datetime import datetime, timedelta

import pandas as pd

from scam_gnn.data.standardize import validate_event_table

BENIGN_SMS = [
    "Lunch at 1pm?",
    "Your package has shipped from https://store.example.org/orders",
    "Meeting moved to 4pm, please confirm.",
]
SCAM_SMS = [
    "Urgent: your bank account is limited. Verify now at {url}",
    "Final warning, claim your refund before suspension: {url}",
    "Your wallet is blocked. Reset immediately via {url}",
]
BENIGN_EMAIL = [
    "Quarterly report attached. Review before Friday.",
    "Invoice is available on the vendor portal: {url}",
    "Team offsite agenda and travel details attached.",
]
SCAM_EMAIL = [
    "Payment failed. Re-authenticate your account at {url}",
    "Mailbox storage exceeded. Keep access using {url}",
    "Payroll update required today. Confirm here: {url}",
]


def generate_synthetic_multichannel_events(
    *,
    seed: int = 7,
    num_events: int = 96,
    start_date: str = "2024-01-01",
) -> pd.DataFrame:
    rng = random.Random(seed)
    base = datetime.fromisoformat(start_date)
    benign_domains = ["portal.example.org", "vendor-payments.com", "docs.example.net"]
    malicious_domains = ["verify-wallet-support.co", "secure-login-alert.net", "payroll-confirm-fast.org"]
    shared_scam_senders = ["ops@secure-login-alert.net", "+2348012341111", "compliance@verify-wallet-support.co"]
    shared_recipients = ["+2348030001111", "finance@example.org", "ceo.office@example.org"]
    benign_senders = ["hr@example.org", "+2348022223333", "support@vendor-payments.com"]
    benign_recipients = ["+2348091112222", "staff@example.org", "ops@example.org"]
    records: list[dict[str, object]] = []
    for index in range(num_events):
        label = 1 if rng.random() < 0.35 else 0
        channel = rng.choices(["sms", "email", "url"], weights=[0.45, 0.4, 0.15], k=1)[0]
        timestamp = base + timedelta(hours=index * 3 + rng.randint(0, 2))
        if label:
            domain = rng.choice(malicious_domains)
            url = f"https://{domain}/session/{rng.randint(1000, 9999)}"
            sender = rng.choice(shared_scam_senders)
            recipient = rng.choice(shared_recipients)
            if channel == "sms":
                text = rng.choice(SCAM_SMS).format(url=url)
            elif channel == "email":
                text = rng.choice(SCAM_EMAIL).format(url=url)
            else:
                text = url
        else:
            domain = rng.choice(benign_domains)
            url = f"https://{domain}/view/{rng.randint(1000, 9999)}"
            sender = rng.choice(benign_senders)
            recipient = rng.choice(benign_recipients)
            if channel == "sms":
                text = rng.choice(BENIGN_SMS).format(url=url)
            elif channel == "email":
                text = rng.choice(BENIGN_EMAIL).format(url=url)
            else:
                text = url
        records.append(
            {
                "event_id": f"{channel}-{index}",
                "timestamp": timestamp,
                "channel": channel,
                "label": label,
                "text_or_url": text,
                "sender_key": sender,
                "recipient_key": recipient,
            }
        )
    return validate_event_table(pd.DataFrame.from_records(records))
