from __future__ import annotations

import numpy as np
import pandas as pd

from scam_gnn.graph import GraphTables


def derive_entity_risk_scores(tables: GraphTables, message_scores: np.ndarray) -> dict[str, pd.DataFrame]:
    message_frame = tables.events.drop_duplicates("message_id")[["message_id", "event_id", "label"]].copy()
    message_frame["score"] = message_scores
    sent_by = tables.relations[tables.relations["edge_type"] == "sent_by"][["src_id", "dst_id"]].rename(
        columns={"src_id": "message_id", "dst_id": "entity_id"}
    )
    contains_url = tables.relations[tables.relations["edge_type"] == "contains_url"][["src_id", "dst_id"]].rename(
        columns={"src_id": "message_id", "dst_id": "url_id"}
    )
    resolves = tables.relations[tables.relations["edge_type"] == "resolves_to_domain"][["src_id", "dst_id"]].rename(
        columns={"src_id": "url_id", "dst_id": "entity_id"}
    )
    sender_scores = (
        sent_by.merge(message_frame[["message_id", "score"]], on="message_id", how="left")
        .groupby("entity_id")["score"]
        .agg(["mean", "max", "count"])
        .rename(columns={"mean": "risk_mean", "max": "risk_max", "count": "linked_messages"})
        .sort_values("risk_mean", ascending=False)
    )
    domain_scores = (
        contains_url.merge(resolves, on="url_id", how="inner")
        .merge(message_frame[["message_id", "score"]], on="message_id", how="left")
        .groupby("entity_id")["score"]
        .agg(["mean", "max", "count"])
        .rename(columns={"mean": "risk_mean", "max": "risk_max", "count": "linked_messages"})
        .sort_values("risk_mean", ascending=False)
    )
    return {"sender": sender_scores, "domain": domain_scores}


def summarize_error_patterns(events: pd.DataFrame, scores: np.ndarray, *, threshold: float = 0.5) -> dict[str, dict[str, object]]:
    frame = events.drop_duplicates("message_id").copy()
    frame["score"] = scores
    frame["prediction"] = (frame["score"] >= threshold).astype(int)
    sender_frequency = frame["sender_key"].value_counts()
    domain_by_message = frame["text_or_url"].astype(str).str.extract(r"https?://([^/\s]+)", expand=False).fillna("")
    domain_label_variety = pd.DataFrame({"domain": domain_by_message, "label": frame["label"]}).groupby("domain")["label"].nunique()
    cross_channel_presence = frame.groupby("sender_key")["channel"].nunique()

    false_positives = frame[(frame["prediction"] == 1) & (frame["label"] == 0)].copy()
    false_negatives = frame[(frame["prediction"] == 0) & (frame["label"] == 1)].copy()
    false_positives["domain"] = false_positives["text_or_url"].astype(str).str.extract(r"https?://([^/\s]+)", expand=False).fillna("")
    false_negatives["domain"] = false_negatives["text_or_url"].astype(str).str.extract(r"https?://([^/\s]+)", expand=False).fillna("")

    benign_shared_infrastructure = false_positives[
        false_positives["domain"].map(domain_label_variety).fillna(0) > 1
    ]
    first_seen_sender = false_negatives[false_negatives["sender_key"].map(sender_frequency).fillna(0) <= 1]
    missed_cross_channel = false_negatives[false_negatives["sender_key"].map(cross_channel_presence).fillna(0) > 1]

    return {
        "benign_shared_infrastructure_false_positives": {
            "count": int(len(benign_shared_infrastructure)),
            "sample_event_ids": benign_shared_infrastructure["event_id"].head(5).tolist(),
        },
        "first_seen_sender_false_negatives": {
            "count": int(len(first_seen_sender)),
            "sample_event_ids": first_seen_sender["event_id"].head(5).tolist(),
        },
        "cross_channel_campaign_false_negatives": {
            "count": int(len(missed_cross_channel)),
            "sample_event_ids": missed_cross_channel["event_id"].head(5).tolist(),
        },
    }
