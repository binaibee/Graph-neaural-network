from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from scam_gnn.data.normalize import (
    canonicalize_domain,
    canonicalize_email,
    canonicalize_phone,
    canonicalize_url,
    extract_domain_from_url,
    extract_urls,
    infer_address_type,
)
from scam_gnn.dependencies import require_optional


@dataclass(slots=True)
class GraphTables:
    events: pd.DataFrame
    entities: pd.DataFrame
    relations: pd.DataFrame


MAX_PAIRWISE_GROUP_SIZE = 128


def _entity_id(entity_type: str, entity_value: str) -> str:
    return f"{entity_type}:{entity_value}"


def _add_entity(
    records: list[dict[str, str]],
    seen: dict[str, dict[str, str]],
    entity_type: str,
    entity_value: str,
) -> str:
    entity_key = _entity_id(entity_type, str(entity_value))
    if entity_key not in seen:
        payload = {
            "entity_id": entity_key,
            "entity_type": entity_type,
            "entity_value": str(entity_value),
        }
        seen[entity_key] = payload
        records.append(payload)
    return entity_key


def _add_relation(
    records: list[dict[str, object]],
    src_id: str,
    src_type: str,
    dst_id: str,
    dst_type: str,
    edge_type: str,
    timestamp: pd.Timestamp,
) -> None:
    records.append(
        {
            "src_id": src_id,
            "src_type": src_type,
            "dst_id": dst_id,
            "dst_type": dst_type,
            "edge_type": edge_type,
            "timestamp": pd.Timestamp(timestamp),
        }
    )


def _balanced_message_sample_weights(message_events: pd.DataFrame) -> np.ndarray:
    ordered = message_events.copy()
    train_frame = ordered[ordered["split"] == "train"]
    if train_frame.empty:
        return np.ones(len(ordered), dtype=np.float32)
    group_counts = train_frame.groupby(["channel", "label"]).size()
    weights = []
    for row in ordered.itertuples():
        count = int(group_counts.get((row.channel, row.label), 0))
        weights.append(1.0 / count if count > 0 else 1.0)
    weights_array = np.asarray(weights, dtype=np.float32)
    train_mask = (ordered["split"] == "train").to_numpy()
    train_mean = float(weights_array[train_mask].mean()) if train_mask.any() else 1.0
    if train_mean > 0:
        weights_array /= train_mean
    return weights_array


def _message_pairs(
    groups: dict[str, list[tuple[str, str, pd.Timestamp]]],
    *,
    edge_type: str,
    include_cross_channel_edges: bool,
) -> list[dict[str, object]]:
    relation_records: list[dict[str, object]] = []
    for members in groups.values():
        if len(members) > MAX_PAIRWISE_GROUP_SIZE:
            continue
        for left, right in combinations(members, 2):
            left_id, left_channel, left_ts = left
            right_id, right_channel, right_ts = right
            if not include_cross_channel_edges and left_channel != right_channel:
                continue
            timestamp = min(left_ts, right_ts)
            relation_records.append(
                {
                    "src_id": left_id,
                    "src_type": "message",
                    "dst_id": right_id,
                    "dst_type": "message",
                    "edge_type": edge_type,
                    "timestamp": timestamp,
                }
            )
            relation_records.append(
                {
                    "src_id": right_id,
                    "src_type": "message",
                    "dst_id": left_id,
                    "dst_type": "message",
                    "edge_type": edge_type,
                    "timestamp": timestamp,
                }
            )
    return relation_records


def build_graph_tables(
    events: pd.DataFrame,
    *,
    include_cross_channel_edges: bool = True,
    include_infrastructure_nodes: bool = True,
) -> GraphTables:
    entity_records: list[dict[str, str]] = []
    seen_entities: dict[str, dict[str, str]] = {}
    relation_records: list[dict[str, object]] = []
    sender_groups: dict[str, list[tuple[str, str, pd.Timestamp]]] = {}
    recipient_groups: dict[str, list[tuple[str, str, pd.Timestamp]]] = {}
    infrastructure_groups: dict[str, list[tuple[str, str, pd.Timestamp]]] = {}
    enriched_events: list[dict[str, object]] = []

    for row in events.sort_values("timestamp").itertuples(index=False):
        message_id = _add_entity(entity_records, seen_entities, "message", row.event_id)
        enriched_events.append({**row._asdict(), "message_id": message_id})

        sender_key = str(row.sender_key or "").strip().lower()
        if sender_key:
            sender_id = _add_entity(entity_records, seen_entities, "sender", sender_key)
            _add_relation(relation_records, message_id, "message", sender_id, "sender", "sent_by", row.timestamp)
            sender_groups.setdefault(sender_key, []).append((message_id, row.channel, row.timestamp))
            address_type = infer_address_type(sender_key)
            if address_type == "phone":
                phone_id = _add_entity(entity_records, seen_entities, "phone", canonicalize_phone(sender_key))
                _add_relation(relation_records, sender_id, "sender", phone_id, "phone", "uses_phone", row.timestamp)
            elif address_type == "email_address":
                email_id = _add_entity(entity_records, seen_entities, "email_address", canonicalize_email(sender_key))
                _add_relation(
                    relation_records,
                    sender_id,
                    "sender",
                    email_id,
                    "email_address",
                    "uses_email",
                    row.timestamp,
                )

        recipient_key = str(row.recipient_key or "").strip().lower()
        if recipient_key:
            recipient_id = _add_entity(entity_records, seen_entities, "recipient", recipient_key)
            _add_relation(relation_records, message_id, "message", recipient_id, "recipient", "sent_to", row.timestamp)
            recipient_groups.setdefault(recipient_key, []).append((message_id, row.channel, row.timestamp))
            address_type = infer_address_type(recipient_key)
            if address_type == "phone":
                phone_id = _add_entity(entity_records, seen_entities, "phone", canonicalize_phone(recipient_key))
                _add_relation(
                    relation_records,
                    recipient_id,
                    "recipient",
                    phone_id,
                    "phone",
                    "uses_phone",
                    row.timestamp,
                )
            elif address_type == "email_address":
                email_id = _add_entity(entity_records, seen_entities, "email_address", canonicalize_email(recipient_key))
                _add_relation(
                    relation_records,
                    recipient_id,
                    "recipient",
                    email_id,
                    "email_address",
                    "uses_email",
                    row.timestamp,
                )

        if include_infrastructure_nodes:
            urls = extract_urls(row.text_or_url)
            if row.channel == "url":
                direct_url = canonicalize_url(row.text_or_url)
                if direct_url and direct_url not in urls:
                    urls.append(direct_url)
            for url in filter(None, urls):
                url_id = _add_entity(entity_records, seen_entities, "url", url)
                _add_relation(relation_records, message_id, "message", url_id, "url", "contains_url", row.timestamp)
                domain = extract_domain_from_url(url)
                if domain:
                    domain_id = _add_entity(entity_records, seen_entities, "domain", canonicalize_domain(domain))
                    _add_relation(relation_records, url_id, "url", domain_id, "domain", "resolves_to_domain", row.timestamp)
                    infrastructure_groups.setdefault(domain, []).append((message_id, row.channel, row.timestamp))

    relation_records.extend(
        _message_pairs(
            sender_groups,
            edge_type="shares_sender",
            include_cross_channel_edges=include_cross_channel_edges,
        )
    )
    relation_records.extend(
        _message_pairs(
            recipient_groups,
            edge_type="shares_recipient",
            include_cross_channel_edges=include_cross_channel_edges,
        )
    )
    if include_infrastructure_nodes:
        relation_records.extend(
            _message_pairs(
                infrastructure_groups,
                edge_type="shares_infrastructure",
                include_cross_channel_edges=include_cross_channel_edges,
            )
        )

    entities = pd.DataFrame.from_records(entity_records).sort_values(["entity_type", "entity_id"]).reset_index(drop=True)
    relations = (
        pd.DataFrame.from_records(relation_records)
        .drop_duplicates(subset=["src_id", "dst_id", "edge_type"])
        .sort_values(["timestamp", "edge_type", "src_id", "dst_id"])
        .reset_index(drop=True)
    )
    return GraphTables(events=pd.DataFrame.from_records(enriched_events), entities=entities, relations=relations)


def project_message_graph(tables: GraphTables) -> tuple[pd.DataFrame, dict[str, int]]:
    message_frame = tables.events[["message_id", "label", "channel", "split"]].drop_duplicates("message_id").copy()
    mapping = {message_id: index for index, message_id in enumerate(message_frame["message_id"])}
    edges = tables.relations[
        (tables.relations["src_type"] == "message") & (tables.relations["dst_type"] == "message")
    ][["src_id", "dst_id", "edge_type"]].copy()
    edges["src_index"] = edges["src_id"].map(mapping)
    edges["dst_index"] = edges["dst_id"].map(mapping)
    return edges.dropna().reset_index(drop=True), mapping


def build_pyg_heterodata(tables: GraphTables, feature_frames: dict[str, pd.DataFrame]):
    torch = require_optional("torch", "research")
    HeteroData = require_optional("torch_geometric.data", "research").HeteroData
    data = HeteroData()
    node_index_maps: dict[str, dict[str, int]] = {}

    for entity_type, frame in tables.entities.groupby("entity_type", sort=False):
        entity_ids = list(frame["entity_id"])
        node_index_maps[entity_type] = {entity_id: index for index, entity_id in enumerate(entity_ids)}
        features = feature_frames.get(entity_type)
        if features is None or features.empty:
            x = torch.zeros((len(entity_ids), 1), dtype=torch.float32)
        else:
            aligned = features.reindex(entity_ids).fillna(0.0)
            x = torch.tensor(aligned.to_numpy(dtype=np.float32), dtype=torch.float32)
        data[entity_type].x = x

    for (src_type, edge_type, dst_type), frame in tables.relations.groupby(["src_type", "edge_type", "dst_type"]):
        src = [node_index_maps[src_type][value] for value in frame["src_id"]]
        dst = [node_index_maps[dst_type][value] for value in frame["dst_id"]]
        data[(src_type, edge_type, dst_type)].edge_index = torch.tensor([src, dst], dtype=torch.long)
        reverse_name = f"rev_{edge_type}"
        data[(dst_type, reverse_name, src_type)].edge_index = torch.tensor([dst, src], dtype=torch.long)

    message_entities = tables.entities[tables.entities["entity_type"] == "message"]["entity_id"].tolist()
    message_events = tables.events.drop_duplicates("message_id").set_index("message_id")
    ordered_message_events = message_events.loc[message_entities].copy()
    data["message"].y = torch.tensor(ordered_message_events["label"].to_numpy(), dtype=torch.long)
    data["message"].sample_weight = torch.tensor(
        _balanced_message_sample_weights(ordered_message_events.reset_index()),
        dtype=torch.float32,
    )
    split_series = ordered_message_events["split"]
    data["message"].train_mask = torch.tensor((split_series == "train").to_numpy(), dtype=torch.bool)
    data["message"].val_mask = torch.tensor((split_series == "validation").to_numpy(), dtype=torch.bool)
    data["message"].test_mask = torch.tensor((split_series == "test").to_numpy(), dtype=torch.bool)
    return data


def build_pyg_homogeneous_data(message_features: pd.DataFrame, tables: GraphTables):
    torch = require_optional("torch", "research")
    Data = require_optional("torch_geometric.data", "research").Data
    projected_edges, mapping = project_message_graph(tables)
    ordered_message_ids = [message_id for message_id, _ in sorted(mapping.items(), key=lambda item: item[1])]
    message_events = tables.events.drop_duplicates("message_id").set_index("message_id")
    x = torch.tensor(
        message_features.reindex(ordered_message_ids).fillna(0.0).to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    edge_index = torch.tensor(
        projected_edges[["src_index", "dst_index"]].to_numpy(dtype=np.int64).T if not projected_edges.empty else np.zeros((2, 0), dtype=np.int64),
        dtype=torch.long,
    )
    data = Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor(message_events.loc[ordered_message_ids, "label"].to_numpy(), dtype=torch.long),
    )
    ordered_message_events = message_events.loc[ordered_message_ids].copy()
    data.sample_weight = torch.tensor(
        _balanced_message_sample_weights(ordered_message_events.reset_index()),
        dtype=torch.float32,
    )
    split_series = ordered_message_events["split"]
    data.train_mask = torch.tensor((split_series == "train").to_numpy(), dtype=torch.bool)
    data.val_mask = torch.tensor((split_series == "validation").to_numpy(), dtype=torch.bool)
    data.test_mask = torch.tensor((split_series == "test").to_numpy(), dtype=torch.bool)
    return data
