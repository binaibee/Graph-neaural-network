"""Stage 2 - Graph construction: turn the events table into nodes and edges.

Node types
    message    one SMS, email or URL record (the thing that gets classified)
    sender     an email sender address
    recipient  an email recipient address
    url        a link found in a message
    domain     the website of a link, or the mail domain of a sender
    phone      a phone number found in the text of a message

Edge types
    sent_by             message -> sender
    sent_to             message -> recipient
    contains_url        message -> url
    contains_phone      message -> phone
    resolves_to_domain  url     -> domain
    uses_domain         sender  -> domain
    shares_sender / shares_recipient / shares_infrastructure
                        message <-> message that has the same sender, the same
                        recipient, or the same phone number or web domain
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import pandas as pd

from scam_gnn.config import Config
from scam_gnn.normalize import canonicalize_url, extract_domain_from_url, extract_phone_numbers, extract_urls

SHARE_EDGE_TYPES = ["shares_sender", "shares_recipient", "shares_infrastructure"]
ENTITY_COLUMNS = ["entity_id", "entity_type", "entity_value"]
RELATION_COLUMNS = ["src_id", "src_type", "dst_id", "dst_type", "edge_type"]


@dataclass
class GraphTables:
    events: pd.DataFrame  # the events table plus a ``message_id`` column
    entities: pd.DataFrame  # one row per node
    relations: pd.DataFrame  # one row per edge


def entity_id(entity_type: str, value: str) -> str:
    return f"{entity_type}:{value}"


def message_links(channel: str, text: str, config: Config) -> tuple[list[str], list[str]]:
    """Links and phone numbers a message refers to."""
    if channel == "url":
        url = canonicalize_url(text)
        return ([url] if url else []), []
    urls = list(dict.fromkeys(extract_urls(text)))[: config.max_urls_per_message]
    phones = list(dict.fromkeys(extract_phone_numbers(text)))
    return urls, phones


def build_graph(events: pd.DataFrame, config: Config) -> GraphTables:
    entities: dict[str, tuple[str, str, str]] = {}
    relations: list[tuple[str, str, str, str, str]] = []
    groups: dict[str, dict[str, list[str]]] = {name: {} for name in SHARE_EDGE_TYPES}

    def node(entity_type: str, value: str) -> str:
        key = entity_id(entity_type, value)
        entities.setdefault(key, (key, entity_type, value))
        return key

    def edge(src: str, src_type: str, dst: str, dst_type: str, edge_type: str) -> None:
        relations.append((src, src_type, dst, dst_type, edge_type))

    message_ids = []
    for row in events.itertuples(index=False):
        message = node("message", row.event_id)
        message_ids.append(message)

        sender = str(row.sender or "").strip().lower()
        if sender:
            sender_node = node("sender", sender)
            edge(message, "message", sender_node, "sender", "sent_by")
            groups["shares_sender"].setdefault(sender, []).append(message)
            if "@" in sender:
                edge(sender_node, "sender", node("domain", sender.split("@", 1)[1]), "domain", "uses_domain")

        recipient = str(row.recipient or "").strip().lower()
        if recipient:
            edge(message, "message", node("recipient", recipient), "recipient", "sent_to")
            groups["shares_recipient"].setdefault(recipient, []).append(message)

        urls, phones = message_links(row.channel, row.text, config)
        shared_keys = set()
        for url in urls:
            url_node = node("url", url)
            edge(message, "message", url_node, "url", "contains_url")
            domain = extract_domain_from_url(url)
            if domain:
                edge(url_node, "url", node("domain", domain), "domain", "resolves_to_domain")
                shared_keys.add(f"domain:{domain}")
        for phone in phones:
            edge(message, "message", node("phone", phone), "phone", "contains_phone")
            shared_keys.add(f"phone:{phone}")
        for key in shared_keys:
            groups["shares_infrastructure"].setdefault(key, []).append(message)

    # Message-to-message edges. Very large groups (for example a mailing-list
    # address or a popular website) say little about any single message and
    # would add millions of edges, so they are skipped.
    for edge_type, members_by_key in groups.items():
        for members in members_by_key.values():
            if not 2 <= len(members) <= config.max_group_size:
                continue
            for left, right in combinations(members, 2):
                edge(left, "message", right, "message", edge_type)
                edge(right, "message", left, "message", edge_type)

    entity_frame = pd.DataFrame(list(entities.values()), columns=ENTITY_COLUMNS)
    relation_frame = pd.DataFrame(relations, columns=RELATION_COLUMNS).drop_duplicates(["src_id", "dst_id", "edge_type"])
    event_frame = events.reset_index(drop=True).copy()
    event_frame["message_id"] = message_ids
    return GraphTables(
        events=event_frame,
        entities=entity_frame.sort_values(["entity_type", "entity_id"]).reset_index(drop=True),
        relations=relation_frame.sort_values(["edge_type", "src_id", "dst_id"]).reset_index(drop=True),
    )


def graph_summary(tables: GraphTables) -> dict:
    """Numbers that describe how connected the graph is."""
    relations = tables.relations
    between_messages = relations[(relations["src_type"] == "message") & (relations["dst_type"] == "message")]
    channel = tables.events.set_index("message_id")["channel"]
    linked = set(between_messages["src_id"])
    cross = between_messages[between_messages["src_id"].map(channel) != between_messages["dst_id"].map(channel)]
    per_channel = {}
    for name, frame in tables.events.groupby("channel"):
        is_linked = frame["message_id"].isin(linked)
        per_channel[name] = {
            "messages": int(len(frame)),
            "linked_to_another_message": int(is_linked.sum()),
            "scam_rate_linked": float(frame[is_linked]["label"].mean()) if is_linked.any() else None,
            "scam_rate_not_linked": float(frame[~is_linked]["label"].mean()) if (~is_linked).any() else None,
        }
    return {
        "nodes": int(len(tables.entities)),
        "edges": int(len(relations)),
        "nodes_by_type": tables.entities["entity_type"].value_counts().to_dict(),
        "edges_by_type": relations["edge_type"].value_counts().to_dict(),
        "message_to_message_edges": int(len(between_messages)),
        "cross_channel_message_edges": int(len(cross)),
        "per_channel": per_channel,
    }
