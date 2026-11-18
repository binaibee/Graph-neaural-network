"""Draws the local graph around one message (matplotlib, works offline)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
EDGE = "#c3c2b7"
# One fixed colour per node type; every node also carries its type as a text label.
NODE_COLORS = {
    "message": "#2a78d6",
    "sender": "#eb6834",
    "recipient": "#1baf7a",
    "phone": "#eda100",
    "email_address": "#e87ba4",
    "url": "#008300",
    "domain": "#4a3aa7",
}
MAX_LINKED_SHOWN = 4
# Plain-language names for the message-to-message edge types.
LINK_WORDS = {
    "shares_sender": "same sender",
    "shares_recipient": "same recipient",
    "shares_infrastructure": "same web domain",
}
ENTITY_ORDER = {"sender": 0, "url": 1, "recipient": 2}


def _shorten(value: str, limit: int = 34) -> str:
    value = str(value)
    # The datasets use placeholder names where the real sender or recipient is not known.
    if value.startswith(("sms_sender_unknown", "sms_recipient_unknown", "url_source_unknown", "url_target_unknown")):
        return "(not known)"
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _spread(count: int, gap: float = 1.0) -> list[float]:
    return [gap * ((count - 1) / 2 - index) for index in range(count)]


def draw_message_graph(analysis):
    """Returns a matplotlib figure of the message, its entities and its linked messages."""
    relations = analysis.relations
    entity_values = dict(zip(analysis.entities["entity_id"], analysis.entities["entity_value"]))
    entity_types = dict(zip(analysis.entities["entity_id"], analysis.entities["entity_type"]))
    focal = analysis.message_id

    direct = relations[(relations["src_id"] == focal) & (relations["dst_type"] != "message")]
    direct = direct.assign(order=direct["dst_type"].map(ENTITY_ORDER).fillna(3)).sort_values(["order", "dst_id"])
    second = relations[relations["src_id"] != focal]

    positions: dict[str, tuple[float, float]] = {focal: (0.0, 0.0)}
    first_ids = list(dict.fromkeys(direct["dst_id"]))
    for node_id, y in zip(first_ids, _spread(len(first_ids))):
        positions[node_id] = (1.0, y)
    second_ids = [node_id for node_id in dict.fromkeys(second["dst_id"]) if node_id not in positions]
    for node_id in second_ids:
        sources = [positions[src][1] for src in second[second["dst_id"] == node_id]["src_id"] if src in positions]
        positions[node_id] = (2.0, sum(sources) / len(sources) if sources else 0.0)

    linked = analysis.linked_messages.sort_values("gnn_score", ascending=False)
    shown = linked.head(MAX_LINKED_SHOWN)
    for message_id, y in zip(shown["message_id"], _spread(len(shown))):
        positions[message_id] = (-1.0, y)

    rows = max(len(first_ids), len(shown), 1)
    fig, ax = plt.subplots(figsize=(11, max(3.4, 1.15 * rows + 1.2)), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    def draw_edge(src, dst, label):
        (x1, y1), (x2, y2) = positions[src], positions[dst]
        ax.plot([x1, x2], [y1, y2], color=EDGE, linewidth=1.4, zorder=1)
        if not label:
            return
        ax.text(
            (x1 + x2) / 2, (y1 + y2) / 2, label, fontsize=8, color=INK_SECONDARY, ha="center", va="center", zorder=2,
            bbox={"boxstyle": "round,pad=0.15", "facecolor": SURFACE, "edgecolor": "none"},
        )

    for row in direct.itertuples():
        draw_edge(focal, row.dst_id, row.edge_type)
    for row in second.itertuples():
        if row.src_id in positions and row.dst_id in positions:
            draw_edge(row.src_id, row.dst_id, row.edge_type)
    for row in shown.itertuples():
        # The edge type goes in the node label: labels on the edges pile up where they converge.
        draw_edge(focal, row.message_id, "")

    def draw_node(node_id, node_type, label, *, focal_node=False):
        x, y = positions[node_id]
        ax.scatter(
            [x], [y], s=620 if focal_node else 300, color=NODE_COLORS.get(node_type, INK_SECONDARY),
            edgecolors=INK if focal_node else SURFACE, linewidths=2.2 if focal_node else 2, zorder=3,
        )
        ax.text(
            x, y - (0.30 if focal_node else 0.24), label, fontsize=8.5, color=INK, ha="center", va="top", zorder=4,
            fontweight="bold" if focal_node else "normal",
        )

    draw_node(focal, "message", f"THIS MESSAGE\nrisk {analysis.score:.0%}", focal_node=True)
    for node_id in first_ids + second_ids:
        node_type = entity_types.get(node_id, node_id.split(":", 1)[0])
        draw_node(node_id, node_type, f"{node_type}\n{_shorten(entity_values.get(node_id, node_id))}")
    for row in shown.itertuples():
        truth = "scam" if row.true_label == 1 else "legitimate"
        reasons = ", ".join(LINK_WORDS.get(name, name) for name in row.linked_by.split(", "))
        kind = "web link" if row.channel == "url" else str(row.channel).upper()
        draw_node(row.message_id, "message", f"earlier {kind} · known {truth}\n({reasons})")
    if len(linked) > len(shown):
        ax.text(
            -1.0, min(y for _, y in positions.values()) - 0.95, f"+ {len(linked) - len(shown)} more connected messages",
            fontsize=8.5, color=INK_SECONDARY, ha="center", va="top",
        )

    present = ["message"] + [t for t in NODE_COLORS if t != "message" and t in set(entity_types.values())]
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=9, markerfacecolor=NODE_COLORS[t], markeredgecolor=SURFACE, label=t)
        for t in present
    ]
    ax.legend(handles=handles, title="Node type", loc="upper right", frameon=False, fontsize=8.5, title_fontsize=8.5, labelcolor=INK)

    ys = [y for _, y in positions.values()]
    ax.set_xlim(-1.75, 2.75)
    ax.set_ylim(min(ys) - 1.35, max(ys) + 0.6)
    ax.axis("off")
    fig.tight_layout(pad=0.4)
    return fig
