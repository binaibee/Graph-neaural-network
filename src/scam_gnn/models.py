"""Stage 4 - Models: two baselines that ignore the graph, and the graph neural network.

Baselines (see only the content features of each message):
    logistic regression       the simplest standard classifier
    gradient-boosted trees    a strong standard classifier for tabular data

Graph model:
    heterogeneous GraphSAGE   sees the same content features *and* the graph
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv

from scam_gnn.config import Config
from scam_gnn.features import Scaler
from scam_gnn.graph import SHARE_EDGE_TYPES, GraphTables

NODE_TYPES = ["message", "sender", "recipient", "url", "domain", "phone"]
# Every edge is stored in both directions; "rev_" is the direction that brings
# information *towards* the node on the right.
SENDER_EDGES = [("domain", "rev_uses_domain", "sender")]
URL_EDGES = [("domain", "rev_resolves_to_domain", "url")]
MESSAGE_EDGES = [
    ("sender", "rev_sent_by", "message"),
    ("recipient", "rev_sent_to", "message"),
    ("url", "rev_contains_url", "message"),
    ("phone", "rev_contains_phone", "message"),
    ("message", "shares_sender", "message"),
    ("message", "shares_recipient", "message"),
    ("message", "shares_infrastructure", "message"),
]
MODEL_EDGES = SENDER_EDGES + URL_EDGES + MESSAGE_EDGES
# Ablation variants: which edges the GNN is allowed to use.
GRAPH_VARIANTS = {
    "full": "all edges",
    "no_message_links": "without message-to-message edges",
    "no_edges": "no edges at all (content features only)",
}
BASELINES = {"logistic_regression": "Logistic regression", "gradient_boosted_trees": "Gradient-boosted trees"}
GNN_NAME = "Heterogeneous GraphSAGE"


class ScamGNN(nn.Module):
    """Message-centric heterogeneous GraphSAGE.

    1. Every node type has its own input layer that maps its features to
       ``hidden_dim`` numbers.
    2. Senders collect information from their mail domain, and links from their
       web domain.
    3. Twice, every message collects information from its sender, recipient,
       links, phone numbers and directly connected messages.
    4. A small classifier turns the message representation into two scores:
       legitimate and scam.
    """

    def __init__(self, input_dims: dict[str, int], hidden_dim: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.encoders = nn.ModuleDict({node_type: nn.Linear(input_dims[node_type], hidden_dim) for node_type in NODE_TYPES})
        self.sender_context = self._block(SENDER_EDGES, hidden_dim)
        self.url_context = self._block(URL_EDGES, hidden_dim)
        self.message_layers = nn.ModuleList([self._block(MESSAGE_EDGES, hidden_dim) for _ in range(layers)])
        self.message_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 2))

    @staticmethod
    def _block(edge_types, hidden_dim: int) -> HeteroConv:
        return HeteroConv({edge_type: SAGEConv(hidden_dim, hidden_dim) for edge_type in edge_types}, aggr="sum")

    @staticmethod
    def _pick(edge_index_dict, edge_types):
        return {edge_type: edge_index_dict[edge_type] for edge_type in edge_types}

    def forward(self, x_dict, edge_index_dict):
        hidden = {node_type: F.relu(self.encoders[node_type](x_dict[node_type])) for node_type in NODE_TYPES}
        for block, edge_types, target in ((self.sender_context, SENDER_EDGES, "sender"), (self.url_context, URL_EDGES, "url")):
            update = block(hidden, self._pick(edge_index_dict, edge_types))[target]
            hidden = {**hidden, target: self.dropout(F.relu(update + hidden[target]))}
        message_input = hidden["message"]
        for norm, layer in zip(self.message_norms, self.message_layers):
            update = layer(hidden, self._pick(edge_index_dict, MESSAGE_EDGES))["message"]
            hidden = {**hidden, "message": self.dropout(F.relu(norm(update + hidden["message"])))}
        return self.classifier(torch.cat([message_input, hidden["message"]], dim=-1))


# ------------------------------------------------------------------- tensors
def to_tensors(tables: GraphTables, features: dict[str, pd.DataFrame], scaler: Scaler):
    """Tables -> the numeric arrays the GNN consumes.

    Returns ``x_dict`` (scaled features per node type), ``edge_index_dict``
    (pairs of node positions per edge type, both directions) and ``node_ids``
    (which node sits at which position).
    """
    x_dict, node_ids, positions = {}, {}, {}
    for node_type in NODE_TYPES:
        ids = tables.entities.loc[tables.entities["entity_type"] == node_type, "entity_id"].tolist()
        node_ids[node_type] = ids
        positions[node_type] = pd.Series(np.arange(len(ids), dtype=np.int64), index=ids)
        frame = features[node_type].reindex(ids) if node_type in features else None
        if frame is None or not len(ids):
            x_dict[node_type] = torch.zeros((len(ids), len(scaler.mean[node_type])), dtype=torch.float32)
        else:
            x_dict[node_type] = torch.from_numpy(scaler.transform(node_type, frame.fillna(0.0)))

    edge_index_dict = {edge_type: torch.zeros((2, 0), dtype=torch.long) for edge_type in MODEL_EDGES}
    for (src_type, edge_type, dst_type), frame in tables.relations.groupby(["src_type", "edge_type", "dst_type"]):
        src = positions[src_type].reindex(frame["src_id"]).to_numpy(dtype=np.int64)
        dst = positions[dst_type].reindex(frame["dst_id"]).to_numpy(dtype=np.int64)
        # Message-to-message edges already exist in both directions.
        forward = (src_type, edge_type, dst_type)
        if forward in edge_index_dict:
            edge_index_dict[forward] = torch.from_numpy(np.stack([src, dst]))
        reverse = (dst_type, f"rev_{edge_type}", src_type)
        if reverse in edge_index_dict:
            edge_index_dict[reverse] = torch.from_numpy(np.stack([dst, src]))
    return x_dict, edge_index_dict, node_ids


def apply_variant(edge_index_dict, variant: str):
    """Edges the GNN may use in an ablation variant."""
    if variant == "full":
        return edge_index_dict
    empty = torch.zeros((2, 0), dtype=torch.long)
    if variant == "no_edges":
        return {edge_type: empty for edge_type in edge_index_dict}
    if variant == "no_message_links":
        return {edge_type: (empty if edge_type[1] in SHARE_EDGE_TYPES else edges) for edge_type, edges in edge_index_dict.items()}
    raise ValueError(f"Unknown graph variant '{variant}'.")


# ------------------------------------------------------------------ training
def balanced_sample_weights(events: pd.DataFrame) -> np.ndarray:
    """Give every (channel, label) group the same total weight during training.

    Without this the largest group (legitimate URLs) would dominate and the
    small groups (for example SMS scams) would be learned poorly.
    """
    train = events[events["split"] == "train"]
    group_size = train.groupby(["channel", "label"]).size()
    keys = list(zip(events["channel"], events["label"]))
    weights = np.array([1.0 / group_size.get(key, 1) for key in keys], dtype=np.float32)
    return weights / weights[(events["split"] == "train").to_numpy()].mean()


def predict_gnn(model: ScamGNN, x_dict, edge_index_dict) -> np.ndarray:
    """Probability of the scam class for every message node."""
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(x_dict, edge_index_dict), dim=-1)[:, 1].numpy()


def train_gnn(x_dict, edge_index_dict, events: pd.DataFrame, config: Config, seed: int) -> tuple[ScamGNN, dict]:
    """Train on the training messages; keep the version that is best on the validation messages.

    ``events`` must be in the same order as the message nodes.
    """
    torch.manual_seed(seed)
    labels = torch.from_numpy(events["label"].to_numpy(dtype=np.int64))
    weights = torch.from_numpy(balanced_sample_weights(events))
    train_mask = torch.from_numpy((events["split"] == "train").to_numpy())
    validation = (events["split"] == "validation").to_numpy()
    y_validation = events["label"].to_numpy()[validation]

    model = ScamGNN({node_type: x.shape[1] for node_type, x in x_dict.items()}, hidden_dim=config.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_state, best_score, best_epoch, stale = None, -1.0, 0, 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x_dict, edge_index_dict)
        losses = F.cross_entropy(logits[train_mask], labels[train_mask], reduction="none")
        loss = (losses * weights[train_mask]).sum() / weights[train_mask].sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        score = average_precision_score(y_validation, predict_gnn(model, x_dict, edge_index_dict)[validation])
        if score > best_score:
            best_state, best_score, best_epoch, stale = copy.deepcopy(model.state_dict()), score, epoch, 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_epoch": best_epoch, "epochs_run": epoch, "validation_pr_auc": float(best_score)}


def train_baseline(name: str, x_train: np.ndarray, y_train: np.ndarray, sample_weight: np.ndarray, seed: int):
    if name == "logistic_regression":
        model = LogisticRegression(max_iter=2000)
    elif name == "gradient_boosted_trees":
        model = HistGradientBoostingClassifier(random_state=seed)
    else:
        raise ValueError(f"Unknown baseline '{name}'.")
    return model.fit(x_train, y_train, sample_weight=sample_weight)
