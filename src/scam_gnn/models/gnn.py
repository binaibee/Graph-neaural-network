from __future__ import annotations

from dataclasses import dataclass

from scam_gnn.dependencies import require_optional


@dataclass(slots=True)
class GNNResult:
    model_name: str
    scores: object
    train_seconds: float
    metadata: dict[str, object]


def _torch_modules():
    torch = require_optional("torch", "research")
    nn = require_optional("torch.nn", "research")
    functional = require_optional("torch.nn.functional", "research")
    tg_nn = require_optional("torch_geometric.nn", "research")
    return torch, nn, functional, tg_nn


def _available_edge_types(metadata, candidates):
    known_edge_types = set(metadata[1])
    return [edge_type for edge_type in candidates if edge_type in known_edge_types]


def _message_centric_edge_groups(metadata):
    sender_edge_types = _available_edge_types(
        metadata,
        [
            ("phone", "rev_uses_phone", "sender"),
            ("email_address", "rev_uses_email", "sender"),
        ],
    )
    recipient_edge_types = _available_edge_types(
        metadata,
        [
            ("phone", "rev_uses_phone", "recipient"),
            ("email_address", "rev_uses_email", "recipient"),
        ],
    )
    url_edge_types = _available_edge_types(
        metadata,
        [
            ("domain", "rev_resolves_to_domain", "url"),
        ],
    )
    message_edge_types = _available_edge_types(
        metadata,
        [
            ("sender", "rev_sent_by", "message"),
            ("recipient", "rev_sent_to", "message"),
            ("url", "rev_contains_url", "message"),
            ("message", "shares_sender", "message"),
            ("message", "shares_recipient", "message"),
            ("message", "shares_infrastructure", "message"),
        ],
    )
    return {
        "sender": sender_edge_types,
        "recipient": recipient_edge_types,
        "url": url_edge_types,
        "message": message_edge_types,
    }


def _message_centric_metadata(metadata):
    edge_groups = _message_centric_edge_groups(metadata)
    edge_types = []
    for group_name in ["sender", "recipient", "url", "message"]:
        for edge_type in edge_groups[group_name]:
            if edge_type not in edge_types:
                edge_types.append(edge_type)
    node_types = []
    for src_type, _, dst_type in edge_types:
        if src_type not in node_types:
            node_types.append(src_type)
        if dst_type not in node_types:
            node_types.append(dst_type)
    return edge_groups, (node_types, edge_types)


def build_homogeneous_model(model_name: str, input_dim: int, hidden_dim: int = 64):
    torch, nn, functional, tg_nn = _torch_modules()

    class HomogeneousClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            conv_cls = tg_nn.GCNConv if model_name == "gcn" else tg_nn.SAGEConv
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.conv1 = conv_cls(input_dim, hidden_dim)
            self.conv2 = conv_cls(hidden_dim, hidden_dim)
            self.dropout = nn.Dropout(0.2)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 2),
            )

        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            base = functional.relu(self.input_proj(x))
            x = self.conv1(x, edge_index)
            x = functional.relu(x)
            x = self.dropout(x)
            x = self.conv2(x, edge_index)
            x = functional.relu(x + base)
            x = self.dropout(x)
            return self.classifier(torch.cat([base, x], dim=-1))

    return HomogeneousClassifier()


def build_heterogeneous_graphsage(metadata, hidden_dim: int = 64, layers: int = 2):
    torch, nn, functional, tg_nn = _torch_modules()
    edge_groups, _ = _message_centric_metadata(metadata)

    class MessageCentricHeterogeneousGraphSAGE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoders = nn.ModuleDict({node_type: nn.LazyLinear(hidden_dim) for node_type in metadata[0]})
            self.dropout = nn.Dropout(0.2)

            self.sender_edge_types = edge_groups["sender"]
            self.recipient_edge_types = edge_groups["recipient"]
            self.url_edge_types = edge_groups["url"]
            self.message_edge_types = edge_groups["message"]

            self.sender_context = self._build_relation_block(self.sender_edge_types)
            self.recipient_context = self._build_relation_block(self.recipient_edge_types)
            self.url_context = self._build_relation_block(self.url_edge_types)
            self.message_convs = nn.ModuleList(
                [self._build_relation_block(self.message_edge_types) for _ in range(layers)]
            )
            self.message_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 2),
            )

        def _build_relation_block(self, edge_types):
            if not edge_types:
                return None
            return tg_nn.HeteroConv(
                {
                    edge_type: tg_nn.SAGEConv((-1, -1), hidden_dim)
                    for edge_type in edge_types
                },
                aggr="sum",
            )

        @staticmethod
        def _edge_subset(edge_index_dict, edge_types):
            return {edge_type: edge_index_dict[edge_type] for edge_type in edge_types if edge_type in edge_index_dict}

        @staticmethod
        def _node_subset(x_dict, edge_types):
            node_types = {src_type for src_type, _, _ in edge_types} | {dst_type for _, _, dst_type in edge_types}
            return {node_type: x_dict[node_type] for node_type in node_types if node_type in x_dict}

        def _apply_context_block(self, x_dict, edge_index_dict, conv, edge_types, target_type):
            if conv is None or not edge_types:
                return x_dict
            updates = conv(self._node_subset(x_dict, edge_types), self._edge_subset(edge_index_dict, edge_types))
            if target_type not in updates:
                return x_dict
            next_dict = dict(x_dict)
            next_dict[target_type] = self.dropout(functional.relu(updates[target_type] + x_dict[target_type]))
            return next_dict

        def forward(self, x_dict, edge_index_dict):
            x_dict = {key: functional.relu(self.encoders[key](value)) for key, value in x_dict.items()}
            x_dict = self._apply_context_block(
                x_dict,
                edge_index_dict,
                self.sender_context,
                self.sender_edge_types,
                "sender",
            )
            x_dict = self._apply_context_block(
                x_dict,
                edge_index_dict,
                self.recipient_context,
                self.recipient_edge_types,
                "recipient",
            )
            x_dict = self._apply_context_block(
                x_dict,
                edge_index_dict,
                self.url_context,
                self.url_edge_types,
                "url",
            )
            message_base = x_dict["message"]
            for norm, conv in zip(self.message_norms, self.message_convs):
                if conv is None:
                    continue
                updated = conv(
                    self._node_subset(x_dict, self.message_edge_types),
                    self._edge_subset(edge_index_dict, self.message_edge_types),
                )
                message_hidden = updated.get("message", x_dict["message"])
                message_hidden = norm(message_hidden + x_dict["message"])
                next_dict = dict(x_dict)
                next_dict["message"] = self.dropout(functional.relu(message_hidden))
                x_dict = next_dict
            return self.classifier(torch.cat([message_base, x_dict["message"]], dim=-1))

    return MessageCentricHeterogeneousGraphSAGE()


def build_heterogeneous_hgt(metadata, hidden_dim: int = 32, layers: int = 1, heads: int = 2):
    torch, nn, functional, tg_nn = _torch_modules()
    _, pruned_metadata = _message_centric_metadata(metadata)
    pruned_node_types, pruned_edge_types = pruned_metadata

    class MessageCentricHGT(nn.Module):
        def __init__(self):
            super().__init__()
            self.node_types = pruned_node_types
            self.edge_types = pruned_edge_types
            self.encoders = nn.ModuleDict({node_type: nn.LazyLinear(hidden_dim) for node_type in self.node_types})
            self.convs = nn.ModuleList(
                [tg_nn.HGTConv(hidden_dim, hidden_dim, pruned_metadata, heads=heads) for _ in range(layers)]
            )
            self.norms = nn.ModuleList(
                [nn.ModuleDict({node_type: nn.LayerNorm(hidden_dim) for node_type in self.node_types}) for _ in range(layers)]
            )
            self.dropout = nn.Dropout(0.2)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 2),
            )

        @staticmethod
        def _edge_subset(edge_index_dict, edge_types):
            return {edge_type: edge_index_dict[edge_type] for edge_type in edge_types if edge_type in edge_index_dict}

        @staticmethod
        def _node_subset(x_dict, node_types):
            return {node_type: x_dict[node_type] for node_type in node_types if node_type in x_dict}

        def forward(self, x_dict, edge_index_dict):
            x_dict = {
                node_type: functional.relu(self.encoders[node_type](value))
                for node_type, value in self._node_subset(x_dict, self.node_types).items()
            }
            message_base = x_dict["message"]
            edge_index_dict = self._edge_subset(edge_index_dict, self.edge_types)
            for norms, conv in zip(self.norms, self.convs):
                updated = conv(x_dict, edge_index_dict)
                next_dict = {}
                for node_type, base in x_dict.items():
                    value = updated.get(node_type)
                    if value is None:
                        value = base
                    value = norms[node_type](value)
                    next_dict[node_type] = self.dropout(functional.relu(value))
                x_dict = next_dict
            return self.classifier(torch.cat([message_base, x_dict["message"]], dim=-1))

    return MessageCentricHGT()
