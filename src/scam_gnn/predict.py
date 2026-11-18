"""Stage 7 - Prediction: score one new communication with the trained GNN.

The GNN classifies *nodes of a graph*, so a new message cannot be scored on its
own.  It is first connected to the stored graph (the same way the training data
was), and then the trained model is run:

    new message -> event row -> nodes and edges -> features -> GNN -> risk score

Nothing is written back: every prediction starts from the same stored graph.
"""
from __future__ import annotations

import pickle
import re
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from scam_gnn.config import CHANNELS, Config, load_config
from scam_gnn.data import EVENT_COLUMNS, make_event
from scam_gnn.features import BASIC_FEATURES, Scaler, TextEncoder, entity_features, message_features, tokenize
from scam_gnn.graph import SHARE_EDGE_TYPES, GraphTables, build_graph, entity_id, message_links
from scam_gnn.models import MODEL_EDGES, NODE_TYPES, ScamGNN, predict_gnn, to_tensors
from scam_gnn.normalize import extract_domain_from_url
from scam_gnn.train import MODEL_FILE, PREDICTIONS_FILE, PREPARED_FILE

NEW_EVENT_ID = "new-message"
NEW_MESSAGE_ID = entity_id("message", NEW_EVENT_ID)
MAX_INPUT_CHARS = 2000
ENTITY_EDGE_TYPES = ["resolves_to_domain", "uses_domain"]  # entity -> entity edges shown around a message
# A host name such as www.example.com (optionally with a port), or an IP address.
DOMAIN_PATTERN = re.compile(r"^(?:[a-z0-9-]+\.)+[a-z0-9-]{2,}(?::\d+)?$")


class SetupError(RuntimeError):
    """The saved model or graph is missing; run ``python -m scam_gnn.train`` first."""


class InvalidInput(ValueError):
    """Input that cannot be analysed."""


@dataclass
class Prediction:
    channel: str
    text: str
    sender: str
    recipient: str
    message_id: str
    score: float  # probability of scam, between 0 and 1
    threshold: float  # decision line chosen on validation data
    relations: pd.DataFrame  # edges around the message
    entities: pd.DataFrame  # nodes around the message
    linked_messages: pd.DataFrame  # other messages connected to it
    features: dict[str, float]  # the basic content features of the message
    true_label: int | None = None
    seconds: float = 0.0
    details: dict = field(default_factory=dict)

    @property
    def is_scam(self) -> bool:
        return self.score >= self.threshold

    @property
    def label(self) -> str:
        return "SCAM" if self.is_scam else "LEGITIMATE"


class Predictor:
    def __init__(self, config: Config | None = None):
        self.config = config or load_config()
        results = self.config.results_path
        for name in (MODEL_FILE, PREPARED_FILE, PREDICTIONS_FILE):
            if not (results / name).exists():
                raise SetupError(f"Missing {self.config.results_dir}/{name}. Train the model first:  python -m scam_gnn.train")
        checkpoint = torch.load(results / MODEL_FILE, map_location="cpu", weights_only=False)
        with (results / PREPARED_FILE).open("rb") as handle:
            prepared = pickle.load(handle)

        self.tables = GraphTables(prepared["events"], prepared["entities"], prepared["relations"])
        self.features: dict[str, pd.DataFrame] = prepared["features"]
        self.scaler = Scaler.from_dict(checkpoint["scaler"])
        self.threshold = float(checkpoint["threshold"])
        self.config.hashed_dim = int(checkpoint["hashed_dim"])
        self.encoder = TextEncoder(self.config, checkpoint["text_embedding"])

        self.x_dict, self.edge_index_dict, node_ids = to_tensors(self.tables, self.features, self.scaler)
        self.positions = {node_type: {node: index for index, node in enumerate(ids)} for node_type, ids in node_ids.items()}
        self.model = ScamGNN(checkpoint["input_dims"], checkpoint["hidden_dim"])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.scores = pd.Series(predict_gnn(self.model, self.x_dict, self.edge_index_dict), index=node_ids["message"])

        # Integrity check: the reloaded model must reproduce the scores saved by training.
        self.predictions = pd.read_csv(results / PREDICTIONS_FILE)
        saved = self.predictions.set_index("message_id")["score_gnn"]
        self.max_score_difference = float((self.scores.reindex(saved.index) - saved).abs().max())

        self.events = self.tables.events.set_index("message_id")
        self.entity_ids = set(self.tables.entities["entity_id"])
        self.channels = [channel for channel in CHANNELS if channel in set(self.events["channel"])]
        # For every sender, recipient, phone number and domain: the messages that refer to it.
        relations = self.tables.relations
        direct = relations[relations["edge_type"].isin(["sent_by", "sent_to", "contains_phone"])]
        self.members = direct.groupby("dst_id")["src_id"].agg(list).to_dict()
        contains = relations[relations["edge_type"] == "contains_url"][["src_id", "dst_id"]]
        resolves = relations[relations["edge_type"] == "resolves_to_domain"][["src_id", "dst_id"]]
        via_url = contains.merge(resolves, left_on="dst_id", right_on="src_id", suffixes=("_message", "_domain"))
        self.members.update(via_url.groupby("dst_id_domain")["src_id_message"].agg(lambda ids: sorted(set(ids))).to_dict())

        train = self.events[self.events["split"] == "train"]
        reference = self.features["message"].loc[train.index, BASIC_FEATURES].copy()
        reference["channel"], reference["label"] = train["channel"].to_numpy(), train["label"].to_numpy()
        self.feature_reference = reference.groupby(["channel", "label"]).mean()

    # ------------------------------------------------------------------ input
    def make_event(self, channel: str, text: str, sender: str = "", recipient: str = "") -> pd.DataFrame:
        if channel not in self.channels:
            raise InvalidInput(f"Unknown channel '{channel}'. Choose one of: {', '.join(self.channels)}.")
        if len(str(text or "")) > MAX_INPUT_CHARS:
            raise InvalidInput(f"The text is too long ({len(text)} characters). The limit is {MAX_INPUT_CHARS}.")
        is_email = channel == "email"  # only the email dataset has sender and recipient addresses
        event = make_event(channel, text, sender if is_email else "", recipient if is_email else "", event_id=NEW_EVENT_ID)
        cleaned = event["text"].iloc[0]
        if not cleaned:
            raise InvalidInput("Please enter a web link to analyse." if channel == "url" else "Please enter a message to analyse.")
        if not tokenize(cleaned):
            raise InvalidInput("The text has no letters or digits, so there is nothing to analyse.")
        if channel == "url" and (" " in cleaned or not DOMAIN_PATTERN.match(extract_domain_from_url(cleaned) or "")):
            raise InvalidInput("That does not look like a web link. Example: https://www.example.com/page")
        return event

    # ---------------------------------------------------------------- scoring
    def _group_keys(self, event: pd.Series) -> list[str]:
        """Graph nodes through which the new message can be linked to stored messages."""
        keys = []
        if event["sender"]:
            keys.append(entity_id("sender", event["sender"]))
        if event["recipient"]:
            keys.append(entity_id("recipient", event["recipient"]))
        urls, phones = message_links(event["channel"], event["text"], self.config)
        keys += [entity_id("domain", domain) for domain in {extract_domain_from_url(url) for url in urls} if domain]
        keys += [entity_id("phone", phone) for phone in phones]
        return keys

    def _local_graph(self, event: pd.DataFrame) -> GraphTables | None:
        """The new message plus every stored message it can connect to, as a small graph.

        Returns ``None`` in the rare case where adding the message would push a
        group over the size limit; the caller then rebuilds the whole graph.
        """
        related: set[str] = set()
        for key in self._group_keys(event.iloc[0]):
            members = self.members.get(key, [])
            if len(members) == self.config.max_group_size:
                return None
            if len(members) < self.config.max_group_size:
                related.update(members)
        stored = self.events.loc[sorted(related)].reset_index()[EVENT_COLUMNS]
        return build_graph(pd.concat([stored, event], ignore_index=True), self.config)

    def score(self, event: pd.DataFrame, *, use_links: bool = True, full_rebuild: bool = False) -> dict:
        """Risk score of a new event.  ``use_links=False`` ignores its links to other messages."""
        local = None if full_rebuild else self._local_graph(event)
        if local is None:
            return self._score_by_full_rebuild(event, use_links)

        new_entities = local.entities[~local.entities["entity_id"].isin(self.entity_ids)]
        new_ids = set(new_entities["entity_id"])
        added = local.relations[local.relations["src_id"].isin(new_ids) | local.relations["dst_id"].isin(new_ids)]
        if not use_links:
            added = added[~added["edge_type"].isin(SHARE_EDGE_TYPES)]

        # Features: the new nodes, plus stored nodes whose number of edges changed.
        new_event = local.events[local.events["event_id"] == NEW_EVENT_ID]
        vector = self.encoder.encode(new_event["text"].tolist())
        new_features = entity_features(GraphTables(new_event, new_entities, added))
        new_features["message"] = message_features(new_event, vector)
        gained_out = added.groupby("src_id").size()
        gained_in = added.groupby("dst_id").size()

        x_dict, positions = {}, {}
        for node_type in NODE_TYPES:
            x = self.x_dict[node_type]
            frame = new_features.get(node_type)
            positions[node_type] = dict(self.positions[node_type])
            if frame is not None and len(frame):
                for offset, node in enumerate(frame.index):
                    positions[node_type][node] = len(x) + offset
                x = torch.cat([x, torch.from_numpy(self.scaler.transform(node_type, frame))])
            if node_type != "message":
                touched = [node for node in (set(gained_out.index) | set(gained_in.index)) - new_ids if node in self.positions[node_type]]
                if touched:
                    updated = self.features[node_type].loc[touched].copy()
                    updated["out_degree"] += pd.Series(touched, index=touched).map(gained_out).fillna(0.0).to_numpy()
                    updated["in_degree"] += pd.Series(touched, index=touched).map(gained_in).fillna(0.0).to_numpy()
                    x = x.clone() if x is self.x_dict[node_type] else x
                    x[[self.positions[node_type][node] for node in touched]] = torch.from_numpy(self.scaler.transform(node_type, updated))
            x_dict[node_type] = x

        edge_index_dict = dict(self.edge_index_dict)
        for (src_type, edge_type, dst_type), frame in added.groupby(["src_type", "edge_type", "dst_type"]):
            src = [positions[src_type][node] for node in frame["src_id"]]
            dst = [positions[dst_type][node] for node in frame["dst_id"]]
            for key, pair in (((src_type, edge_type, dst_type), [src, dst]), ((dst_type, f"rev_{edge_type}", src_type), [dst, src])):
                if key in edge_index_dict:
                    edge_index_dict[key] = torch.cat([edge_index_dict[key], torch.tensor(pair, dtype=torch.long)], dim=1)

        scores = predict_gnn(self.model, x_dict, edge_index_dict)
        return {
            "score": float(scores[positions["message"][NEW_MESSAGE_ID]]),
            "event": new_event.iloc[0],
            "added": added,
            "new_entities": new_entities,
            "message_features": new_features["message"].iloc[0],
            "path": "incremental update",
        }

    def _score_by_full_rebuild(self, event: pd.DataFrame, use_links: bool) -> dict:
        """Reference path: rebuild the complete graph and all features from scratch."""
        events = pd.concat([self.tables.events[EVENT_COLUMNS], event], ignore_index=True)
        tables = build_graph(events, self.config)
        if not use_links:
            relations = tables.relations
            drop = relations["edge_type"].isin(SHARE_EDGE_TYPES) & ((relations["src_id"] == NEW_MESSAGE_ID) | (relations["dst_id"] == NEW_MESSAGE_ID))
            tables = GraphTables(tables.events, tables.entities, relations[~drop].reset_index(drop=True))
        new_event = tables.events[tables.events["event_id"] == NEW_EVENT_ID]
        new_message_features = message_features(new_event, self.encoder.encode(new_event["text"].tolist()))
        features = entity_features(tables)
        features["message"] = pd.concat([self.features["message"], new_message_features])
        x_dict, edge_index_dict, node_ids = to_tensors(tables, features, self.scaler)
        scores = pd.Series(predict_gnn(self.model, x_dict, edge_index_dict), index=node_ids["message"])
        new_ids = set(tables.entities["entity_id"]) - self.entity_ids
        relations = tables.relations
        return {
            "score": float(scores[NEW_MESSAGE_ID]),
            "event": new_event.iloc[0],
            "added": relations[relations["src_id"].isin(new_ids) | relations["dst_id"].isin(new_ids)],
            "new_entities": tables.entities[tables.entities["entity_id"].isin(new_ids)],
            "message_features": new_message_features.iloc[0],
            "path": "full rebuild",
        }

    # ------------------------------------------------------------- public API
    def predict(self, channel: str, text: str, sender: str = "", recipient: str = "") -> Prediction:
        started = time.perf_counter()
        event = self.make_event(channel, text, sender, recipient)
        result = self.score(event)
        entities = pd.concat([self.tables.entities, result["new_entities"]], ignore_index=True)
        prediction = self._describe(NEW_MESSAGE_ID, result["event"], result["score"], result["added"], entities, result["message_features"], None)
        prediction.seconds = time.perf_counter() - started
        prediction.details = {"path": result["path"], "new_nodes": int(len(result["new_entities"])), "new_edges": int(len(result["added"]))}
        return prediction

    def lookup(self, event_id: str) -> Prediction:
        """A message that is already in the graph (for example a test record) with its score."""
        message_id = entity_id("message", event_id)
        if message_id not in self.scores.index:
            raise InvalidInput(f"Unknown record '{event_id}'.")
        event = self.events.loc[message_id]
        relations = self.tables.relations
        own = relations[relations["src_id"] == message_id]
        return self._describe(message_id, event, float(self.scores[message_id]), own, self.tables.entities, self.features["message"].loc[message_id], int(event["label"]))

    def _describe(self, message_id, event, score, own_relations, entities, feature_row, true_label) -> Prediction:
        direct = own_relations[own_relations["src_id"] == message_id]
        neighbours = direct[direct["dst_type"] != "message"]["dst_id"].tolist()
        base = self.tables.relations
        second = pd.concat(
            [
                own_relations[own_relations["src_id"].isin(neighbours) & own_relations["edge_type"].isin(ENTITY_EDGE_TYPES)],
                base[base["src_id"].isin(neighbours) & base["edge_type"].isin(ENTITY_EDGE_TYPES)],
            ]
        ).drop_duplicates(["src_id", "dst_id", "edge_type"])
        around = pd.concat([direct[direct["dst_type"] != "message"], second], ignore_index=True)
        node_ids = set(around["src_id"]) | set(around["dst_id"])

        links = direct[(direct["dst_type"] == "message") & (direct["dst_id"] != message_id)]
        rows = []
        for other_id, group in links.groupby("dst_id", sort=False):
            other = self.events.loc[other_id]
            rows.append(
                {
                    "message_id": other_id,
                    "linked_by": ", ".join(sorted(group["edge_type"].unique())),
                    "channel": other["channel"],
                    "true_label": int(other["label"]),
                    "score": float(self.scores[other_id]),
                    "text": str(other["text"]),
                }
            )
        linked = pd.DataFrame(rows, columns=["message_id", "linked_by", "channel", "true_label", "score", "text"])
        return Prediction(
            channel=str(event["channel"]),
            text=str(event["text"]),
            sender=str(event["sender"]),
            recipient=str(event["recipient"]),
            message_id=message_id,
            score=score,
            threshold=self.threshold,
            relations=around,
            entities=entities[entities["entity_id"].isin(node_ids)].reset_index(drop=True),
            linked_messages=linked.sort_values("score", ascending=False).reset_index(drop=True),
            features={name: float(feature_row[name]) for name in BASIC_FEATURES},
            true_label=true_label,
        )

    def sample_test_records(self, per_group: int = 2, random_state: int = 0) -> pd.DataFrame:
        """A random draw from the test split, the same number from every channel and class."""
        test = self.predictions[self.predictions["split"] == "test"]
        sample = pd.concat(
            [group.sample(n=min(per_group, len(group)), random_state=random_state) for _, group in test.groupby(["channel", "label"], sort=True)]
        )
        sample["text"] = sample["message_id"].map(self.events["text"])
        return sample.reset_index(drop=True)
