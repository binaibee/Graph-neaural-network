"""Stage 8 - Explanation: which parts of a message drive its risk score?

Method: *remove one factor and score again* (perturbation analysis).  For each
factor present in the message, the message is scored a second time without that
factor.  The difference between the two scores is how much the factor
contributed.  This uses the trained model itself, so the explanation always
reflects what the model actually does.

It answers "what happens to the score if this is taken away?".  It does not
describe the model's internal reasoning, and the contributions of several
factors do not have to add up to the total.
"""
from __future__ import annotations

from dataclasses import dataclass

from scam_gnn.features import URGENT_KEYWORDS, tokenize
from scam_gnn.normalize import extract_phone_numbers, extract_urls, remove_phone_numbers, remove_urls
from scam_gnn.predict import InvalidInput, Prediction, Predictor


@dataclass
class Factor:
    name: str  # what was removed
    detail: str  # the concrete value, for display
    score_without: float  # risk score after removing it

    def effect(self, score: float) -> float:
        """Positive: the factor pushed the score up (towards scam)."""
        return score - self.score_without


def _remove_urgent_words(text: str) -> str:
    return " ".join(word for word in text.split() if not (set(tokenize(word)) & URGENT_KEYWORDS))


def explain(predictor: Predictor, prediction: Prediction) -> list[Factor]:
    """Factors of a *new* message, largest effect first."""
    channel, text, sender, recipient = prediction.channel, prediction.text, prediction.sender, prediction.recipient
    candidates: list[tuple[str, str, dict]] = []
    if channel != "url":
        phones = list(dict.fromkeys(extract_phone_numbers(text)))
        if phones:
            candidates.append(("Phone number in the text", ", ".join(phones), {"text": remove_phone_numbers(text)}))
        urls = list(dict.fromkeys(extract_urls(text)))
        if urls:
            candidates.append(("Link in the text", ", ".join(urls[:3]), {"text": remove_urls(text)}))
        urgent = sorted({token for token in tokenize(text) if token in URGENT_KEYWORDS})
        if urgent:
            candidates.append(("Urgent words", ", ".join(urgent), {"text": _remove_urgent_words(text)}))
    if channel == "email" and sender:
        candidates.append(("Sender address", sender, {"sender": ""}))
    if channel == "email" and recipient:
        candidates.append(("Recipient address", recipient, {"recipient": ""}))
    if len(prediction.linked_messages):
        count = len(prediction.linked_messages)
        candidates.append(("Connections to other messages", f"{count} connected message{'s' if count != 1 else ''}", {"use_links": False}))

    factors = []
    for name, detail, change in candidates:
        use_links = change.pop("use_links", True)
        try:
            event = predictor.make_event(channel, change.get("text", text), change.get("sender", sender), change.get("recipient", recipient))
        except InvalidInput:
            continue  # nothing is left of the message once this factor is removed
        factors.append(Factor(name, detail, predictor.score(event, use_links=use_links)["score"]))
    return sorted(factors, key=lambda factor: abs(factor.effect(prediction.score)), reverse=True)
