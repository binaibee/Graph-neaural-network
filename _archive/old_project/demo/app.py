"""Live demonstration of the scam-detection GNN (Streamlit).

Start it from the repository root with  run_demo.bat  or:

    python -m streamlit run demo\\app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

DEMO_DIR = Path(__file__).resolve().parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import engine  # noqa: E402
from graph_view import draw_message_graph  # noqa: E402

TEST_CASES_PATH = DEMO_DIR / "sample_data" / "test_cases.json"
SUMMARY_NAME = "uci_sms_phiusiil_multichannel_summary.json"
SUMMARY_PATHS = [
    engine.ROOT / "artifacts" / "uci_sms_phiusiil_multichannel" / SUMMARY_NAME,
    engine.MODEL_DIR / "experiment_summary.json",
]
# One-click examples on the first tab: (button label, test case id).
EXAMPLES = [
    ("Scam with a phone number", "T3"),
    ("Scam with a link", "T4"),
    ("Normal message", "T1"),
    ("Scam with no number or link", "T5"),
]
MODEL_LABELS = {
    "logistic_regression": ("Logistic regression", "No graph"),
    "mlp": ("MLP", "No graph"),
    "tree_ensemble": ("Gradient-boosted trees", "No graph"),
    "gcn": ("GCN", "GNN"),
    "graphsage": ("GraphSAGE", "GNN"),
    "hetero_graphsage": ("Heterogeneous GraphSAGE (this demo)", "GNN"),
    "hetero_hgt": ("HGT", "GNN"),
}
METRIC_LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "recall_at_precision": "Recall @ 90% precision",
}
FEATURE_DESCRIPTIONS = {
    "length": "Number of characters",
    "digit_ratio": "Share of characters that are digits",
    "uppercase_ratio": "Share of characters that are capitals",
    "url_count": "Number of links (http/https) in the text",
    "urgent_keyword_count": "Count of urgent keywords",
    "out_degree": "Edges leaving this message node",
    "in_degree": "Edges arriving at this message node",
    "infrastructure_degree": "Edges to links / messages on the same domain",
}

st.set_page_config(page_title="Scam detection with a Graph Neural Network", layout="wide", initial_sidebar_state="collapsed")


@st.cache_resource(show_spinner=False)
def load_engine() -> engine.DemoEngine:
    return engine.DemoEngine()


@st.cache_data
def load_test_cases() -> list[dict]:
    return json.loads(TEST_CASES_PATH.read_text(encoding="utf-8"))["cases"]


def label_text(value: int) -> str:
    return "SCAM" if int(value) == 1 else "LEGITIMATE"


def result_mark(expected: str, predicted: str) -> str:
    return "✓ Correct" if expected == predicted else "✗ Wrong"


# ----------------------------------------------------------------- rendering
def render_verdict(analysis: engine.Analysis) -> None:
    """Large coloured banner: icon + label + risk score."""
    scam = analysis.prediction == 1
    border, background, icon = ("#d03b3b", "#fdecec", "⚠") if scam else ("#0ca30c", "#e8f6e8", "✓")
    st.markdown(
        f"""
<div style="border-left: 10px solid {border}; background: {background}; color: #0b0b0b; padding: 18px 24px; border-radius: 6px; margin: 8px 0 4px 0;">
  <div style="font-size: 2.1rem; font-weight: 700; line-height: 1.2;">{icon} {analysis.label_text}</div>
  <div style="font-size: 1.25rem;">Risk score: <b>{analysis.score:.0%}</b></div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.caption("The risk score is the model's estimate of how likely the message is a scam. 50% or more is marked SCAM.")


def render_analysis(analysis: engine.Analysis, eng: engine.DemoEngine) -> None:
    st.markdown(f"**Message:** {analysis.event['text_or_url']}")
    render_verdict(analysis)

    # Plain-language reasons, built only from facts about this message.
    linked = analysis.linked_messages
    sender = str(analysis.event["sender_key"])
    has_contact = not sender.startswith("sms_sender_unknown")
    st.markdown("#### Why?")
    reasons = []
    if has_contact:
        reasons.append(f"The message contains a {'link' if sender.lower().startswith('http') else 'phone number'}: `{sender}`.")
    else:
        reasons.append("The message contains no phone number and no link.")
    if len(linked):
        scam_count = int((linked["true_label"] == 1).sum())
        reasons.append(f"That same {'link' if sender.lower().startswith('http') else 'phone number'} appears in **{len(linked)} earlier messages**, and {scam_count} of them are known scams.")
    else:
        reasons.append("It is not connected to any earlier message in the data.")
    st.markdown("\n".join(f"- {reason}" for reason in reasons))
    if analysis.prediction == 0 and not has_contact:
        st.info(
            "If this message looks like a scam to you but came out LEGITIMATE: the model reacts mainly to phone numbers and links, "
            "because in its training data almost only scam messages contain them. Scam wording on its own is often missed. "
            "This is a known limitation."
        )

    st.markdown("#### The graph the model looked at")
    st.caption("The big dot is your message. The other dots are the things it is connected to. The model passes information along the lines to work out the risk score.")
    figure = draw_message_graph(analysis)
    st.pyplot(figure, width="stretch")
    plt.close(figure)

    with st.expander("Technical details (open only if you are asked)"):
        st.markdown("**Steps the system performed**")
        sender_rule = (
            "taken from the first phone number or link found in the text (the rule used for the UCI SMS data, which has no real sender field)"
            if has_contact
            else "no phone number or link found in the text, so a placeholder 'unknown' sender is used"
        )
        added_nodes = 1 + int((~analysis.entities["entity_id"].isin(eng.tables.entities["entity_id"])).sum())
        steps = pd.DataFrame(
            [
                ("1. Preprocessing", f"Whitespace cleaned; channel = sms; sender = `{sender}` ({sender_rule})."),
                ("2. Feature extraction", f"{len(eng.meta['feature_columns'])} numbers for the message: 5 text counts, channel flags, a text vector ({eng.embedding_backend}), graph degree counts."),
                ("3. Graph construction", f"{added_nodes} new node(s) and {len(analysis.relations) + len(linked)} new edge(s) added to the existing graph of {eng.meta['dataset']['num_nodes']:,} nodes."),
                ("4. GNN model", "Forward pass of the trained heterogeneous GraphSAGE over the whole graph."),
                ("5. Risk prediction", f"Softmax probability of the scam class for the new message node = {analysis.score:.4f} (SCAM if 0.5 or more). Confidence band: {analysis.confidence}."),
            ],
            columns=["Step", "What happened"],
        )
        st.table(steps.set_index("Step"))
        st.caption(f"Time taken: {sum(analysis.timings.values()):.1f} seconds. The message is not saved into the graph.")

        if len(linked):
            st.markdown("**Earlier messages connected to this one**")
            shown = linked.copy()
            shown["true_label"] = shown["true_label"].map(label_text)
            shown["gnn_score"] = shown["gnn_score"].map("{:.0%}".format)
            st.dataframe(
                shown[["linked_by", "channel", "true_label", "gnn_score", "text"]].rename(
                    columns={"linked_by": "Connected by", "channel": "Channel", "true_label": "Known label", "gnn_score": "Model risk", "text": "Text"}
                ),
                hide_index=True,
                width="stretch",
            )
        if analysis.score_without_links is not None:
            st.write(
                f"**Effect of the connections (measured with the model):** with them removed, the same model gives "
                f"{analysis.score_without_links:.1%} instead of {analysis.score:.1%}."
            )

        st.markdown("**Numbers the model received for this message**")
        st.caption("These are model inputs, not an explanation generated by the GNN. No explainability method is implemented.")
        reference = eng.feature_reference
        channel = analysis.event["channel"]
        rows = [
            {
                "Model input": FEATURE_DESCRIPTIONS.get(name, name),
                "This message": round(value, 3),
                "Average legitimate (training)": round(float(reference.loc[(channel, 0), name]), 3),
                "Average scam (training)": round(float(reference.loc[(channel, 1), name]), 3),
            }
            for name, value in analysis.features.items()
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def analyse_into_state(eng: engine.DemoEngine, text: str) -> None:
    try:
        with st.spinner("Adding the message to the graph and running the model…"):
            st.session_state["live_analysis"] = eng.analyze_sms(text)
    except engine.InvalidMessage as error:
        st.session_state.pop("live_analysis", None)
        st.warning(str(error))
    except Exception as error:  # noqa: BLE001 - never show a stack trace during a presentation
        st.session_state.pop("live_analysis", None)
        st.error(f"The message could not be analysed: {error}")


# ---------------------------------------------------------------------- page
st.title("Scam detection with a Graph Neural Network")
st.write("The system reads a message, connects it to earlier messages in a graph, and a trained Graph Neural Network decides: **scam or legitimate?**")

with st.sidebar:
    st.header("Demo controls")
    if st.button("Reload model and data", help="Use this if something looks stuck. It loads the saved model and data again."):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.session_state.clear()
        st.rerun()

try:
    with st.spinner("Loading the graph and the trained model (a few seconds)…"):
        eng = load_engine()
except engine.DemoSetupError as error:
    st.error(str(error))
    st.stop()
except Exception as error:  # noqa: BLE001
    st.error(f"The demo could not start: {error}")
    st.info("Close this window, then start run_demo.bat again from the project folder.")
    st.stop()

meta = eng.meta
cases = load_test_cases()
case_by_id = {case["id"]: case for case in cases}

with st.sidebar:
    st.success("Model loaded")
    st.markdown(
        f"**Model:** heterogeneous GraphSAGE  \n"
        f"**Graph:** {meta['dataset']['num_nodes']:,} nodes, {meta['dataset']['num_edges']:,} edges  \n"
        f"**Messages:** {meta['dataset']['num_messages']:,}"
    )
    if eng.max_score_difference >= 1e-4:
        st.warning("The loaded model does not match its saved scores. Re-run demo\\train_demo_model.py.")

tab_try, tab_tests, tab_results = st.tabs(["1 · Try a message", "2 · Run the tests", "3 · Results"])

# -------------------------------------------------------------- try a message
with tab_try:
    st.subheader("Step 1 — Click an example")

    def _use_example(case_id: str) -> None:
        st.session_state["message_text"] = case_by_id[case_id]["text"]
        st.session_state["run_now"] = True

    for column, (label, case_id) in zip(st.columns(len(EXAMPLES)), EXAMPLES):
        column.button(label, on_click=_use_example, args=(case_id,), width="stretch")

    st.markdown("**…or type your own SMS message**")
    # A form sends the typed text together with the button press, so nothing typed is lost.
    with st.form("live_form", border=False):
        st.text_area(
            "Message",
            key="message_text",
            height=90,
            max_chars=engine.MAX_MESSAGE_CHARS,
            placeholder="Type or paste an SMS message here",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Analyze Message", type="primary")
    if submitted or st.session_state.pop("run_now", False):
        analyse_into_state(eng, st.session_state.get("message_text", ""))

    if "live_analysis" in st.session_state:
        st.divider()
        st.subheader("Step 2 — The result")
        render_analysis(st.session_state["live_analysis"], eng)

# -------------------------------------------------------------------- tests
with tab_tests:
    st.subheader("Test A — Ten prepared messages")
    st.write("Ten SMS messages with the correct answer written down beforehand. Press the button and the model answers each one.")
    if st.button("Run the 10 tests", type="primary"):
        results = []
        progress = st.progress(0.0, text="Running…")
        for index, case in enumerate(cases):
            try:
                analysis = eng.analyze_sms(case["text"])
                prediction, risk = analysis.label_text, f"{analysis.score:.0%}"
            except Exception:  # noqa: BLE001
                prediction, risk = "error", "-"
            results.append(
                {
                    "Message": case["text"],
                    "Correct answer": case["expected"],
                    "Model said": prediction,
                    "Risk score": risk,
                    "Result": result_mark(case["expected"], prediction),
                }
            )
            progress.progress((index + 1) / len(cases), text=f"Running… {index + 1} of {len(cases)}")
        progress.empty()
        st.session_state["case_results"] = pd.DataFrame(results)
    if "case_results" in st.session_state:
        results = st.session_state["case_results"]
        correct = int((results["Result"] == "✓ Correct").sum())
        st.markdown(f"### {correct} of {len(results)} correct")
        st.dataframe(results, hide_index=True, width="stretch")
        st.caption(
            "The wrong answers are shown as they are. The model relies on phone numbers and links: scam wording without one is missed, "
            "and a harmless message containing a number used by scammers is flagged."
        )

    st.divider()
    st.subheader("Test B — Real records the model was never taught")
    st.write(
        "Twelve real records (SMS and web links) from the **test set**: data whose answers were hidden from the model during training. "
        "The computer picks them at random."
    )
    if st.button("Pick 12 different records"):
        st.session_state["sample_number"] = st.session_state.get("sample_number", 0) + 1
    sample = eng.sample_held_out(per_group=3, random_state=int(st.session_state.get("sample_number", 0)))
    predicted = (sample["score"] >= engine.THRESHOLD).astype(int).map(label_text)
    expected = sample["label"].map(label_text)
    table = pd.DataFrame(
        {
            "Type": sample["channel"].map({"sms": "SMS", "url": "Web link"}).fillna(sample["channel"]),
            "Text": sample["text"],
            "Correct answer": expected,
            "Model said": predicted,
            "Risk score": sample["score"].map("{:.0%}".format),
            "Result": [result_mark(e, p) for e, p in zip(expected, predicted)],
        }
    )
    st.markdown(f"### {int((table['Result'] == '✓ Correct').sum())} of {len(table)} correct")
    st.dataframe(table, hide_index=True, width="stretch", height=38 + 35 * len(table))

# ------------------------------------------------------------------ results
with tab_results:
    st.subheader("How good is the model?")
    counts = meta["dataset"]["split_counts"]
    st.write(
        f"Measured on the test set: {counts['test/sms']['messages']:,} SMS and {counts['test/url']['messages']:,} web links "
        "that the model was never taught the answers to. All numbers are computed by the project's code."
    )
    rows = []
    for name, key in [("SMS", "sms"), ("Web links (URLs)", "url"), ("Both together", "overall")]:
        source = meta["test_metrics"] if key == "overall" else meta["test_metrics"]["per_channel"][key]
        derived = meta["test_metrics_derived_by_demo"][key]
        rows.append(
            {
                "Channel": name,
                "Accuracy": f"{derived['accuracy']:.1%}",
                "Precision": f"{source['precision']:.1%}",
                "Recall": f"{source['recall']:.1%}",
                "F1-score": f"{derived['f1']:.1%}",
                "ROC-AUC": f"{source['roc_auc']:.3f}",
                "PR-AUC": f"{source['pr_auc']:.3f}",
            }
        )
    st.table(pd.DataFrame(rows).set_index("Channel"))

    overall = meta["test_metrics_derived_by_demo"]["overall"]
    st.markdown("**What happened to the 4,115 test records** (confusion matrix)".replace("4,115", f"{overall['tp'] + overall['fp'] + overall['fn'] + overall['tn']:,}"))
    first, second, third, fourth = st.columns(4)
    first.metric("Scams caught", f"{overall['tp']:,}")
    second.metric("Scams missed", f"{overall['fn']:,}")
    third.metric("False alarms", f"{overall['fp']:,}")
    fourth.metric("Legitimate, correctly passed", f"{overall['tn']:,}")

    with st.expander("What do these numbers mean?"):
        st.markdown(
            """
- **Accuracy** — the share of all answers that were right.
- **Precision** — of the messages the model called scam, the share that really were scams.
- **Recall** — of the real scams, the share the model caught.
- **F1-score** — one number that balances precision and recall.
- **ROC-AUC** — the chance that a random scam gets a higher risk score than a random legitimate message (1.0 is perfect, 0.5 is guessing).
- **PR-AUC** — a summary of precision and recall over every possible decision line. The project's main metric, because scams are the minority.

Accuracy, F1 and the confusion matrix are worked out by the demo from the saved test predictions; the others come from the project's `metrics.py`.
"""
        )

    with st.expander("Comparison with the other models in the project"):
        summary_path = next((path for path in SUMMARY_PATHS if path.exists()), None)
        if summary_path is None:
            st.warning("No experiment summary found. Create it with:  python -m scam_gnn.cli run-research --config examples\\configs\\uci_sms_phiusiil_multichannel.json")
        else:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            scope = st.radio("Show", ["SMS", "Web links (URLs)", "Both together"], horizontal=True)
            scope_key = {"SMS": "sms", "Web links (URLs)": "url", "Both together": "overall"}[scope]
            comparison = []
            for name, payload in summary["results"].items():
                source = payload["metrics"] if scope_key == "overall" else payload["per_channel"].get(scope_key, {})
                label, kind = MODEL_LABELS.get(name, (name, ""))
                comparison.append({"Model": label, "Type": kind} | {METRIC_LABELS[m]: f"{source[m]['mean']:.3f} ± {source[m]['std']:.3f}" for m in METRIC_LABELS if m in source})
            st.dataframe(pd.DataFrame(comparison), hide_index=True, width="stretch")
            st.caption("Average ± spread over 3 training runs. On this data the models without a graph score higher than the GNNs, most clearly on SMS.")

    with st.expander("About the graph"):
        st.markdown(
            """
**Dots (nodes):** `message`, `sender`, `recipient`, `phone`, `email_address`, `url`, `domain`.

**Lines (edges):** `sent_by`, `sent_to`, `uses_phone`, `uses_email`, `contains_url`, `resolves_to_domain`, and three that join
one message to another: `shares_sender`, `shares_recipient`, `shares_infrastructure` (same web domain).
"""
        )
        left, right = st.columns(2)
        left.dataframe(pd.Series(meta["dataset"]["nodes_by_type"], name="Count").rename_axis("Node type").reset_index(), hide_index=True, width="stretch")
        right.dataframe(pd.Series(meta["dataset"]["edges_by_type"], name="Count").rename_axis("Edge type").reset_index(), hide_index=True, width="stretch")

    with st.expander("Limitations"):
        st.markdown(
            f"""
- The model relies mainly on phone numbers and links; scam wording without one is often missed.
- The SMS data has no sender field, so the "sender" is a phone number or link found inside the text.
- The SMS and web-link datasets share nothing, so no line in the graph joins an SMS to a web link.
- Web-link scores are near-perfect because that dataset is easy: its ready-made columns separate the classes almost completely.
- The models without a graph score higher than the GNNs on this data.
- A message is scored in a few seconds on a laptop. That is on-demand scoring, not a high-volume real-time system.
- The datasets have no timestamps, so the train/test split follows dataset order.
- Text is represented by: {eng.embedding_backend}.
- No explainability method is implemented.
"""
        )
