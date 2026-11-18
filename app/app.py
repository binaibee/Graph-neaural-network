"""Demonstration of the scam-detection system (Streamlit, one page).

Start it with  run_demo.bat  or, from the repository root:

    python -m streamlit run app\\app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
for folder in (ROOT / "src", APP_DIR):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from graph_view import draw_message_graph  # noqa: E402
from scam_gnn.explain import explain  # noqa: E402
from scam_gnn.features import BASIC_FEATURES, URGENT_KEYWORDS  # noqa: E402
from scam_gnn.predict import MAX_INPUT_CHARS, InvalidInput, Prediction, Predictor, SetupError  # noqa: E402
from scam_gnn.train import RESULTS_FILE  # noqa: E402

CHANNEL_NAMES = {"sms": "SMS", "email": "Email", "url": "Web link"}
INPUT_HINTS = {"sms": "Type or paste an SMS message", "email": "Type or paste the subject and body of an email", "url": "Type or paste a web link, for example https://www.example.com/page"}
FEATURE_NAMES = {
    "length": "Number of characters",
    "digit_ratio": "Share of characters that are digits",
    "uppercase_ratio": "Share of characters that are capitals",
    "url_count": "Number of links",
    "phone_count": "Number of phone numbers",
    "urgent_keyword_count": "Number of urgent words",
}
SCOPES = {"sms": "SMS", "email": "Email", "url": "Web links", "overall": "All channels"}

st.set_page_config(page_title="Scam detection with a Graph Neural Network", layout="wide", initial_sidebar_state="collapsed")


@st.cache_resource(show_spinner=False)
def load_predictor() -> Predictor:
    return Predictor()


@st.cache_data
def load_results() -> dict:
    return json.loads((load_predictor().config.results_path / RESULTS_FILE).read_text(encoding="utf-8"))


@st.cache_data
def load_test_cases() -> list[dict]:
    return json.loads((APP_DIR / "test_cases.json").read_text(encoding="utf-8"))["cases"]


def label_text(value: int) -> str:
    return "SCAM" if int(value) == 1 else "LEGITIMATE"


def change_text(change: float) -> str:
    """How the risk score moved, in percentage points."""
    return "no change" if abs(change) < 0.0005 else f"{change * 100:+.1f} points"


def shorten(text: str, limit: int = 400) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + " …"


def mark(expected: str, predicted: str) -> str:
    return "✓ Correct" if expected == predicted else "✗ Wrong"


# ----------------------------------------------------------------- rendering
def render_verdict(prediction: Prediction) -> None:
    border, background, icon = ("#d03b3b", "#fdecec", "⚠") if prediction.is_scam else ("#0ca30c", "#e8f6e8", "✓")
    st.markdown(
        f"""
<div style="border-left: 10px solid {border}; background: {background}; color: #0b0b0b; padding: 18px 24px; border-radius: 6px; margin: 8px 0 4px 0;">
  <div style="font-size: 2.1rem; font-weight: 700; line-height: 1.2;">{icon} {prediction.label}</div>
  <div style="font-size: 1.25rem;">Risk score: <b>{prediction.score:.0%}</b></div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.caption(
        f"The risk score is the model's estimate of how likely this is a scam. {prediction.threshold:.0%} or more is marked SCAM "
        "(that line was chosen on validation data)."
    )


def render_result(prediction: Prediction, factors: list | None, predictor: Predictor) -> None:
    st.markdown(f"**{CHANNEL_NAMES[prediction.channel]}:**")
    st.text(shorten(prediction.text))  # plain text, so nothing in a message is turned into a clickable link
    render_verdict(prediction)
    if prediction.true_label is not None:
        correct = prediction.is_scam == bool(prediction.true_label)
        st.info(f"Correct answer in the dataset: **{label_text(prediction.true_label)}**. The model is **{'right' if correct else 'wrong'}**.")

    if factors is not None:
        st.markdown("#### Why? Remove one thing and score again")
        if factors:
            st.write("Each row removes one part of the message and asks the model again. A large drop means that part mattered.")
            table = pd.DataFrame(
                {
                    "What was removed": [factor.name for factor in factors],
                    "Found in this message": [factor.detail for factor in factors],
                    "Risk score without it": [f"{factor.score_without:.1%}" for factor in factors],
                    "Change": [change_text(-factor.effect(prediction.score)) for factor in factors],
                }
            )
            # A dataframe shows links as plain text, so a suspicious link can never be clicked by accident.
            st.dataframe(table, hide_index=True, width="stretch")
        else:
            st.write(
                "This message has no phone number, link, urgent word or connection that could be removed, "
                "so the score comes from the wording of the text as a whole."
            )

    st.markdown("#### The graph around this message")
    st.caption("The big dot is this message. The other dots are what it refers to and the stored messages it is connected to. The model passes information along the lines.")
    figure = draw_message_graph(prediction)
    st.pyplot(figure, width="stretch")
    plt.close(figure)

    with st.expander("Technical details"):
        if prediction.details:
            st.markdown(
                f"- **Preprocessing:** text cleaned; channel = `{prediction.channel}`.\n"
                f"- **Graph construction:** {prediction.details['new_nodes']} new node(s) and {prediction.details['new_edges']} new edge(s) "
                f"connected to the stored graph ({prediction.details['path']}).\n"
                f"- **Features:** {predictor.features['message'].shape[1]} numbers describe the message.\n"
                f"- **Model:** one forward pass of the trained heterogeneous GraphSAGE.\n"
                f"- **Output:** probability of scam = {prediction.score:.4f}; decision line = {prediction.threshold:.4f}.\n"
                f"- **Time taken:** {prediction.seconds:.2f} seconds. The message is not stored."
            )
        reference = predictor.feature_reference
        rows = [
            {
                "Model input": FEATURE_NAMES[name],
                "This message": round(prediction.features[name], 3),
                "Average legitimate (training data)": round(float(reference.loc[(prediction.channel, 0), name]), 3),
                "Average scam (training data)": round(float(reference.loc[(prediction.channel, 1), name]), 3),
            }
            for name in BASIC_FEATURES
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Urgent words the system counts: " + ", ".join(sorted(URGENT_KEYWORDS)) + ". The model also receives a numeric summary of the words in the text.")
        if len(prediction.linked_messages):
            linked = prediction.linked_messages.copy()
            linked["true_label"] = linked["true_label"].map(label_text)
            linked["score"] = linked["score"].map("{:.0%}".format)
            st.markdown("**Stored messages connected to this one**")
            st.dataframe(
                linked[["linked_by", "channel", "true_label", "score", "text"]].rename(
                    columns={"linked_by": "Connected by", "channel": "Channel", "true_label": "Known label", "score": "Model risk", "text": "Text"}
                ),
                hide_index=True,
                width="stretch",
            )


def analyse(predictor: Predictor, channel: str, text: str, sender: str = "", recipient: str = "") -> None:
    try:
        with st.spinner("Connecting the message to the graph and running the model…"):
            prediction = predictor.predict(channel, text, sender, recipient)
            st.session_state["result"] = (prediction, explain(predictor, prediction))
    except InvalidInput as error:
        st.session_state.pop("result", None)
        st.warning(str(error))
    except Exception as error:  # noqa: BLE001 - never show a stack trace during a presentation
        st.session_state.pop("result", None)
        st.error(f"This input could not be analysed: {error}")


def metric_table(entries: dict[str, dict], metric: str, scopes: list[str]) -> pd.DataFrame:
    rows = []
    for entry in entries.values():
        row = {"Model": entry["label"]}
        for scope in scopes:
            value = entry["summary"][scope][metric]
            row[SCOPES[scope]] = f"{value['mean']:.3f}" + (f" ± {value['std']:.3f}" if value["std"] >= 0.0005 else "")
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model")


# ---------------------------------------------------------------------- page
st.title("Scam detection with a Graph Neural Network")
st.write("A message comes in, the system connects it to stored messages in a graph, and a trained Graph Neural Network decides: **scam or legitimate?**")

try:
    with st.spinner("Loading the graph and the trained model…"):
        predictor = load_predictor()
        results = load_results()
except SetupError as error:
    st.error(str(error))
    st.stop()
except Exception as error:  # noqa: BLE001
    st.error(f"The demo could not start: {error}")
    st.info("Close this window and start run_demo.bat again from the project folder.")
    st.stop()

with st.sidebar:
    st.header("Demo controls")
    if st.button("Reload model and data", help="Use this if something looks stuck."):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.session_state.clear()
        st.rerun()
    if predictor.max_score_difference >= 1e-4:
        st.warning("The loaded model does not reproduce its saved scores. Run  python -m scam_gnn.train  again.")

cases = [case for case in load_test_cases() if case["channel"] in predictor.channels]
scopes = [scope for scope in SCOPES if scope in predictor.channels or scope == "overall"]

# ------------------------------------------------------------ 1. the message
st.header("1 · Analyse a message")
channel = st.radio("Channel", predictor.channels, format_func=CHANNEL_NAMES.get, horizontal=True, key="channel")

def _use_example(case: dict) -> None:
    st.session_state["input_text"] = case["text"]
    st.session_state["input_sender"] = case.get("sender", "")
    st.session_state["run_now"] = True

def _use_real_record(channel_name: str) -> None:
    """A random record of this channel from the test data, shown with its true label."""
    test = predictor.predictions[(predictor.predictions["split"] == "test") & (predictor.predictions["channel"] == channel_name)]
    st.session_state["record_number"] = st.session_state.get("record_number", 0) + 1
    event_id = test.sample(n=1, random_state=st.session_state["record_number"])["event_id"].iloc[0]
    st.session_state["result"] = (predictor.lookup(event_id), None)

examples = [case for case in cases if case["channel"] == channel and case.get("example")]
st.caption("Click an example, or type your own below.")
columns = st.columns(len(examples) + 1)
for column, case in zip(columns, examples):
    column.button(case["example"], on_click=_use_example, args=(case,), width="stretch", key=f"example_{case['id']}")
columns[-1].button("Random real record from the test data", on_click=_use_real_record, args=(channel,), width="stretch", key=f"real_{channel}")

# A form sends the typed text together with the button press, so nothing typed is lost.
with st.form("input_form", border=False):
    if channel == "email":
        left, right = st.columns(2)
    st.text_area("Input", key="input_text", height=90, max_chars=MAX_INPUT_CHARS, placeholder=INPUT_HINTS[channel], label_visibility="collapsed")
    submitted = st.form_submit_button("Analyze", type="primary")
if submitted or st.session_state.pop("run_now", False):
    analyse(predictor, channel, st.session_state.get("input_text", ""), st.session_state.get("input_sender", ""), st.session_state.get("input_recipient", ""))

if "result" in st.session_state:
    st.header("2 · Result")
    render_result(*st.session_state["result"], predictor)

# ---------------------------------------------------------------- 3. results
st.divider()
st.header("3 · How good is the system?")
total_test = sum(value["messages"] for key, value in results["splits"].items() if key.endswith("/test"))
st.write(
    f"Measured on **{total_test:,} test records** that were kept away from training. "
    f"Each model was trained {len(results['config']['seeds'])} times; the table shows the average F1-score (1.000 is perfect)."
)
st.markdown("**The GNN compared with two standard models that do not use the graph**")
st.table(metric_table(results["models"], "f1", scopes))
st.markdown("**Does the graph help? The same GNN with parts of the graph switched off**")
st.table(metric_table(results["ablation"], "f1", scopes))

demo = results["demo_model"]["test_metrics"]["overall"]
st.markdown(f"**What the demo model did with the {total_test:,} test records**")
first, second, third, fourth = st.columns(4)
first.metric("Scams caught", f"{demo['tp']:,}")
second.metric("Scams missed", f"{demo['fn']:,}")
third.metric("False alarms", f"{demo['fp']:,}")
fourth.metric("Legitimate, correctly passed", f"{demo['tn']:,}")

with st.expander("More metrics: precision, recall, accuracy, ROC-AUC, PR-AUC"):
    choice = st.radio("Metric", ["precision", "recall", "accuracy", "roc_auc", "pr_auc"], format_func=lambda name: name.replace("_", "-").upper() if "auc" in name else name.capitalize(), horizontal=True)
    st.table(metric_table({**results["models"], **{key: value for key, value in results["ablation"].items() if key != "full"}}, choice, scopes))
    st.markdown(
        """
- **Accuracy** — the share of all answers that were right.
- **Precision** — of the messages the model called scam, the share that really were scams.
- **Recall** — of the real scams, the share the model caught.
- **F1-score** — one number that balances precision and recall.
- **ROC-AUC** — the chance that a random scam gets a higher risk score than a random legitimate message (0.5 is guessing).
- **PR-AUC** — precision and recall summarised over every possible decision line.
"""
    )

with st.expander("Run the tests live"):
    st.markdown("**Test A — prepared inputs.** The correct answers were written down before the model was run.")
    if st.button("Run the prepared tests"):
        rows = []
        progress = st.progress(0.0, text="Running…")
        for index, case in enumerate(cases):
            try:
                predicted = predictor.predict(case["channel"], case["text"], case.get("sender", ""), case.get("recipient", ""))
                answer, risk = predicted.label, f"{predicted.score:.0%}"
            except Exception:  # noqa: BLE001
                answer, risk = "error", "-"
            rows.append({"Channel": CHANNEL_NAMES[case["channel"]], "Input": case["text"], "Correct answer": case["expected"], "Model said": answer, "Risk score": risk, "Result": mark(case["expected"], answer)})
            progress.progress((index + 1) / len(cases), text=f"Running… {index + 1} of {len(cases)}")
        progress.empty()
        st.session_state["test_rows"] = pd.DataFrame(rows)
    if "test_rows" in st.session_state:
        table = st.session_state["test_rows"]
        st.markdown(f"**{int((table['Result'] == '✓ Correct').sum())} of {len(table)} correct.** Wrong answers are shown as they are.")
        st.dataframe(table, hide_index=True, width="stretch", height=38 + 35 * len(table))

    st.markdown("**Test B — real test records.** Picked at random by the computer from the test data.")
    if st.button("Pick different records"):
        st.session_state["sample_number"] = st.session_state.get("sample_number", 0) + 1
    sample = predictor.sample_test_records(per_group=2, random_state=int(st.session_state.get("sample_number", 0)))
    said = (sample["score_gnn"] >= predictor.threshold).astype(int).map(label_text)
    truth = sample["label"].map(label_text)
    table = pd.DataFrame(
        {
            "Channel": sample["channel"].map(CHANNEL_NAMES),
            "Text": sample["text"].str.slice(0, 160),
            "Correct answer": truth,
            "Model said": said,
            "Risk score": sample["score_gnn"].map("{:.0%}".format),
            "Result": [mark(e, p) for e, p in zip(truth, said)],
        }
    )
    st.markdown(f"**{int((table['Result'] == '✓ Correct').sum())} of {len(table)} correct.**")
    st.dataframe(table, hide_index=True, width="stretch", height=38 + 35 * len(table))

with st.expander("About the data and the graph"):
    data_rows = [
        {"Channel": CHANNEL_NAMES[channel_name], "Records used": report["rows_used"], "Scam": report["scam"], "Legitimate": report["legitimate"], "Duplicates removed": report["duplicates_or_empty_removed"]}
        for channel_name, report in results["data"].items()
    ]
    st.dataframe(pd.DataFrame(data_rows), hide_index=True, width="stretch")
    graph = results["graph"]
    st.write(
        f"The graph has **{graph['nodes']:,} nodes** and **{graph['edges']:,} edges**. "
        f"{graph['message_to_message_edges']:,} edges join one message to another; {graph['cross_channel_message_edges']:,} of those join messages from different channels."
    )
    left, right = st.columns(2)
    left.dataframe(pd.Series(graph["nodes_by_type"], name="Count").rename_axis("Node type").reset_index(), hide_index=True, width="stretch")
    right.dataframe(pd.Series(graph["edges_by_type"], name="Count").rename_axis("Edge type").reset_index(), hide_index=True, width="stretch")
    st.caption(f"Text is represented as: {results['text_representation']}.")

with st.expander("Limitations"):
    st.markdown(
        """
- Phone calls are not part of this version; see the project handbook for the reasons.
- The SMS and web-link datasets contain no sender information, so those messages are connected only through the phone numbers and links they contain.
- A message is scored against a stored graph on one laptop. This demonstrates on-demand detection, not a high-volume live deployment.
- The explanation shows what happens when a part of the message is removed. It does not show the model's internal reasoning.
- The datasets are public research collections and several years old, so recent scam styles are under-represented.
"""
    )
