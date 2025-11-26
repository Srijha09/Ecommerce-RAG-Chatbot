import json
from pathlib import Path
from typing import Any, Dict, List
import altair as alt
import requests
import streamlit as st
import pandas as pd

API_URL = "http://localhost:8000/chat"
EVAL_RESULTS_PATH = Path("data/offline_eval_results.json")

st.set_page_config(
    page_title="Customer Support RAG Chatbot",
    page_icon="🧭",
    layout="wide",
)

# ---------------------------
# Sidebar – global controls
# ---------------------------
st.sidebar.title("🧭 RAG Console")

mode = st.sidebar.radio(
    "Mode",
    ["Chatbot", "Offline Evaluation"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption("Switch between live chat and offline evaluation.")


# ===========================
# Helpers for offline eval
# ===========================
@st.cache_data
def load_eval_results(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Offline eval results not found at {path}. "
            "Run `python -m scripts.run_offline_eval` first."
        )
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def results_to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "id": r.get("id"),
                "question": r.get("question"),
                "reference_answer": r.get("reference_answer"),
                "model_answer": r.get("model_answer"),
                "judge_label": r.get("judge_label"),
                "bleu": r.get("bleu"),
                "meteor": r.get("meteor"),
                "rouge_l": r.get("rouge_l"),
            }
        )
    return pd.DataFrame(rows)


# ===========================
# Mode 1: Chatbot UI
# ===========================
if mode == "Chatbot":
    # Sidebar controls specific to chat
    with st.sidebar:
        st.header("⚙️ RAG Controls")

        top_k = st.slider(
            "Top K (retrieved chunks)",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            help="How many chunks to retrieve from the vector store.",
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.5,
            value=0.1,
            step=0.05,
            help="Higher values = more creative, lower = more deterministic.",
        )

        st.markdown("---")
        st.caption("These settings are sent to the backend on each request.")

    # ---------------------------
    # Main layout – Chat
    # ---------------------------
    st.title("Customer Support RAG Chatbot")
    st.write(
        "Ask any question based on the Everstorm policies. "
        "Responses are grounded in retrieved documentation."
    )

    # Two columns: chat (wide) + source panel (narrow)
    chat_col, side_col = st.columns([3, 2])

    if "messages" not in st.session_state:
        st.session_state.messages = []        # [{"role": "user"/"assistant", "content": str}]
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []    # cache last response sources
    if "last_judge" not in st.session_state:
        st.session_state.last_judge = None    # {"label": str, "cycles": [...]}

    # ---------------------------
    # Chat history display
    # ---------------------------
    with chat_col:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_input = st.chat_input("Ask a question…")

        if user_input:
            # show user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            # build history as (user, assistant) tuples
            history = []
            last_user = None
            for m in st.session_state.messages:
                if m["role"] == "user":
                    last_user = m["content"]
                elif m["role"] == "assistant" and last_user is not None:
                    history.append((last_user, m["content"]))
                    last_user = None

            payload = {
                "question": user_input,
                "history": history,
                "top_k": top_k,
                "temperature": float(temperature),
            }

            try:
                resp = requests.post(API_URL, json=payload, timeout=300)

                if not resp.ok:
                    # surface backend error details
                    try:
                        detail = resp.json()
                    except Exception:
                        detail = resp.text
                    answer = (
                        f"⚠️ Backend returned {resp.status_code}\n\n"
                        f"Details: {detail}"
                    )
                    sources = []
                    judge_label = None
                    judge_cycles = []
                else:
                    data = resp.json()
                    answer = data.get("answer", "")
                    sources = data.get("sources", [])
                    judge_label = data.get("judge_label")
                    judge_cycles = data.get("judge_cycles", [])

            except Exception as e:
                answer = f"⚠️ Error calling backend: {e}"
                sources = []
                judge_label = None
                judge_cycles = []

            # store latest results
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.last_sources = sources
            st.session_state.last_judge = {
                "label": judge_label,
                "cycles": judge_cycles,
            }

            # render assistant answer
            with st.chat_message("assistant"):
                st.write(answer)

                # 🔍 LLM-as-Judge expander directly under the answer
                if judge_label is not None:
                    with st.expander("🔍 LLM Evaluation"):
                        st.write(f"**Judge Label:** `{judge_label}`")
                        if judge_cycles:
                            st.write("**Evaluation Cycles:**")
                            for c in judge_cycles:
                                crit = c.get("critique") or "(no critique – judged correct)"
                                st.markdown(
                                    f"- Cycle {c.get('cycle', '?')}: **{crit}**"
                                )
                        else:
                            st.caption("No detailed cycles available.")

                # 📘 References
                if sources:
                    st.markdown("### 📘 References:")
                    for s in sources:
                        meta = s.get("metadata", {})
                        file = meta.get("source", "unknown")
                        page = meta.get("page_number", "?")
                        st.markdown(f"- `{file}`, page **{page}**")

    # ---------------------------
    # Right column – source chunks
    # ---------------------------
    with side_col:
        st.subheader("🔍 Retrieved Context")

        sources = st.session_state.get("last_sources", [])
        if not sources:
            st.caption("Ask a question to see which chunks were retrieved.")
        else:
            for idx, src in enumerate(sources):
                with st.expander(f"Source {idx+1}"):
                    st.write(src.get("page_content", ""))
                    meta = src.get("metadata", {})
                    if meta:
                        st.json(meta)


# ===========================
# Mode 2: Offline Evaluation UI
# ===========================
elif mode == "Offline Evaluation":
    st.title("📊 RAG Offline Evaluation Dashboard")
    st.caption(
        "Visualization of offline evaluation results from `scripts/run_offline_eval.py`."
    )

    try:
        eval_data = load_eval_results(EVAL_RESULTS_PATH)
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("Run `python -m scripts.run_offline_eval` and refresh this page.")
        st.stop()

    summary = eval_data.get("summary", {})
    results = eval_data.get("results", [])

    df = results_to_dataframe(results)

    # Summary metrics
    st.subheader("High-level Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Examples", summary.get("num_examples", len(results)))

    with col2:
        st.metric("Avg BLEU", f"{summary.get('avg_bleu', 0.0):.3f}")

    with col3:
        st.metric("Avg METEOR", f"{summary.get('avg_meteor', 0.0):.3f}")

    with col4:
        st.metric("Avg ROUGE-L", f"{summary.get('avg_rouge_l', 0.0):.3f}")

    label_counts = summary.get("label_counts", {})
    total = summary.get("num_examples", len(results)) or 1

    st.markdown("### Judge Label Distribution")
    lc_cols = st.columns(max(len(label_counts), 1))
    for (label, count), col in zip(label_counts.items(), lc_cols):
        col.metric(
            label,
            f"{count}",
            f"{(count / total) * 100:.1f}%",
        )

    # 🔵 NEW: Judge label bar chart
    if label_counts:
        chart_df = (
            pd.DataFrame(
                [{"judge_label": lbl, "count": cnt} for lbl, cnt in label_counts.items()]
            )
            .sort_values("judge_label")
        )
        st.markdown("#### Judge Label Distribution (Chart)")
        label_chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("judge_label:N", title="Label"),
                y=alt.Y("count:Q", title="Count"),
                color="judge_label:N",
                tooltip=["judge_label", "count"],
            )
            .properties(height=300, width="container")
        )
        st.altair_chart(label_chart, use_container_width=True)

    # Sidebar filters for eval mode
    st.sidebar.header("🔎 Eval Filters")

    all_labels = sorted(df["judge_label"].dropna().unique().tolist())
    selected_labels = st.sidebar.multiselect(
        "Judge Labels",
        options=all_labels,
        default=all_labels,
    )

    min_bleu = st.sidebar.slider(
        "Min BLEU",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
    )

    min_meteor = st.sidebar.slider(
        "Min METEOR",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
    )

    min_rouge = st.sidebar.slider(
        "Min ROUGE-L",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
    )

    search_text = st.sidebar.text_input(
        "Search in question text",
        value="",
    )

    filtered_df = df.copy()

    if selected_labels:
        filtered_df = filtered_df[filtered_df["judge_label"].isin(selected_labels)]

    filtered_df = filtered_df[
        (filtered_df["bleu"].fillna(0.0) >= min_bleu)
        & (filtered_df["meteor"].fillna(0.0) >= min_meteor)
        & (filtered_df["rouge_l"].fillna(0.0) >= min_rouge)
    ]

    if search_text.strip():
        mask = filtered_df["question"].str.contains(
            search_text.strip(), case=False, na=False
        )
        filtered_df = filtered_df[mask]

    st.markdown("### Filtered Examples")
    st.caption(
        f"Showing {len(filtered_df)} of {len(df)} examples "
        f"({len(filtered_df) / max(len(df),1) * 100:.1f}%)."
    )

    st.dataframe(
        filtered_df[
            ["id", "judge_label", "bleu", "meteor", "rouge_l", "question"]
        ].reset_index(drop=True),
        use_container_width=True,
        height=300,
    )

    # 🔵 NEW: Metric trend chart over filtered examples
    if not filtered_df.empty:
        st.markdown("#### Metric Trends (Filtered Examples)")
        # reshape for altair
        metrics_long = filtered_df.melt(
            id_vars=["id"],
            value_vars=["bleu", "meteor", "rouge_l"],
            var_name="metric",
            value_name="value",
        )
        metric_chart = (
            alt.Chart(metrics_long)
            .mark_line(point=True)
            .encode(
                x=alt.X("id:N", title="Example ID", sort=None),
                y=alt.Y("value:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("metric:N", title="Metric"),
                tooltip=["id", "metric", "value"],
            )
            .properties(height=300, width="container")
        )
        st.altair_chart(metric_chart, use_container_width=True)

    st.markdown("### Per-Query Details")

    if filtered_df.empty:
        st.info("No examples match the current filters.")
    else:
        ids = filtered_df["id"].tolist()
        selected_id = st.selectbox(
            "Select example ID",
            options=ids,
            index=0,
        )

        selected_row = filtered_df[filtered_df["id"] == selected_id].iloc[0]
        full_result = next((r for r in results if r.get("id") == selected_id), None)

        q_col, score_col = st.columns([3, 2])

        with q_col:
            st.markdown("**Question**")
            st.write(selected_row["question"])

            st.markdown("**Reference Answer**")
            st.write(selected_row["reference_answer"])

            st.markdown("**Model Answer**")
            st.write(selected_row["model_answer"])

        with score_col:
            st.markdown("**Scores**")
            st.write(f"- BLEU: `{selected_row['bleu']:.4f}`")
            st.write(f"- METEOR: `{selected_row['meteor']:.4f}`")
            st.write(f"- ROUGE-L: `{selected_row['rouge_l']:.4f}`")

            st.markdown("**Judge**")
            st.write(f"- Label: `{selected_row['judge_label']}`")

        if full_result is not None:
            sources = full_result.get("sources", [])
            judge_cycles = full_result.get("judge_cycles", [])

            with st.expander("📘 Retrieved Sources", expanded=False):
                if not sources:
                    st.caption("No sources recorded for this example.")
                else:
                    for i, s in enumerate(sources, 1):
                        meta = s.get("metadata", {})
                        file = meta.get("source", "unknown")
                        page = meta.get("page_number", "?")

                        st.markdown(f"**Source {i}** – `{file}`, page **{page}**")
                        st.write(s.get("page_content", ""))
                        st.json(meta)

            with st.expander("🔍 LLM-as-Judge Cycles", expanded=False):
                if not judge_cycles:
                    st.caption("No judge cycle details available.")
                else:
                    for c in judge_cycles:
                        cycle_no = c.get("cycle", "?")
                        crit = c.get("critique") or "(no critique – judged correct)"
                        st.markdown(f"**Cycle {cycle_no}**")
                        st.write(crit)
