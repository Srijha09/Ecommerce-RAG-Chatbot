import streamlit as st
import requests

API_URL = "http://localhost:8000/chat"

st.set_page_config(
    page_title="Customer Support RAG Chatbot",
    page_icon="🧭",
    layout="wide",
)

# ---------------------------
# Sidebar – hyperparameters
# ---------------------------
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
# Main layout
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
                with st.expander("LLM Evaluation"):
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
