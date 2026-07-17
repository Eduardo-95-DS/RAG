"""Thin Streamlit client for the NVIDIA 2025 Annual Report Assistant (item 8).

This is now a thin UI: it holds NO pipeline code (no LangChain, FAISS, torch,
Groq). It POSTs questions to the FastAPI backend's /query endpoint and renders
the JSON it gets back. Its only dependencies are `streamlit` and `requests`.

Config via env:
  BACKEND_URL   base URL of the rag-api service (default http://localhost:8000)
  API_KEY       shared secret sent as the X-API-Key header (optional locally)
"""
import os
import time

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("API_KEY", "")
REQUEST_TIMEOUT = 120  # the pipeline can take a few seconds; be generous

st.set_page_config(
    page_title="NVIDIA 2025 Annual Report Assistant",
    page_icon="📊",
    layout="centered",
)

st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background-color: #76b900;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


def _headers():
    return {"X-API-Key": API_KEY} if API_KEY else {}


def call_query(question: str, history: list) -> dict:
    """POST /query. history is a list of {'q','a'} for the last few turns."""
    resp = requests.post(
        f"{BACKEND_URL}/query",
        json={"question": question, "history": history[-3:]},
        headers=_headers(),
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def call_feedback(query: str, rewritten_query: str, answer: str, rating: int) -> None:
    """POST /feedback. Best-effort — never breaks the UI."""
    try:
        requests.post(
            f"{BACKEND_URL}/feedback",
            json={"query": query, "rewritten_query": rewritten_query,
                  "answer": answer, "rating": rating},
            headers=_headers(),
            timeout=10,
        )
    except requests.RequestException:
        pass


def init_session_state():
    if 'history' not in st.session_state:
        st.session_state.history = []


def render_transparency_panel(res: dict):
    """Show how the answer was produced, from the /query JSON payload."""
    sources = res.get('sources') or []
    route = res.get('route', 'retrieve')
    route_labels = {
        'retrieve': "📚 Retrieve — answered from the annual report",
        'conversational': "💬 Conversational — answered directly, no document lookup",
        'refuse': "🚫 Refused — off-topic or out-of-scope for this assistant",
    }
    with st.expander("🔎 How this answer was produced"):
        st.markdown(f"**Original question:** {res.get('question', '')}")
        st.markdown(f"**Route:** {route_labels.get(route, route)}")

        if route != 'retrieve':
            st.markdown(
                "This question skipped retrieval, so there are no source chunks or "
                "grounding check for it."
            )
            return

        rewritten = res.get('rewritten_query') or "(no rewrite)"
        st.markdown(f"**Rewritten query:** {rewritten}")

        if sources:
            st.markdown(
                f"**Retrieved {len(sources)} chunks** "
                "(hybrid FAISS + BM25, cross-encoder reranked)"
            )
            for i, d in enumerate(sources, start=1):
                title = d['text'][:80].replace("\n", " ").strip()
                with st.expander(f"Chunk {i}: {title}…"):
                    st.markdown(d['text'])
                    meta_bits = []
                    if d.get('source'):
                        meta_bits.append(f"source: {d['source']}")
                    if d.get('page') not in ('', None):
                        meta_bits.append(f"page: {d['page']}")
                    if meta_bits:
                        st.caption(" | ".join(meta_bits))
        else:
            st.markdown("**Retrieved chunks:** none")

        st.markdown("---")
        if res.get('grounded'):
            st.markdown("✅ **Grounding check passed** — answer is supported by the retrieved passages.")
        else:
            st.markdown(
                "⚠️ **Guardrail rejected the draft answer** — it wasn't grounded in the "
                "retrieved passages, so a fallback message was returned instead."
            )


def main():
    init_session_state()

    with st.sidebar:
        st.markdown("### About this document")
        st.markdown(
            "This app is grounded in NVIDIA's 2025 Annual Report (Form 10-K + Proxy "
            "Statement), covering fiscal year ended January 26, 2025."
        )
        st.markdown("**You can ask about:**")
        st.markdown(
            "- **Financial results** — revenue ($130.5B, up 114% YoY), net income, "
            "earnings per share, gross margin, operating cash flow, segment breakdowns\n"
            "- **Business segments** — Data Center, Gaming, Professional Visualization, "
            "and Automotive revenue and operating income\n"
            "- **Products and architecture** — Blackwell, Hopper, H100/H200, NVLink, "
            "CUDA, DRIVE, Jetson, and more\n"
            "- **Strategy** — NVIDIA's accelerated computing platform, AI factory "
            "vision, software stack, and partnerships\n"
            "- **Risks** — export controls and China exposure, competition from AMD "
            "and Intel, TSMC manufacturing dependency\n"
            "- **Corporate** — executive compensation, board composition, capital "
            "return to shareholders, R&D investment, employee headcount"
        )

    st.title("📊 NVIDIA 2025 Annual Report Assistant")
    st.markdown(
        "This is NVIDIA's Fiscal Year 2025 Annual Report, covering the company's "
        "financial results, AI and data center business, product lines, and leadership. "
        "Ask a question below and get an answer sourced directly from the report."
    )

    st.markdown("---")

    with st.form("search_form"):
        typed_question = st.text_input(
            "Enter your question:",
            placeholder="e.g. What were NVIDIA's 2025 revenues?"
        )
        submit = st.form_submit_button("🔍 Search")

    st.markdown("**Or try one of these:**")
    col1, col2 = st.columns(2)
    with col1:
        q1 = st.button("💰 What were NVIDIA's 2025 revenues?")
        q2 = st.button("🤖 What is NVIDIA's AI strategy?")
    with col2:
        q3 = st.button("📦 What are NVIDIA's main products?")
        q4 = st.button("👤 Who leads NVIDIA?")

    question_to_process = None
    if submit and typed_question:
        question_to_process = typed_question.strip()
    elif q1:
        question_to_process = "What were NVIDIA's 2025 revenues?"
    elif q2:
        question_to_process = "What is NVIDIA's AI strategy?"
    elif q3:
        question_to_process = "What are NVIDIA's main products?"
    elif q4:
        question_to_process = "Who leads NVIDIA?"

    answer_area = st.empty()

    if question_to_process:
        if len(question_to_process) > 500:
            answer_area.warning("Question too long. Please keep it under 500 characters.")
        else:
            with st.spinner("Retrieving and generating answer (this may take a few seconds)..."):
                try:
                    history_for_api = [
                        {"q": h["question"], "a": h["answer"]}
                        for h in st.session_state.history[-3:]
                    ]
                    result = call_query(question_to_process, history_for_api)
                    elapsed_time = result.get("elapsed_s", 0.0)
                    st.session_state.history.append({
                        'question': question_to_process,
                        'answer': result['answer'],
                        'time': elapsed_time,
                    })
                    st.session_state.last_result = {
                        'question': question_to_process,
                        'rewritten_query': result.get('rewritten_query', ''),
                        'route': result.get('route', 'retrieve'),
                        'answer': result['answer'],
                        'sources': result.get('sources', []),
                        'grounded': result.get('grounded', True),
                    }
                    st.session_state.feedback_key = f"feedback_{len(st.session_state.history)}"
                    with answer_area.container():
                        st.markdown("### 💡 Answer")
                        st.success(result['answer'])
                        st.caption(f"⏱️ Response time: {elapsed_time:.2f} seconds")
                        render_transparency_panel(st.session_state.last_result)
                except requests.HTTPError as e:
                    status = e.response.status_code if e.response is not None else "?"
                    if status == 401:
                        answer_area.error("Backend rejected the request (auth). Check API_KEY.")
                    elif status == 429:
                        answer_area.error("Too many requests. Please wait a moment and try again.")
                    else:
                        answer_area.error(f"Backend error ({status}). Please try again.")
                except requests.RequestException as e:
                    answer_area.error(f"Couldn't reach the backend: {e}")

    # Feedback for the most recent answer only.
    if st.session_state.get("last_result"):
        rating = st.feedback("thumbs", key=st.session_state.feedback_key)
        if rating is not None:
            last_saved_key = st.session_state.get("last_saved_feedback_key")
            if last_saved_key != st.session_state.feedback_key:
                call_feedback(
                    query=st.session_state.last_result['question'],
                    rewritten_query=st.session_state.last_result['rewritten_query'],
                    answer=st.session_state.last_result['answer'],
                    rating=1 if rating == 1 else -1,
                )
                st.session_state.last_saved_feedback_key = st.session_state.feedback_key

    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📜 Recent Searches")
        for item in reversed(st.session_state.history[-3:]):
            with st.container():
                st.markdown(f"**Q:** {item['question']}")
                preview = item['answer'][:200]
                if len(item['answer']) > 200:
                    preview += "..."
                st.markdown(f"**A:** {preview}")
                st.caption(f"Time: {item['time']:.2f}s")
                st.markdown("")


if __name__ == "__main__":
    main()
