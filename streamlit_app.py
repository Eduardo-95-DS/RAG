"""Streamlit UI for NVIDIA 2025 Annual Report Assistant"""

# import os
# os.environ["HF_HUB_OFFLINE"] = "1"

import streamlit as st
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder

FAISS_INDEX_PATH = "faiss_index"

st.set_page_config(
    page_title="NVIDIA 2025 Annual Report Assistant",
    page_icon="📊",
    layout="centered"
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


def init_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'history' not in st.session_state:
        st.session_state.history = []


@st.cache_resource
def initialize_rag():
    """Initialize the RAG system (cached across all sessions)"""
    try:
        llm = Config.get_llm()
        doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        vector_store = VectorStore()

        if Path(FAISS_INDEX_PATH).exists():
            vector_store.load(FAISS_INDEX_PATH)
            num_chunks = "cached"
        else:
            documents = doc_processor.process_urls(Config.SOURCES)
            vector_store.create_vectorstore(documents)
            vector_store.save(FAISS_INDEX_PATH)
            num_chunks = len(documents)

        graph_builder = GraphBuilder(
            retriever=vector_store.get_hybrid_retriever(),
            llm=llm
        )
        graph_builder.build()

        return graph_builder, num_chunks
    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")
        return None, 0


def main():
    """Main application"""
    init_session_state()

    st.title("📊 NVIDIA 2025 Annual Report Assistant")
    st.markdown("Ask any question about NVIDIA's 2025 Annual Report.")

    if not st.session_state.initialized:
        with st.spinner("Loading system..."):
            rag_system, num_chunks = initialize_rag()
            if rag_system:
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True
                if num_chunks == "cached":
                    st.success("✅ System ready! (index loaded from cache)")
                else:
                    st.success(f"✅ System ready! ({num_chunks} document chunks loaded)")

    st.markdown("---")

    # Search form
    with st.form("search_form"):
        typed_question = st.text_input(
            "Enter your question:",
            placeholder="e.g. What were NVIDIA's 2025 revenues?"
        )
        submit = st.form_submit_button("🔍 Search")

    # Suggested questions
    st.markdown("**Or try one of these:**")
    col1, col2 = st.columns(2)
    with col1:
        q1 = st.button("💰 What were NVIDIA's 2025 revenues?")
        q2 = st.button("🤖 What is NVIDIA's AI strategy?")
    with col2:
        q3 = st.button("📦 What are NVIDIA's main products?")
        q4 = st.button("👤 Who leads NVIDIA?")

    # Determine which question to process
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

    # Fixed answer area
    answer_area = st.empty()

    # Process question
    if question_to_process:
        if len(question_to_process) > 500:
            answer_area.warning("Question too long. Please keep it under 500 characters.")
        elif st.session_state.rag_system:
            with st.spinner("Retrieving and generating answer (this may take a few seconds)..."):
                try:
                    start_time = time.time()
                    result = st.session_state.rag_system.run(question_to_process)
                    elapsed_time = time.time() - start_time
                    st.session_state.history.append({
                        'question': question_to_process,
                        'answer': result['answer'],
                        'time': elapsed_time
                    })
                    with answer_area.container():
                        st.markdown("### 💡 Answer")
                        st.success(result['answer'])
                        st.caption(f"⏱️ Response time: {elapsed_time:.2f} seconds")
                except Exception as e:
                    if "rate_limit" in str(e).lower():
                        answer_area.error("Too many requests. Please wait a moment and try again.")
                    else:
                        answer_area.error(f"Failed to answer: {str(e)}")

    # History
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
