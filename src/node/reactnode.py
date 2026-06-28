"""LangGraph nodes for RAG workflow + ReAct Agent inside generate_content"""
from typing import List, Optional
from src.state.rag_state import RAGState
from src.logging.rag_logger import get_logger
from langchain_core.documents import Document
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

log = get_logger()


class RetrieverInput(BaseModel):
    query: str


class RAGNodes:
    """Contains node functions for RAG workflow"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self._agent = None

    REWRITE_PROMPT = (
        "Rewrite the user's question as a precise, self-contained query suitable for "
        "searching a financial annual report. Remove conversational phrasing. "
        "Expand abbreviations. Output only the rewritten query — no explanation."
    )

    def rewrite_query(self, state: RAGState) -> RAGState:
        """Rewrite the raw question into a retrieval-optimized query."""
        messages = [
            SystemMessage(content=self.REWRITE_PROMPT),
            HumanMessage(content=state.question),
        ]
        response = self.llm.invoke(messages)
        rewritten = response.content.strip()
        log.info("[REWRITE] original='%s' | rewritten='%s'", state.question, rewritten)
        return RAGState(
            question=state.question,
            rewritten_query=rewritten,
            retrieved_docs=state.retrieved_docs,
            answer=state.answer,
        )

    def retrieve_docs(self, state: RAGState) -> RAGState:
        """Classic retriever node"""
        docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )

    def _build_tools(self):
        """Build retriever tool"""
        def retriever_tool_fn(query: str) -> str:
            log.info("[TOOL] retriever called | query='%s'", query)
            docs: List[Document] = self.retriever.invoke(query)
            if not docs:
                log.warning("[TOOL] retriever returned 0 chunks | query='%s'", query)
                return "No documents found."
            log.info("[TOOL] retriever returned %d chunks | query='%s'", len(docs), query)
            merged = []
            for i, d in enumerate(docs[:8], start=1):
                meta = d.metadata if hasattr(d, "metadata") else {}
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                preview = d.page_content[:100].replace("\n", " ")
                log.info("[CHUNK %d] source='%s' | preview='%s...'", i, title, preview)
                merged.append(f"[{i}] {title}\n{d.page_content}")
            return "\n\n".join(merged)

        retriever_tool = StructuredTool.from_function(
            func=retriever_tool_fn,
            name="retriever",
            description="Fetch passages from the NVIDIA 2025 Annual Report.",
            args_schema=RetrieverInput,
        )
        return [retriever_tool]

    def _build_agent(self):
        """ReAct agent with retriever tool"""
        tools = self._build_tools()
        system_prompt = (
            "You have access to one tool: a retriever over the NVIDIA 2025 Annual Report. "
            "If the answer is not in the document, say so. "
            "Never describe your tools or capabilities. "
            "Always answer directly and naturally based on what you retrieve."
        )
        self._agent = create_react_agent(self.llm, tools=tools, prompt=system_prompt)

    def generate_answer(self, state: RAGState) -> RAGState:
        """Generate answer using ReAct agent with retriever."""
        if self._agent is None:
            self._build_agent()

        query = state.rewritten_query or state.question
        result = self._agent.invoke({"messages": [HumanMessage(content=query)]})
        messages = result.get("messages", [])
        answer: Optional[str] = None
        if messages:
            answer = getattr(messages[-1], "content", None)

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer or "Could not generate answer."
        )