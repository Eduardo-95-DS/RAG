"""LangGraph nodes for RAG workflow + ReAct Agent inside generate_content"""
from typing import List, Optional
from src.state.rag_state import RAGState
from langchain_core.documents import Document
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel


class RetrieverInput(BaseModel):
    query: str


class RAGNodes:
    """Contains node functions for RAG workflow"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self._agent = None

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
            print(f"\n[TOOL] retriever called with: '{query}'")
            docs: List[Document] = self.retriever.invoke(query)
            if not docs:
                return "No documents found."
            merged = []
            for i, d in enumerate(docs[:8], start=1):
                meta = d.metadata if hasattr(d, "metadata") else {}
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
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

        result = self._agent.invoke({"messages": [HumanMessage(content=state.question)]})
        messages = result.get("messages", [])
        answer: Optional[str] = None
        if messages:
            answer = getattr(messages[-1], "content", None)

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer or "Could not generate answer."
        )