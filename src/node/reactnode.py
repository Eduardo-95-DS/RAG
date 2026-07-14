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

    FALLBACK_ANSWER = (
        "I wasn't able to find a reliable answer in the NVIDIA 2025 Annual Report "
        "for that question."
    )

    # Positive polarity (YES = grounded/good). The previous inverted phrasing
    # (YES = unsupported) is a classic footgun: models frequently answer "YES"
    # meaning "yes, it's grounded", the opposite of intended, causing correct
    # answers to be rejected. Kept simple and one-word so parsing is unambiguous.
    GROUND_CHECK_PROMPT = (
        "You are a grounding checker. Given retrieved document passages and an "
        "AI-generated answer, decide whether the answer is fully supported by the "
        "passages.\n"
        "Answer YES if every claim in the answer is supported by the passages.\n"
        "Answer NO if the answer contains any claim not supported by the passages.\n"
        "Output only the single word YES or NO."
    )

    def __init__(self, retriever, llm):
        self.retriever = retriever
        # Plain primary (with retries): used by the ReAct agent, which calls
        # bind_tools() on it. Must NOT be a fallback-wrapped runnable.
        self.llm = llm
        # Fallback-wrapped (primary -> smaller model) for the plain .invoke()
        # sites (rewriter, ground check), where bind_tools is never called so a
        # RunnableWithFallbacks is fine. Built here rather than passed in so the
        # GraphBuilder/streamlit call signature is unchanged.
        from src.config.config import Config
        self.llm_fallback = Config.get_llm_with_fallback()
        self._agent = None
        self._last_retrieved: List[Document] = []

    REWRITE_PROMPT = (
        "You are a query rewriter for a financial document search system.\n"
        "You will receive a current question and, optionally, recent conversation history.\n\n"
        "Do two things in order:\n"
        "1. Resolve any references that depend on prior turns. "
        "For example, if the history shows the last question was about data center revenue "
        "and the current question is 'How does that compare to last year?', "
        "rewrite it as 'How did NVIDIA data center revenue in FY2025 compare to FY2024?'\n"
        "2. Reformulate the result as a precise, self-contained query optimized for "
        "searching a financial annual report: remove conversational phrasing, expand abbreviations.\n\n"
        "If there is no conversation history, or the question is already self-contained, "
        "just do step 2.\n\n"
        "Output only the rewritten query — no explanation, no preamble."
    )

    def rewrite_query(self, state: RAGState) -> RAGState:
        """Rewrite the raw question into a retrieval-optimized, self-contained query.

        If conversation_history is present, the prompt instructs the model to first
        resolve any references to prior turns (e.g. 'that', 'them', 'last year')
        before reformulating for retrieval.
        """
        # Build an optional history block to prepend to the user message.
        history_block = ""
        if state.conversation_history:
            turns = state.conversation_history[-3:]  # last 3 turns is enough context
            formatted = "\n".join(f"Q: {t['q']}\nA: {t['a']}" for t in turns)
            history_block = f"Conversation history:\n{formatted}\n\n"

        user_content = f"{history_block}Current question: {state.question}"

        messages = [
            SystemMessage(content=self.REWRITE_PROMPT),
            HumanMessage(content=user_content),
        ]
        # Fallback-wrapped: a transient primary failure here retries, then falls
        # back to the smaller model rather than erroring at the user.
        response = self.llm_fallback.invoke(messages)
        rewritten = response.content.strip()
        log.info("[REWRITE] original='%s' | rewritten='%s'", state.question, rewritten)
        return RAGState(
            question=state.question,
            rewritten_query=rewritten,
            retrieved_docs=state.retrieved_docs,
            answer=state.answer,
            conversation_history=state.conversation_history,
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
                self._last_retrieved = []
                return "No documents found."
            log.info("[TOOL] retriever returned %d chunks | query='%s'", len(docs), query)
            self._last_retrieved = docs
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
            rewritten_query=state.rewritten_query,
            retrieved_docs=self._last_retrieved,
            answer=answer or "Could not generate answer.",
        )

    def ground_check(self, state: RAGState) -> RAGState:
        """Check whether the answer is grounded in the retrieved chunks."""
        if not state.retrieved_docs:
            log.warning("[GROUND] no retrieved docs — returning fallback")
            return RAGState(
                question=state.question,
                rewritten_query=state.rewritten_query,
                retrieved_docs=state.retrieved_docs,
                answer=self.FALLBACK_ANSWER,
            )

        context = "\n\n".join(d.page_content for d in state.retrieved_docs[:8])
        user_msg = (
            f"Passages:\n{context}\n\n"
            f"Answer:\n{state.answer}\n\n"
            "Is the answer fully supported by the passages? YES or NO."
        )
        # Fallback-wrapped: same rationale as the rewriter.
        response = self.llm_fallback.invoke([
            SystemMessage(content=self.GROUND_CHECK_PROMPT),
            HumanMessage(content=user_msg),
        ])
        verdict = response.content.strip().upper()
        # Strict parse: grounded only if the verdict *starts with* YES. A bare
        # substring match ("YES" in verdict) misfires on any stray "yes"; and
        # anything ambiguous or empty should fail closed to the fallback. So we
        # keep the original answer only on an explicit YES, else use the fallback.
        grounded = verdict.startswith("YES")
        log.info("[GROUND] verdict='%s' grounded=%s | answer='%s...'",
                 verdict, grounded, state.answer[:80])

        final_answer = state.answer if grounded else self.FALLBACK_ANSWER
        return RAGState(
            question=state.question,
            rewritten_query=state.rewritten_query,
            retrieved_docs=state.retrieved_docs,
            answer=final_answer,
        )