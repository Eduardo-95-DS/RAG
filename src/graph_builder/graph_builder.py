"""Graph builder for LangGraph workflow"""
import time
from langgraph.graph import StateGraph, END
from src.state.rag_state import RAGState
from src.node.reactnode import RAGNodes
from src.logging.rag_logger import get_logger

log = get_logger()


class GraphBuilder:
    """Builds and manages the LangGraph workflow"""

    def __init__(self, retriever, llm):
        self.nodes = RAGNodes(retriever, llm)
        self.graph = None

    def build(self):
        """Build the RAG workflow graph"""
        builder = StateGraph(RAGState)
        builder.add_node("rewriter", self.nodes.rewrite_query)
        builder.add_node("responder", self.nodes.generate_answer)
        builder.add_node("guardrail", self.nodes.ground_check)
        builder.set_entry_point("rewriter")
        builder.add_edge("rewriter", "responder")
        builder.add_edge("responder", "guardrail")
        builder.add_edge("guardrail", END)
        self.graph = builder.compile()
        return self.graph

    def run(self, question: str, history: list = None) -> dict:
        """Run the RAG workflow.

        Args:
            question: The user's current question.
            history: Optional list of prior turns, each a dict with keys 'q' and 'a'.
                     Used by the rewriter to resolve conversational references.
        """
        if self.graph is None:
            self.build()
        log.info("[QUERY] '%s'", question)
        t0 = time.monotonic()
        initial_state = RAGState(question=question, conversation_history=history or [])
        result = self.graph.invoke(initial_state)
        elapsed = time.monotonic() - t0
        answer = result.get("answer", "")
        log.info("[ANSWER] elapsed=%.2fs | answer='%s...'", elapsed, answer[:120].replace("\n", " "))
        return result
