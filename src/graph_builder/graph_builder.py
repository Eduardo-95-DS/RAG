"""Graph builder for LangGraph workflow"""
from langgraph.graph import StateGraph, END
from src.state.rag_state import RAGState
from src.node.reactnode import RAGNodes


class GraphBuilder:
    """Builds and manages the LangGraph workflow"""

    def __init__(self, retriever, llm):
        self.nodes = RAGNodes(retriever, llm)
        self.graph = None

    def build(self):
        """Build the RAG workflow graph"""
        builder = StateGraph(RAGState)
        builder.add_node("responder", self.nodes.generate_answer)
        builder.set_entry_point("responder")
        builder.add_edge("responder", END)
        self.graph = builder.compile()
        return self.graph

    def run(self, question: str) -> dict:
        """Run the RAG workflow"""
        if self.graph is None:
            self.build()
        initial_state = RAGState(question=question)
        return self.graph.invoke(initial_state)
