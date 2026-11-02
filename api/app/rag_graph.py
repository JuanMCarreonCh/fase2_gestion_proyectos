"""
LangGraph RAG pipeline.

Flow:
  retrieve → select_best → respond

- retrieve:     query FAISS, get top-K calaveritas
- select_best:  LLM chooses the most fitting one (no modifications)
- respond:      return the selected calaverita + session memory update
"""

import logging
from typing import TypedDict, Annotated
import operator

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

from app.config import LLM_MODEL, OLLAMA_BASE_URL
from app.vector_store import get_retriever

logger = logging.getLogger(__name__)


# ── State ──────────────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    query: str
    session_id: str
    candidates: Annotated[list[str], operator.add]  # top-K retrieved texts
    selected: str                                    # best calaverita chosen by LLM
    answer: str                                      # final answer to user


# ── Nodes ──────────────────────────────────────────────────────────────────────

def retrieve_node(state: RAGState) -> dict:
    """Retrieve top-K calaveritas from FAISS."""
    retriever = get_retriever()
    docs = retriever.invoke(state["query"])
    candidates = [doc.page_content for doc in docs]
    logger.info("Retrieved %d candidates for query: %s", len(candidates), state["query"])
    return {"candidates": candidates}


def select_best_node(state: RAGState) -> dict:
    """Use LLM to select the single most fitting calaverita. No additions."""
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

    numbered = "\n\n".join(
        f"[{i+1}]\n{text}" for i, text in enumerate(state["candidates"])
    )
    prompt = (
        f"El usuario busca una calaverita sobre: '{state['query']}'.\n\n"
        f"Tienes estas opciones:\n\n{numbered}\n\n"
        "Elige SOLAMENTE el número de la calaverita que mejor se adapta al tema. "
        "Responde ÚNICAMENTE con el número (1, 2 o 3), sin explicación."
    )

    response = llm.invoke(prompt)
    raw = response.content.strip()

    # Parse selection index robustly
    selected_idx = 0
    for char in raw:
        if char.isdigit():
            idx = int(char) - 1
            if 0 <= idx < len(state["candidates"]):
                selected_idx = idx
                break

    selected = state["candidates"][selected_idx]
    logger.info("LLM selected candidate index %d", selected_idx)
    return {"selected": selected}


def respond_node(state: RAGState) -> dict:
    """Build the final answer from the selected calaverita."""
    return {"answer": state["selected"]}


# ── Graph ──────────────────────────────────────────────────────────────────────

def build_rag_graph():
    """Compile and return the LangGraph RAG pipeline."""
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("select_best", select_best_node)
    graph.add_node("respond", respond_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "select_best")
    graph.add_edge("select_best", "respond")
    graph.add_edge("respond", END)

    return graph.compile()


# Singleton compiled graph
_rag_graph = None


def get_rag_graph():
    global _rag_graph
    if _rag_graph is None:
        _rag_graph = build_rag_graph()
    return _rag_graph
