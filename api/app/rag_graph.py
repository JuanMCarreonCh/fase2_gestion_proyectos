"""
LangGraph RAG pipeline.

Flow:
  retrieve → select_best → respond

- retrieve:     query FAISS, get top-K calaveritas
- select_best:  LLM elige la mejor candidata y genera una variación nueva,
                tomando en cuenta el historial de la sesión para no repetir.
- respond:      devuelve la calaverita generada.
"""

import logging
from typing import TypedDict, Annotated
import operator

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

from app.config import LLM_MODEL, OLLAMA_BASE_URL
from app.vector_store import get_retriever
from app.memory import get_history

logger = logging.getLogger(__name__)


# ── State ──────────────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    query: str
    session_id: str
    candidates: Annotated[list[str], operator.add]  # top-K retrieved texts
    selected: str                                    # calaverita generada por el LLM
    answer: str                                      # respuesta final al usuario


# ── Nodes ──────────────────────────────────────────────────────────────────────

def retrieve_node(state: RAGState) -> dict:
    """Retrieve top-K calaveritas from FAISS."""
    retriever = get_retriever()
    docs = retriever.invoke(state["query"])
    candidates = [doc.page_content for doc in docs]
    logger.info("Retrieved %d candidates for query: %s", len(candidates), state["query"])
    return {"candidates": candidates}


def select_best_node(state: RAGState) -> dict:
    """
    LLM selects the best candidate and generates a new calaverita based on it.
    Uses session history to avoid repeating previously delivered calaveritas.
    """
    # temperatura > 0 para que haya variación en la generación
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.7)

    numbered = "\n\n".join(
        f"[{i+1}]\n{text}" for i, text in enumerate(state["candidates"])
    )

    # Extraer solo los mensajes del asistente (calaveritas ya entregadas)
    history = get_history(state["session_id"])
    session_messages = [
        msg["content"]
        for msg in history
        if msg["role"] == "assistant"
    ]

    prompt = (
        f"El usuario busca una calaverita sobre: '{state['query']}'.\n\n"
        f"Tienes estas opciones:\n\n{numbered}\n\n"
        "Elige la que mejor se adapte a lo que quiere el usuario y genera una en base a la elección. "
        f"Las calaveritas no se deben repetir, aunque pida lo mismo, así que toma en cuenta el historial: {session_messages}\n\n"
        "Responde ÚNICAMENTE con la calaverita generada."
        "En caso de que el usuario solo pida una nueva u otra, devuelve solo una variacion de la ultima que generaste"
    )

    response = llm.invoke(prompt)
    selected = response.content.strip()
    logger.info("LLM generated calaverita for session %s", state["session_id"])
    return {"selected": selected}


def respond_node(state: RAGState) -> dict:
    """Devuelve la calaverita generada como respuesta final."""
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
