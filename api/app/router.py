"""
FastAPI router — exposes RAG endpoints.
"""

import uuid
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.memory import add_message, get_history, clear_session
from app.rag_graph import get_rag_graph

router = APIRouter()
logger = logging.getLogger(__name__)


# ── Request / Response schemas ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., description="Lo que el usuario quiere que trate la calaverita")
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class QueryResponse(BaseModel):
    session_id: str
    calaverita: str
    history: list[dict]


class ResetRequest(BaseModel):
    session_id: str


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query_calaverita(request: QueryRequest):
    """
    Main RAG endpoint.
    Retrieves the most fitting calaverita for the given query.
    """
    try:
        rag = get_rag_graph()
        result = rag.invoke(
            {
                "query": request.query,
                "session_id": request.session_id,
                "candidates": [],
                "selected": "",
                "answer": "",
            }
        )
        calaverita = result["answer"]

        # Persist to session memory
        add_message(request.session_id, "user", request.query)
        add_message(request.session_id, "assistant", calaverita)

        return QueryResponse(
            session_id=request.session_id,
            calaverita=calaverita,
            history=get_history(request.session_id),
        )
    except Exception as exc:
        logger.exception("Error processing query: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/reset")
async def reset_session(request: ResetRequest):
    """Clear session memory (nueva conversación en Streamlit)."""
    clear_session(request.session_id)
    return {"message": "Sesión reiniciada", "session_id": request.session_id}


@router.get("/history/{session_id}")
async def get_session_history(session_id: str):
    """Return current session history."""
    return {"session_id": session_id, "history": get_history(session_id)}
