"""
Session-scoped in-memory store.
Each session_id gets its own conversation history.
When a new session is created (or reset), its history is cleared.
"""

from collections import defaultdict
from typing import Any

# Internal store: { session_id: [{"role": ..., "content": ...}, ...] }
_session_store: dict[str, list[dict[str, Any]]] = defaultdict(list)


def get_history(session_id: str) -> list[dict[str, Any]]:
    """Return the message history for a session."""
    return list(_session_store[session_id])


def add_message(session_id: str, role: str, content: str) -> None:
    """Append a message to the session history."""
    _session_store[session_id].append({"role": role, "content": content})


def clear_session(session_id: str) -> None:
    """Wipe all history for a given session (called on 'Nueva conversación')."""
    _session_store.pop(session_id, None)


def list_sessions() -> list[str]:
    """Return all active session IDs (useful for debugging)."""
    return list(_session_store.keys())
