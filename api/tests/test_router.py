"""
Tests unitarios para los endpoints FastAPI (app/router.py + main.py)

Todos los tests mockean el grafo LangGraph para no depender de Ollama.

Cubre:
- GET  /health
- POST /api/v1/query  → respuesta correcta, historial actualizado, session_id generado
- POST /api/v1/query  → múltiples consultas acumulan historial
- POST /api/v1/reset  → limpia la sesión
- GET  /api/v1/history/{session_id}  → devuelve historial correcto
- Manejo de errores 500 cuando el grafo falla
"""

import uuid
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

FAKE_CALAVERITA = (
    "El tester llegó al más allá\n"
    "con sus asserts sin pasar,\n"
    "la Catrina dijo: ¡Acá\n"
    "siempre hay bugs que corregir!"
)


# ── Fixture local: mock del grafo ─────────────────────────────────────────────

@pytest.fixture
def mock_graph():
    mock = MagicMock()
    mock.invoke.return_value = {"answer": FAKE_CALAVERITA}
    with patch("app.router.get_rag_graph", return_value=mock):
        yield mock


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_response_contains_status_ok(self):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"

    def test_response_contains_service_name(self):
        response = client.get("/health")
        data = response.json()
        assert "service" in data
        assert data["service"] == "calaverita-rag-api"


# ── POST /api/v1/query ────────────────────────────────────────────────────────

class TestQueryEndpoint:
    def test_returns_200_with_valid_query(self, mock_graph):
        response = client.post(
            "/api/v1/query",
            json={"query": "un programador", "session_id": "test-session-1"},
        )
        assert response.status_code == 200

    def test_response_contains_calaverita(self, mock_graph):
        response = client.post(
            "/api/v1/query",
            json={"query": "un médico", "session_id": "test-session-2"},
        )
        data = response.json()
        assert "calaverita" in data
        assert data["calaverita"] == FAKE_CALAVERITA

    def test_response_contains_session_id(self, mock_graph):
        session_id = str(uuid.uuid4())
        response = client.post(
            "/api/v1/query",
            json={"query": "un maestro", "session_id": session_id},
        )
        data = response.json()
        assert data["session_id"] == session_id

    def test_auto_generates_session_id_when_not_provided(self, mock_graph):
        response = client.post(
            "/api/v1/query",
            json={"query": "un futbolista"},
        )
        data = response.json()
        assert "session_id" in data
        assert len(data["session_id"]) > 0

    def test_history_updated_after_query(self, mock_graph):
        session_id = "test-history-update"
        response = client.post(
            "/api/v1/query",
            json={"query": "un abogado", "session_id": session_id},
        )
        data = response.json()
        history = data["history"]

        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "un abogado"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == FAKE_CALAVERITA

    def test_multiple_queries_accumulate_history(self, mock_graph):
        session_id = "test-accumulate"
        for query in ["un músico", "un ingeniero"]:
            client.post(
                "/api/v1/query",
                json={"query": query, "session_id": session_id},
            )
        response = client.post(
            "/api/v1/query",
            json={"query": "un artista", "session_id": session_id},
        )
        data = response.json()
        # 3 queries × 2 mensajes (user + assistant) = 6 mensajes
        assert len(data["history"]) == 6

    def test_rag_graph_invoked_with_correct_state(self, mock_graph):
        session_id = "test-graph-invoke"
        client.post(
            "/api/v1/query",
            json={"query": "un cocinero", "session_id": session_id},
        )
        mock_graph.invoke.assert_called_once()
        call_args = mock_graph.invoke.call_args[0][0]
        assert call_args["query"] == "un cocinero"
        assert call_args["session_id"] == session_id

    def test_returns_500_when_graph_raises(self):
        broken_graph = MagicMock()
        broken_graph.invoke.side_effect = RuntimeError("Ollama no disponible")
        with patch("app.router.get_rag_graph", return_value=broken_graph):
            response = client.post(
                "/api/v1/query",
                json={"query": "un deportista", "session_id": "test-error"},
            )
        assert response.status_code == 500
        assert "Ollama no disponible" in response.json()["detail"]

    def test_empty_query_still_calls_graph(self, mock_graph):
        """Una query vacía es válida para el endpoint (validación es del modelo)."""
        response = client.post(
            "/api/v1/query",
            json={"query": "", "session_id": "test-empty"},
        )
        assert response.status_code == 200


# ── POST /api/v1/reset ────────────────────────────────────────────────────────

class TestResetEndpoint:
    def test_reset_returns_200(self, mock_graph):
        session_id = "test-reset-1"
        client.post("/api/v1/query", json={"query": "algo", "session_id": session_id})
        response = client.post("/api/v1/reset", json={"session_id": session_id})
        assert response.status_code == 200

    def test_reset_clears_history(self, mock_graph):
        session_id = "test-reset-2"
        client.post("/api/v1/query", json={"query": "algo", "session_id": session_id})
        client.post("/api/v1/reset", json={"session_id": session_id})

        # Verificar que el historial está vacío después del reset
        history_response = client.get(f"/api/v1/history/{session_id}")
        assert history_response.json()["history"] == []

    def test_reset_nonexistent_session_returns_200(self):
        """Resetear una sesión que no existe no debe fallar."""
        response = client.post(
            "/api/v1/reset",
            json={"session_id": "sesion-inexistente-xyz"},
        )
        assert response.status_code == 200

    def test_reset_response_contains_session_id(self, mock_graph):
        session_id = "test-reset-response"
        response = client.post("/api/v1/reset", json={"session_id": session_id})
        data = response.json()
        assert data["session_id"] == session_id

    def test_can_query_after_reset(self, mock_graph):
        session_id = "test-reuse-after-reset"
        client.post("/api/v1/query", json={"query": "antes", "session_id": session_id})
        client.post("/api/v1/reset", json={"session_id": session_id})
        response = client.post(
            "/api/v1/query",
            json={"query": "después del reset", "session_id": session_id},
        )
        data = response.json()
        # Solo 2 mensajes (user + assistant del query post-reset)
        assert len(data["history"]) == 2
        assert data["history"][0]["content"] == "después del reset"


# ── GET /api/v1/history/{session_id} ─────────────────────────────────────────

class TestHistoryEndpoint:
    def test_empty_history_for_new_session(self):
        response = client.get("/api/v1/history/sesion-nueva-xyz")
        assert response.status_code == 200
        assert response.json()["history"] == []

    def test_history_reflects_messages(self, mock_graph):
        session_id = "test-history-reflect"
        client.post("/api/v1/query", json={"query": "un estudiante", "session_id": session_id})
        response = client.get(f"/api/v1/history/{session_id}")
        data = response.json()
        assert data["session_id"] == session_id
        assert len(data["history"]) == 2

    def test_history_contains_correct_roles(self, mock_graph):
        session_id = "test-history-roles"
        client.post("/api/v1/query", json={"query": "un político", "session_id": session_id})
        response = client.get(f"/api/v1/history/{session_id}")
        history = response.json()["history"]
        roles = [msg["role"] for msg in history]
        assert roles == ["user", "assistant"]

