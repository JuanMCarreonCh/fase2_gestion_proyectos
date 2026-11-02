"""
Fixtures compartidos para todos los tests del proyecto Calaverita RAG.

- Limpia el estado de memoria entre tests para garantizar aislamiento.
- Provee el cliente HTTP de FastAPI sin levantar el servidor real.
- Provee mocks de Ollama (LLM y embeddings) para tests sin infraestructura.
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# ── Limpieza de memoria entre tests ───────────────────────────────────────────

@pytest.fixture(autouse=True)
def clear_memory_store():
    """
    Limpia el store de sesiones antes de cada test.
    Evita que el estado de un test contamine al siguiente.
    """
    import app.memory as memory_module
    memory_module._session_store.clear()
    yield
    memory_module._session_store.clear()


# ── Cliente FastAPI ────────────────────────────────────────────────────────────

@pytest.fixture
def api_client():
    """
    Cliente de prueba de FastAPI. No requiere levantar el servidor.
    El grafo RAG es mockeado por defecto; úsalo junto a mock_rag_graph.
    """
    from main import app
    return TestClient(app)


# ── Mock del grafo LangGraph ───────────────────────────────────────────────────

@pytest.fixture
def mock_rag_graph():
    """
    Reemplaza el grafo LangGraph compilado con un mock que devuelve
    una calaverita de prueba fija. Evita llamadas reales a Ollama.
    """
    fake_calaverita = (
        "El tester llegó al más allá\n"
        "con sus asserts sin pasar,\n"
        "la Catrina dijo: ¡Acá\n"
        "siempre hay bugs que corregir!"
    )
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {"answer": fake_calaverita}

    with patch("app.router.get_rag_graph", return_value=mock_graph):
        yield mock_graph, fake_calaverita


# ── Mock de embeddings Ollama ──────────────────────────────────────────────────

@pytest.fixture
def mock_embeddings():
    """
    Mock de OllamaEmbeddings que devuelve vectores de ceros.
    Permite construir el índice FAISS sin Ollama corriendo.
    """
    import numpy as np

    mock_emb = MagicMock()
    # embed_documents: lista de textos → lista de vectores
    mock_emb.embed_documents.side_effect = lambda texts: [
        np.zeros(768).tolist() for _ in texts
    ]
    # embed_query: texto → vector
    mock_emb.embed_query.return_value = np.zeros(768).tolist()
    return mock_emb

