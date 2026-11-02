"""
Tests unitarios para app/rag_graph.py

Mockea ChatOllama y el retriever FAISS para no depender de infraestructura.

Cubre:
- retrieve_node: invoca el retriever y devuelve candidatos
- retrieve_node: manejo cuando el retriever devuelve lista vacía
- select_best_node: construye el prompt con query, candidatos e historial
- select_best_node: devuelve el texto generado por el LLM limpio (strip)
- select_best_node: con historial vacío no incluye calaveritas previas
- select_best_node: con historial filtra solo mensajes del asistente
- respond_node: pasa el campo 'selected' a 'answer' sin modificar
- build_rag_graph: el grafo compila sin errores
- Flujo completo: retrieve → select_best → respond con mocks
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from app.rag_graph import retrieve_node, select_best_node, respond_node, build_rag_graph, RAGState


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_state(**kwargs) -> RAGState:
    """Crea un RAGState con valores por defecto sobreescribibles."""
    defaults: RAGState = {
        "query": "un programador",
        "session_id": "test-session",
        "candidates": [],
        "selected": "",
        "answer": "",
    }
    defaults.update(kwargs)
    return defaults


def make_doc(text: str, tema: str = "test") -> Document:
    return Document(page_content=text, metadata={"id": "t001", "tema": tema})


# ── retrieve_node ─────────────────────────────────────────────────────────────

class TestRetrieveNode:
    def test_returns_candidates_from_retriever(self):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            make_doc("Calaverita A"),
            make_doc("Calaverita B"),
            make_doc("Calaverita C"),
        ]
        with patch("app.rag_graph.get_retriever", return_value=mock_retriever):
            state = make_state(query="un médico")
            result = retrieve_node(state)

        assert "candidates" in result
        assert len(result["candidates"]) == 3
        assert result["candidates"][0] == "Calaverita A"

    def test_retriever_called_with_query(self):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        with patch("app.rag_graph.get_retriever", return_value=mock_retriever):
            state = make_state(query="un futbolista")
            retrieve_node(state)

        mock_retriever.invoke.assert_called_once_with("un futbolista")

    def test_empty_retriever_result_returns_empty_candidates(self):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        with patch("app.rag_graph.get_retriever", return_value=mock_retriever):
            result = retrieve_node(make_state())

        assert result["candidates"] == []


# ── select_best_node ──────────────────────────────────────────────────────────

class TestSelectBestNode:
    GENERATED = "El programador voló al cielo\ncon su teclado sin compilar."

    def _mock_llm(self, content: str = None):
        mock_response = MagicMock()
        mock_response.content = content or self.GENERATED
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        return mock_llm

    def test_returns_selected_with_llm_output(self):
        mock_llm = self._mock_llm()
        with patch("app.rag_graph.ChatOllama", return_value=mock_llm), \
             patch("app.rag_graph.get_history", return_value=[]):
            state = make_state(candidates=["Calaverita A", "Calaverita B", "Calaverita C"])
            result = select_best_node(state)

        assert result["selected"] == self.GENERATED

    def test_strips_whitespace_from_llm_output(self):
        mock_llm = self._mock_llm(content=f"  {self.GENERATED}\n\n")
        with patch("app.rag_graph.ChatOllama", return_value=mock_llm), \
             patch("app.rag_graph.get_history", return_value=[]):
            result = select_best_node(make_state(candidates=["A", "B", "C"]))

        assert result["selected"] == self.GENERATED

    def test_prompt_contains_query(self):
        mock_llm = self._mock_llm()
        with patch("app.rag_graph.ChatOllama", return_value=mock_llm), \
             patch("app.rag_graph.get_history", return_value=[]):
            select_best_node(make_state(query="un músico", candidates=["A"]))

        prompt_sent = mock_llm.invoke.call_args[0][0]
        assert "un músico" in prompt_sent

    def test_prompt_contains_all_candidates(self):
        mock_llm = self._mock_llm()
        candidates = ["Calaverita Uno", "Calaverita Dos", "Calaverita Tres"]
        with patch("app.rag_graph.ChatOllama", return_value=mock_llm), \
             patch("app.rag_graph.get_history", return_value=[]):
            select_best_node(make_state(candidates=candidates))

        prompt_sent = mock_llm.invoke.call_args[0][0]
        for c in candidates:
            assert c in prompt_sent

    def test_prompt_includes_assistant_history(self):
        mock_llm = self._mock_llm()
        history = [
            {"role": "user", "content": "un médico"},
            {"role": "assistant", "content": "El doctor llegó al infierno..."},
        ]
        with patch("app.rag_graph.ChatOllama", return_value=mock_llm), \
             patch("app.rag_graph.get_history", return_value=history):
            select_best_node(make_state(candidates=["A"]))

        prompt_sent = mock_llm.invoke.call_args[0][0]
        assert "El doctor llegó al infierno..." in prompt_sent

    def test_prompt_excludes_user_messages_from_history(self):
        mock_llm = self._mock_llm()
        history = [
            {"role": "user", "content": "MENSAJE_USUARIO_NO_DEBE_APARECER"},
            {"role": "assistant", "content": "respuesta del asistente"},
        ]
        with patch("app.rag_graph.ChatOllama", return_value=mock_llm), \
             patch("app.rag_graph.get_history", return_value=history):
            select_best_node(make_state(candidates=["A"]))

        prompt_sent = mock_llm.invoke.call_args[0][0]
        assert "MENSAJE_USUARIO_NO_DEBE_APARECER" not in prompt_sent

    def test_empty_history_works_without_error(self):
        mock_llm = self._mock_llm()
        with patch("app.rag_graph.ChatOllama", return_value=mock_llm), \
             patch("app.rag_graph.get_history", return_value=[]):
            result = select_best_node(make_state(candidates=["Única candidata"]))

        assert result["selected"] == self.GENERATED

    def test_get_history_called_with_session_id(self):
        mock_llm = self._mock_llm()
        mock_get_history = MagicMock(return_value=[])
        with patch("app.rag_graph.ChatOllama", return_value=mock_llm), \
             patch("app.rag_graph.get_history", mock_get_history):
            select_best_node(make_state(session_id="mi-sesion-123", candidates=["A"]))

        mock_get_history.assert_called_once_with("mi-sesion-123")


# ── respond_node ──────────────────────────────────────────────────────────────

class TestRespondNode:
    def test_passes_selected_to_answer(self):
        calaverita = "El ingeniero calculó la distancia hasta el infierno..."
        state = make_state(selected=calaverita)
        result = respond_node(state)
        assert result["answer"] == calaverita

    def test_does_not_modify_content(self):
        content = "  Calaverita con espacios  \n"
        state = make_state(selected=content)
        result = respond_node(state)
        assert result["answer"] == content  # sin strip, sin modificar

    def test_empty_selected_returns_empty_answer(self):
        state = make_state(selected="")
        result = respond_node(state)
        assert result["answer"] == ""


# ── build_rag_graph ───────────────────────────────────────────────────────────

class TestBuildRagGraph:
    def test_graph_compiles_without_error(self):
        graph = build_rag_graph()
        assert graph is not None

    def test_graph_has_invoke_method(self):
        graph = build_rag_graph()
        assert callable(getattr(graph, "invoke", None))


# ── Flujo completo (integración ligera) ───────────────────────────────────────

class TestFullGraphFlow:
    """
    Prueba el grafo compilado de extremo a extremo con todos los
    componentes externos mockeados (retriever + LLM + memory).
    """

    CALAVERITA_GENERADA = "El tester voló al cielo\ncon sus pruebas sin correr."

    def test_full_flow_returns_answer(self):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            make_doc("Calaverita 1"),
            make_doc("Calaverita 2"),
            make_doc("Calaverita 3"),
        ]

        mock_response = MagicMock()
        mock_response.content = self.CALAVERITA_GENERADA
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        with patch("app.rag_graph.get_retriever", return_value=mock_retriever), \
             patch("app.rag_graph.ChatOllama", return_value=mock_llm), \
             patch("app.rag_graph.get_history", return_value=[]):

            graph = build_rag_graph()
            result = graph.invoke({
                "query": "un tester",
                "session_id": "test-flow",
                "candidates": [],
                "selected": "",
                "answer": "",
            })

        assert result["answer"] == self.CALAVERITA_GENERADA

    def test_full_flow_retriever_is_called(self):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [make_doc("Solo una")]

        mock_response = MagicMock()
        mock_response.content = "generada"
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        with patch("app.rag_graph.get_retriever", return_value=mock_retriever), \
             patch("app.rag_graph.ChatOllama", return_value=mock_llm), \
             patch("app.rag_graph.get_history", return_value=[]):

            graph = build_rag_graph()
            graph.invoke({
                "query": "un artista",
                "session_id": "test-retriever",
                "candidates": [],
                "selected": "",
                "answer": "",
            })

        mock_retriever.invoke.assert_called_once_with("un artista")

