"""
Tests unitarios para app/memory.py

Cubre:
- Historial vacío en sesión nueva
- Agregar mensajes de distintos roles
- Aislamiento entre sesiones distintas
- Clear de sesión borra solo esa sesión
- Clear de sesión inexistente no lanza error
- list_sessions refleja el estado actual
"""

import pytest
from app.memory import add_message, get_history, clear_session, list_sessions


SESSION_A = "session-test-a"
SESSION_B = "session-test-b"


class TestGetHistory:
    def test_empty_session_returns_empty_list(self):
        """Una sesión nueva no debe tener historial."""
        result = get_history(SESSION_A)
        assert result == []

    def test_returns_copy_not_reference(self):
        """get_history debe devolver una copia, no la lista interna."""
        add_message(SESSION_A, "user", "hola")
        history = get_history(SESSION_A)
        history.append({"role": "hacker", "content": "inyectado"})
        # El store interno no debe haberse modificado
        assert len(get_history(SESSION_A)) == 1


class TestAddMessage:
    def test_add_single_user_message(self):
        add_message(SESSION_A, "user", "un programador")
        history = get_history(SESSION_A)
        assert len(history) == 1
        assert history[0] == {"role": "user", "content": "un programador"}

    def test_add_user_and_assistant_messages(self):
        add_message(SESSION_A, "user", "un médico")
        add_message(SESSION_A, "assistant", "El doctor llegó al infierno...")
        history = get_history(SESSION_A)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_messages_preserve_order(self):
        messages = [
            ("user", "primera"),
            ("assistant", "respuesta1"),
            ("user", "segunda"),
            ("assistant", "respuesta2"),
        ]
        for role, content in messages:
            add_message(SESSION_A, role, content)

        history = get_history(SESSION_A)
        for i, (role, content) in enumerate(messages):
            assert history[i]["role"] == role
            assert history[i]["content"] == content

    def test_different_sessions_are_independent(self):
        add_message(SESSION_A, "user", "mensaje de A")
        add_message(SESSION_B, "user", "mensaje de B")

        history_a = get_history(SESSION_A)
        history_b = get_history(SESSION_B)

        assert len(history_a) == 1
        assert len(history_b) == 1
        assert history_a[0]["content"] == "mensaje de A"
        assert history_b[0]["content"] == "mensaje de B"


class TestClearSession:
    def test_clear_removes_all_messages(self):
        add_message(SESSION_A, "user", "algo")
        add_message(SESSION_A, "assistant", "respuesta")
        clear_session(SESSION_A)
        assert get_history(SESSION_A) == []

    def test_clear_nonexistent_session_does_not_raise(self):
        """Borrar una sesión que no existe no debe lanzar excepción."""
        clear_session("session-que-no-existe")  # no debe fallar

    def test_clear_only_affects_target_session(self):
        add_message(SESSION_A, "user", "A sigue aquí")
        add_message(SESSION_B, "user", "B se borra")
        clear_session(SESSION_B)

        assert len(get_history(SESSION_A)) == 1
        assert get_history(SESSION_B) == []

    def test_can_add_messages_after_clear(self):
        add_message(SESSION_A, "user", "antes del clear")
        clear_session(SESSION_A)
        add_message(SESSION_A, "user", "después del clear")

        history = get_history(SESSION_A)
        assert len(history) == 1
        assert history[0]["content"] == "después del clear"


class TestListSessions:
    def test_empty_when_no_sessions(self):
        assert list_sessions() == []

    def test_lists_active_sessions(self):
        add_message(SESSION_A, "user", "x")
        add_message(SESSION_B, "user", "y")
        sessions = list_sessions()
        assert SESSION_A in sessions
        assert SESSION_B in sessions

    def test_cleared_session_not_listed(self):
        add_message(SESSION_A, "user", "x")
        clear_session(SESSION_A)
        assert SESSION_A not in list_sessions()

