"""
Streamlit frontend for Calaverita RAG.
- Maintains session_id in st.session_state
- 'Nueva conversación' resets session (calls /reset endpoint)
"""

import uuid
import requests
import streamlit as st

# ── Config ──────────────────────────────────────────────────────────────────────
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_QUERY = f"{API_URL}/api/v1/query"
API_RESET = f"{API_URL}/api/v1/reset"

# ── Page setup ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎃 Calaveritas RAG",
    page_icon="💀",
    layout="centered",
)

# ── Session state init ──────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💀 Calaveritas RAG")
    st.markdown("*Servicio RAG · Día de Muertos*")
    st.divider()

    if st.button("🕯️ Nueva conversación", use_container_width=True):
        # Tell the API to clear the session
        try:
            requests.post(API_RESET, json={"session_id": st.session_state.session_id}, timeout=5)
        except Exception:
            pass
        # Reset local state and generate new session
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(f"Sesión: `{st.session_state.session_id[:8]}...`")
    st.caption("Modelos: qwen2.5:3b · nomic-embed-text")

# ── Header ───────────────────────────────────────────────────────────────────────
st.title("🎃 Calaveritas de Día de Muertos")
st.markdown(
    "Dime **quién es** o **de qué trata** la calaverita y te devolveré la más adecuada."
)
st.divider()

# ── Chat history display ─────────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Chat input ───────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ej: un programador, un médico, un maestro…"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Consultando al oráculo de los muertos..."):
            try:
                response = requests.post(
                    API_QUERY,
                    json={
                        "query": prompt,
                        "session_id": st.session_state.session_id,
                    },
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()
                calaverita = data["calaverita"]
                st.markdown(f"```\n{calaverita}\n```")
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"```\n{calaverita}\n```"}
                )
            except requests.exceptions.ConnectionError:
                error_msg = "❌ No se pudo conectar con la API. ¿Está corriendo el servicio?"
                st.error(error_msg)
            except requests.exceptions.Timeout:
                error_msg = "⏳ La API tardó demasiado. Los modelos pueden estar cargando, intenta de nuevo."
                st.warning(error_msg)
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
