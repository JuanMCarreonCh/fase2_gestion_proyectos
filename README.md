# 💀 Calaveritas RAG · Día de Muertos

Servicio RAG (Retrieval-Augmented Generation) que genera **calaveritas literarias temáticas de Día de Muertos** personalizadas y sin repeticiones, usando **LangGraph + FAISS + Ollama**.

---

## Índice

1. [¿Qué hace este proyecto?](#-qué-hace-este-proyecto)
2. [Arquitectura](#-arquitectura)
3. [Flujo detallado de LangGraph](#-flujo-detallado-de-langgraph)
4. [Requisitos previos](#-requisitos-previos)
   - [Docker y Docker Compose](#docker-y-docker-compose)
   - [Python y Conda (solo desarrollo local)](#python-y-conda-solo-para-desarrollo-local)
   - [Poetry (solo desarrollo local)](#poetry-solo-para-desarrollo-local)
   - [Ollama (solo desarrollo local)](#ollama-solo-para-desarrollo-local)
5. [Estructura del proyecto](#-estructura-del-proyecto)
6. [Inicio rápido con Docker](#-inicio-rápido-con-docker-recomendado)
7. [Desarrollo local sin Docker](#-desarrollo-local-sin-docker)
8. [Uso de la aplicación](#-uso-de-la-aplicación)
9. [Referencia de la API](#-referencia-de-la-api)
10. [Pruebas unitarias](#-pruebas-unitarias)
11. [Extender el proyecto](#-extender-el-proyecto)
12. [Comandos útiles](#-comandos-útiles)
13. [Solución de problemas](#-solución-de-problemas)

---

## 🎃 ¿Qué hace este proyecto?

El usuario escribe un tema, personaje o profesión (p. ej. *"un programador"*, *"una maestra"*, *"un doctor"*), y el sistema ejecuta un flujo RAG inteligente:

1. **Recupera (Retrieval):** Consulta la base vectorial (FAISS) con el modelo de embeddings `nomic-embed-text` para obtener las **top-3 calaveritas base** más afines semánticamente.
2. **Consulta la Memoria de Sesión:** Extrae las calaveritas entregadas previamente en la conversación activa.
3. **Selecciona y Genera (Generation):** El modelo de lenguaje (`qwen2.5:3b`, temperature=0.7) analiza las candidatas, selecciona la más idónea y **compone una nueva calaverita adaptada**. Gracias al contexto del historial, **garantiza que no se repita**, incluso si el usuario vuelve a pedir el mismo tema.
4. **Responde y Persiste:** Entrega la calaverita generada al frontend de Streamlit y almacena la interacción en la memoria temporal de la sesión.

---

## 🏗️ Arquitectura

```
┌────────────────────────────────────────────────────────────────────────┐
│                          Docker Compose                                │
│                                                                        │
│  ┌──────────────┐         ┌──────────────────────────────────────┐     │
│  │  Streamlit   │────────▶│            FastAPI (API)             │     │
│  │  :8501       │         │                :8000                 │     │
│  │  (Frontend)  │◀────────│                                      │     │
│  └──────────────┘         │   ┌──────────────────────────────┐   │     │
│                           │   │       LangGraph Pipeline     │   │     │
│                           │   │ retrieve ─▶ select ─▶ respond│   │     │
│                           │   └───────────────┬──────────────┘   │     │
│                           │                   │                  │     │
│                           │    ┌──────────────┴──────────────┐   │     │
│                           │    │ FAISS (Vector Store en RAM) │   │     │
│                           │    │ Memoria Temporal de Sesión  │   │     │
│                           │    └──────────────┬──────────────┘   │     │
│                           └───────────────────┼──────────────────┘     │
│                                               │ HTTP / REST            │
│                           ┌───────────────────▼──────────────────┐     │
│                           │             Ollama Server            │     │
│                           │                :11434                │     │
│                           │   • qwen2.5:3b (SLM Generador)       │     │
│                           │   • nomic-embed-text (Embeddings)    │     │
│                           │   • Volumen local: ./ollama-data     │     │
│                           └──────────────────────────────────────┘     │
└────────────────────────────────────────────────────────────────────────┘
```

| Componente | Tecnología | Rol en el sistema |
|---|---|---|
| **Frontend** | Streamlit | Chat reactivo, gestión de sesión y botón de reinicio |
| **Backend & Grafo** | FastAPI + LangGraph | Orquestación del estado, endpoints REST y pipeline |
| **Vector Store** | FAISS (in-memory) | Indexación y búsqueda semántica de documentos base |
| **LLM Generador** | `qwen2.5:3b` via Ollama | Selección contextual y composición de versos únicos |
| **Embeddings** | `nomic-embed-text` | Vectorización en español (768 dimensiones) |
| **Gestión de Paquetes** | Poetry | Aislamiento y control estricto de dependencias |

---

## 🔄 Flujo detallado de LangGraph

El pipeline se implementa mediante un grafo dirigido acíclico (`StateGraph`) con memoria de sesión:

```
                  ┌────────────────────────┐
                  │         START          │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │     retrieve_node      │
                  │   Búsqueda en FAISS    │
                  │   (Top 3 candidatos)   │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │    select_best_node    │
                  │  1. Lee historial      │
                  │  2. Prompt con reglas  │
                  │  3. LLM genera nueva   │
                  │     (anti-repetición)  │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │      respond_node      │
                  │  Empaqueta respuesta   │
                  │  final en state        │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │          END           │
                  └────────────────────────┘
```

### Nodos del Grafo

1. **`retrieve_node`**:
   - Recibe `state["query"]`.
   - El retriever de FAISS convierte la consulta a vector usando `nomic-embed-text`.
   - Recupera los 3 textos de calaveritas más afines y los almacena en `state["candidates"]`.

2. **`select_best_node`**:
   - Consulta el historial de la sesión (`app.memory.get_history`) y extrae únicamente las respuestas previas generadas (`role == "assistant"`).
   - Genera el prompt estructurado:
     ```python
     f"El usuario busca una calaverita sobre: '{state['query']}'.\n\n"
     f"Tienes estas opciones:\n\n{candidates_numbered}\n\n"
     "Elige la que mejor se adapte a lo que quiere el usuario y genera una en base a la elección. "
     f"Las calaveritas no se deben repetir, aunque pida lo mismo, así que toma en cuenta el historial: {session_messages}\n\n"
     "Responde ÚNICAMENTE con la calaverita generada, sin explicaciones, sin numeración, sin comentarios adicionales."
     ```
   - Invoca a `ChatOllama(model="qwen2.5:3b", temperature=0.7)` para permitir variación poética sin inventar formatos extraños.
   - Guarda el verso resultante en `state["selected"]`.

3. **`respond_node`**:
   - Coloca `state["selected"]` en `state["answer"]`.
   - Finaliza el grafo y retorna el resultado al router de FastAPI, el cual almacena el turno en la memoria de la sesión.

---

## 📋 Requisitos previos

### Docker y Docker Compose

Docker es el requisito principal para ejecutar todo el stack contenedorizado de forma homogénea.

#### Windows
1. Descarga **Docker Desktop** desde https://www.docker.com/products/docker-desktop/
2. Ejecuta el instalador marcando **"Use WSL 2 instead of Hyper-V"**.
3. Reinicia tu equipo.
4. Abre Docker Desktop y espera a que el estado sea **"Engine running"**.
5. Verifica en PowerShell:
   ```powershell
   docker --version
   docker compose version
   ```

#### Linux (Ubuntu / Debian)
```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
# Añadir repo oficial de Docker y luego:
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo usermod -aG docker $USER
```

#### macOS
Descarga e instala Docker Desktop para Mac (Apple Silicon o Intel) desde el sitio oficial.

---

### Python y Conda (solo para desarrollo local)

Si prefieres trabajar fuera de Docker:

```bash
# Crear entorno conda con Python 3.11
conda create -n fase2gestion python=3.11 -y
conda activate fase2gestion
```

### Poetry (solo para desarrollo local)

```bash
conda activate fase2gestion
pip install poetry
poetry --version
```

### Ollama (solo para desarrollo local)

Instala Ollama en tu sistema operativo desde https://ollama.com/download y levanta el servicio:

```bash
ollama serve
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

---

## 📁 Estructura del proyecto

```
fase2_gestion_proyectos/
│
├── docker-compose.yml          ← Orquesta Ollama, descargador de modelos, API y Streamlit
├── README.md                   ← Documentación completa del proyecto
├── .gitignore                  ← Ignora entornos, caches y carpeta de modelos ./ollama-data
│
├── api/                        ← Microservicio Backend (FastAPI + LangGraph)
│   ├── Dockerfile              ← Dockerfile optimizado para producción con Poetry
│   ├── pyproject.toml          ← Especificación de dependencias (fastapi, langgraph, pytest, etc.)
│   ├── poetry.lock             ← Lockfile de versiones exactas
│   ├── main.py                 ← Punto de entrada FastAPI y middlewares CORS
│   ├── .env.example            ← Plantilla de configuración de variables de entorno
│   │
│   ├── app/                    ← Código fuente de la aplicación
│   │   ├── __init__.py
│   │   ├── config.py           ← Parámetros de configuración (modelos, URLs, K)
│   │   ├── knowledge_base.py   ← Corpus de 12 calaveritas temáticas y conversor a Documents
│   │   ├── vector_store.py     ← Inicializador singleton de FAISS y OllamaEmbeddings
│   │   ├── memory.py           ← Memoria temporal en RAM segmentada por session_id
│   │   ├── rag_graph.py        ← Grafo LangGraph (retrieve -> select_best -> respond)
│   │   └── router.py           ← Endpoints REST (/query, /reset, /history)
│   │
│   ├── tests/                  ← Suite de Pruebas Unitarias
│   │   ├── __init__.py
│   │   ├── conftest.py         ← Fixtures compartidos, mocks de Ollama y TestClient
│   │   ├── test_memory.py      ← Pruebas del almacén de sesiones
│   │   ├── test_knowledge_base.py ← Pruebas de integridad del corpus
│   │   ├── test_router.py      ← Pruebas de endpoints HTTP y manejo de errores
│   │   └── test_rag_graph.py   ← Pruebas de nodos, prompt, anti-repetición y flujo
│   │
│   └── data/
│       └── .gitkeep            ← Directorio reservado para persistencia vectorial
│
└── frontend/
    └── app.py                  ← Interfaz de usuario interactiva en Streamlit
```

---

## 🚀 Inicio rápido con Docker (Recomendado)

### 1. Clonar y posicionarse en el proyecto
```bash
cd "c:\Users\juani\Desktop\MasterIA\Curso5. Gestion de Proyectos\Fase2\fase2_gestion_proyectos"
```

### 2. Levantar todos los contenedores
```bash
docker compose up --build
```

**Comportamiento de persistencia de modelos:**
- Los modelos se descargan automáticamente dentro de `./ollama-data/` en la raíz del proyecto.
- Ejecutar `docker compose down` o `docker compose down -v` **NO** borra los modelos descargados, evitando descargas innecesarias en el futuro.

### 3. Acceso a los servicios
- 💀 **Frontend Streamlit**: [http://localhost:8501](http://localhost:8501)
- 📖 **Documentación Swagger API**: [http://localhost:8000/docs](http://localhost:8000/docs)
- ❤️ **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## 💻 Desarrollo local sin Docker

```bash
# 1. Activar entorno Conda
conda activate fase2gestion

# 2. Instalar dependencias con Poetry
cd api
poetry install

# 3. Configurar entorno
cp .env.example .env
# Asegurarse que OLLAMA_BASE_URL=http://localhost:11434

# 4. Iniciar Backend
poetry run uvicorn main:app --reload --port 8000

# 5. Iniciar Frontend (en otra terminal)
conda activate fase2gestion
pip install streamlit requests
cd frontend
streamlit run app.py
```

---

## 🖥️ Uso de la aplicación

1. Abre [http://localhost:8501](http://localhost:8501).
2. Escribe una profesión o tema en el chat: *"un programador"*, *"un bombero"*, *"un estudiante"*.
3. La aplicación responderá con una calaverita en verso inspirada en la base de datos pero generada exclusivamente para la solicitud.
4. Si vuelves a escribir *"un programador"*, el modelo utilizará el historial de la sesión para componer una variante diferente sin repetir la anterior.
5. Haz clic en **🕯️ Nueva conversación** en la barra lateral para borrar la memoria y reiniciar la sesión.

---

## 📡 Referencia de la API

### `POST /api/v1/query`
Envía una consulta y obtiene una calaverita generada manteniendo el contexto de la sesión.

**Request:**
```json
{
  "query": "un programador",
  "session_id": "f83a45c2-b912-4d2b-9871-38290123abcd"
}
```

**Response:**
```json
{
  "session_id": "f83a45c2-b912-4d2b-9871-38290123abcd",
  "calaverita": "El programador tecleaba con pasión,\nla Catrina le hizo un pull request,\nse lo llevó directo al panteón\nsin darle tiempo de hacer el test.",
  "history": [
    {"role": "user", "content": "un programador"},
    {"role": "assistant", "content": "El programador tecleaba..."}
  ]
}
```

### `POST /api/v1/reset`
Limpia la memoria asociada a una sesión específica.

```json
{
  "session_id": "f83a45c2-b912-4d2b-9871-38290123abcd"
}
```

### `GET /api/v1/history/{session_id}`
Consulta el historial de mensajes de la sesión indicada.

---

## 🧪 Pruebas unitarias

El proyecto cuenta con una batería completa de pruebas unitarias implementadas con `pytest`. Todas las pruebas mockean Ollama y los modelos de lenguaje, permitiendo una ejecución instantánea y determinista sin requerir servicios externos corriendo.

### Ejecución de tests

#### Con Poetry (local):
```bash
cd api
poetry run pytest -v
```

#### Dentro del contenedor Docker:
```bash
docker exec -it calaverita_api pytest -v
```

### Cobertura de pruebas

| Archivo de Test | Módulo Evaluado | Aspectos Verificados |
|---|---|---|
| `test_memory.py` | `app/memory.py` | Aislamiento entre sesiones, preservación de orden, eliminación controlada, inmutabilidad de lecturas. |
| `test_knowledge_base.py` | `app/knowledge_base.py` | Integridad de campos (`id`, `tema`, `texto`), unicidad de IDs y temas, conversión adecuada a `Document` de LangChain. |
| `test_router.py` | `app/router.py` | Códigos HTTP (200, 500), auto-generación de `session_id`, acumulación del historial, endpoint `/reset` y `/history`. |
| `test_rag_graph.py` | `app/rag_graph.py` | Invocación de `retrieve_node`, construcción del prompt anti-repetición con historial en `select_best_node`, ejecución de `respond_node` y compilación integral del grafo. |

---

## 🔧 Extender el proyecto

### Modificar o agregar calaveritas base
Edita `api/app/knowledge_base.py` y agrega nuevos registros a la lista `CALAVERITAS`. El índice FAISS se recrea en memoria al reiniciar la API.

### Cambiar parámetros del modelo o creatividad
En `api/app/rag_graph.py`, puedes ajustar el parámetro `temperature` de `ChatOllama`:
- `temperature=0.2`: Calaveritas más apegadas literalmente a los candidatos base.
- `temperature=0.7`: Mayor variedad léxica y originalidad en los versos.

---

## ⚡ Comandos útiles

```bash
# Ver estado de los contenedores
docker compose ps

# Monitorear logs en vivo
docker compose logs -f api
docker compose logs -f ollama

# Parar la aplicación conservando los modelos descargados
docker compose down

# Ejecutar las pruebas unitarias
docker exec -it calaverita_api pytest -v
```

---

## 🐛 Solución de problemas

- **Ollama tarda en responder la primera consulta:** La primera vez que el SLM `qwen2.5:3b` se carga en la memoria RAM del contenedor puede tardar unos segundos. Las llamadas subsecuentes responderán con mucha mayor velocidad.
- **Error de conexión con Ollama en desarrollo local:** Asegúrate de ejecutar `ollama serve` en una terminal y validar que `http://localhost:11434` responda.
