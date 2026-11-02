# 💀 Calaveritas RAG · Día de Muertos

Servicio RAG (Retrieval-Augmented Generation) que devuelve **calaveritas temáticas de Día de Muertos** usando LangGraph + FAISS + Ollama.

---

## Índice

1. [¿Qué hace este proyecto?](#-qué-hace-este-proyecto)
2. [Arquitectura](#-arquitectura)
3. [Requisitos previos](#-requisitos-previos)
   - [Docker y Docker Compose](#docker-y-docker-compose)
   - [Python y Conda](#python-y-conda-solo-para-desarrollo-local)
   - [Poetry](#poetry-solo-para-desarrollo-local)
4. [Estructura del proyecto](#-estructura-del-proyecto)
5. [Inicio rápido con Docker](#-inicio-rápido-con-docker-recomendado)
6. [Desarrollo local sin Docker](#-desarrollo-local-sin-docker)
7. [Uso de la aplicación](#-uso-de-la-aplicación)
8. [Referencia de la API](#-referencia-de-la-api)
9. [Extender el proyecto](#-extender-el-proyecto)
10. [Comandos útiles](#-comandos-útiles)
11. [Solución de problemas](#-solución-de-problemas)

---

## 🎃 ¿Qué hace este proyecto?

El usuario escribe un tema (p. ej. *"un programador"*, *"una maestra"*, *"un futbolista"*) y el sistema:

1. **Busca** en la base de conocimientos (FAISS) las 3 calaveritas más similares al tema
2. **Elige** la más adecuada usando un modelo de lenguaje ligero (qwen2.5:3b)
3. **Devuelve** la calaverita exacta sin modificarla ni inventar nada nuevo

El historial de la conversación se mantiene **por sesión**. Al crear una nueva conversación en Streamlit, la memoria se borra completamente.

---

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────┐
│                  Docker Compose                      │
│                                                     │
│  ┌──────────────┐     ┌───────────────────────┐    │
│  │  Streamlit   │────▶│     FastAPI + RAG      │    │
│  │  :8501       │     │     :8000              │    │
│  │  (frontend)  │     │  LangGraph Pipeline:   │    │
│  └──────────────┘     │  retrieve→select→resp  │    │
│                       │  FAISS (in-memory)      │    │
│                       │  Memoria por sesión     │    │
│                       └──────────┬────────────-─┘    │
│                                  │                   │
│                       ┌──────────▼──────────┐       │
│                       │      Ollama         │       │
│                       │      :11434         │       │
│                       │  qwen2.5:3b  (LLM)  │       │
│                       │  nomic-embed-text   │       │
│                       └─────────────────────┘       │
└─────────────────────────────────────────────────────┘
```

| Componente | Tecnología | Para qué |
|---|---|---|
| Frontend | Streamlit | Interfaz de chat |
| Backend / RAG | FastAPI + LangGraph | Lógica del RAG |
| Vector store | FAISS (in-memory) | Búsqueda semántica |
| LLM selector | qwen2.5:3b via Ollama | Elegir la mejor calaverita |
| Embeddings | nomic-embed-text via Ollama | Convertir texto a vectores |
| Dependencias Python | Poetry | Gestión de paquetes del backend |

---

## 📋 Requisitos previos

### Docker y Docker Compose

Docker es la única dependencia **obligatoria** para correr el proyecto completo. Incluye Docker Compose a partir de la versión 20.10.

#### Windows

1. Descarga **Docker Desktop** desde https://www.docker.com/products/docker-desktop/
2. Ejecuta el instalador `.exe` y sigue el asistente (acepta usar WSL 2 si te lo pregunta)
3. Reinicia el equipo cuando se te indique
4. Abre Docker Desktop y espera a que el estado diga **"Engine running"**
5. Verifica la instalación abriendo PowerShell o CMD:
   ```powershell
   docker --version
   docker compose version
   ```
   Deberías ver algo como:
   ```
   Docker version 27.x.x, build ...
   Docker Compose version v2.x.x
   ```

> **WSL 2 (Windows Subsystem for Linux):** Docker Desktop lo instalará automáticamente si no lo tienes. Si te pide activar virtualización en BIOS, reinicia y activa `Intel VT-x` o `AMD-V` desde la configuración de firmware.

#### macOS

1. Descarga **Docker Desktop para Mac** desde https://www.docker.com/products/docker-desktop/
   - Elige la versión correcta: **Apple Silicon (M1/M2/M3)** o **Intel**
2. Arrastra Docker.app a la carpeta Aplicaciones
3. Ábrelo y sigue el asistente de configuración
4. Verifica en Terminal:
   ```bash
   docker --version
   docker compose version
   ```

#### Linux (Ubuntu / Debian)

```bash
# 1. Actualizar paquetes
sudo apt-get update

# 2. Instalar dependencias
sudo apt-get install -y ca-certificates curl gnupg lsb-release

# 3. Agregar repositorio oficial de Docker
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 4. Instalar Docker Engine y Compose
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# 5. Permitir usar Docker sin sudo (requiere cerrar y abrir sesión)
sudo usermod -aG docker $USER

# 6. Verificar
docker --version
docker compose version
```

---

### Python y Conda (solo para desarrollo local)

Solo necesitas Python/Conda si quieres ejecutar el proyecto **sin Docker** (modo desarrollo).

#### Instalar Conda (Miniconda recomendado)

1. Descarga Miniconda desde https://docs.conda.io/en/latest/miniconda.html  
   (elige la versión para tu OS y arquitectura)
2. Instala siguiendo el asistente
3. Abre una terminal nueva y verifica:
   ```bash
   conda --version
   ```

#### Crear el entorno conda del proyecto

```bash
# Crear entorno con Python 3.11
conda create -n fase2gestion python=3.11 -y

# Activar el entorno
conda activate fase2gestion

# Verificar
python --version   # Python 3.11.x
```

---

### Poetry (solo para desarrollo local)

Poetry gestiona las dependencias de Python del backend.

```bash
# Con el entorno conda activado:
conda activate fase2gestion

# Instalar Poetry
pip install poetry

# Verificar
poetry --version
```

> Poetry instalado dentro del entorno conda queda aislado y no afecta otros entornos.

---

### Ollama (solo para desarrollo local)

Si quieres correr sin Docker necesitas Ollama instalado localmente.

- **Windows / macOS:** Descarga desde https://ollama.com/download
- **Linux:**
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

---

## 📁 Estructura del proyecto

```
fase2_gestion_proyectos/
│
├── docker-compose.yml          ← Orquesta los 3 contenedores
├── README.md                   ← Este archivo
│
├── api/                        ← Servicio FastAPI (tiene su propio Dockerfile)
│   ├── Dockerfile              ← Imagen del backend
│   ├── pyproject.toml          ← Dependencias gestionadas con Poetry
│   ├── poetry.lock             ← Versiones exactas (se genera con poetry install)
│   ├── main.py                 ← Punto de entrada: crea la app FastAPI
│   ├── .env.example            ← Plantilla de variables de entorno
│   ├── data/
│   │   └── .gitkeep            ← Reservado para el índice FAISS persistente
│   └── app/
│       ├── __init__.py
│       ├── config.py           ← Lee variables de entorno (.env)
│       ├── knowledge_base.py   ← Corpus: 12 calaveritas temáticas
│       ├── vector_store.py     ← FAISS + OllamaEmbeddings (singleton lru_cache)
│       ├── memory.py           ← Memoria temporal por session_id en RAM
│       ├── rag_graph.py        ← Pipeline LangGraph: retrieve→select_best→respond
│       └── router.py           ← Endpoints FastAPI: /query /reset /history
│
└── frontend/
    └── app.py                  ← Interfaz Streamlit (chat + gestión de sesión)
```

**Archivos clave para entender el RAG:**

| Archivo | Por qué es importante |
|---|---|
| [`rag_graph.py`](api/app/rag_graph.py) | Corazón del sistema. Define el grafo LangGraph con los 3 nodos |
| [`knowledge_base.py`](api/app/knowledge_base.py) | Corpus de calaveritas. Aquí agregas/editas contenido |
| [`vector_store.py`](api/app/vector_store.py) | Crea el índice FAISS y expone el retriever |
| [`memory.py`](api/app/memory.py) | Gestión de sesiones en memoria RAM |
| [`router.py`](api/app/router.py) | Define los endpoints HTTP de la API |

---

## 🚀 Inicio rápido con Docker (recomendado)

Este método levanta todo automáticamente: Ollama, la API y el frontend.

### Paso 1 — Clonar / ubicarse en el proyecto

```bash
cd "fase2_gestion_proyectos"
```

### Paso 2 — Levantar los contenedores

```bash
docker compose up --build
```

**¿Qué ocurre en este momento?**

1. Docker construye la imagen de la API (instala Poetry + dependencias Python)
2. Descarga las imágenes de Ollama y Python para el frontend
3. Ollama arranca y descarga los modelos:
   - `qwen2.5:3b` (~2 GB) — modelo de lenguaje selector
   - `nomic-embed-text` (~274 MB) — embeddings
4. La API espera a que Ollama esté listo (healthcheck automático)
5. El frontend arranca en el puerto 8501

> ⏳ **La primera vez puede tardar 5 a 15 minutos** dependiendo de tu conexión a internet (se descargan ~2.3 GB de modelos). Las siguientes veces es inmediato porque los modelos quedan en caché en el volumen `ollama_data`.

### Paso 3 — Verificar que todo está corriendo

```bash
docker compose ps
```

Deberías ver los 3 contenedores con estado `running` o `healthy`:

```
NAME                  STATUS
ollama                running (healthy)
calaverita_api        running
calaverita_frontend   running
```

### Paso 4 — Abrir la aplicación

| Servicio | URL |
|---|---|
| 💀 Frontend Streamlit | http://localhost:8501 |
| 📖 API Docs (Swagger) | http://localhost:8000/docs |
| ❤️ Health check API | http://localhost:8000/health |

### Parar los contenedores

```bash
# Parar y mantener volúmenes (los modelos Ollama se conservan)
docker compose down

# Parar y eliminar TODO incluyendo modelos descargados
docker compose down -v
```

---

## 💻 Desarrollo local sin Docker

Útil para modificar el código y ver los cambios en tiempo real sin reconstruir imágenes.

### Requisitos

- Conda con el entorno `fase2gestion` creado (ver [sección anterior](#python-y-conda-solo-para-desarrollo-local))
- Poetry instalado en el entorno
- Ollama instalado y corriendo localmente

### Paso 1 — Activar entorno y preparar variables

```bash
conda activate fase2gestion

cd api
cp .env.example .env
```

Edita `.env` y cambia `OLLAMA_BASE_URL` a `http://localhost:11434`:

```env
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=qwen2.5:3b
EMBED_MODEL=nomic-embed-text
RETRIEVER_TOP_K=3
```

### Paso 2 — Instalar dependencias con Poetry

```bash
cd api
poetry install
```

Esto crea el `poetry.lock` con versiones exactas y las instala en el entorno conda activo.

### Paso 3 — Descargar modelos en Ollama local

```bash
# En otra terminal, asegúrate de que Ollama está corriendo:
ollama serve

# Descargar modelos (primera vez):
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

### Paso 4 — Correr la API

```bash
cd api
poetry run uvicorn main:app --reload --port 8000
```

### Paso 5 — Correr el frontend Streamlit

```bash
# En otra terminal, con el entorno activado:
conda activate fase2gestion
pip install streamlit requests

cd frontend
API_URL=http://localhost:8000 streamlit run app.py
```

---

## 🖥️ Uso de la aplicación

### Interfaz Streamlit

1. Abre http://localhost:8501
2. Escribe en el cuadro de chat **el tema** de la calaverita:
   - *"un programador"*
   - *"una maestra de primaria"*
   - *"un futbolista famoso"*
   - *"un médico"*
3. El sistema devuelve la calaverita más adecuada de la base de conocimientos
4. Para empezar una conversación nueva haz clic en **🕯️ Nueva conversación** en el panel lateral — esto borra el historial de la sesión actual

### Flujo interno (LangGraph)

```
Usuario escribe query
       │
       ▼
[FAISS Retriever]
Convierte la query a vector (nomic-embed-text)
Busca los 3 documentos más similares en el índice
       │
       ▼
[LLM Selector - qwen2.5:3b]
Recibe las 3 calaveritas candidatas
Elige SOLO el número de la mejor (temperature=0)
No inventa ni modifica nada
       │
       ▼
[Respond]
Devuelve la calaverita elegida
Guarda en memoria de sesión
       │
       ▼
Streamlit muestra la calaverita
```

---

## 📡 Referencia de la API

### `GET /health`

Verifica que el servicio está activo.

```bash
curl http://localhost:8000/health
# {"status":"ok","service":"calaverita-rag-api"}
```

### `POST /api/v1/query`

Consulta principal RAG. Devuelve la calaverita más adecuada.

**Request:**
```json
{
  "query": "un programador",
  "session_id": "opcional-uuid-de-sesion"
}
```

**Response:**
```json
{
  "session_id": "abc123-...",
  "calaverita": "Un programador llegó a la muerte\ncon su código sin compilar,\nla Catrina le dijo: ¡Suerte!\nen el cielo vas a debuggear.",
  "history": [
    {"role": "user", "content": "un programador"},
    {"role": "assistant", "content": "Un programador llegó a la muerte..."}
  ]
}
```

### `POST /api/v1/reset`

Limpia el historial de una sesión (llamado por Streamlit al crear nueva conversación).

```json
{"session_id": "abc123-..."}
```

### `GET /api/v1/history/{session_id}`

Devuelve el historial completo de una sesión.

```bash
curl http://localhost:8000/api/v1/history/abc123-...
```

> La documentación interactiva completa está en http://localhost:8000/docs (Swagger UI).

---

## 🔧 Extender el proyecto

### Agregar más calaveritas

Edita [`api/app/knowledge_base.py`](api/app/knowledge_base.py) y agrega entradas al array `CALAVERITAS`:

```python
{
    "id": "c013",
    "tema": "chef",
    "texto": (
        "El chef llegó al paraíso\n"
        "con su sazón celestial,\n"
        "la Catrina en improviso\n"
        "le pidió mole especial."
    ),
},
```

El índice FAISS se reconstruye automáticamente al reiniciar el contenedor.

### Cambiar el número de candidatos (top-K)

En `docker-compose.yml`, en el servicio `api`:

```yaml
environment:
  - RETRIEVER_TOP_K=5   # Cambiar de 3 a 5
```

### Cambiar el modelo LLM

```yaml
environment:
  - LLM_MODEL=llama3.2:3b   # Otro modelo disponible en Ollama
```

Y en el servicio `ollama`, agregar el pull del nuevo modelo.

### Agregar un nodo al pipeline LangGraph

En [`api/app/rag_graph.py`](api/app/rag_graph.py), los nodos se agregan así:

```python
def mi_nuevo_nodo(state: RAGState) -> dict:
    # Tu lógica aquí
    return {"campo_del_state": valor}

graph.add_node("mi_nodo", mi_nuevo_nodo)
graph.add_edge("select_best", "mi_nodo")   # Insertar en el flujo
graph.add_edge("mi_nodo", "respond")
```

### Hacer persistente el índice FAISS

En [`api/app/vector_store.py`](api/app/vector_store.py), el índice es in-memory y se reconstruye cada vez. Para hacerlo persistente:

```python
# Guardar
vector_store.save_local(FAISS_INDEX_PATH)

# Cargar (si ya existe)
FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
```

---

## ⚡ Comandos útiles

```bash
# ── Docker ────────────────────────────────────────────────────────────────────
docker compose up --build          # Primera vez o después de cambios en el código
docker compose up -d               # Levantar en background (detached)
docker compose down                # Parar todo (conserva volúmenes)
docker compose down -v             # Parar todo y borrar volúmenes (incluye modelos)
docker compose ps                  # Ver estado de los contenedores
docker compose logs -f             # Ver todos los logs en tiempo real
docker compose logs -f api         # Logs solo del backend
docker compose logs -f ollama      # Ver progreso de descarga de modelos
docker compose restart api         # Reiniciar solo la API

# ── Ollama (dentro del contenedor) ───────────────────────────────────────────
docker exec -it ollama ollama list               # Modelos instalados
docker exec -it ollama ollama pull qwen2.5:3b    # Forzar descarga

# ── Poetry ────────────────────────────────────────────────────────────────────
conda activate fase2gestion
cd api
poetry install                     # Instalar todas las dependencias
poetry add <paquete>               # Agregar nueva dependencia
poetry add --group dev <paquete>   # Dependencia solo de desarrollo
poetry update                      # Actualizar dependencias
poetry show                        # Listar paquetes instalados

# ── Pruebas rápidas de la API ─────────────────────────────────────────────────
curl http://localhost:8000/health

curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"un programador\"}"

curl -X POST http://localhost:8000/api/v1/reset \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"TU-SESSION-ID\"}"
```

---

## 🐛 Solución de problemas

### La API no arranca y hay errores de conexión a Ollama

Ollama puede tardar en descargar los modelos. El `healthcheck` en el compose hace que la API espere, pero si el timeout es muy largo puedes ver el progreso con:

```bash
docker compose logs -f ollama
```

### Error `PyYAML` al instalar paquetes pip

Ocurre si se usa la imagen `python:3.11-slim` sin herramientas de compilación. Este proyecto usa `python:3.11` (imagen completa) para el frontend que incluye `gcc` y `build-essential`. Si ves este error en otro contexto, instala:

```bash
# En Debian/Ubuntu
apt-get install -y build-essential python3-dev
```

### Puerto 8000 o 8501 ya en uso

```bash
# Ver qué proceso usa el puerto
netstat -ano | findstr :8000   # Windows
lsof -i :8000                   # macOS/Linux

# Cambiar el puerto en docker-compose.yml
ports:
  - "8080:8000"   # Usar 8080 en lugar de 8000
```

### Los modelos de Ollama no se descargan

Verifica que el contenedor de Ollama tiene acceso a internet:

```bash
docker exec -it ollama curl -I https://ollama.com
```

Si estás detrás de un proxy corporativo, configura `HTTP_PROXY` y `HTTPS_PROXY` como variables de entorno en el servicio `ollama` del compose.

### Reconstruir desde cero

```bash
docker compose down -v                    # Borrar todo incluyendo modelos
docker compose build --no-cache           # Reconstruir imágenes sin caché
docker compose up                         # Levantar de nuevo
```

---

## 📌 Versiones de modelos

| Modelo | Versión | Tamaño | Propósito |
|---|---|---|---|
| `qwen2.5:3b` | 3B params | ~2.0 GB | Selector del mejor resultado |
| `nomic-embed-text` | v1.5 | ~274 MB | Embeddings multilingüe (español) |

> Los modelos se descargan automáticamente la primera vez y quedan guardados en el volumen Docker `ollama_data`.
