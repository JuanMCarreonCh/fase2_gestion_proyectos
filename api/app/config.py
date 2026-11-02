"""
Configuration settings loaded from environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen2.5:3b")
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "nomic-embed-text")

# FAISS index persistence path (inside container)
FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "/app/data/faiss_index")

# Number of documents to retrieve
RETRIEVER_TOP_K: int = int(os.getenv("RETRIEVER_TOP_K", "3"))
