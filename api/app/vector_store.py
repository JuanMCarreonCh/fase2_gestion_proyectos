"""
FAISS vector store initialization and retriever factory.
Singleton pattern: index is built once on startup.
"""

import logging
from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

from app.config import EMBED_MODEL, OLLAMA_BASE_URL, RETRIEVER_TOP_K
from app.knowledge_base import get_documents

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_vector_store() -> FAISS:
    """
    Build (or load) the FAISS vector store from the calaveritas corpus.
    Uses lru_cache so it's built only once per process lifetime.
    """
    logger.info("Building FAISS index with model: %s", EMBED_MODEL)
    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    documents = get_documents()
    vector_store = FAISS.from_documents(documents, embeddings)
    logger.info("FAISS index built with %d documents.", len(documents))
    return vector_store


def get_retriever():
    """Return a configured retriever from the FAISS vector store."""
    store = get_vector_store()
    return store.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})
