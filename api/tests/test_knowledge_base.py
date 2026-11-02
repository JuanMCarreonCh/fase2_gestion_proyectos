"""
Tests unitarios para app/knowledge_base.py

Cubre:
- El corpus CALAVERITAS tiene el mínimo de entradas esperadas
- Cada entrada tiene los campos requeridos (id, tema, texto)
- Los IDs son únicos
- Los temas son únicos (no hay calaveritas duplicadas por tema)
- get_documents() devuelve Documents de LangChain bien formados
- Los metadatos de cada Document coinciden con el corpus
"""

import pytest
from langchain_core.documents import Document

from app.knowledge_base import CALAVERITAS, get_documents

EXPECTED_MIN_COUNT = 10  # mínimo de calaveritas que debe tener el corpus


class TestCalaveritas:
    def test_corpus_has_minimum_entries(self):
        assert len(CALAVERITAS) >= EXPECTED_MIN_COUNT, (
            f"El corpus debe tener al menos {EXPECTED_MIN_COUNT} calaveritas"
        )

    def test_all_entries_have_required_fields(self):
        for calaverita in CALAVERITAS:
            assert "id" in calaverita, f"Falta 'id' en: {calaverita}"
            assert "tema" in calaverita, f"Falta 'tema' en: {calaverita}"
            assert "texto" in calaverita, f"Falta 'texto' en: {calaverita}"

    def test_all_fields_are_non_empty_strings(self):
        for calaverita in CALAVERITAS:
            assert isinstance(calaverita["id"], str) and calaverita["id"].strip()
            assert isinstance(calaverita["tema"], str) and calaverita["tema"].strip()
            assert isinstance(calaverita["texto"], str) and calaverita["texto"].strip()

    def test_ids_are_unique(self):
        ids = [c["id"] for c in CALAVERITAS]
        assert len(ids) == len(set(ids)), "Existen IDs duplicados en el corpus"

    def test_temas_are_unique(self):
        temas = [c["tema"] for c in CALAVERITAS]
        assert len(temas) == len(set(temas)), "Existen temas duplicados en el corpus"

    def test_texto_has_multiple_lines(self):
        """Cada calaverita debe tener al menos 2 versos (líneas)."""
        for calaverita in CALAVERITAS:
            lines = calaverita["texto"].strip().split("\n")
            assert len(lines) >= 2, (
                f"La calaverita '{calaverita['id']}' tiene menos de 2 versos"
            )


class TestGetDocuments:
    def test_returns_list_of_documents(self):
        docs = get_documents()
        assert isinstance(docs, list)
        assert all(isinstance(doc, Document) for doc in docs)

    def test_document_count_matches_corpus(self):
        docs = get_documents()
        assert len(docs) == len(CALAVERITAS)

    def test_document_content_matches_corpus(self):
        docs = get_documents()
        corpus_textos = {c["texto"] for c in CALAVERITAS}
        for doc in docs:
            assert doc.page_content in corpus_textos, (
                f"Texto de Document no encontrado en corpus: {doc.page_content[:40]}"
            )

    def test_document_metadata_has_id_and_tema(self):
        docs = get_documents()
        for doc in docs:
            assert "id" in doc.metadata
            assert "tema" in doc.metadata

    def test_document_metadata_matches_corpus(self):
        docs = get_documents()
        corpus_by_texto = {c["texto"]: c for c in CALAVERITAS}
        for doc in docs:
            original = corpus_by_texto[doc.page_content]
            assert doc.metadata["id"] == original["id"]
            assert doc.metadata["tema"] == original["tema"]

    def test_get_documents_is_deterministic(self):
        """Llamar dos veces debe devolver los mismos documentos en el mismo orden."""
        docs1 = get_documents()
        docs2 = get_documents()
        assert [d.page_content for d in docs1] == [d.page_content for d in docs2]

