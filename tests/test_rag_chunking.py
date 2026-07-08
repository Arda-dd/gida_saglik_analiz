from pathlib import Path

import pytest

from src.rag.chunking import chunk_knowledge_base, chunk_markdown_file

KB_DOCS_DIR = Path(__file__).resolve().parents[1] / "data" / "knowledge_base" / "docs"


def test_chunk_markdown_file_splits_by_h2_sections():
    chunks = chunk_markdown_file(KB_DOCS_DIR / "who_salt_reduction.md")
    sections = [c.section for c in chunks]

    assert "Özet" in sections
    assert "Bu Projede Kullanımı" in sections
    assert all(c.doc_id == "who_salt_reduction" for c in chunks)


def test_chunk_markdown_file_extracts_metadata():
    chunks = chunk_markdown_file(KB_DOCS_DIR / "who_salt_reduction.md")

    assert all(c.verified is True for c in chunks)
    assert all(c.source is not None and "World Health Organization" in c.source for c in chunks)
    assert all(c.source_url == "https://www.who.int/news-room/fact-sheets/detail/salt-reduction" for c in chunks)


def test_chunk_markdown_file_marks_draft_docs_as_unverified():
    chunks = chunk_markdown_file(KB_DOCS_DIR / "tgk_etiketleme_ozet.md")

    assert all(c.verified is False for c in chunks)


def test_chunk_markdown_file_preserves_preamble_content_before_first_h2():
    # glossary dokumaninda H1 ile ilk "## " arasinda bir TR/EN terim tablosu var - bu
    # sadece metadata degil, gercek icerik oldugundan kaybolmamali.
    chunks = chunk_markdown_file(KB_DOCS_DIR / "glossary_nutrition_terms.md")
    preamble_chunks = [c for c in chunks if c.chunk_id.endswith("::preamble")]

    assert len(preamble_chunks) == 1
    assert "Enerji" in preamble_chunks[0].text
    assert "Energy" in preamble_chunks[0].text


def test_chunk_id_is_unique_within_document():
    chunks = chunk_markdown_file(KB_DOCS_DIR / "who_sugars_intake.md")
    ids = [c.chunk_id for c in chunks]

    assert len(ids) == len(set(ids))


def test_embedding_text_includes_title_and_section():
    chunks = chunk_markdown_file(KB_DOCS_DIR / "who_salt_reduction.md")
    chunk = chunks[0]

    assert chunk.title in chunk.embedding_text
    assert chunk.section in chunk.embedding_text
    assert chunk.text in chunk.embedding_text


def test_chunk_knowledge_base_covers_all_docs():
    chunks = chunk_knowledge_base(KB_DOCS_DIR)
    doc_ids = {c.doc_id for c in chunks}

    expected_docs = {p.stem for p in KB_DOCS_DIR.glob("*.md")}
    assert doc_ids == expected_docs
    assert len(chunks) >= len(expected_docs)


def test_chunk_markdown_file_no_empty_sections():
    chunks = chunk_knowledge_base(KB_DOCS_DIR)
    assert all(c.text.strip() for c in chunks)
