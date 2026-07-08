from unittest.mock import MagicMock

import pytest

from src.common.schema import NutritionFacts
from src.rag.chunking import Chunk
from src.rag.generate import (
    _extract_cited_ids,
    _valid_citation_ratio,
    build_prompt,
    generate_explanation,
)
from src.rag.retriever import RetrievalResult


def make_chunk(chunk_id: str, text: str = "ornek metin") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("::")[0],
        title="Test Baslik",
        section="Test Bolum",
        text=text,
        verified=True,
        source="Test Kaynak",
        source_url=None,
    )


def make_retrieval_result(chunk_id: str, score: float = 0.9) -> RetrievalResult:
    return RetrievalResult(chunk=make_chunk(chunk_id), score=score, dense_score=score, bm25_score=score)


def test_extract_cited_ids_finds_all_citation_tags():
    text = "Yuksek seker riski var [Kaynak: doc_a::0]. Ayrica [Kaynak: doc_b::1] da destekler."
    assert _extract_cited_ids(text) == ["doc_a::0", "doc_b::1"]


def test_extract_cited_ids_returns_empty_list_when_no_citations():
    assert _extract_cited_ids("Hicbir kaynak yok burada.") == []


def test_valid_citation_ratio_all_valid():
    ratio = _valid_citation_ratio(["a::0", "b::1"], {"a::0", "b::1", "c::2"})
    assert ratio == 1.0


def test_valid_citation_ratio_partial():
    ratio = _valid_citation_ratio(["a::0", "hallucinated::9"], {"a::0", "b::1"})
    assert ratio == pytest.approx(0.5)


def test_valid_citation_ratio_no_citations_is_zero():
    assert _valid_citation_ratio([], {"a::0"}) == 0.0


def test_build_prompt_includes_risk_descriptions_and_sources():
    nutrition = NutritionFacts(sugar_g=35.0)
    retrieved = [make_retrieval_result("who_sugars_intake::0")]
    prompt = build_prompt(nutrition, ["yuksek_seker"], retrieved)

    assert "Yuksek seker icerir" in prompt
    assert "who_sugars_intake::0" in prompt
    assert "35.0" in prompt


def test_build_prompt_handles_no_risk_flags():
    nutrition = NutritionFacts(sugar_g=2.0)
    prompt = build_prompt(nutrition, [], [])
    assert "Tespit edilen bir risk bayragi yok" in prompt


def test_generate_explanation_accepts_valid_citations_without_regenerating():
    retrieved = [make_retrieval_result("who_sugars_intake::0")]
    retriever = MagicMock()
    retriever.retrieve.return_value = retrieved

    llm = MagicMock()
    llm.generate.return_value = "Bu urun yuksek seker icerir [Kaynak: who_sugars_intake::0]."

    nutrition = NutritionFacts(sugar_g=35.0)
    result = generate_explanation(nutrition, ["yuksek_seker"], retriever, llm, top_k=3)

    assert result.regenerated is False
    assert result.valid_citation_ratio == 1.0
    assert llm.generate.call_count == 1


def test_generate_explanation_regenerates_when_citation_is_hallucinated():
    retrieved = [make_retrieval_result("who_sugars_intake::0")]
    retriever = MagicMock()
    retriever.retrieve.return_value = retrieved

    llm = MagicMock()
    llm.generate.side_effect = [
        "Bu urun riskli [Kaynak: uydurma_kaynak::99].",
        "Bu urun riskli [Kaynak: who_sugars_intake::0].",
    ]

    nutrition = NutritionFacts(sugar_g=35.0)
    result = generate_explanation(nutrition, ["yuksek_seker"], retriever, llm, top_k=3)

    assert result.regenerated is True
    assert result.valid_citation_ratio == 1.0
    assert llm.generate.call_count == 2


def test_generate_explanation_does_not_regenerate_when_no_chunks_retrieved():
    retriever = MagicMock()
    retriever.retrieve.return_value = []

    llm = MagicMock()
    llm.generate.return_value = "Herhangi bir kaynak bulunamadi."

    nutrition = NutritionFacts(sugar_g=2.0)
    result = generate_explanation(nutrition, [], retriever, llm, top_k=3)

    assert result.regenerated is False
    assert llm.generate.call_count == 1
