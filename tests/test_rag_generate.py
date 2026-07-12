from unittest.mock import MagicMock

import pytest

from src.common.schema import NutritionFacts
from src.rag.chunking import Chunk
from src.rag.generate import (
    _extract_cited_ids,
    _extract_citation_segments,
    _valid_citation_ratio,
    build_prompt,
    compute_numeric_grounding,
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


def test_extract_citation_segments_pairs_preceding_text_with_chunk_id():
    text = "Once bu var [Kaynak: a::0]. Sonra bu var [Kaynak: b::1]."
    segments = _extract_citation_segments(text)

    assert len(segments) == 2
    assert segments[0] == ("Once bu var ", "a::0")
    assert segments[1] == (". Sonra bu var ", "b::1")


def test_extract_citation_segments_empty_when_no_citations():
    assert _extract_citation_segments("Hic kaynak yok.") == []


def test_compute_numeric_grounding_number_present_in_cited_chunk():
    text = "WHO gunluk %10'dan az onerir [Kaynak: who_sugars_intake::0]."
    retrieved = [make_retrieval_result("who_sugars_intake::0")]
    retrieved[0].chunk.text = "Serbest sekerin gunluk enerjinin %10'undan az olmasi onerilir."

    ratio = compute_numeric_grounding(text, retrieved, NutritionFacts())
    assert ratio == 1.0


def test_compute_numeric_grounding_derived_number_not_in_source_is_flagged():
    # Kaynakta "%10" yaziyor ama model kendi hesabiyla "25 gram" turetip sunmus - bu sayi
    # ne kaynakta ne de urun verisinde var, dogru sekilde "dayanaksiz" olarak isaretlenmeli.
    text = "WHO gunluk seker tuketimini 25 grama indirmeyi onerir [Kaynak: who_sugars_intake::0]."
    retrieved = [make_retrieval_result("who_sugars_intake::0")]
    retrieved[0].chunk.text = "Serbest sekerin gunluk enerjinin %10'undan az olmasi onerilir."

    ratio = compute_numeric_grounding(text, retrieved, NutritionFacts())
    assert ratio == 0.0


def test_compute_numeric_grounding_accepts_product_nutrition_numbers():
    # Urunun kendi girdi verisindeki bir sayiyi (35g seker) tekrarlamak halusinasyon degildir,
    # atif yapilan kaynakta gecmese bile "gecerli" (dayanakli) sayilmali.
    text = "Bu urun 35g seker icerir [Kaynak: who_sugars_intake::0]."
    retrieved = [make_retrieval_result("who_sugars_intake::0")]
    retrieved[0].chunk.text = "Serbest sekerin gunluk enerjinin %10'undan az olmasi onerilir."

    ratio = compute_numeric_grounding(text, retrieved, NutritionFacts(sugar_g=35.0))
    assert ratio == 1.0


def test_compute_numeric_grounding_flags_misattributed_number_from_wrong_chunk():
    # "5" sayisi GERCEKTEN retrieval'da var ama baska bir chunk'ta (b::0) - a::0'a atif
    # yapilirken kullanilmasi yanlis kaynak gosterme (misattribution) sayilmali.
    text = "Gunluk tuz alimi 5g'dan az olmali [Kaynak: a::0]."
    chunk_a = make_retrieval_result("a::0")
    chunk_a.chunk.text = "Bu bolumde sayisal bir deger yok."
    chunk_b = make_retrieval_result("b::0")
    chunk_b.chunk.text = "WHO gunluk 5g'dan az tuz onerir."

    ratio = compute_numeric_grounding(text, [chunk_a, chunk_b], NutritionFacts())
    assert ratio == 0.0


def test_compute_numeric_grounding_no_numbers_is_vacuously_grounded():
    text = "Bu urun risklidir [Kaynak: a::0]."
    retrieved = [make_retrieval_result("a::0")]

    ratio = compute_numeric_grounding(text, retrieved, NutritionFacts())
    assert ratio == 1.0


def test_generate_explanation_regenerates_when_numeric_claim_unsupported():
    retrieved = [make_retrieval_result("who_sugars_intake::0")]
    retrieved[0].chunk.text = "Serbest sekerin gunluk enerjinin %10'undan az olmasi onerilir."
    retriever = MagicMock()
    retriever.retrieve.return_value = retrieved

    llm = MagicMock()
    llm.generate.side_effect = [
        # Ilk yanit: gecerli bir chunk_id'ye atif yapiyor ama kaynakta olmayan "25 gram" sayisini turetmis.
        "WHO gunluk seker tuketimini 25 grama indirmeyi onerir [Kaynak: who_sugars_intake::0].",
        # Ikinci yanit: sadece kaynaktaki sayiyi (%10) kullanan duzeltilmis versiyon.
        "WHO gunluk enerjinin %10'undan azini seker olarak onerir [Kaynak: who_sugars_intake::0].",
    ]

    nutrition = NutritionFacts(sugar_g=35.0)
    result = generate_explanation(nutrition, ["yuksek_seker"], retriever, llm, top_k=3)

    assert result.regenerated is True
    assert result.numeric_grounding_ratio == 1.0
    assert llm.generate.call_count == 2
