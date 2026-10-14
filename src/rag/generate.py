"""RAG generation katmani: retrieval sonuclarini + kural motoru risk bayraklarini LLM'e vererek
kaynak referansli, Turkce saglik yorumu uretir.

Self-consistency katmani (oneri formu 2.4) IKI ayri kontrol yapar:
1. Kaynak gecerliligi (valid_citation_ratio): LLM'in urettigi [Kaynak: chunk_id] etiketleri
   gercekten retrieval'da donen chunk'lara ait mi (uydurma kaynak/halusinasyon kontrolu).
2. Sayisal dayanak (numeric_grounding_ratio): bir cumlede atif yapilan kaynagin GERCEKTEN o
   sayiyi icerip icermedigi (ya da sayinin urunun kendi besin verisinden gelip gelmedigi).
   Bu ikinci kontrol, Faz 4'un gercek degerlendirmesinde bulunan somut bir soruna karsi
   eklendi: kucuk/ucretsiz modeller gecerli bir chunk'a atif yaparken bile o chunk'ta
   YAZMAYAN bir sayi "turetip" (ornegin kaynakta "%5-10" yazarken kendi hesabiyla "25 gram"
   UYDURARAK) sunabiliyor - chunk_id gecerli oldugu icin (1) numarali kontrol bunu yakalamaz.

Her iki oran da esigin altindaysa, daha siki (ve HANGI sorunun oldugunu belirten) bir
talimatla YENIDEN URETIM yapilir.

Sayisal esik/risk hesaplari asla LLM'e birakilmaz - bunlar src/ocr/risk_engine.py'de kural
tabanli hesaplanip prompt'a hazir bayrak olarak verilir (form 2.3 taahhudu + halusinasyon riski
B-plani); LLM'den yalnizca bu hazir bilgiyi dogal dile dokmesi istenir.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.common.schema import NutritionFacts
from src.ocr.risk_engine import describe_risks
from src.rag.llm_provider import LLMProvider
from src.rag.retriever import RetrievalResult, Retriever

SYSTEM_PROMPT = (
    "Sen bir gida ve saglik asistanisin. SADECE sana verilen KAYNAK PARCALARI icindeki bilgiyi "
    "kullanarak Turkce, kisa ve anlasilir bir saglik degerlendirmesi yaz. Hicbir sayisal esik "
    "veya oran UYDURMA - sadece sana verilen risk bayraklarini ve kaynak parcalarini yorumla. "
    "Bir kaynaktaki yuzde/orani kendi basina baska bir birime (ornegin grama) CEVIRME - kaynakta "
    "birebir yazan sayiyi kullan. Her iddiani, ilgili kaynagin chunk_id'siyle "
    "[Kaynak: <chunk_id>] seklinde etiketle: urunun kendi besin degerleri icin "
    "[Kaynak: urun_etiketi], kural motorunun risk bayraklari icin [Kaynak: kural_motoru], "
    "diger her sey icin SADECE KAYNAK PARCALARI'ndaki chunk_id'ler. Kaynak parcalarinda olmayan "
    "bir bilgiyi kesinlikle ekleme."
)

# Urunun kendi verisi ve kural motoru da mesru birer kaynaktir; model onlari kaynak gostermek
# istiyordu ama kimlikleri olmadigi icin bolum basliklarini ("URUN BESIN DEGERLERI") etiket
# olarak yaziyor ve uydurma kaynak sayiliyordu (gpt-4.1-mini'nin kalan FC kaybinin tamami,
# 2026-10-06 kiyasi). Yasaklamak yerine (denendi, basliklari ANMAK hatayi artirdi) gercek
# kimlik verildi; uydurma kaynak tespiti diger her kimlik icin aynen calisir.
PRODUCT_SOURCE_ID = "urun_etiketi"
RULE_ENGINE_SOURCE_ID = "kural_motoru"
INTERNAL_SOURCE_IDS = {PRODUCT_SOURCE_ID, RULE_ENGINE_SOURCE_ID}

CITATION_PATTERN = re.compile(r"\[Kaynak:\s*([^\]]+)\]")
NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)?")
# "100 g'da", "100 ml icin" gibi olcu bazi ifadeleri bir iddia degil, birim; sayisal dayanak
# kontrolunden once cikarilir (aksi halde "100" kaynakta gecmedigi icin her modelde haksiz
# yere dayanaksiz sayiliyordu - 2026-10-06 kiyasinda risksiz_urun'u 0.80'e sabitleyen artefakt).
BASIS_PATTERN = re.compile(r"\b100\s*(?:gram\w*|mililitre\w*|gr\b|g\b|ml\b)", re.IGNORECASE)


@dataclass
class GenerationResult:
    text: str
    retrieved: list[RetrievalResult]
    cited_chunk_ids: list[str]
    valid_citation_ratio: float
    numeric_grounding_ratio: float
    regenerated: bool


def _build_context_block(retrieved: list[RetrievalResult]) -> str:
    blocks = []
    for r in retrieved:
        c = r.chunk
        confidence = "DOGRULANMIS" if c.verified else "TASLAK/DOGRULANMAMIS"
        blocks.append(
            f"[chunk_id: {c.chunk_id}] ({confidence}, kaynak: {c.source or 'ic referans'})\n"
            f"{c.title} — {c.section}\n{c.text}"
        )
    return "\n\n".join(blocks)


def build_prompt(
    nutrition: NutritionFacts,
    risk_flags: list[str],
    retrieved: list[RetrievalResult],
) -> str:
    risk_lines = describe_risks(risk_flags) if risk_flags else ["Tespit edilen bir risk bayragi yok."]
    nutrition_summary = nutrition.model_dump(exclude_none=True)

    return (
        f"URUN BESIN DEGERLERI [chunk_id: {PRODUCT_SOURCE_ID}] (100g bazinda): {nutrition_summary}\n\n"
        f"KURAL MOTORU RISK BAYRAKLARI [chunk_id: {RULE_ENGINE_SOURCE_ID}]:\n"
        + "\n".join(f"- {line}" for line in risk_lines)
        + "\n\n"
        f"KAYNAK PARCALARI:\n{_build_context_block(retrieved)}\n\n"
        "Yukaridaki bilgiyi kullanarak kullaniciya 3-5 cumlelik bir saglik degerlendirmesi yaz. "
        "Her cumlede kullandigin kaynagi [Kaynak: chunk_id] ile belirt."
    )


def _split_citation_ids(raw: str) -> list[str]:
    """Tek etiketteki birden fazla kaynagi ayirir: "[Kaynak: a::0, b::1]" -> ["a::0", "b::1"].

    Daha guclu modeller (gpt-4.1-mini, 2026-10-06 kiyasi) ayni cumleyi iki chunk'a birden
    dayandirip tek etikette virgulle yazabiliyor; ayrilmazsa "a::0, b::1" tek bir (gecersiz)
    chunk_id sanilir ve hem kaynak gecerliligi hem sayisal dayanak haksiz yere dusuk cikar.
    chunk_id'ler virgul/noktali virgul icermez, bu yuzden bu ayrim guvenlidir.
    """
    return [part.strip() for part in re.split(r"[,;]", raw) if part.strip()]


def _extract_cited_ids(text: str) -> list[str]:
    return [cid for m in CITATION_PATTERN.findall(text) for cid in _split_citation_ids(m)]


def _valid_citation_ratio(cited_ids: list[str], valid_ids: set[str]) -> float:
    if not cited_ids:
        return 0.0
    valid_count = sum(1 for cid in cited_ids if cid in valid_ids)
    return valid_count / len(cited_ids)


def _extract_citation_segments(text: str) -> list[tuple[str, list[str]]]:
    """Her [Kaynak: ...] etiketinden once gelen metin parcasini o etiketteki chunk_id'lerle esler.

    Boylece "hangi cumle hangi kaynaga atif yapiyor" bilgisi korunur - sayisal dayanak
    kontrolu (compute_numeric_grounding) bu eslesmeyi kullanir.
    """
    segments = []
    last_end = 0
    for match in CITATION_PATTERN.finditer(text):
        segment_text = text[last_end : match.start()]
        segments.append((segment_text, _split_citation_ids(match.group(1))))
        last_end = match.end()
    return segments


def _numbers_in(text: str) -> set[float]:
    """Metindeki sayilari float'a normalize eder (ondalik virgul/nokta farki, "35" ile
    NutritionFacts.model_dump()'un urettigi "35.0" gibi bicim farklarini elemek icin -
    aksi halde ayni sayi farkli yazildiginda yanlislikla "dayanaksiz" sayilirdi)."""
    normalized = set()
    for match in NUMBER_PATTERN.findall(text):
        try:
            normalized.add(float(match.replace(",", ".")))
        except ValueError:
            continue
    return normalized


def compute_numeric_grounding(
    text: str,
    retrieved: list[RetrievalResult],
    nutrition: NutritionFacts,
    risk_flags: list[str] | None = None,
) -> float:
    """Her atiftaki sayisal degerin, o SPESIFIK atif yapilan chunk'in metninde ya da urunun
    kendi girdi besin verisinde (prompt'ta dogrudan verilir, retrieval disi ama mesru) birebir
    gecip gecmedigini kontrol eder.

    Ne ikisinde de gecmeyen bir sayi = modelin kendi turettigi, kaynaksiz bir iddia (ornegin
    kaynakta "%5-10 enerji" yazarken "25 gram" gibi kendi hesapladigi bir deger sunmasi).
    Bir cumlede hic sayi yoksa kontrol edilecek bir sey yoktur (cezalandirilmaz).
    [Kaynak: kural_motoru] atfindaki sayilar risk bayraklarinin aciklamalarina karsi kontrol edilir.
    """
    nutrition_text = str(nutrition.model_dump(exclude_none=True))
    chunk_text_by_id = {r.chunk.chunk_id: r.chunk.text for r in retrieved}
    chunk_text_by_id[PRODUCT_SOURCE_ID] = nutrition_text
    chunk_text_by_id[RULE_ENGINE_SOURCE_ID] = " ".join(describe_risks(risk_flags or []))
    nutrition_numbers = _numbers_in(nutrition_text)

    grounded_count = 0
    total_count = 0
    for segment_text, chunk_ids in _extract_citation_segments(text):
        segment_numbers = _numbers_in(BASIS_PATTERN.sub(" ", segment_text))
        if not segment_numbers:
            continue
        # Birden fazla kaynak gosterildiyse sayi bunlardan HERHANGI birinde gecmesi yeterli.
        chunk_numbers = set().union(*(_numbers_in(chunk_text_by_id.get(cid, "")) for cid in chunk_ids))
        for number in segment_numbers:
            total_count += 1
            if number in chunk_numbers or number in nutrition_numbers:
                grounded_count += 1

    return grounded_count / total_count if total_count else 1.0


def _build_retry_suffix(citation_ok: bool, numeric_ok: bool, valid_ids: set[str]) -> str:
    parts = ["\n\nONEMLI UYARI: Onceki yanitinla ilgili sorun(lar) tespit edildi."]
    if not citation_ok:
        parts.append(
            " (1) Kaynaklarda olmayan veya hatali bir [Kaynak: ...] etiketi kullandin - SADECE "
            f"asagida listelenen chunk_id'leri kullan: {', '.join(sorted(valid_ids))}."
        )
    if not numeric_ok:
        parts.append(
            " (2) Bir cumlede, o cumledeki [Kaynak: ...] etiketinin ait oldugu kaynak "
            "parcasinda GERCEKTEN YAZMAYAN bir sayi/oran kullandin (ornegin kaynakta '%5-10' "
            "yaziyorsa kendi basina '25 gram' gibi turetilmis bir deger UYDURMA). Sadece kaynak "
            "parcasinda birebir yazan sayilari veya urunun kendi besin degerlerini kullan."
        )
    return "".join(parts)


def generate_explanation(
    nutrition: NutritionFacts,
    risk_flags: list[str],
    retriever: Retriever,
    llm: LLMProvider,
    top_k: int = 5,
    consistency_threshold: float = 0.7,
) -> GenerationResult:
    """Uctan uca: retrieval -> prompt -> LLM uretim -> self-consistency kontrolu (kaynak
    gecerliligi + sayisal dayanak) -> (gerekirse) yeniden uretim."""
    query = " ".join(describe_risks(risk_flags)) if risk_flags else "genel beslenme degerlendirmesi"
    retrieved = retriever.retrieve(query, top_k=top_k)
    retrieved_ids = {r.chunk.chunk_id for r in retrieved}
    valid_ids = retrieved_ids | INTERNAL_SOURCE_IDS

    prompt = build_prompt(nutrition, risk_flags, retrieved)
    text = llm.generate(prompt, system=SYSTEM_PROMPT)
    cited_ids = _extract_cited_ids(text)
    citation_ratio = _valid_citation_ratio(cited_ids, valid_ids)
    numeric_ratio = compute_numeric_grounding(text, retrieved, nutrition, risk_flags)
    regenerated = False

    citation_ok = citation_ratio >= consistency_threshold
    numeric_ok = numeric_ratio >= consistency_threshold

    if (not citation_ok or not numeric_ok) and retrieved_ids:
        retry_prompt = prompt + _build_retry_suffix(citation_ok, numeric_ok, valid_ids)
        text = llm.generate(retry_prompt, system=SYSTEM_PROMPT)
        cited_ids = _extract_cited_ids(text)
        citation_ratio = _valid_citation_ratio(cited_ids, valid_ids)
        numeric_ratio = compute_numeric_grounding(text, retrieved, nutrition, risk_flags)
        regenerated = True

    return GenerationResult(
        text=text,
        retrieved=retrieved,
        cited_chunk_ids=cited_ids,
        valid_citation_ratio=citation_ratio,
        numeric_grounding_ratio=numeric_ratio,
        regenerated=regenerated,
    )
