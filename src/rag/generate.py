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
    "[Kaynak: <chunk_id>] seklinde etiketle. Kaynak parcalarinda olmayan bir bilgiyi kesinlikle "
    "ekleme."
)

CITATION_PATTERN = re.compile(r"\[Kaynak:\s*([^\]]+)\]")
NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)?")


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
        f"URUN BESIN DEGERLERI (100g bazinda): {nutrition_summary}\n\n"
        "KURAL MOTORU RISK BAYRAKLARI:\n"
        + "\n".join(f"- {line}" for line in risk_lines)
        + "\n\n"
        f"KAYNAK PARCALARI:\n{_build_context_block(retrieved)}\n\n"
        "Yukaridaki bilgiyi kullanarak kullaniciya 3-5 cumlelik bir saglik degerlendirmesi yaz. "
        "Her cumlede kullandigin kaynagi [Kaynak: chunk_id] ile belirt."
    )


def _extract_cited_ids(text: str) -> list[str]:
    return [m.strip() for m in CITATION_PATTERN.findall(text)]


def _valid_citation_ratio(cited_ids: list[str], valid_ids: set[str]) -> float:
    if not cited_ids:
        return 0.0
    valid_count = sum(1 for cid in cited_ids if cid in valid_ids)
    return valid_count / len(cited_ids)


def _extract_citation_segments(text: str) -> list[tuple[str, str]]:
    """Her [Kaynak: chunk_id] etiketinden once gelen metin parcasini o chunk_id ile esler.

    Boylece "hangi cumle hangi kaynaga atif yapiyor" bilgisi korunur - sayisal dayanak
    kontrolu (compute_numeric_grounding) bu eslesmeyi kullanir.
    """
    segments = []
    last_end = 0
    for match in CITATION_PATTERN.finditer(text):
        segment_text = text[last_end : match.start()]
        segments.append((segment_text, match.group(1).strip()))
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
) -> float:
    """Her atiftaki sayisal degerin, o SPESIFIK atif yapilan chunk'in metninde ya da urunun
    kendi girdi besin verisinde (prompt'ta dogrudan verilir, retrieval disi ama mesru) birebir
    gecip gecmedigini kontrol eder.

    Ne ikisinde de gecmeyen bir sayi = modelin kendi turettigi, kaynaksiz bir iddia (ornegin
    kaynakta "%5-10 enerji" yazarken "25 gram" gibi kendi hesapladigi bir deger sunmasi).
    Bir cumlede hic sayi yoksa kontrol edilecek bir sey yoktur (cezalandirilmaz).
    """
    chunk_text_by_id = {r.chunk.chunk_id: r.chunk.text for r in retrieved}
    nutrition_numbers = _numbers_in(str(nutrition.model_dump(exclude_none=True)))

    grounded_count = 0
    total_count = 0
    for segment_text, chunk_id in _extract_citation_segments(text):
        segment_numbers = _numbers_in(segment_text)
        if not segment_numbers:
            continue
        chunk_numbers = _numbers_in(chunk_text_by_id.get(chunk_id, ""))
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
    valid_ids = {r.chunk.chunk_id for r in retrieved}

    prompt = build_prompt(nutrition, risk_flags, retrieved)
    text = llm.generate(prompt, system=SYSTEM_PROMPT)
    cited_ids = _extract_cited_ids(text)
    citation_ratio = _valid_citation_ratio(cited_ids, valid_ids)
    numeric_ratio = compute_numeric_grounding(text, retrieved, nutrition)
    regenerated = False

    citation_ok = citation_ratio >= consistency_threshold
    numeric_ok = numeric_ratio >= consistency_threshold

    if (not citation_ok or not numeric_ok) and valid_ids:
        retry_prompt = prompt + _build_retry_suffix(citation_ok, numeric_ok, valid_ids)
        text = llm.generate(retry_prompt, system=SYSTEM_PROMPT)
        cited_ids = _extract_cited_ids(text)
        citation_ratio = _valid_citation_ratio(cited_ids, valid_ids)
        numeric_ratio = compute_numeric_grounding(text, retrieved, nutrition)
        regenerated = True

    return GenerationResult(
        text=text,
        retrieved=retrieved,
        cited_chunk_ids=cited_ids,
        valid_citation_ratio=citation_ratio,
        numeric_grounding_ratio=numeric_ratio,
        regenerated=regenerated,
    )
