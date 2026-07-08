"""RAG generation katmani: retrieval sonuclarini + kural motoru risk bayraklarini LLM'e vererek
kaynak referansli, Turkce saglik yorumu uretir.

Self-consistency katmani (oneri formu 2.4): LLM'in urettigi [Kaynak: chunk_id] etiketleri
gercekten retrieval'da donen chunk'lara ait mi diye dogrulanir (halusinasyon kontrolu). Bu oran
(valid_citation_ratio) esigin altindaysa, daha siki bir talimatla YENIDEN URETIM yapilir (form:
"dusuk tutarlilik -> yeniden uretim").

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
    "Her iddiani, ilgili kaynagin chunk_id'siyle [Kaynak: <chunk_id>] seklinde etiketle. "
    "Kaynak parcalarinda olmayan bir bilgiyi kesinlikle ekleme."
)

STRICT_RETRY_SUFFIX = (
    "\n\nONEMLI UYARI: Onceki yanitinda kaynaklarda olmayan veya hatali bir [Kaynak: ...] "
    "etiketi kullandin. Bu sefer SADECE asagida listelenen chunk_id'leri kullan, baska hicbir "
    "kaynak etiketi UYDURMA: {valid_ids}"
)

CITATION_PATTERN = re.compile(r"\[Kaynak:\s*([^\]]+)\]")


@dataclass
class GenerationResult:
    text: str
    retrieved: list[RetrievalResult]
    cited_chunk_ids: list[str]
    valid_citation_ratio: float
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


def generate_explanation(
    nutrition: NutritionFacts,
    risk_flags: list[str],
    retriever: Retriever,
    llm: LLMProvider,
    top_k: int = 5,
    consistency_threshold: float = 0.7,
) -> GenerationResult:
    """Uctan uca: retrieval -> prompt -> LLM uretim -> self-consistency kontrolu -> (gerekirse) yeniden uretim."""
    query = " ".join(describe_risks(risk_flags)) if risk_flags else "genel beslenme degerlendirmesi"
    retrieved = retriever.retrieve(query, top_k=top_k)
    valid_ids = {r.chunk.chunk_id for r in retrieved}

    prompt = build_prompt(nutrition, risk_flags, retrieved)
    text = llm.generate(prompt, system=SYSTEM_PROMPT)
    cited_ids = _extract_cited_ids(text)
    ratio = _valid_citation_ratio(cited_ids, valid_ids)
    regenerated = False

    if ratio < consistency_threshold and valid_ids:
        retry_prompt = prompt + STRICT_RETRY_SUFFIX.format(valid_ids=", ".join(sorted(valid_ids)))
        text = llm.generate(retry_prompt, system=SYSTEM_PROMPT)
        cited_ids = _extract_cited_ids(text)
        ratio = _valid_citation_ratio(cited_ids, valid_ids)
        regenerated = True

    return GenerationResult(
        text=text,
        retrieved=retrieved,
        cited_chunk_ids=cited_ids,
        valid_citation_ratio=ratio,
        regenerated=regenerated,
    )
