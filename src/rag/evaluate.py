"""RAG retrieval + generation degerlendirme scripti.

Retrieval metrikleri (oneri formu 2.4): Top-k Recall, MRR - data/rag_eval/queries.json
icindeki elle etiketlenmis (query, relevant_chunk_ids) ciftleriyle olculur.

Generation metrikleri: Factual Consistency Score (LLM'in urettigi [Kaynak: ...] etiketlerinin
gercekten retrieval'da donen chunk'lara ait olma orani - src/rag/generate.py'deki
valid_citation_ratio ile ayni mekanizma) ve Ground Truth Alignment Ratio (uretilen metindeki
sayisal degerlerin retrieval baglaminda GERCEKTEN var olma orani - halusinasyon/uydurma sayisal
deger kontrolu, ek bir ground-truth esik veritabani gerektirmez).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from src.common.schema import NutritionFacts
from src.rag.generate import GenerationResult, generate_explanation
from src.rag.llm_provider import get_llm_provider
from src.rag.retriever import Retriever

QUERIES_PATH = Path(__file__).resolve().parents[2] / "data" / "rag_eval" / "queries.json"
REPORT_PATH = Path(__file__).resolve().parents[2] / "docs" / "rag_evaluation_report.json"
NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)?")


@dataclass
class RetrievalEvalResult:
    query: str
    recall_at_k: float
    reciprocal_rank: float
    retrieved_ids: list[str]


def load_eval_queries(path: Path = QUERIES_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_retrieval(
    retriever: Retriever, queries: list[dict], top_k: int = 5
) -> tuple[list[RetrievalEvalResult], float, float]:
    """Her sorgu icin Recall@k ve Reciprocal Rank hesaplar; ortalama Recall@k ve MRR doner."""
    results = []
    for item in queries:
        relevant = set(item["relevant_chunk_ids"])
        retrieved = retriever.retrieve(item["query"], top_k=top_k)
        retrieved_ids = [r.chunk.chunk_id for r in retrieved]

        hits = {cid for cid in retrieved_ids if cid in relevant}
        recall = len(hits) / len(relevant) if relevant else 0.0

        reciprocal_rank = 0.0
        for rank, cid in enumerate(retrieved_ids, start=1):
            if cid in relevant:
                reciprocal_rank = 1.0 / rank
                break

        results.append(
            RetrievalEvalResult(
                query=item["query"],
                recall_at_k=recall,
                reciprocal_rank=reciprocal_rank,
                retrieved_ids=retrieved_ids,
            )
        )

    mean_recall = sum(r.recall_at_k for r in results) / len(results) if results else 0.0
    mrr = sum(r.reciprocal_rank for r in results) / len(results) if results else 0.0
    return results, mean_recall, mrr


def _numbers_in(text: str) -> list[str]:
    return NUMBER_PATTERN.findall(text)


def ground_truth_alignment_ratio(generated_text: str, retrieved_context: str) -> float:
    """Uretilen metindeki sayilarin kac tanesi retrieval baglaminda GERCEKTEN geciyor.

    Uretilen metinde hic sayi yoksa 1.0 doner (desteklenmeyen bir sayisal iddia YOK demektir,
    bu guvenli bir durumdur - cezalandirilmaz).
    """
    generated_numbers = _numbers_in(generated_text)
    if not generated_numbers:
        return 1.0
    context_numbers = set(_numbers_in(retrieved_context))
    aligned = sum(1 for n in generated_numbers if n in context_numbers)
    return aligned / len(generated_numbers)


@dataclass
class GenerationEvalCase:
    name: str
    nutrition: NutritionFacts
    risk_flags: list[str]


DEFAULT_GENERATION_CASES: list[GenerationEvalCase] = [
    GenerationEvalCase(
        name="yuksek_seker_urun",
        nutrition=NutritionFacts(energy_kcal=450, sugar_g=35.0, fat_g=12.0, salt_g=0.4),
        risk_flags=["yuksek_seker"],
    ),
    GenerationEvalCase(
        name="yuksek_tuz_ve_sodyum_urun",
        nutrition=NutritionFacts(energy_kcal=250, sugar_g=3.0, salt_g=2.8, sodium_mg=1120),
        risk_flags=["yuksek_tuz", "yuksek_sodyum"],
    ),
    GenerationEvalCase(
        name="yuksek_doymus_yag_urun",
        nutrition=NutritionFacts(energy_kcal=520, fat_g=30.0, saturated_fat_g=14.0, sugar_g=8.0),
        risk_flags=["yuksek_doymus_yag"],
    ),
    GenerationEvalCase(
        name="risksiz_urun",
        nutrition=NutritionFacts(energy_kcal=180, sugar_g=2.0, fat_g=3.0, salt_g=0.2),
        risk_flags=[],
    ),
]


def evaluate_generation(
    retriever: Retriever,
    cases: list[GenerationEvalCase] = DEFAULT_GENERATION_CASES,
    top_k: int = 5,
) -> tuple[list[dict], float, float]:
    """Her senaryo icin generate_explanation calistirir; Factual Consistency + Ground Truth
    Alignment ortalamalarini doner (gercek LLM API cagrisi gerektirir)."""
    llm = get_llm_provider()
    rows = []
    for case in cases:
        result: GenerationResult = generate_explanation(
            case.nutrition, case.risk_flags, retriever, llm, top_k=top_k
        )
        # "Grounding" kaynagi hem retrieval baglamini HEM DE urunun kendi besin degerlerini
        # (prompt'ta dogrudan verilen girdi verisi) icermeli - LLM'in urunun kendi 35g seker
        # gibi degerlerini dogru sekilde tekrarlamasi halusinasyon DEGILDIR, sadece retrieval
        # disi (ama yine de meşru) bir kaynaktan gelir.
        context_text = "\n".join(r.chunk.text for r in result.retrieved)
        context_text += "\n" + str(case.nutrition.model_dump(exclude_none=True))
        alignment = ground_truth_alignment_ratio(result.text, context_text)
        rows.append(
            {
                "name": case.name,
                "valid_citation_ratio": result.valid_citation_ratio,
                "ground_truth_alignment_ratio": alignment,
                "regenerated": result.regenerated,
                "text": result.text,
            }
        )

    mean_consistency = sum(r["valid_citation_ratio"] for r in rows) / len(rows) if rows else 0.0
    mean_alignment = sum(r["ground_truth_alignment_ratio"] for r in rows) / len(rows) if rows else 0.0
    return rows, mean_consistency, mean_alignment


def main() -> None:
    print("RAG index yukleniyor...")
    retriever = Retriever.load()

    queries = load_eval_queries()
    print(f"{len(queries)} sorguyla retrieval degerlendiriliyor (top_k=5)...")
    retrieval_results, mean_recall, mrr = evaluate_retrieval(retriever, queries, top_k=5)
    for r in retrieval_results:
        print(f"  [{r.recall_at_k:.2f} recall, RR={r.reciprocal_rank:.2f}] {r.query}")
    print(f"--> Ortalama Recall@5: %{mean_recall * 100:.1f} | MRR: {mrr:.3f}")

    print("\nGeneration degerlendiriliyor (gercek LLM API cagrisi)...")
    generation_rows, mean_consistency, mean_alignment = evaluate_generation(retriever)
    for row in generation_rows:
        print(
            f"  {row['name']}: valid_citation_ratio={row['valid_citation_ratio']:.2f}, "
            f"ground_truth_alignment={row['ground_truth_alignment_ratio']:.2f}, "
            f"regenerated={row['regenerated']}"
        )
    print(f"--> Factual Consistency Score: %{mean_consistency * 100:.1f}")
    print(f"--> Ground Truth Alignment Ratio: %{mean_alignment * 100:.1f}")

    report = {
        "retrieval": {
            "mean_recall_at_5": mean_recall,
            "mrr": mrr,
            "per_query": [
                {
                    "query": r.query,
                    "recall_at_5": r.recall_at_k,
                    "reciprocal_rank": r.reciprocal_rank,
                    "retrieved_ids": r.retrieved_ids,
                }
                for r in retrieval_results
            ],
        },
        "generation": {
            "mean_factual_consistency_score": mean_consistency,
            "mean_ground_truth_alignment_ratio": mean_alignment,
            "cases": generation_rows,
        },
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRapor kaydedildi: {REPORT_PATH}")


if __name__ == "__main__":
    main()
