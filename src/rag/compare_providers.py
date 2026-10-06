"""LLM saglayici/model kiyasi: generation metriklerini her model icin N tekrarla olcer.

4 senaryoluk tek bir evaluate kosusu cok oynaktir (ayni HuggingFace modeli ayni gun %76 ile
%92 arasi Ground Truth Alignment verebiliyor), bu yuzden her model birkac kez calistirilip
ortalamasi alinir. Retrieval (FAISS index + embedding) tum modeller icin ortaktir, sadece LLM
degisir.

Kullanim:
    python -m src.rag.compare_providers --runs 3 \
        --model openai:gpt-4.1-mini --model openai:gpt-4o-mini --model huggingface

Not: metrikler icerik dogrulugunu tam yakalamaz - ornegin bir urun sayisinin YANLIS besin
ogesine baglanmasi (30 g yag -> "30 g doymus yag") Ground Truth Alignment'ta dayanakli sayilir,
cunku sayi urun verisinde geciyor. Kiyas sonuclari mutlaka uretilen metinler (rapordaki `text`
alanlari) okunarak yorumlanmalidir (bkz. docs/rag_results_notes.md).
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from src.common.config import get_config
from src.rag.evaluate import evaluate_generation
from src.rag.llm_provider import get_llm_provider
from src.rag.retriever import Retriever

REPORT_PATH = Path(__file__).resolve().parents[2] / "docs" / "rag_llm_comparison_report.json"


def parse_model_spec(spec: str, config: dict) -> tuple[str, str]:
    """"openai:gpt-4.1-mini" -> ("openai", "gpt-4.1-mini"); model verilmezse config'teki varsayilan."""
    provider, _, model = spec.partition(":")
    return provider, model or config["llm"][f"model_{provider}"]


def compare(specs: list[str], runs: int) -> dict:
    base_config = get_config()
    retriever = Retriever.load()
    results = []
    for spec in specs:
        provider, model = parse_model_spec(spec, base_config)
        config = copy.deepcopy(base_config)
        config["llm"]["provider"] = provider
        config["llm"][f"model_{provider}"] = model
        llm = get_llm_provider(config)

        run_rows = []
        for i in range(runs):
            rows, consistency, alignment = evaluate_generation(retriever, llm=llm)
            print(
                f"[{provider}:{model}] kosu {i + 1}/{runs}: "
                f"Factual Consistency=%{consistency * 100:.1f} GTA=%{alignment * 100:.1f}",
                flush=True,
            )
            run_rows.append(
                {"factual_consistency": consistency, "ground_truth_alignment": alignment, "cases": rows}
            )

        mean_consistency = sum(r["factual_consistency"] for r in run_rows) / runs
        mean_alignment = sum(r["ground_truth_alignment"] for r in run_rows) / runs
        print(
            f"--> {provider}:{model} ortalama ({runs} kosu): "
            f"Factual Consistency=%{mean_consistency * 100:.1f} GTA=%{mean_alignment * 100:.1f}\n"
        )
        results.append(
            {
                "provider": provider,
                "model": model,
                "runs": runs,
                "mean_factual_consistency": mean_consistency,
                "mean_ground_truth_alignment": mean_alignment,
                "per_run": run_rows,
            }
        )
    return {"models": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", action="append", required=True, help="saglayici[:model]")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    report = compare(args.model, args.runs)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Rapor kaydedildi: {REPORT_PATH}")


if __name__ == "__main__":
    main()
