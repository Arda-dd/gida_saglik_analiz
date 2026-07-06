"""RAG bilgi tabani icin data/knowledge_base/thresholds.json uretimi.

config/config.yaml -> thresholds bolumundeki sayisal degerler (tek dogruluk kaynagi) ile
buradaki THRESHOLD_METADATA (kaynak/birim/dogrulama notu) birlestirilerek zengin bir JSON
uretilir. Kaynaklarin gercek WebFetch ile dogrulanip dogrulanmadigi 'verified' alaniyla
acikca isaretlenir (bkz. data/knowledge_base/docs/*.md).
"""

from __future__ import annotations

import json
from pathlib import Path

from src.common.config import load_config

SCHEMA_VERSION = "1.0"

# Her esik icin: unit, source, source_note, verified (WebFetch ile dogrulandi mi?)
THRESHOLD_METADATA: dict[str, dict] = {
    "sugar_high_g_per_100g": {
        "unit": "g/100g",
        "source": "UK FSA traffic-light / Nutri-Score front-of-pack konvansiyonu",
        "source_note": "TODO: TGK'nin kendi esikleriyle karsilastirilip dogrulanmali. "
        "Bkz. docs/nutriscore_thresholds_overview.md",
        "verified": False,
    },
    "salt_high_g_per_100g": {
        "unit": "g/100g",
        "source": "UK FSA traffic-light / Nutri-Score front-of-pack konvansiyonu",
        "source_note": "TODO: TGK'nin kendi esikleriyle karsilastirilip dogrulanmali. "
        "Bkz. docs/nutriscore_thresholds_overview.md",
        "verified": False,
    },
    "saturated_fat_high_g_per_100g": {
        "unit": "g/100g",
        "source": "UK FSA traffic-light / Nutri-Score front-of-pack konvansiyonu",
        "source_note": "TODO: TGK'nin kendi esikleriyle karsilastirilip dogrulanmali. "
        "Bkz. docs/nutriscore_thresholds_overview.md",
        "verified": False,
    },
    "sodium_high_mg_per_100g": {
        "unit": "mg/100g",
        "source": "UK FSA traffic-light / Nutri-Score front-of-pack konvansiyonu",
        "source_note": "TODO: TGK'nin kendi esikleriyle karsilastirilip dogrulanmali. "
        "Bkz. docs/nutriscore_thresholds_overview.md",
        "verified": False,
    },
    "who_salt_daily_intake_g": {
        "unit": "g/gun",
        "source": "WHO 'Salt reduction' Fact Sheet",
        "source_note": "https://www.who.int/news-room/fact-sheets/detail/salt-reduction "
        "(WebFetch ile 2026-07-06 tarihinde dogrulandi)",
        "verified": True,
    },
    "who_free_sugars_energy_pct": {
        "unit": "% gunluk toplam enerji",
        "source": "WHO 'Healthy diet' Fact Sheet",
        "source_note": "https://www.who.int/news-room/fact-sheets/detail/healthy-diet "
        "(WebFetch ile 2026-07-06 tarihinde dogrulandi)",
        "verified": True,
    },
    "who_fat_energy_pct_max": {
        "unit": "% gunluk toplam enerji",
        "source": "WHO 'Healthy diet' Fact Sheet",
        "source_note": "https://www.who.int/news-room/fact-sheets/detail/healthy-diet "
        "(WebFetch ile 2026-07-06 tarihinde dogrulandi)",
        "verified": True,
    },
    "who_saturated_fat_energy_pct_max": {
        "unit": "% gunluk toplam enerji",
        "source": "WHO 'Healthy diet' Fact Sheet",
        "source_note": "https://www.who.int/news-room/fact-sheets/detail/healthy-diet "
        "(WebFetch ile 2026-07-06 tarihinde dogrulandi)",
        "verified": True,
    },
    "energy_kcal_max_plausible_per_100g": {
        "unit": "kcal/100g",
        "source": "Teorik maksimum (saf yag ~9kcal/g x 100g)",
        "source_note": "Saglik esigi degil, veri dogrulama (validation.py) icin mantik sinirdir.",
        "verified": True,
    },
}

REQUIRED_FIELDS = {"value", "unit", "source", "source_note", "verified"}


def build_thresholds_json(config_path: Path, out_path: Path) -> None:
    """config.yaml + THRESHOLD_METADATA -> zenginlestirilmis thresholds.json."""
    config = load_config(config_path)
    raw_thresholds: dict = config.get("thresholds", {})

    enriched = {}
    for key, value in raw_thresholds.items():
        metadata = THRESHOLD_METADATA.get(key)
        if metadata is None:
            raise ValueError(f"'{key}' esigi icin THRESHOLD_METADATA tanimli degil")
        enriched[key] = {"value": value, **metadata}

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_from": str(config_path),
        "thresholds": enriched,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_thresholds(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_thresholds_schema(data: dict) -> list[str]:
    """thresholds.json yapisini dogrular, hata mesajlari listesi doner (bos ise gecerli)."""
    errors: list[str] = []

    if "schema_version" not in data:
        errors.append("'schema_version' alani eksik")
    if "thresholds" not in data:
        errors.append("'thresholds' alani eksik")
        return errors

    for key, entry in data["thresholds"].items():
        missing = REQUIRED_FIELDS - entry.keys()
        if missing:
            errors.append(f"'{key}' icin eksik alanlar: {sorted(missing)}")

    return errors
