"""Uctan uca analiz pipeline'i (Faz 6).

Oneri formu mimarisi: Foto -> On isleme -> [CNN Siniflandirma + OCR+Normalizasyon] ->
Kural Motoru (risk+alerjen) -> RAG (retriever+LLM) -> Kisisel Profil Filtresi -> 3 katmanli
cikti. Bu modul, api/main.py (FastAPI) ve demo/app.py (Streamlit) tarafindan ORTAK kullanilir -
boylece HTTP katmani sadece ince bir sarmalayici olur, is mantigi tek yerde yasar.

RAG (Faz 4) ve kisisel filtre (Faz 5) katmanlari OPSIYONELDIR: RAG bir API anahtari
gerektirdiginden basarisiz olursa (anahtar yok, rate limit, ag hatasi), pipeline halusinasyon
riskine girmeden None donup KURAL TABANLI (Faz 3) sonuclarla devam eder - form Risk Yonetimi
B-plani ile tutarlidir. Kisisel filtre de sadece bir HealthProfile verilmisse calisir.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.barcode.detect import barcodes_from_text, decode_barcodes
from src.barcode.lookup import BarcodeProduct, lookup_barcode
from src.barcode.merge import merge_nutrition
from src.common.schema import Allergen, NutritionFacts, ProductCategory, ProductRecord
from src.data.image_preprocessing import preprocess_label_image
from src.health.profile import HealthProfile
from src.health.recommend import HealthAssessment, build_health_assessment, recommend_alternatives
from src.ocr.allergens import detect_allergens
from src.ocr.extract import extract_text_live
from src.ocr.normalize import extract_and_normalize
from src.ocr.risk_engine import assess_risks, describe_risks
from src.rag.generate import GenerationResult, generate_explanation
from src.rag.llm_provider import get_llm_provider
from src.rag.retriever import Retriever
from src.vision.infer import load_best_model_from_report, predict_category

logger = logging.getLogger(__name__)

DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "dataset.csv"


@dataclass
class AnalysisResult:
    category: str
    category_confidence: float
    nutrition: NutritionFacts
    detected_allergens: list[Allergen]
    risk_flags: list[str]
    risk_messages: list[str]
    ocr_confidence: float
    ocr_engine: str | None = None
    # Opsiyonel katmanlarin (RAG, yedek OCR) sessiz basarisizliklarini kullaniciya/istemciye
    # gostermek icin (bkz. api/health.py'deki HuggingFace 402 olayi).
    warnings: list[str] = field(default_factory=list)
    explanation: GenerationResult | None = None
    health_assessment: HealthAssessment | None = None
    alternatives: list[ProductRecord] = field(default_factory=list)
    # Barkod bulunup urun Open Food Facts'te varsa besin/kategori/alerjen oradan gelir.
    barcode: str | None = None
    product_name: str | None = None
    nutrition_source: str = "ocr"  # "ocr" | "open_food_facts"
    category_source: str = "cnn"  # "cnn" | "open_food_facts"


@lru_cache(maxsize=1)
def _get_vision_model():
    return load_best_model_from_report()


@lru_cache(maxsize=1)
def _get_retriever() -> Retriever:
    return Retriever.load()


@lru_cache(maxsize=1)
def _get_llm():
    return get_llm_provider()


def _none_if_nan(value) -> float | None:
    return None if pd.isna(value) else float(value)


@lru_cache(maxsize=1)
def get_candidate_products() -> tuple[ProductRecord, ...]:
    """Alternatif urun onerisi (Faz 5) icin aday havuzu - Faz 1'in dataset.csv'sinden,
    besin verisi bulunan (OFF) kayitlar. Yerel/henuz-OCR'lanmamis kayitlar (besin verisi NaN)
    atlanir - eksik veriyle yanlis "daha az riskli" karsilastirmasi yapilmasin diye."""
    if not DATASET_PATH.exists():
        return ()

    df = pd.read_csv(DATASET_PATH)
    records = []
    for _, row in df.iterrows():
        if pd.isna(row.get("energy_kcal")) and pd.isna(row.get("sugar_g")):
            continue

        # Bos allergens hucresi pandas tarafindan NaN (float) olarak okunur - NaN Python'da
        # truthy oldugundan `or ""` bunu yakalamaz ve str(nan)="nan" gibi gecersiz bir
        # Allergen degerine donusurdu; bu yuzden once pd.isna ile acikca kontrol edilir.
        allergens_cell = row.get("allergens")
        allergens_str = "" if pd.isna(allergens_cell) else str(allergens_cell)
        allergens = [Allergen(a) for a in allergens_str.split(";") if a]

        try:
            category = ProductCategory(row["category"])
        except ValueError:
            category = ProductCategory.BILINMIYOR

        nutrition = NutritionFacts(
            energy_kcal=_none_if_nan(row.get("energy_kcal")),
            energy_kj=_none_if_nan(row.get("energy_kj")),
            fat_g=_none_if_nan(row.get("fat_g")),
            saturated_fat_g=_none_if_nan(row.get("saturated_fat_g")),
            carbohydrate_g=_none_if_nan(row.get("carbohydrate_g")),
            sugar_g=_none_if_nan(row.get("sugar_g")),
            fiber_g=_none_if_nan(row.get("fiber_g")),
            protein_g=_none_if_nan(row.get("protein_g")),
            salt_g=_none_if_nan(row.get("salt_g")),
            sodium_mg=_none_if_nan(row.get("sodium_mg")),
        )
        records.append(
            ProductRecord(
                product_id=row["product_id"],
                category=category,
                nutrition=nutrition,
                allergens=allergens,
                source=row.get("source"),
            )
        )
    return tuple(records)


def _lookup_first(barcodes: list[str], warnings: list[str]) -> BarcodeProduct | None:
    """Bulunan barkodlari sirayla Open Food Facts'te arar; ilk bulunan urunu doner.

    Ag/API hatasi pipeline'i durdurmaz (OCR sonucuyla devam edilir) ama uyari olarak eklenir."""
    for barcode in barcodes:
        try:
            product = lookup_barcode(barcode)
        except Exception as exc:
            logger.warning("Barkod sorgusu basarisiz (%s): %s", barcode, exc)
            warnings.append(f"Barkod {barcode} sorgulanamadi ({type(exc).__name__}) - etiket okumasi kullanildi.")
            continue
        if product is not None:
            return product
    return None


def analyze_label_image(
    image_path: Path,
    profile: HealthProfile | None = None,
    generate_llm_explanation: bool = True,
    include_alternatives: bool = True,
) -> AnalysisResult:
    """Tam pipeline: on isleme -> kategori (CNN) -> OCR/besin -> risk motoru ->
    (opsiyonel) RAG aciklamasi -> (opsiyonel, profil verilmisse) kisisel filtre + alternatifler."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        processed_path = preprocess_label_image(image_path, Path(tmp_dir) / "processed.jpg")

        model, categories = _get_vision_model()
        category, category_confidence = predict_category(model, processed_path, categories)

        # Barkod once ORIJINAL gorselde aranir (on isleme perspektifi/kontrasti degistirir).
        barcodes = decode_barcodes(image_path)

        ocr_result = extract_text_live(processed_path)
        nutrition, _basis = extract_and_normalize(ocr_result.text)
        detected_allergens = detect_allergens(ocr_result.text)

    warnings: list[str] = []
    if ocr_result.fallback_notes:
        warnings.append(
            f"OCR yedek motora dustu ({'; '.join(ocr_result.fallback_notes)}), "
            f"{ocr_result.engine} kullanildi - dogruluk dusuk olabilir."
        )
    if not ocr_result.text.strip():
        warnings.append("Etiketten metin okunamadi - daha yakindan ve net bir fotograf cekin.")

    if not barcodes:
        barcodes = barcodes_from_text(ocr_result.text)
    barcode_product = _lookup_first(barcodes, warnings)
    nutrition_source, category_source = "ocr", "cnn"
    if barcode_product is not None and barcode_product.has_usable_nutrition:
        nutrition, _filled, conflicts = merge_nutrition(barcode_product.record.nutrition, nutrition)
        nutrition_source = "open_food_facts"
        if conflicts:
            warnings.append(
                "Etiketten okunan degerler urun veritabanindan farkli (" + "; ".join(conflicts)
                + ") - veritabani degeri kullanildi; tarif degismis olabilir, etiketi kontrol edin."
            )
    if barcode_product is not None and barcode_product.record.category != ProductCategory.BILINMIYOR:
        category, category_confidence = barcode_product.record.category.value, 1.0
        category_source = "open_food_facts"
    if barcode_product is not None:
        # Alerjende guvenli taraf: etiket + veritabani + "eser miktarda icerebilir" birlesimi.
        detected_allergens = sorted(
            set(detected_allergens) | set(barcode_product.record.allergens) | set(barcode_product.traces),
            key=lambda a: a.value,
        )
        if barcode_product.traces:
            warnings.append(
                "Eser miktarda icerebilir: " + ", ".join(a.value for a in barcode_product.traces) + "."
            )

    risk_flags = assess_risks(nutrition)
    risk_messages = describe_risks(risk_flags)

    explanation = None
    if generate_llm_explanation:
        try:
            explanation = generate_explanation(
                nutrition, risk_flags, _get_retriever(), _get_llm()
            )
        except Exception as exc:
            # RAG katmani opsiyoneldir (API anahtari/aginin durumuna bagli) - basarisiz
            # olursa kural tabanli (Faz 3) sonuclarla devam edilir, tum istek COKMEZ. Ama hata
            # artik loglanir ve yanita uyari olarak eklenir (once tamamen sessizdi).
            logger.warning("RAG aciklamasi uretilemedi: %s", exc, exc_info=True)
            warnings.append(f"Saglik aciklamasi uretilemedi ({type(exc).__name__}: {str(exc)[:120]}).")
            explanation = None

    health_assessment = None
    alternatives: list[ProductRecord] = []
    if profile is not None:
        health_assessment = build_health_assessment(nutrition, detected_allergens, profile)
        if include_alternatives:
            try:
                resolved_category = ProductCategory(category)
            except ValueError:
                resolved_category = ProductCategory.BILINMIYOR
            current_product = ProductRecord(
                product_id="uploaded_image",
                category=resolved_category,
                nutrition=nutrition,
                allergens=detected_allergens,
                source="upload",
            )
            alternatives = recommend_alternatives(
                current_product, list(get_candidate_products()), profile
            )

    return AnalysisResult(
        category=category,
        category_confidence=category_confidence,
        nutrition=nutrition,
        detected_allergens=detected_allergens,
        risk_flags=risk_flags,
        risk_messages=risk_messages,
        ocr_confidence=ocr_result.mean_confidence,
        ocr_engine=ocr_result.engine,
        warnings=warnings,
        barcode=barcode_product.barcode if barcode_product else (barcodes[0] if barcodes else None),
        product_name=barcode_product.product_name if barcode_product else None,
        nutrition_source=nutrition_source,
        category_source=category_source,
        explanation=explanation,
        health_assessment=health_assessment,
        alternatives=alternatives,
    )
