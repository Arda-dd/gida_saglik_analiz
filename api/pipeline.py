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
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.barcode.detect import barcodes_from_text, decode_barcodes
from src.barcode.lookup import BarcodeProduct, lookup_barcode
from src.barcode.merge import FIELD_LABELS_TR, merge_nutrition
from src.common.schema import Allergen, NutritionFacts, ProductCategory, ProductRecord
from src.data.image_preprocessing import preprocess_label_image
from src.health.profile import HealthProfile
from src.health.recommend import HealthAssessment, build_health_assessment, recommend_alternatives
from src.ocr.allergens import detect_allergens
from src.ocr.extract import REQUIRED_API_KEYS, extract_text_live, extract_text_ocrspace
from src.ocr.verify import assess_image_quality, verify_nutrition
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
    # OpenAI OCR'in okudugu her alan ikinci motorla dogrulandi mi (None: dogrulama yapilmadi).
    nutrition_verification: dict[str, bool] | None = None


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


VLM_OCR_ENGINE = "openai_vision"


def _reference_ocr(image_path: Path) -> tuple[str | None, str | None]:
    """Capraz dogrulama motorunu calistirir: (ham metin, None) ya da (None, hata ozeti).

    Motor config.yaml -> ocr.cross_check_engine (null: kapali -> (None, None)). Birincil motor
    OpenAI degilse (yedege dusulmusse) sonuc kullanilmaz ama paralel calistigi icin beklenmez."""
    from src.common.config import get_config

    engine = get_config().get("ocr", {}).get("cross_check_engine")
    if not engine:
        return None, None
    key_name = REQUIRED_API_KEYS.get(engine)
    if key_name and not os.environ.get(key_name):
        return None, f"{engine}: {key_name} yok"
    if engine != "ocrspace":
        return None, f"{engine} capraz dogrulama icin desteklenmiyor"
    try:
        return extract_text_ocrspace(image_path).text, None
    except Exception as exc:
        logger.warning("Capraz dogrulama OCR'i basarisiz: %s", exc)
        return None, f"{engine}: {type(exc).__name__}"


def _verification_warnings(verification: dict[str, bool], nutrition: NutritionFacts) -> list[str]:
    unverified = [f for f, ok in verification.items() if not ok]
    if not unverified:
        return []
    details = ", ".join(f"{FIELD_LABELS_TR.get(f, f)} ({getattr(nutrition, f):g})" for f in unverified)
    warnings = [f"Su degerler ikinci bir OCR motoruyla dogrulanamadi, etiketten kontrol edin: {details}."]
    if len(verification) >= 3 and len(unverified) * 2 >= len(verification):
        warnings.append(
            "Okunan degerlerin yarisindan fazlasi dogrulanamadi - etiket net okunamamis olabilir; "
            "daha yakindan ve net bir fotograf cekin."
        )
    return warnings


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

        # Birincil OCR (OpenAI) ile capraz dogrulama motoru (OCR.space) PARALEL calisir.
        with ThreadPoolExecutor(max_workers=2) as pool:
            primary_future = pool.submit(extract_text_live, processed_path)
            reference_future = pool.submit(_reference_ocr, processed_path)
            ocr_result = primary_future.result()
            reference_text, reference_error = reference_future.result()
        nutrition, _basis = extract_and_normalize(ocr_result.text)
        detected_allergens = detect_allergens(ocr_result.text)

    warnings: list[str] = assess_image_quality(image_path)
    if ocr_result.fallback_notes:
        warnings.append(
            f"OCR yedek motora dustu ({'; '.join(ocr_result.fallback_notes)}), "
            f"{ocr_result.engine} kullanildi - dogruluk dusuk olabilir."
        )
    if not ocr_result.text.strip():
        warnings.append("Etiketten metin okunamadi - daha yakindan ve net bir fotograf cekin.")

    verification = None
    if ocr_result.engine == VLM_OCR_ENGINE and ocr_result.text.strip():
        if reference_text is not None:
            verification = verify_nutrition(nutrition, ocr_result.text, reference_text)
        elif reference_error is not None:
            warnings.append(
                f"Besin degerleri ikinci bir OCR motoruyla dogrulanamadi ({reference_error}) - "
                "okunamayan etiketlerde yapay zeka OCR'i deger uydurabilir, etiketten kontrol edin."
            )

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

    if verification and nutrition_source == "ocr":
        warnings.extend(_verification_warnings(verification, nutrition))
    if ocr_result.text.strip() and not nutrition.model_dump(exclude_none=True):
        warnings.append(
            "Etiketten besin degeri cikarilamadi - besin tablosunun net gorundugu bir fotograf cekin."
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
        nutrition_verification=verification,
        explanation=explanation,
        health_assessment=health_assessment,
        alternatives=alternatives,
    )
