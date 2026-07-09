"""FastAPI backend (Faz 6): etiket fotografi yukleme -> tam analiz pipeline -> 3 katmanli cikti.

Oneri formu 2.6: POST /analyze, POST /profile, GET /recommend uc noktalari. Profil deposu
bu prototipte bellek-ici bir sozluktur (kalici veritabani formun kapsami disi - Faz 6 hedefi
uctan uca akisi gostermektir, uretim veritabani degil).
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api.pipeline import AnalysisResult, analyze_label_image, get_candidate_products
from src.common.schema import Allergen
from src.health.profile import ChronicCondition, HealthProfile
from src.health.recommend import recommend_alternatives

app = FastAPI(
    title="Gida & Saglik Asistani API",
    description="TUBITAK 2209-A - gorsel + metin analizi ile besin bilgilendirme sistemi",
    version="0.1.0",
)

_PROFILE_STORE: dict[str, HealthProfile] = {}


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request, exc: Exception) -> JSONResponse:
    """Form Risk Yonetimi B-plani: beklenmeyen hatalar ham stack trace yerine yapilandirilmis
    bir JSON hatasi olarak donmeli (kullaniciya guvenli, taninabilir bir mesaj)."""
    return JSONResponse(
        status_code=500,
        content={"detail": f"Beklenmeyen bir hata olustu: {exc.__class__.__name__}"},
    )


class ProfileCreateRequest(BaseModel):
    chronic_conditions: list[ChronicCondition] = Field(default_factory=list)
    allergens: list[Allergen] = Field(default_factory=list)
    daily_calorie_target_kcal: float | None = None
    daily_protein_target_g: float | None = None
    daily_fat_target_g: float | None = None
    daily_carbohydrate_target_g: float | None = None


class ProfileResponse(BaseModel):
    profile_id: str


@app.post("/profile", response_model=ProfileResponse)
def create_profile(request: ProfileCreateRequest) -> ProfileResponse:
    """Anonim bir saglik profili olusturur (isim/iletisim bilgisi ALINMAZ, bkz. HealthProfile)."""
    profile_id = f"anon_{uuid.uuid4().hex[:12]}"
    profile = HealthProfile(profile_id=profile_id, **request.model_dump())
    _PROFILE_STORE[profile_id] = profile
    return ProfileResponse(profile_id=profile_id)


def _serialize_result(result: AnalysisResult) -> dict:
    payload = {
        "category": result.category,
        "category_confidence": result.category_confidence,
        "nutrition": result.nutrition.model_dump(exclude_none=True),
        "detected_allergens": [a.value for a in result.detected_allergens],
        "risk_flags": result.risk_flags,
        "risk_messages": result.risk_messages,
        "ocr_confidence": result.ocr_confidence,
    }

    if result.explanation is not None:
        payload["explanation"] = {
            "text": result.explanation.text,
            "sources": [
                {
                    "chunk_id": r.chunk.chunk_id,
                    "title": r.chunk.title,
                    "section": r.chunk.section,
                    "verified": r.chunk.verified,
                }
                for r in result.explanation.retrieved
            ],
            "valid_citation_ratio": result.explanation.valid_citation_ratio,
        }

    if result.health_assessment is not None:
        payload["health_assessment"] = {
            "health_risk_messages": result.health_assessment.health_risk_messages,
            "diet_compliance_score": result.health_assessment.diet_compliance_score,
            "allergen_warning": result.health_assessment.allergen_warning,
            "allergen_conflicts": [a.value for a in result.health_assessment.allergen_conflicts],
        }

    if result.alternatives:
        payload["alternatives"] = [
            {"product_id": p.product_id, "category": p.category.value}
            for p in result.alternatives
        ]

    return payload


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    profile_id: str | None = None,
    generate_explanation: bool = True,
) -> dict:
    """Yuklenen etiket fotografini tam pipeline'dan (Faz 2-5) gecirip 3 katmanli sonuc doner."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Sadece gorsel dosyalari kabul edilir")

    profile = None
    if profile_id is not None:
        profile = _PROFILE_STORE.get(profile_id)
        if profile is None:
            raise HTTPException(status_code=404, detail=f"Profil bulunamadi: {profile_id}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        upload_path = Path(tmp_dir) / (file.filename or "upload.jpg")
        with upload_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        try:
            result = analyze_label_image(
                upload_path, profile=profile, generate_llm_explanation=generate_explanation
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    return _serialize_result(result)


@app.get("/recommend")
def recommend(product_id: str, profile_id: str, max_results: int = 3) -> dict:
    """Verilen urune (dataset.csv'deki bir product_id) ve profile gore alternatif urun onerir."""
    profile = _PROFILE_STORE.get(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Profil bulunamadi: {profile_id}")

    candidates = get_candidate_products()
    current = next((p for p in candidates if p.product_id == product_id), None)
    if current is None:
        raise HTTPException(status_code=404, detail=f"Urun bulunamadi: {product_id}")

    alternatives = recommend_alternatives(current, list(candidates), profile, max_results=max_results)
    return {
        "product_id": product_id,
        "alternatives": [
            {"product_id": p.product_id, "category": p.category.value} for p in alternatives
        ],
    }


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}
