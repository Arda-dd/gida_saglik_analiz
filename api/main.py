from __future__ import annotations

from datetime import timedelta
import hashlib
import json
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.database import init_db, get_db, User as DBUser, Profile as DBProfile, ScanHistory as DBScanHistory
from api.auth import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
    get_current_user_optional,
)
from api.pipeline import AnalysisResult, analyze_label_image, get_candidate_products
from src.common.schema import Allergen, NutritionFacts, ProductRecord, UserObjective
from src.health.profile import ChronicCondition, HealthProfile
from src.health.recommend import build_health_assessment, recommend_alternatives
from src.ocr.risk_engine import describe_risks

app = FastAPI(
    title="Gida & Saglik Asistani API",
    description="TUBITAK 2209-A - gorsel + metin analizi ile besin bilgilendirme sistemi",
    version="0.1.0",
)

_PROFILE_STORE: dict[str, HealthProfile] = {}


@app.on_event("startup")
def startup_event():
    init_db()


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request, exc: Exception) -> JSONResponse:
    """Form Risk Yonetimi B-plani: beklenmeyen hatalar ham stack trace yerine yapilandirilmis
    bir JSON hatasi olarak donmeli (kullaniciya guvenli, taninabilir bir mesaj)."""
    return JSONResponse(
        status_code=500,
        content={"detail": f"Beklenmeyen bir hata olustu: {exc.__class__.__name__}"},
    )


class UserRegisterRequest(BaseModel):
    email: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class ProfileCreateRequest(BaseModel):
    chronic_conditions: list[ChronicCondition] = Field(default_factory=list)
    allergens: list[Allergen] = Field(default_factory=list)
    daily_calorie_target_kcal: float | None = None
    daily_protein_target_g: float | None = None
    daily_fat_target_g: float | None = None
    daily_carbohydrate_target_g: float | None = None
    objective: UserObjective | None = None


class ProfileResponse(BaseModel):
    profile_id: str


@app.post("/auth/register", response_model=dict)
def register(request: UserRegisterRequest, db: Session = Depends(get_db)):
    user = db.query(DBUser).filter(DBUser.email == request.email).first()
    if user:
        raise HTTPException(status_code=400, detail="Bu e-posta adresiyle kayıtlı bir kullanıcı zaten var.")

    hashed = hash_password(request.password)
    new_user = DBUser(email=request.email, hashed_password=hashed)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    profile = DBProfile(user_id=new_user.id, objective=None)
    db.add(profile)
    db.commit()

    return {"message": "Kayıt başarıyla tamamlandı."}


@app.post("/auth/login", response_model=Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    remember_me: bool = False,
    db: Session = Depends(get_db),
):
    user = db.query(DBUser).filter(DBUser.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Hatalı e-posta veya şifre.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    expires_delta = timedelta(days=30) if remember_me else timedelta(days=1)
    access_token = create_access_token(data={"sub": user.email}, expires_delta=expires_delta)
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/auth/me")
def get_me(current_user: DBUser = Depends(get_current_user)):
    profile_data = {}
    if current_user.profile:
        p = current_user.profile
        profile_data = {
            "chronic_conditions": [c for c in p.chronic_conditions.split(",") if c],
            "allergens": [a for a in p.allergens.split(",") if a],
            "daily_calorie_target_kcal": p.daily_calorie_target_kcal,
            "daily_protein_target_g": p.daily_protein_target_g,
            "daily_fat_target_g": p.daily_fat_target_g,
            "daily_carbohydrate_target_g": p.daily_carbohydrate_target_g,
            "objective": p.objective,
        }
    return {"id": current_user.id, "email": current_user.email, "profile": profile_data}


@app.post("/profile", response_model=ProfileResponse)
def create_profile(
    request: ProfileCreateRequest,
    current_user: DBUser | None = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
) -> ProfileResponse:
    """Kimlik doğrulanmış kullanıcı varsa veritabanındaki profilini günceller, yoksa anonim profil oluşturur."""
    if current_user:
        profile = current_user.profile
        if not profile:
            profile = DBProfile(user_id=current_user.id)
            db.add(profile)

        profile.chronic_conditions = ",".join(c.value for c in request.chronic_conditions)
        profile.allergens = ",".join(a.value for a in request.allergens)
        profile.daily_calorie_target_kcal = request.daily_calorie_target_kcal
        profile.daily_protein_target_g = request.daily_protein_target_g
        profile.daily_fat_target_g = request.daily_fat_target_g
        profile.daily_carbohydrate_target_g = request.daily_carbohydrate_target_g
        profile.objective = request.objective.value if request.objective else None
        db.commit()
        return ProfileResponse(profile_id=f"user_{current_user.id}")

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
            "numeric_grounding_ratio": result.explanation.numeric_grounding_ratio,
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
    current_user: DBUser | None = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
) -> dict:
    """Yuklenen etiket fotografini tam pipeline'dan gecirir. Caching ve ScanHistory destegi sunar."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Sadece gorsel dosyalari kabul edilir")

    # Dosya hash'ini olustur
    file_bytes = await file.read()
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    await file.seek(0)

    profile = None
    if current_user:
        db_prof = current_user.profile
        if db_prof:
            profile = HealthProfile(
                profile_id=f"db_{current_user.id}",
                chronic_conditions=[ChronicCondition(c) for c in db_prof.chronic_conditions.split(",") if c],
                allergens=[Allergen(a) for a in db_prof.allergens.split(",") if a],
                daily_calorie_target_kcal=db_prof.daily_calorie_target_kcal,
                daily_protein_target_g=db_prof.daily_protein_target_g,
                daily_fat_target_g=db_prof.daily_fat_target_g,
                daily_carbohydrate_target_g=db_prof.daily_carbohydrate_target_g,
                objective=UserObjective(db_prof.objective) if db_prof.objective else None
            )
    elif profile_id is not None:
        profile = _PROFILE_STORE.get(profile_id)
        if profile is None:
            raise HTTPException(status_code=404, detail=f"Profil bulunamadi: {profile_id}")

    # Caching kontrolü
    if current_user:
        cached_scan = db.query(DBScanHistory).filter(
            DBScanHistory.user_id == current_user.id,
            DBScanHistory.file_hash == file_hash
        ).first()
        if cached_scan:
            nutrition_data = json.loads(cached_scan.nutrition_json)
            nutrition = NutritionFacts(**nutrition_data)
            detected_allergens_list = [Allergen(a) for a in cached_scan.detected_allergens.split(",") if a]
            risk_flags_list = [rf for rf in cached_scan.risk_flags.split(",") if rf]

            payload = {
                "category": cached_scan.category,
                "category_confidence": cached_scan.category_confidence,
                "nutrition": nutrition_data,
                "detected_allergens": [a.value for a in detected_allergens_list],
                "risk_flags": risk_flags_list,
                "risk_messages": describe_risks(risk_flags_list),
                "ocr_confidence": cached_scan.ocr_confidence,
                "cached": True
            }

            if cached_scan.explanation_text:
                payload["explanation"] = {
                    "text": cached_scan.explanation_text,
                    "sources": [],
                    "valid_citation_ratio": 1.0,
                }

            if profile:
                health_assessment = build_health_assessment(nutrition, detected_allergens_list, profile)
                payload["health_assessment"] = {
                    "health_risk_messages": health_assessment.health_risk_messages,
                    "diet_compliance_score": health_assessment.diet_compliance_score,
                    "allergen_warning": health_assessment.allergen_warning,
                    "allergen_conflicts": [a.value for a in health_assessment.allergen_conflicts],
                }

                candidates = get_candidate_products()
                try:
                    from src.common.schema import ProductCategory
                    current_cat = ProductCategory(cached_scan.category)
                except ValueError:
                    current_cat = ProductCategory.BILINMIYOR

                current_prod = ProductRecord(
                    product_id="cached_image",
                    category=current_cat,
                    nutrition=nutrition,
                    allergens=detected_allergens_list,
                    source="cached"
                )
                alternatives = recommend_alternatives(current_prod, list(candidates), profile)
                payload["alternatives"] = [
                    {"product_id": p.product_id, "category": p.category.value} for p in alternatives
                ]

            return payload

    with tempfile.TemporaryDirectory() as tmp_dir:
        upload_path = Path(tmp_dir) / (file.filename or "upload.jpg")
        with upload_path.open("wb") as f:
            f.write(file_bytes)

        try:
            result = analyze_label_image(
                upload_path, profile=profile, generate_llm_explanation=generate_explanation
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    # Veritabanına kaydet
    if current_user:
        nutrition_json = json.dumps(result.nutrition.model_dump(exclude_none=True))
        detected_allergens_str = ",".join(a.value for a in result.detected_allergens)
        risk_flags_str = ",".join(result.risk_flags)
        explanation_text = result.explanation.text if result.explanation else None

        new_scan = DBScanHistory(
            user_id=current_user.id,
            product_id=f"scan_{uuid.uuid4().hex[:8]}",
            category=result.category,
            category_confidence=result.category_confidence,
            nutrition_json=nutrition_json,
            detected_allergens=detected_allergens_str,
            risk_flags=risk_flags_str,
            ocr_confidence=result.ocr_confidence,
            file_hash=file_hash,
            explanation_text=explanation_text
        )
        db.add(new_scan)
        db.commit()

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
