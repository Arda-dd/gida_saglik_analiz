"""Kisisel saglik profili modeli (Faz 5 - Kisisel Saglik Profili ve Karar Destek).

Oneri formu 2.5: "kronik hastalik (diyabet/hipertansiyon), alerjenler, gunluk kalori/makro
hedefleri" iceren bir profil, retrieval ile generation arasindaki Kisisel Profil Filtreleme
Katmani'nin girdisidir (bkz. src/health/personal_filter.py). Profil anonimdir - isim/iletisim
gibi kisisel kimlik bilgisi tutulmaz, sadece anonim bir profile_id + saglik nitelikleri.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from src.common.schema import Allergen


class ChronicCondition(str, Enum):
    DIYABET = "diyabet"
    HIPERTANSIYON = "hipertansiyon"
    BOBREK_HASTALIGI = "bobrek_hastaligi"
    KALP_HASTALIGI = "kalp_hastaligi"


class HealthProfile(BaseModel):
    """Anonim kullanici saglik profili - isim/iletisim bilgisi ICERMEZ."""

    profile_id: str
    chronic_conditions: list[ChronicCondition] = Field(default_factory=list)
    allergens: list[Allergen] = Field(default_factory=list)
    daily_calorie_target_kcal: float | None = Field(default=None, gt=0)
    daily_protein_target_g: float | None = Field(default=None, ge=0)
    daily_fat_target_g: float | None = Field(default=None, ge=0)
    daily_carbohydrate_target_g: float | None = Field(default=None, ge=0)
