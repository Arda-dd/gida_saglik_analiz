from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from api import pipeline as pipeline_module
from src.common.schema import Allergen, ProductCategory
from src.health.profile import ChronicCondition, HealthProfile
from src.ocr.extract import OCRResult
from src.rag.generate import GenerationResult


class TestGetCandidateProducts:
    def test_parses_csv_and_skips_rows_without_nutrition(self, tmp_path, monkeypatch):
        csv_path = tmp_path / "dataset.csv"
        pd.DataFrame(
            [
                {
                    "product_id": "off_1",
                    "category": "atistirmalik",
                    "source": "open_food_facts",
                    "image_path": "x.jpg",
                    "energy_kcal": 100.0,
                    "energy_kj": 418.0,
                    "fat_g": 5.0,
                    "saturated_fat_g": 1.0,
                    "carbohydrate_g": 10.0,
                    "sugar_g": 8.0,
                    "fiber_g": 1.0,
                    "protein_g": 2.0,
                    "salt_g": 0.5,
                    "sodium_mg": 200.0,
                    "allergens": "findik;gluten",
                },
                {
                    "product_id": "local_1",
                    "category": "icecek",
                    "source": "local_market",
                    "image_path": "y.jpg",
                    "energy_kcal": None,
                    "energy_kj": None,
                    "fat_g": None,
                    "saturated_fat_g": None,
                    "carbohydrate_g": None,
                    "sugar_g": None,
                    "fiber_g": None,
                    "protein_g": None,
                    "salt_g": None,
                    "sodium_mg": None,
                    "allergens": "",
                },
            ]
        ).to_csv(csv_path, index=False)

        monkeypatch.setattr(pipeline_module, "DATASET_PATH", csv_path)
        pipeline_module.get_candidate_products.cache_clear()

        products = pipeline_module.get_candidate_products()

        assert len(products) == 1
        assert products[0].product_id == "off_1"
        assert products[0].category == ProductCategory.ATISTIRMALIK
        assert set(products[0].allergens) == {Allergen.FINDIK, Allergen.GLUTEN}

    def test_returns_empty_tuple_when_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pipeline_module, "DATASET_PATH", tmp_path / "does_not_exist.csv")
        pipeline_module.get_candidate_products.cache_clear()

        assert pipeline_module.get_candidate_products() == ()

    def test_handles_blank_allergens_cell_read_as_nan_by_pandas(self, tmp_path, monkeypatch):
        # Regresyon testi: bos bir CSV hucresi pandas tarafindan NaN (float) olarak okunur.
        # NaN Python'da truthy oldugundan `or ""` ile yakalanamaz - gercek dataset.csv'de
        # bu, Allergen("nan") ValueError'ina yol aciyordu (bkz. git gecmisi).
        csv_path = tmp_path / "dataset.csv"
        csv_path.write_text(
            "product_id,category,source,image_path,energy_kcal,energy_kj,fat_g,"
            "saturated_fat_g,carbohydrate_g,sugar_g,fiber_g,protein_g,salt_g,sodium_mg,allergens\n"
            "off_2,icecek,open_food_facts,z.jpg,50.0,209.0,0.0,0.0,12.0,12.0,0.0,0.0,0.0,0.0,\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(pipeline_module, "DATASET_PATH", csv_path)
        pipeline_module.get_candidate_products.cache_clear()

        products = pipeline_module.get_candidate_products()

        assert len(products) == 1
        assert products[0].allergens == []


@pytest.fixture
def mocked_pipeline_dependencies(monkeypatch):
    """Vision/OCR/RAG bagimliliklarini mock'lar - gercek model/gorsel/API cagrisi yapmadan
    orkestrasyon mantigini (dogru veri akisi, opsiyonel katmanlarin dogru dallanmasi) test
    etmeyi saglar."""
    monkeypatch.setattr(pipeline_module, "preprocess_label_image", lambda src, dest: dest)
    monkeypatch.setattr(
        pipeline_module,
        "_get_vision_model",
        lambda: (MagicMock(), ["sut_urunu", "atistirmalik", "icecek", "hazir_gida", "konserve"]),
    )
    monkeypatch.setattr(
        pipeline_module, "predict_category", lambda model, path, categories: ("atistirmalik", 0.91)
    )
    monkeypatch.setattr(
        pipeline_module,
        "extract_text_easyocr",
        lambda path: OCRResult(text="Seker: 35 g Tuz: 2 g Findik icerir", mean_confidence=82.0, engine="easyocr"),
    )
    yield


class TestAnalyzeLabelImage:
    def test_without_profile_skips_personal_filter(self, mocked_pipeline_dependencies, monkeypatch):
        monkeypatch.setattr(pipeline_module, "generate_explanation", lambda *a, **kw: None)

        result = pipeline_module.analyze_label_image(
            Path("fake.jpg"), profile=None, generate_llm_explanation=False
        )

        assert result.category == "atistirmalik"
        assert result.category_confidence == pytest.approx(0.91)
        assert result.nutrition.sugar_g == pytest.approx(35.0)
        assert "yuksek_seker" in result.risk_flags
        assert Allergen.FINDIK in result.detected_allergens
        assert result.health_assessment is None
        assert result.alternatives == []

    def test_with_profile_computes_health_assessment_and_alternatives(
        self, mocked_pipeline_dependencies, monkeypatch
    ):
        monkeypatch.setattr(
            pipeline_module,
            "get_candidate_products",
            lambda: (),
        )

        profile = HealthProfile(
            profile_id="p1", chronic_conditions=[ChronicCondition.DIYABET], allergens=[Allergen.FINDIK]
        )

        result = pipeline_module.analyze_label_image(
            Path("fake.jpg"), profile=profile, generate_llm_explanation=False
        )

        assert result.health_assessment is not None
        assert result.health_assessment.allergen_warning is True
        assert any("Diyabet" in msg for msg in result.health_assessment.health_risk_messages)
        assert result.alternatives == []

    def test_generate_llm_explanation_false_skips_rag_entirely(
        self, mocked_pipeline_dependencies, monkeypatch
    ):
        called = MagicMock()
        monkeypatch.setattr(pipeline_module, "generate_explanation", called)

        result = pipeline_module.analyze_label_image(
            Path("fake.jpg"), profile=None, generate_llm_explanation=False
        )

        called.assert_not_called()
        assert result.explanation is None

    def test_rag_failure_gracefully_degrades_instead_of_crashing(
        self, mocked_pipeline_dependencies, monkeypatch
    ):
        def _raise(*args, **kwargs):
            raise RuntimeError("API anahtari yok / rate limit")

        monkeypatch.setattr(pipeline_module, "_get_retriever", lambda: MagicMock())
        monkeypatch.setattr(pipeline_module, "_get_llm", lambda: MagicMock())
        monkeypatch.setattr(pipeline_module, "generate_explanation", _raise)

        result = pipeline_module.analyze_label_image(
            Path("fake.jpg"), profile=None, generate_llm_explanation=True
        )

        assert result.explanation is None
        assert result.category == "atistirmalik"

    def test_rag_success_populates_explanation(self, mocked_pipeline_dependencies, monkeypatch):
        fake_result = GenerationResult(
            text="Bu urun yuksek seker icerir [Kaynak: who_sugars_intake::0].",
            retrieved=[],
            cited_chunk_ids=["who_sugars_intake::0"],
            valid_citation_ratio=1.0,
            numeric_grounding_ratio=1.0,
            regenerated=False,
        )
        monkeypatch.setattr(pipeline_module, "_get_retriever", lambda: MagicMock())
        monkeypatch.setattr(pipeline_module, "_get_llm", lambda: MagicMock())
        monkeypatch.setattr(pipeline_module, "generate_explanation", lambda *a, **kw: fake_result)

        result = pipeline_module.analyze_label_image(
            Path("fake.jpg"), profile=None, generate_llm_explanation=True
        )

        assert result.explanation is fake_result
