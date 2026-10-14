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
    monkeypatch.setattr(pipeline_module, "decode_barcodes", lambda path: [])
    monkeypatch.setattr(pipeline_module, "lookup_barcode", lambda code: None)
    monkeypatch.setattr(pipeline_module, "_reference_ocr", lambda path: (None, None))
    monkeypatch.setattr(pipeline_module, "assess_image_quality", lambda path: [])
    monkeypatch.setattr(pipeline_module, "_vlm_category", lambda path: None)
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
        "extract_text_live",
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
        assert result.ocr_engine == "easyocr"
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
        # Hata artik sessiz degil: yanitta gorunur bir uyari var.
        assert any("aciklamasi uretilemedi" in w and "rate limit" in w for w in result.warnings)

    def test_ocr_fallback_is_reported_as_warning(self, mocked_pipeline_dependencies, monkeypatch):
        monkeypatch.setattr(
            pipeline_module,
            "extract_text_live",
            lambda path: OCRResult(
                text="Seker: 35 g", mean_confidence=82.0, engine="easyocr",
                fallback_notes=["openai_vision: TimeoutError"],
            ),
        )
        result = pipeline_module.analyze_label_image(Path("fake.jpg"), generate_llm_explanation=False)
        assert any("yedek motora dustu" in w and "TimeoutError" in w for w in result.warnings)

    def test_unreadable_label_asks_for_better_photo(self, mocked_pipeline_dependencies, monkeypatch):
        monkeypatch.setattr(
            pipeline_module, "extract_text_live",
            lambda path: OCRResult(text="", mean_confidence=0.0, engine="easyocr"),
        )
        result = pipeline_module.analyze_label_image(Path("fake.jpg"), generate_llm_explanation=False)
        assert any("metin okunamadi" in w for w in result.warnings)

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


def _barcode_product(nutrition, category=ProductCategory.ATISTIRMALIK, allergens=(), traces=()):
    from src.barcode.lookup import BarcodeProduct
    from src.common.schema import NutritionFacts, ProductRecord

    return BarcodeProduct(
        barcode="3046920022606",
        record=ProductRecord(
            product_id="off_3046920022606", category=category,
            nutrition=NutritionFacts(**nutrition), allergens=list(allergens),
        ),
        product_name="Test cikolata",
        traces=list(traces),
    )


class TestBarcodePath:
    def test_barcode_product_overrides_ocr_nutrition_and_category(self, mocked_pipeline_dependencies, monkeypatch):
        # OCR (fikstur): "Seker: 35 g Tuz: 2 g Findik icerir"; CNN: atistirmalik.
        monkeypatch.setattr(pipeline_module, "decode_barcodes", lambda path: ["3046920022606"])
        monkeypatch.setattr(
            pipeline_module, "lookup_barcode",
            lambda code: _barcode_product(
                {"energy_kcal": 120, "sugar_g": 34, "fat_g": 1, "salt_g": 2.1},
                category=ProductCategory.ICECEK, allergens=[Allergen.LAKTOZ], traces=[Allergen.SOYA],
            ),
        )
        result = pipeline_module.analyze_label_image(Path("fake.jpg"), generate_llm_explanation=False)

        assert result.barcode == "3046920022606"
        assert result.product_name == "Test cikolata"
        assert result.nutrition_source == "open_food_facts"
        assert result.nutrition.energy_kcal == 120
        assert result.nutrition.sugar_g == 34  # 35 (etiket) ile uyumlu -> uyari yok
        assert result.category == "icecek" and result.category_source == "open_food_facts"
        # Alerjen: etiket (findik) + veritabani (laktoz) + eser (soya) birlesimi
        assert set(result.detected_allergens) == {Allergen.FINDIK, Allergen.LAKTOZ, Allergen.SOYA}
        assert any("Eser miktarda" in w for w in result.warnings)
        assert not any("farkli" in w for w in result.warnings)

    def test_label_database_conflict_is_warned(self, mocked_pipeline_dependencies, monkeypatch):
        monkeypatch.setattr(pipeline_module, "decode_barcodes", lambda path: ["3046920022606"])
        monkeypatch.setattr(
            pipeline_module, "lookup_barcode",
            lambda code: _barcode_product({"energy_kcal": 120, "sugar_g": 10, "fat_g": 1}),
        )
        result = pipeline_module.analyze_label_image(Path("fake.jpg"), generate_llm_explanation=False)
        assert result.nutrition.sugar_g == 10
        assert any("seker: etiket 35, veritabani 10" in w for w in result.warnings)

    def test_lookup_failure_falls_back_to_ocr_with_warning(self, mocked_pipeline_dependencies, monkeypatch):
        def _raise(code):
            raise TimeoutError("OFF yavas")

        monkeypatch.setattr(pipeline_module, "decode_barcodes", lambda path: ["3046920022606"])
        monkeypatch.setattr(pipeline_module, "lookup_barcode", _raise)
        result = pipeline_module.analyze_label_image(Path("fake.jpg"), generate_llm_explanation=False)
        assert result.nutrition_source == "ocr"
        assert result.nutrition.sugar_g == 35
        assert any("sorgulanamadi" in w for w in result.warnings)

    def test_barcode_read_from_ocr_text_when_image_decoder_finds_none(self, mocked_pipeline_dependencies, monkeypatch):
        monkeypatch.setattr(
            pipeline_module, "extract_text_live",
            lambda path: OCRResult(text="Seker 35 g\n3 046920 022606", mean_confidence=100.0, engine="openai_vision"),
        )
        seen = []
        monkeypatch.setattr(pipeline_module, "lookup_barcode", lambda code: seen.append(code))
        pipeline_module.analyze_label_image(Path("fake.jpg"), generate_llm_explanation=False)
        assert seen == ["3046920022606"]

    def test_sparse_database_record_does_not_replace_ocr(self, mocked_pipeline_dependencies, monkeypatch):
        monkeypatch.setattr(pipeline_module, "decode_barcodes", lambda path: ["3046920022606"])
        monkeypatch.setattr(pipeline_module, "lookup_barcode", lambda code: _barcode_product({"energy_kcal": 120}))
        result = pipeline_module.analyze_label_image(Path("fake.jpg"), generate_llm_explanation=False)
        assert result.nutrition_source == "ocr"
        assert result.category_source == "open_food_facts"  # kategori yine de kullanilabilir


class TestCrossCheck:
    def _openai(self, monkeypatch, text="Seker 35 g\nTuz 2 g"):
        monkeypatch.setattr(
            pipeline_module, "extract_text_live",
            lambda path: OCRResult(text=text, mean_confidence=100.0, engine="openai_vision"),
        )

    def test_values_missing_from_reference_text_are_flagged(self, mocked_pipeline_dependencies, monkeypatch):
        self._openai(monkeypatch)
        monkeypatch.setattr(pipeline_module, "_reference_ocr", lambda path: ("Sucres 35 g", None))
        result = pipeline_module.analyze_label_image(Path("fake.jpg"), generate_llm_explanation=False)

        assert result.nutrition_verification == {"sugar_g": True, "salt_g": False}
        assert any("dogrulanamadi" in w and "tuz (2)" in w for w in result.warnings)

    def test_mostly_unverified_asks_for_better_photo(self, mocked_pipeline_dependencies, monkeypatch):
        self._openai(monkeypatch, "Seker 35 g\nTuz 2 g\nYag 12 g\nProtein 4 g")
        monkeypatch.setattr(pipeline_module, "_reference_ocr", lambda path: ("Protein 4 g", None))
        result = pipeline_module.analyze_label_image(Path("fake.jpg"), generate_llm_explanation=False)
        assert any("yarisindan fazlasi" in w for w in result.warnings)

    def test_reference_failure_is_reported(self, mocked_pipeline_dependencies, monkeypatch):
        self._openai(monkeypatch)
        monkeypatch.setattr(pipeline_module, "_reference_ocr", lambda path: (None, "ocrspace: HTTPError"))
        result = pipeline_module.analyze_label_image(Path("fake.jpg"), generate_llm_explanation=False)
        assert result.nutrition_verification is None
        assert any("dogrulanamadi (ocrspace: HTTPError)" in w for w in result.warnings)

    def test_no_cross_check_for_non_vlm_engines(self, mocked_pipeline_dependencies, monkeypatch):
        # Fikstur OCR'i easyocr - uydurma riski yok, dogrulama yapilmaz.
        monkeypatch.setattr(pipeline_module, "_reference_ocr", lambda path: ("bambaska metin", None))
        result = pipeline_module.analyze_label_image(Path("fake.jpg"), generate_llm_explanation=False)
        assert result.nutrition_verification is None
        assert not any("dogrulanamadi" in w for w in result.warnings)

    def test_image_quality_warnings_are_included(self, mocked_pipeline_dependencies, monkeypatch):
        monkeypatch.setattr(pipeline_module, "assess_image_quality", lambda path: ["Fotograf bulanik"])
        result = pipeline_module.analyze_label_image(Path("fake.jpg"), generate_llm_explanation=False)
        assert "Fotograf bulanik" in result.warnings


def test_text_without_nutrition_values_warns(mocked_pipeline_dependencies, monkeypatch):
    monkeypatch.setattr(
        pipeline_module, "extract_text_live",
        lambda path: OCRResult(text="Icindekiler: domates, tuz", mean_confidence=100.0, engine="easyocr"),
    )
    result = pipeline_module.analyze_label_image(Path("fake.jpg"), generate_llm_explanation=False)
    assert any("besin degeri cikarilamadi" in w for w in result.warnings)


class TestCategoryVote:
    def test_vlm_overrides_cnn_when_they_disagree(self, mocked_pipeline_dependencies, monkeypatch):
        monkeypatch.setattr(pipeline_module, "_vlm_category", lambda path: "hazir_gida")
        result = pipeline_module.analyze_label_image(Path("fake.jpg"), generate_llm_explanation=False)
        assert result.category == "hazir_gida"
        assert result.category_source == "vlm"

    def test_agreement_strengthens_cnn_decision(self, mocked_pipeline_dependencies, monkeypatch):
        monkeypatch.setattr(pipeline_module, "_vlm_category", lambda path: "atistirmalik")
        result = pipeline_module.analyze_label_image(Path("fake.jpg"), generate_llm_explanation=False)
        assert result.category == "atistirmalik"
        assert result.category_source == "cnn+vlm"
        assert result.category_confidence > 0.91

    def test_cnn_alone_when_vlm_unavailable(self, mocked_pipeline_dependencies):
        result = pipeline_module.analyze_label_image(Path("fake.jpg"), generate_llm_explanation=False)
        assert result.category_source == "cnn"
        assert result.category_confidence == pytest.approx(0.91)

    def test_barcode_category_beats_both_models(self, mocked_pipeline_dependencies, monkeypatch):
        monkeypatch.setattr(pipeline_module, "_vlm_category", lambda path: "hazir_gida")
        monkeypatch.setattr(pipeline_module, "decode_barcodes", lambda path: ["3046920022606"])
        monkeypatch.setattr(
            pipeline_module, "lookup_barcode",
            lambda code: _barcode_product({"energy_kcal": 1}, category=ProductCategory.ICECEK),
        )
        result = pipeline_module.analyze_label_image(Path("fake.jpg"), generate_llm_explanation=False)
        assert result.category == "icecek"
        assert result.category_source == "open_food_facts"
