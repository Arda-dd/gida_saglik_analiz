import io

import pytest
from fastapi.testclient import TestClient

from api import main as main_module
from api.pipeline import AnalysisResult
from src.common.schema import Allergen, NutritionFacts, ProductCategory, ProductRecord

client = TestClient(main_module.app)


@pytest.fixture(autouse=True)
def _clear_profile_store():
    main_module._PROFILE_STORE.clear()
    yield
    main_module._PROFILE_STORE.clear()


def _fake_image_bytes() -> bytes:
    # Gercek bir JPEG olmasi gerekmez - analyze_label_image mock'landigi icin icerik
    # okunmayacak, sadece content_type/dosya akisi test edilecek.
    return b"fake-image-bytes"


class TestHealthCheck:
    def test_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestCreateProfile:
    def test_returns_profile_id_and_stores_profile(self):
        response = client.post(
            "/profile",
            json={"chronic_conditions": ["diyabet"], "allergens": ["findik"], "daily_calorie_target_kcal": 2000},
        )

        assert response.status_code == 200
        profile_id = response.json()["profile_id"]
        assert profile_id.startswith("anon_")
        assert profile_id in main_module._PROFILE_STORE

    def test_defaults_work_with_empty_body(self):
        response = client.post("/profile", json={})
        assert response.status_code == 200


class TestAnalyze:
    def test_rejects_non_image_content_type(self):
        response = client.post(
            "/analyze",
            files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
        )
        assert response.status_code == 400

    def test_returns_404_for_unknown_profile_id(self):
        response = client.post(
            "/analyze",
            files={"file": ("test.jpg", io.BytesIO(_fake_image_bytes()), "image/jpeg")},
            params={"profile_id": "does_not_exist"},
        )
        assert response.status_code == 404

    def test_returns_serialized_result_on_success(self, monkeypatch):
        fake_result = AnalysisResult(
            category="atistirmalik",
            category_confidence=0.9,
            nutrition=NutritionFacts(sugar_g=35.0),
            detected_allergens=[Allergen.FINDIK],
            risk_flags=["yuksek_seker"],
            risk_messages=["Yuksek seker icerir (100g'da esigin uzerinde)."],
            ocr_confidence=80.0,
        )
        monkeypatch.setattr(main_module, "analyze_label_image", lambda *a, **kw: fake_result)

        response = client.post(
            "/analyze",
            files={"file": ("test.jpg", io.BytesIO(_fake_image_bytes()), "image/jpeg")},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["category"] == "atistirmalik"
        assert body["nutrition"]["sugar_g"] == 35.0
        assert body["detected_allergens"] == ["findik"]
        assert "yuksek_seker" in body["risk_flags"]
        assert "health_assessment" not in body

    def test_propagates_value_error_as_422(self, monkeypatch):
        def _raise(*args, **kwargs):
            raise ValueError("Gorsel okunamadi")

        monkeypatch.setattr(main_module, "analyze_label_image", _raise)

        response = client.post(
            "/analyze",
            files={"file": ("test.jpg", io.BytesIO(_fake_image_bytes()), "image/jpeg")},
        )

        assert response.status_code == 422


class TestRecommend:
    def test_returns_404_for_unknown_profile(self, monkeypatch):
        monkeypatch.setattr(main_module, "get_candidate_products", lambda: ())
        response = client.get("/recommend", params={"product_id": "x", "profile_id": "missing"})
        assert response.status_code == 404

    def test_returns_404_for_unknown_product(self):
        profile_id = client.post("/profile", json={}).json()["profile_id"]
        response = client.get("/recommend", params={"product_id": "missing", "profile_id": profile_id})
        assert response.status_code == 404

    def test_returns_alternatives_for_known_product(self, monkeypatch):
        current = ProductRecord(
            product_id="cur",
            category=ProductCategory.ATISTIRMALIK,
            nutrition=NutritionFacts(sugar_g=40.0),
            allergens=[],
            source="test",
        )
        better = ProductRecord(
            product_id="better",
            category=ProductCategory.ATISTIRMALIK,
            nutrition=NutritionFacts(sugar_g=1.0),
            allergens=[],
            source="test",
        )
        monkeypatch.setattr(main_module, "get_candidate_products", lambda: (current, better))

        profile_id = client.post("/profile", json={}).json()["profile_id"]
        response = client.get("/recommend", params={"product_id": "cur", "profile_id": profile_id})

        assert response.status_code == 200
        body = response.json()
        assert body["alternatives"] == [{"product_id": "better", "category": "atistirmalik"}]
