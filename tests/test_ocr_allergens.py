from src.common.schema import Allergen
from src.ocr.allergens import detect_allergens


def test_detect_allergens_exact_lactose_keyword():
    assert Allergen.LAKTOZ in detect_allergens("Icindekiler: sut, seker, un")


def test_detect_allergens_exact_gluten_keyword():
    assert Allergen.GLUTEN in detect_allergens("Bugday unu, su, tuz")


def test_detect_allergens_exact_hazelnut_keyword():
    assert Allergen.FINDIK in detect_allergens("kakao, findik, seker")


def test_detect_allergens_soy_short_synonym_exact_match():
    assert Allergen.SOYA in detect_allergens("soya lesitini, seker")


def test_detect_allergens_egg_keyword():
    assert Allergen.YUMURTA in detect_allergens("yumurta, un, sut")


def test_detect_allergens_fish_keyword():
    assert Allergen.BALIK in detect_allergens("balik unu, tuz")


def test_detect_allergens_tolerates_minor_ocr_typo_in_long_word():
    # "findik" -> OCR hatasiyla "findlk" (tek harf degisikligi, uzun kelime - fuzzy tolerans devrede)
    assert Allergen.FINDIK in detect_allergens("kakao, findlk parcalari, seker")


def test_detect_allergens_no_false_positive_on_unrelated_text():
    result = detect_allergens("su, seker, tuz, karbonhidrat")
    assert result == []


def test_detect_allergens_short_synonym_does_not_fuzzy_match_unrelated_word():
    # "un" kisa bir sinonim (gluten gostergesi degil, bu listede yok ama benzer risk icin
    # soya'nin kisa sinonimi "soy" un rastgele bir kelimeyle fuzzy eslesmemesi test edilir
    result = detect_allergens("bu urun soguk yerde saklanmalidir")
    assert Allergen.SOYA not in result


def test_detect_allergens_multiple_allergens_in_same_text():
    text = "Bugday unu, sut tozu, findik, yumurta"
    result = detect_allergens(text)

    assert Allergen.GLUTEN in result
    assert Allergen.LAKTOZ in result
    assert Allergen.FINDIK in result
    assert Allergen.YUMURTA in result


def test_detect_allergens_empty_text_returns_empty_list():
    assert detect_allergens("") == []


def test_detect_allergens_case_insensitive():
    assert Allergen.GLUTEN in detect_allergens("BUGDAY UNU")
