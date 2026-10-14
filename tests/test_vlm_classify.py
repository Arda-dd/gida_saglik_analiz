from unittest.mock import MagicMock

from PIL import Image

from src.vision.vlm_classify import classify_with_openai, combine_category_predictions


def _client(answer: str):
    client = MagicMock()
    client.chat.completions.create.return_value.choices = [MagicMock()]
    client.chat.completions.create.return_value.choices[0].message.content = answer
    return client


def _image(tmp_path):
    path = tmp_path / "x.png"
    Image.new("RGB", (64, 64), "white").save(path)
    return path


def test_classify_normalizes_answer(tmp_path):
    assert classify_with_openai(_image(tmp_path), client=_client(" Atistirmalik. ")) == "atistirmalik"


def test_classify_unknown_answer_becomes_bilinmiyor(tmp_path):
    assert classify_with_openai(_image(tmp_path), client=_client("dondurma")) == "bilinmiyor"


def test_classify_uses_low_detail_image(tmp_path):
    client = _client("icecek")
    classify_with_openai(_image(tmp_path), client=client)
    content = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert content[1]["image_url"]["detail"] == "low"


def test_combine_rules():
    assert combine_category_predictions("icecek", 0.6, None) == ("icecek", 0.6, "cnn")
    assert combine_category_predictions("icecek", 0.6, "bilinmiyor") == ("icecek", 0.6, "cnn")
    category, confidence, source = combine_category_predictions("icecek", 0.6, "icecek")
    assert (category, source) == ("icecek", "cnn+vlm") and confidence > 0.9
    category, confidence, source = combine_category_predictions("sut_urunu", 0.99, "hazir_gida")
    assert (category, source) == ("hazir_gida", "vlm") and confidence < 0.99
