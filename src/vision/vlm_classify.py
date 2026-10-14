"""Gorsel-dil modeliyle (OpenAI) urun kategorisi tahmini - CNN'i DESTEKLEMEK icin.

Oneri formu kategori siniflandirmasi icin CNN taahhut ediyor; bu modul CNN'in yerini almaz,
onunla birlikte oylama icin ikinci bir gorus saglar (bkz. combine_category_predictions).
CNN'in test dogrulugu %75 civarinda kaldi (394 gorsellik kucuk veri seti); canli testte hazir
gidayi "sut_urunu" dedi. Kategori tanimlari egitim etiketlerini ureten OFF eslemesiyle
(src/data/off_category_map.py) ayni tutulur - aksi halde model "dogru" ama farkli bir
taksonomiyle cevap verir.
"""

from __future__ import annotations

import os
from pathlib import Path

from src.ocr.extract import _encode_image_for_vlm

CATEGORY_DESCRIPTIONS = {
    "konserve": "konserve ve kavanoz urunleri: konserve sebze/baklagil/balik/ton, recel, tursu",
    "sut_urunu": "sut urunleri: sut, yogurt, peynir, ayran/kefir, sutlu tatlilar",
    "icecek": "icecekler: su, maden suyu, gazli icecek, meyve suyu, cay, enerji icecegi, bitkisel sut",
    "atistirmalik": "atistirmaliklar: biskuvi, kek, cikolata, cips, kraker, gofret, tahil bari, sekerleme",
    "hazir_gida": "hazir gidalar: hazir yemek, dondurulmus urun, pizza, makarna, hazir corba, sandvic, salata",
}
VLM_CATEGORY_PROMPT = (
    "Bu bir gida urununun fotografi. Urunu asagidaki kategorilerden TAM OLARAK BIRINE ata. "
    "Bir urun birden fazlasina uyuyorsa listede ONCE gelen kategoriyi sec.\n"
    + "\n".join(f"- {name}: {desc}" for name, desc in CATEGORY_DESCRIPTIONS.items())
    + "\nSadece kategori adini yaz (ornegin: atistirmalik). Emin degilsen yine en olasi olani yaz."
)


def classify_with_openai(image_path: Path, model: str = "gpt-4.1-mini", client=None) -> str:
    """Kategori adini doner; cevap gecerli bir kategori degilse 'bilinmiyor'."""
    if client is None:
        import openai

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY bulunamadi (.env dosyasini kontrol edin)")
        client = openai.OpenAI(api_key=api_key, timeout=30)

    mime, encoded = _encode_image_for_vlm(image_path, max_side=1024)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=10,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VLM_CATEGORY_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}", "detail": "low"}},
                ],
            }
        ],
    )
    answer = (response.choices[0].message.content or "").strip().lower().strip(".`'\" ")
    return answer if answer in CATEGORY_DESCRIPTIONS else "bilinmiyor"


# Birlestirilmis kararin guven degerleri, CNN test bolmesinde (60 gorsel, 2026-10-14) olculen
# dogruluklardan gelir: iki model ayni kategoriyi soylediginde %97.7, ayristiklarinda OpenAI'in
# kategorisi %88 (17'de 15) dogruydu. Kucuk orneklem - kaba bir kalibrasyondur.
AGREEMENT_CONFIDENCE = 0.977
DISAGREEMENT_CONFIDENCE = 0.88


def combine_category_predictions(
    cnn_category: str, cnn_confidence: float, vlm_category: str | None
) -> tuple[str, float, str]:
    """(kategori, guven, kaynak) doner. Kaynak: "cnn+vlm" (hemfikir), "vlm" (ayristilar,
    OpenAI kazanir - testte ayrisan 17 gorselin 15'inde dogruydu) ya da "cnn" (OpenAI yok)."""
    if vlm_category is None or vlm_category == "bilinmiyor":
        return cnn_category, cnn_confidence, "cnn"
    if vlm_category == cnn_category:
        return cnn_category, max(cnn_confidence, AGREEMENT_CONFIDENCE), "cnn+vlm"
    return vlm_category, DISAGREEMENT_CONFIDENCE, "vlm"
