# Beslenme Terimleri Sözlüğü (TR/EN) — İç Referans

**Durum:** İÇ REFERANS (dış kaynak iddiası içermez, düşük doğrulama riski). Bu doküman RAG
retriever'ının Türkçe/İngilizce terim eşleştirmesinde yardımcı bağlam olarak kullanılması için
hazırlanmıştır.

| Türkçe            | İngilizce           | Birim         |
|-------------------|----------------------|---------------|
| Enerji            | Energy               | kcal / kJ     |
| Yağ               | Fat                  | g / 100g      |
| Doymuş yağ        | Saturated fat        | g / 100g      |
| Karbonhidrat      | Carbohydrate         | g / 100g      |
| Şeker             | Sugar(s)             | g / 100g      |
| Lif / Diyet lifi  | Fiber / Dietary fiber| g / 100g      |
| Protein           | Protein              | g / 100g      |
| Tuz               | Salt                 | g / 100g      |
| Sodyum            | Sodium               | mg / 100g     |
| Porsiyon          | Serving / Portion    | g veya mL     |
| Alerjen           | Allergen             | -             |
| Laktoz            | Lactose              | -             |
| Gluten            | Gluten               | -             |

## Dönüşüm Notları

- `1 kcal = 4.184 kJ`
- `Tuz (g) = Sodyum (g) × 2.5` ⟺ `Sodyum (mg) = Tuz (g) / 2.5 × 1000`

(Bu iki formül `src/common/units.py` içinde zaten kod olarak uygulanmış ve test edilmiştir —
bkz. `tests/test_units.py`.)
