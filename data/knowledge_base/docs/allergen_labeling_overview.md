# Alerjen Beyanı — Genel Çerçeve

**Durum:** TASLAK — DOĞRULANMALI. TGK'nin alerjen tebliğinin birebir metni bu oturumda
çekilmemiştir (bkz. `tgk_etiketleme_ozet.md` — aynı erişim kısıtı geçerlidir).

**TODO:** Türk Gıda Kodeksi ile uyumlu güncel "alerjen ve intoleransa neden olan maddeler" listesi
resmi kaynaktan doğrulanmalı (AB Gıda Bilgilendirme Yönetmeliği 1169/2011 Ek II ile büyük ölçüde
örtüştüğü bilinmektedir, ancak Türkiye'deki güncel tebliğ metniyle teyit edilmelidir).

## Bu Projede Kullanılan Alerjen Listesi (config.yaml)

`laktoz`, `gluten`, `findik`, `soya`, `yumurta`, `balik` — bu 6 kategori, projenin `Allergen` enum'ında
(`src/common/schema.py`) tanımlıdır. AB/TGK'nin tam 14 majör alerjen listesinden (süt/laktoz, gluten
içeren tahıllar, kabuklu deniz ürünleri, yumurta, balık, yer fıstığı, soya, süt, sert kabuklu
yemişler [fındık dahil], kereviz, hardal, susam, kükürt dioksit/sülfitler, lupin, yumuşakçalar)
**alt küme** olarak seçilmiştir — proje kapsamı gereği en sık karşılaşılan ve Türkiye'deki paketli
gıdalarda en yaygın görülen alerjenlere odaklanılmıştır.

## Bu Projede Kullanımı

Faz 3'te (`src/ocr/allergens.py`) OCR ile çıkarılan içindekiler listesinden bu alerjenlerin
sinonim/fuzzy eşleşmesi (rapidfuzz) yapılacaktır. Liste genişletilmek istenirse, hem
`config.yaml -> allergens` hem de `src/common/schema.py -> Allergen` enum'ı güncellenmelidir.
