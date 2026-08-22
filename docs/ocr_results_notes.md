# Faz 3 Sonuçları ve Sınırlamalar — OCR ve Metin Normalizasyonu

**Durum:** Pipeline uçtan uca çalışıyor (Tesseract + EasyOCR ile metin çıkarımı → regex tabanlı
besin değeri ayrıştırma → 100g normalizasyonu → risk motoru → alerjen tespiti), tüm modüller
pytest ile test edilmiş (135+ test). Ancak gerçek besin tablosu görsellerinde ölçülen alan bazlı
doğruluk, öneri formunun hedefine (**≥%90**) henüz ulaşmadı. Bu doküman sonuçları ve kök nedeni
şeffaf şekilde raporlar.

## Sonuçlar (2026-07-08, 30 gerçek OFF besin tablosu görseli, ground-truth OFF nutriments ile kıyas)

| Motor      | Alan bazlı doğruluk | Ortalama OCR güveni |
|------------|---------------------|----------------------|
| Tesseract  | %7.5 (20/265 alan)  | %37.0                |
| EasyOCR    | %16.2 (43/265 alan) | %51.0                |

**Seçilen motor:** EasyOCR daha iyi performans gösterdi (hem doğruluk hem güven skorunda).

## Kök Neden Analizi (neden ≥%90 değil?)

Üç ayrı tanı adımı yapıldı, her biri gerçek bir katkı sağladı:

1. **Dil uyumsuzluğu (kısmen çözüldü):** Open Food Facts Fransa kökenli bir platform; kategori
   bazlı arama (ülke filtresi olmadan) büyük ölçüde **Fransızca etiketli** ürünler getirdi
   (örnek gerçek OCR çıktısı: *"VALEURSNUTRITIONMELLES ... Energie. 298 ... Matieres grasses ...
   Glucides"*). `normalize.py`'nin ilk hali sadece Türkçe+İngilizce anahtar kelime içeriyordu.
   Fransızca eş anlamlılar eklenince (`energie`, `matières grasses`/`lipides`,
   `acides gras saturés`, `glucides`, `sucres`, `fibres alimentaires`, `protéines`, `sel`)
   EasyOCR doğruluğu **%10.2 → %16.2**'ye çıktı (bazı ürünlerde 2/9 → 8/9'a sıçrama görüldü).
2. **Görsel kalitesi (küçük katkı):** 30 görselin sadece 2'si aşırı düşük çözünürlüklüydü
   (400×114, 185×400) — bozuk/okunamaz dosya yoktu. Demek ki görsel kalitesi ana darboğaz değil.
3. **Çok sütunlu tablo yapısı (asıl darboğaz):** Gerçek etiketlerin çoğu **"Pour/Per 100g" ve
   "Pour/Per porsiyon" şeklinde çok sütunlu** bir tablo düzenindedir. OCR motorları metni düz bir
   dizi olarak çıkardığında (bizim `extract_text_easyocr`/`extract_text_tesseract` fonksiyonlarımız
   konum bilgisini atıp sadece birleştirilmiş metin döndürüyor), sütun yapısı kayboluyor ve
   değerler etiketlerinden kopuyor. Örnek gerçek çıktı: *"638 kJ 1s84 k 159 kcal 300 kcal ...
   EneRCiE MaTIERES ... Acides ... Sucres ..."* — sayılar bir arada, isimler baska bir yerde.
   Bizim regex tabanlı ayrıştırıcımız "anahtar kelime → yakın sayı" varsayımına dayandığından, bu
   düzende basarisiz oluyor. **Bu, literatürde bilinen bir problemdir** (bkz. öneri formu
   kaynakçası, Romero-Tapiador ve ark. 2025 — vizyon-dil modellerinin porsiyon/veri çeşitliliği
   sınırlamaları).

## Layout-Aware Satır Gruplama Denendi (2026-07-19) — Beklenen İyileşme Gerçekleşmedi

Aşağıdaki 1 numaralı öneri (konum-farkında ayrıştırma) Semih tarafından uygulandı
(`src/ocr/extract.py::_group_boxes_into_rows` — tespitleri y-koordinatına göre satırlara
gruplayıp her satırı x-koordinatına göre soldan sağa sıralıyor, hem Tesseract hem EasyOCR için).
`python -m src.ocr.evaluate` aynı 30 gerçek görsel + ground-truth setiyle **tekrar çalıştırıldı**
(2026-07-19) ve sonuç, hipotezin aksine, **ölçülebilir bir iyileşme göstermedi**:

| Motor      | Önceki (satır gruplama öncesi) | Sonraki (satır gruplama sonrası) |
|------------|-------------------------------|-----------------------------------|
| Tesseract  | %7.5 (20/265 alan)             | %7.9 (21/265 alan)                 |
| EasyOCR    | %16.2 (43/265 alan)            | %15.8 (42/265 alan) — **hafif gerileme** |

Fark, gürültü seviyesinde (±1 alan) — istatistiksel olarak anlamlı bir kazanç yok. Muhtemel
neden: kök neden analizindeki asıl darboğaz (madde 3), etiketlerin **"Per 100g" / "Per porsiyon"
şeklinde yan yana iki sütun** içermesiydi — satır gruplama, bir satırdaki tüm kutucukları
birleştirdiğinde bu iki sütunu da AYNI satıra dahil ediyor (örn. "Energie 638kJ 159kcal 300kcal"),
yani etiket başına iki değer yan yana duruyor ve regex hâlâ hangisinin "100g" hangisinin
"porsiyon" değeri olduğunu ayırt edemiyor — satır gruplama sütun içi karışıklığı çözmüyor, sadece
satırlar arası karışıklığı çözüyor (ki bu örneklemde asıl sorun zaten sütunlar arasıydı). Bu
yüzden **madde 1 kapandı sayılmıyor** — gerçek çözüm için sütun sınırlarını da (x-koordinatı
kümelemesi ile "100g sütunu" / "porsiyon sütunu" ayrımı) tespit eden bir sonraki iterasyon
gerekiyor; bkz. güncel yol haritası aşağıda. Ham veri: `docs/ocr_evaluation_report.json`.

## PaddleOCR / PP-Structure'a Gecis (2026-08-22) — Ilk Olculebilir Kazanc

Satir gruplamasinin cozemedigi sorun (sutunlar arasi karisiklik) icin tablo YAPISINI
taniyan bir motora gecildi: **PaddleOCR 3.7 / PP-StructureV3**. PP-Structure tabloyu
HTML olarak dondurur, yani etiket-deger bagi kaynakta korunur:

```html
<tr><td>Glucides</td><td>35g</td></tr>
<tr><td>dont sucres</td><td>30g</td></tr>
```

### Sonuclar (ayni 30 gercek OFF besin tablosu gorseli, ayni ground-truth)

| Motor | Alan bazli dogruluk | Ortalama OCR guveni |
|---|---|---|
| Tesseract | %7.9 (21/265) | %37.0 |
| EasyOCR (onceki varsayilan) | %15.5 (41/264) | %51.8 |
| **PaddleOCR / PP-Structure** | **%33.3 (88/264)** | **%78.7** |

Olcum, `src/ocr/evaluate.py`'nin kendi `GT_FIELD_MAP` ve `_values_match` fonksiyonlariyla,
ayni `data/raw/openfoodfacts_ocr_samples/manifest.json` uzerinde yapildi. **Ancak
`docs/ocr_evaluation_report.json` henuz yeniden uretilmedi** - icindeki ham veri onceki
(Tesseract/EasyOCR, 2026-07-19) kosusuna aittir. Kanonik raporu tazelemek icin
`python -m src.ocr.evaluate` calistirilmalidir (PP-Structure yavas oldugundan ~45 dakika).

Not: Alan sayisi 265 -> 264'e dustu (OFF'ta bir ground-truth alani degismis); EasyOCR
taban cizgisi bu yeni setle yeniden olculdu (%15.8 -> %15.5), yani kiyas ayni set uzerinde.

**Bu, bu problem alanindaki ilk gercek, olculebilir kazanctir** (dogruluk ~2.15 kat,
guven ~1.5 kat). Yine de formun >=%90 hedefinin cok altindadir.

### Kalan hatanin kok nedeni: artik OCR degil, AYRISTIRICI

Sonuc cift tepeli: 11 gorselde 0 alan, 7 gorselde 6+ alan (1 gorselde 9/9). Sifir cikan
gorseller tek tek incelendi ve **cogunda OCR metni kusursuzdu** - sorun `normalize.py`'de:

1. **Birim-sayi sirasi (asil neden).** Ornek `off_6111242101180` (OCR guveni %96.1, 0/8):
   OCR ciktisi `Valeur energetique Kcal 58`, `Proteines g 3,0`, `Lipides g 3,0` seklinde,
   yani duzen `etiket BIRIM SAYI`. `normalize.py`'nin `_search_value` regex'i ise
   `etiket ... SAYI BIRIM` bekliyor ve hicbirini yakalayamiyor. Bu, PP-Structure'in
   basariyla cikardigi bir tablonun ayristiricida bosa gitmesi demektir.
2. **Degerlendirme setinin veri kalitesi.** Ornek `off_6111035502828` (guven %96.2, 0/9):
   OFF'un `image_nutrition_url`'i bir MADEN SUYU mineral analiz tablosunu gosteriyor
   (`Sodium / Calcium / Bicarbonates / Sulfates / Nitrates`) - goruntude besin tablosu
   hic yok, ama ground-truth 9 alan bekliyor. Bu gorseller ne yapilsa 0 verir.
3. Birkac gorselde OCR gercekten bir sey bulamadi (guven 0.0) - dusuk kaliteli goruntu.

0 alan cikan 11 gorselin ortalama OCR guveni %56.9 iken, 1+ alan cikanlarinki %91.2 -
yani guven skoru basarisiz vakalar icin ise yarar bir gosterge.

### Bedeli: gecikme

PP-Structure varsayilan (server sinifi) modellerle CPU'da **gorsel basina ~75 saniye**
surdu (30 gorsel ~38 dakika). EasyOCR birkac saniyede donuyordu. Form 2.6 `inference
latency`'yi olculen bir metrik sayiyor ve mobil uygulama anlik yanit vaat ediyor -
bu haliyle server modelleri mobil senaryo icin uygun DEGILDIR. Kullanilmayan alt
moduller (formul/muhur/grafik/bolge tespiti) zaten kapatildi; sonraki adim
`PP-OCRv5_mobile_det/rec` varyantlarini olcmektir.

### Ortam: iki gercek hata bulundu ve koda gomuldu

1. **ASCII olmayan model yolu.** Paddle'in C++ inference motoru
   `C:/Users/Semih Erdogan/.paddlex` gibi ASCII disi karakter iceren yollarda dosyayi
   acamiyor ve bos girdi okudugu icin
   `[json.exception.parse_error.101] ... attempting to parse an empty input` veriyor.
   `src/ocr/extract.py::_ensure_paddle_env` yolu Windows 8.3 kisa adina cevirir.
2. **oneDNN + PIR yurutucu uyumsuzlugu.** Layout modelinde
   `NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support` cokmesi;
   MKLDNN varsayilan olarak kapatilir.

Ayrica **`paddlepaddle` Python 3.14 icin wheel yayinlamiyor** (sadece cp39-cp313) -
proje bu yuzden Python 3.10-3.13 gerektirir (bkz. README).

### Forma gore sapma notu (sonuc raporunda belirtilmeli)

Oneri formu 2.1 OCR araci olarak "Tesseract ve EasyOCR" diyor; Risk Yonetimi B-plani
yetersizlik halinde **Google Cloud Vision / Azure** ongoruyor. PaddleOCR formda adi
gecmeyen bir aractir. Gerekce: acik kaynak kalinarak (formun tercih ettigi cizgi)
ticari/odemeli buluta gitmeden B-planinin amaci saglandi ve dogruluk 2 katina cikti.
## Güncel Yol Haritası

0. ~~Tablo yapisini taniyan motora gec (PaddleOCR/PP-Structure)~~ — **uygulandi,
   %15.5 -> %33.3 (2026-08-22)**. Siradaki darbogaz OCR degil `normalize.py`:
   (a) `etiket BIRIM SAYI` duzenini destekle, (b) degerlendirme setinden besin
   tablosu OLMAYAN gorselleri ayikla, (c) mobil model varyantlarini olcup gecikmeyi
   kabul edilebilir seviyeye cek.
1. ~~Konum-farkında (layout-aware) satır gruplama~~ — **uygulandı, ölçülebilir kazanç yok**
   (yukarı bakınız). Bir sonraki iterasyon satır İÇİNDEKİ sütun ayrımını (x-koordinatı bazlı
   kümeleme) hedeflemeli.
2. **Türkçe yerel etiketlerle yeniden değerlendirme:** Bu değerlendirme örneklemi tesadüfen büyük
   ölçüde Fransızca/çok sütunlu global ürünlerden oluştu. Projenin asıl hedefi Türkiye marketlerinden
   toplanan etiketlerdir (Faz 1'in bekleyen insan görevi) — bunlar genellikle tek sütunlu, Türkçe
   etiketlerdir ve muhtemelen çok daha yüksek doğruluk verecektir. Yerel fotoğraflar toplanınca
   `python -m src.ocr.evaluate` benzeri bir değerlendirme Türkçe etiketlerle tekrarlanmalıdır.
3. Doğruluk hâlâ hedefin altında kalırsa, formun **Risk Yönetimi B-Planı** devreye girer: Google
   Cloud Vision veya Azure Cognitive Services OCR gibi ticari servislere geçiş (tablo yapısını daha
   iyi koruyan gelişmiş düzen analizi sunarlar).

## Pipeline'ın Kendisi Hakkında

Kök neden analizinden bağımsız olarak, şu bileşenler **doğru ve test edilmiş** şekilde çalışıyor:
Tesseract + EasyOCR entegrasyonu (güven skoruyla), regex tabanlı besin değeri çıkarımı (Türkçe +
İngilizce + Fransızca eş anlamlılar, virgül/nokta ondalık ayracı, kcal↔kJ, sodyum↔tuz tutarlılığı),
100g/porsiyon bazı tespiti ve normalizasyonu, eşik tabanlı risk motoru (şeker/tuz/doymuş yağ/sodyum),
rapidfuzz tabanlı alerjen tespiti (OCR yazım hatalarına toleranslı). Bu modüller sentetik ve gerçek
veri karışımıyla 135+ pytest testiyle doğrulanmıştır ve Faz 4/6'da doğrudan kullanılabilir.
