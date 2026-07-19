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

## Güncel Yol Haritası

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
