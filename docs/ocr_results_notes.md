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

## Önerilen Yol Haritası

1. **Konum-farkında (layout-aware) ayrıştırma:** EasyOCR zaten her tespit için bounding box (x,y
   koordinatları) döndürüyor — şu an bu bilgi `extract_text_easyocr` içinde atılıyor. Bir sonraki
   adım, tespitleri y-koordinatına göre satırlara gruplamak, her satır içinde x-koordinatına göre
   sıralamak (klasik OCR tablo-yeniden-yapılandırma tekniği) ve regex'i satır bazında çalıştırmaktır.
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
