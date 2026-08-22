# Faz 3 Sonuçları ve Sınırlamalar — OCR ve Metin Normalizasyonu

**Durum:** Pipeline uçtan uca çalışıyor (Tesseract + EasyOCR ile metin çıkarımı → regex tabanlı
besin değeri ayrıştırma → 100g normalizasyonu → risk motoru → alerjen tespiti), tüm modüller
pytest ile test edilmiş (135+ test). Canlı pipeline hâlâ EasyOCR kullanıyor (%16 civarı alan
doğruluğu, hedef ≥%90'ın altında) ama **2026-08-15'te eklenen OCR.space bulut API'si kısmi bir
örneklemde %50.6 alan doğruluğu gösterdi** (bkz. aşağıdaki ilgili bölüm) - henüz tam
doğrulanmadığı ve dağıtım riski taşıdığı için üretime alınmadı, sadece karşılaştırma amaçlı
entegre edildi. Bu doküman sonuçları ve kök nedeni şeffaf şekilde raporlar.

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

## OCR.space Bulut API'si Eklendi (2026-08-15) — Büyük, Kısmen Doğrulanmış Kazanç

Formun **Risk Yönetimi B-Planı** (ticari OCR servisine geçiş) erken denendi: `src/ocr/extract.py`'ye
`extract_text_ocrspace()` eklendi (ücretsiz katman, `OCR_SPACE_API_KEY`). `isTable=True` +
`OCREngine=3` parametreleriyle satırları TAB (Engine 3'te Markdown tablo) ile ayırarak
döndürüyor — bizim `_group_boxes_into_rows`'un çözemediği **sütun içi** karışıklığı (yukarıdaki
bölüm) doğrudan API'nin kendisi çözüyor.

`python -m src.ocr.evaluate` aynı 30 görsel + ground-truth setiyle çalıştırıldı:

| Motor      | Alan bazlı doğruluk | Ortalama "güven" |
|------------|---------------------|----------------------|
| Tesseract  | %7.9 (21/265 alan)  | %37.0                |
| EasyOCR    | %15.8 (42/265 alan) | %51.8                |
| **OCR.space** | **%50.6 (89/176 alan)** | %66.7 (bkz. not) |

**Sonuç dramatik şekilde daha iyi** — ama iki önemli çekince ile:

1. **Örneklem eksik:** Ücretsiz katmanın istek limiti, değerlendirme sırasında 30 görselin
   10'unda `429 Too Many Requests` hatasına yol açtı (istekler arasına 3sn bekleme eklense de
   değişmedi — bu bir hız degil, muhtemelen gunluk/saatlik bir kota siniri). Yani OCR.space
   rakamı 265 değil **176 alan** üzerinden — tam 30 görsellik bir kıyas için kota sıfırlandığında
   (`python -m src.ocr.evaluate`) tekrar çalıştırılıp bu tabloya eklenmelidir.
2. **"Güven" gerçek bir güven skoru değil:** OCR.space'in ücretsiz API'si Tesseract/EasyOCR'in
   aksine kelime bazlı bir güven skoru döndürmüyor - `mean_confidence` burada sadece istegin
   basarili olup olmadigini (100.0/0.0) yansitan bir vekil, gercek bir OCR kalite olcusu degil.
3. **Yan bulgu (fixlendi):** OCR.space, aksanlı Latin harflerini (é, è vb.) sistematik olarak
   "�" (Unicode replacement karakteri) ile değiştiriyor - bu, Fransızca anahtar kelimelerimizin
   (`matières`, `saturées`, `protéines`) regex'ini kırıyordu. Karakter sınıflarına "�"
   toleransı eklendi (`src/ocr/normalize.py`) ve ayrıca "acides gras saturées" yerine kısa
   "dont saturées" formunun da tanınması sağlandı (gerçek OCR.space çıktısında görüldü).

**Dağıtım riski:** Serbest katmanın hız/kota sınırı, canlı demo pipeline'ında (kullanıcı yükledigi
her fotograf icin anlık bir istek) güvenilir bir birincil motor olarak kullanmayı riskli kılıyor -
şu an sadece **karşılaştırma/değerlendirme amaçlı** entegre edildi, `api/pipeline.py`'deki
canlı akış hâlâ EasyOCR kullanıyor. Üretime almadan önce ya ücretli bir plana geçilmeli ya da
EasyOCR'a otomatik geri düşen (fallback) bir deneme mekanizması eklenmelidir.

## Güncel Yol Haritası

1. ~~Konum-farkında (layout-aware) satır gruplama~~ — **uygulandı, ölçülebilir kazanç yok**
   (yukarı bakınız).
2. **OCR.space'i tam 30 görsellik ornekte tekrar olcmek** (kota sifirlandiginda) ve gercekten
   %50+ dogrulaniyorsa, hiz/kota sinirina karsi bir fallback stratejisiyle (EasyOCR'a otomatik
   dusme) canli pipeline'a (`api/pipeline.py`) tasimak.
3. **Türkçe yerel etiketlerle yeniden değerlendirme:** Bu değerlendirme örneklemi tesadüfen büyük
   ölçüde Fransızca/çok sütunlu global ürünlerden oluştu. Projenin asıl hedefi Türkiye marketlerinden
   toplanan etiketlerdir (Faz 1'in bekleyen insan görevi) — bunlar genellikle tek sütunlu, Türkçe
   etiketlerdir ve muhtemelen çok daha yüksek doğruluk verecektir. Yerel fotoğraflar toplanınca
   `python -m src.ocr.evaluate` benzeri bir değerlendirme Türkçe etiketlerle tekrarlanmalıdır.
4. Doğruluk hâlâ hedefin altında kalırsa, formun **Risk Yönetimi B-Planı**'nın diğer adayları
   (Google Cloud Vision, Azure Document Intelligence) da denenebilir.

## Pipeline'ın Kendisi Hakkında

Kök neden analizinden bağımsız olarak, şu bileşenler **doğru ve test edilmiş** şekilde çalışıyor:
Tesseract + EasyOCR entegrasyonu (güven skoruyla), regex tabanlı besin değeri çıkarımı (Türkçe +
İngilizce + Fransızca eş anlamlılar, virgül/nokta ondalık ayracı, kcal↔kJ, sodyum↔tuz tutarlılığı),
100g/porsiyon bazı tespiti ve normalizasyonu, eşik tabanlı risk motoru (şeker/tuz/doymuş yağ/sodyum),
rapidfuzz tabanlı alerjen tespiti (OCR yazım hatalarına toleranslı). Bu modüller sentetik ve gerçek
veri karışımıyla 135+ pytest testiyle doğrulanmıştır ve Faz 4/6'da doğrudan kullanılabilir.
