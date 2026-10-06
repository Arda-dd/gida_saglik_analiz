# OCR Motorlarının Karşılaştırmalı Değerlendirmesi

**Proje:** Gıda Etiketi Görüntülerinden Sağlık Analizi (TÜBİTAK 2209-A)
**Tarih:** 6 Ekim 2026
**Kapsam:** Besin değeri tablosu fotoğraflarından metin çıkarımı için 5 OCR motorunun aynı test
seti ve aynı ölçütlerle karşılaştırılması (öneri formu 2.1 ve 2.6)

---

## 1. Özet

| Motor | Tür | Alan doğruluğu (30 görsel) | Sayı yakalama | Ortak 20 görselde alan doğruluğu | Süre (medyan) | Maliyet (30 görsel) |
|---|---|---|---|---|---|---|
| Tesseract 5.4 | Yerel, klasik OCR | %7.9 | %14.4 | %7.8 | 0.17 sn | 0 |
| EasyOCR 1.7 | Yerel, derin öğrenme | %16.2 | %36.3 | %11.7 | 0.33 sn | 0 |
| Windows OCR | Yerel, işletim sistemine gömülü | %9.4 | %28.4 | %8.4 | 0.32 sn | 0 |
| OCR.space | Bulut API (ücretsiz katman) | %38.9* | %58.7* | **%57.5** | 2.35 sn | 0 |
| OpenAI gpt-4.1-mini (görsel) | Bulut, görsel-dil modeli | **%50.6**† | **%72.6** | %55.9 | 1.85 sn | $0.011 |

\* OCR.space 30 görselin 10'unda ücretsiz kota sınırına (HTTP 429) takıldı; bu görsellerin
alanları yanlış sayıldı. Kotaya takılmadığı 20 görselde alan doğruluğu %57.5'tir (bkz. 4.1).
† Bu değerin 6 alanı, modelin okunamayan bir görsel için **uydurduğu** besin tablosundan gelir
(bkz. Bölüm 5). Uydurma hariç: 128/265 = %48.3.

**Ana bulgular:**

1. **Bulut tabanlı iki motor, yerel motorların 3–6 katı doğruluk sağlıyor.** Yerel motorların en
   iyisi EasyOCR (%16.2) iken OCR.space ve OpenAI %50 civarında.
2. **OCR.space ile OpenAI pratikte başa baş.** İkisinin de başarıyla işlediği 20 görselde
   OCR.space biraz önde (%57.5'e karşı %55.9). OpenAI'ın 30 görsellik toplamdaki üstünlüğü, büyük
   ölçüde OCR.space'in kotaya takıldığı görsellerden geliyor.
3. **Görsel-dil modeli (OpenAI) okuyamadığı etikette içerik uydurabiliyor.** Bu, sağlık
   uygulaması için en ciddi risk. Klasik OCR motorları okuyamadığında anlamsız karakterler
   üretir, yani hata görünür olur. OpenAI ise makul görünen ama sahte besin değerleri yazdı.
   Prompt iyileştirmesi bunu azaltıyor ama tamamen ortadan kaldırmıyor.
4. **Darboğaz yalnızca OCR değil.** OpenAI'ın doğru okuduğu sayıların yaklaşık %20'si, kural
   tabanlı ayrıştırma (regex) katmanı etiket formatını tanımadığı için kaybediliyor.

---

## 2. Test Seti ve Yöntem

### 2.1 Test seti

- Open Food Facts (OFF) veritabanından 30 gerçek ürün fotoğrafı: 5 kategori × 6 ürün
  (atıştırmalık, hazır gıda, içecek, konserve, süt ürünü).
- Her ürün için OFF'un `nutriments` alanı **ground truth** olarak kullanıldı. Bu alan 9 besin
  değeri içerir: enerji (kcal, kJ), yağ, doymuş yağ, karbonhidrat, şeker, lif, protein ve tuz.
  Toplam 265 değerlendirme alanı vardır, bunların 64'ünün değeri 0'dır.
- Ürünlerin kökeni (barkod önekine göre): 12 Fas, 14 Fransa, 2 İsviçre, 2 diğer. Etiketler
  çoğunlukla Fransızca, bir kısmı Arapça-Fransızca çift dilli. Türk market ürünü yok (bkz.
  Sınırlamalar).
- **Görseller düşük çözünürlüklü:** En uzun kenar 288–400 piksel, dosya boyutu 12–47 KB.

### 2.2 Değerlendirme hattı

Her motor için akış aynıdır:

```
görsel → OCR motoru → ham metin → ortak regex ayrıştırıcı (src/ocr/normalize.py) → besin değerleri → ground truth ile karşılaştırma
```

Bütün motorların çıktısı aynı ayrıştırıcıdan geçer. OpenAI modelinden yalnızca **metni olduğu
gibi yazıya dökmesi** istendi, yapılandırılmış besin değeri çıkarması istenmedi. Böylece
karşılaştırma "metni hangi motor daha iyi okuyor" sorusunu ölçer.

### 2.3 Ölçütler

| Ölçüt | Tanım | Neyi ölçer |
|---|---|---|
| **Alan doğruluğu** | Ayrıştırılan değer ground truth'a yakın mı? (tolerans: max(±1.0, ±%15)) | OCR + ayrıştırıcı birlikte, yani uçtan uca başarı |
| **Sayı yakalama** | Sıfır olmayan ground truth değeri OCR'ın ham metninde sayı olarak geçiyor mu? (±0.05) | Yalnızca OCR'ın okuma başarısı, ayrıştırıcıdan bağımsız |
| Süre | Görsel başına duvar saati süresi (medyan) | Hız |
| Maliyet | Ücretli API'ler için token tabanlı tahmin | İşletme maliyeti |

**Hata yönetimi:** Bir motor bir görselde hata verirse (kota, ağ hatası) o görselin bütün
alanları **yanlış** sayılır. Önceki raporlarda hata alan görseller paydadan düşürülüyordu, bu da
OCR.space'in Ağustos'taki %50.6'lık sonucunu 265 yerine 176 alan üzerinden hesaplatıp
şişiriyordu.

### 2.4 Karşılaştırılan motorlar

| Motor | Çalışma yeri | İnternet gerekir mi? | Ücret | Dil ayarı | Not |
|---|---|---|---|---|---|
| Tesseract 5.4 | Yerel (CPU) | Hayır | Ücretsiz, açık kaynak | `tur+eng` | Kelime kutuları satırlara gruplanır |
| EasyOCR 1.7 | Yerel (GPU/CPU) | Hayır | Ücretsiz, açık kaynak | `tr, en` | Kelime kutuları satırlara gruplanır |
| Windows OCR | Yerel (Windows 10/11'e gömülü) | Hayır | Ücretsiz | `tr` | Ek kurulum yok; PowerShell üzerinden çağrılır |
| OCR.space | Bulut | Evet | Ücretsiz katman (kotalı) | `tur`, Engine 3, tablo modu | Tablo satırlarını kendisi düzenler |
| OpenAI gpt-4.1-mini | Bulut | Evet | ~$0.0004/görsel | Prompt ile | Görsel-dil modeli (VLM) |

---

## 3. Genel Sonuçlar

Özet tablo Bölüm 1'dedir. Ek gözlemler:

- **Tam doğru okunan görsel sayısı** (bütün alanları doğru): OpenAI 8/30 (1'i uydurma tablo, bkz.
  Bölüm 5), OCR.space 6/30. Yerel motorların hiçbiri tek bir görselde bile bütün alanları
  doğru okuyamadı.
- **Hiç doğru alan bulunamayan görsel sayısı:** Tesseract 21, Windows 19, EasyOCR 18, OCR.space 15
  (10'u kota), OpenAI 9.
- **Motorların birbirini tamamlaması:** Herhangi bir motorun doğru bulduğu alanların birleşimi
  155/265'tir (%58.5). Yalnızca yerel motorlar birleştirilse bile en fazla %20.0'a ulaşılıyor.
  Yerel motorları bir arada kullanmak bulut motorlarının seviyesine yaklaştırmıyor.

---

## 4. Ayrıntılı Sonuçlar

### 4.1 OCR.space ile adil karşılaştırma (ortak 20 görsel)

OCR.space'in kotaya takıldığı 10 görsel (6 süt ürününün tamamı, 3 konserve, 1 içecek) dışarıda
bırakıldığında bütün motorlar aynı 20 görsel (179 alan) üzerinde karşılaştırılabilir:

| Motor | Alan doğruluğu | Sayı yakalama |
|---|---|---|
| Tesseract | %7.8 | %17.4 |
| EasyOCR | %11.7 | %37.7 |
| Windows OCR | %8.4 | %27.5 |
| **OCR.space** | **%57.5** | **%85.5** |
| OpenAI gpt-4.1-mini | %55.9 | %81.2 |

Aynı gün yapılan ilk tam koşuda OCR.space yalnızca 2 görselde kotaya takılmış ve başarılı 28
görselde %50.4 alan doğruluğu vermişti. Kotanın sıfırlanmasından sonra eksik 10 görsel
`python -m src.ocr.evaluate --engines ocrspace --retry-failed` ile tamamlanacaktır.

### 4.2 Besin öğesi bazında (doğru / değerlendirilen)

| Alan | Tesseract | EasyOCR | Windows | OCR.space* | OpenAI |
|---|---|---|---|---|---|
| Enerji (kcal) | 8/30 | 11/30 | 5/30 | 14/30 | 19/30 |
| Enerji (kJ) | 9/30 | 11/30 | 6/30 | 15/30 | 20/30 |
| Yağ | 0/30 | 5/30 | 0/30 | 11/30 | 16/30 |
| Doymuş yağ | 0/29 | 3/29 | 0/29 | 11/29 | 12/29 |
| Karbonhidrat | 2/30 | 3/30 | 4/30 | 10/30 | 13/30 |
| Şeker | 0/29 | 4/29 | 2/29 | 10/29 | 12/29 |
| Lif | 0/27 | 1/27 | 2/27 | 9/27 | 9/27 |
| Protein | 1/30 | 3/30 | 0/30 | 11/30 | 16/30 |
| Tuz | 1/30 | 2/30 | 6/30 | 12/30 | 17/30 |

Enerji değerleri bütün motorlarda en kolay okunan alan. Büyük ve belirgin yazıldıkları için
beklenen bir sonuç. Yerel motorlar "yağ / doymuş yağ" alt satırlarını neredeyse hiç
çıkaramıyor.

### 4.3 Kategori bazında alan doğruluğu

| Kategori | Tesseract | EasyOCR | Windows | OCR.space* | OpenAI |
|---|---|---|---|---|---|
| Atıştırmalık | %13 | %20 | %15 | %74 | %74 |
| Hazır gıda | %7 | %11 | %7 | %80 | %85 |
| İçecek | %0 | %0 | %0 | %0 | %12 |
| Konserve | %9 | %39 | %19 | %37 | %63 |
| Süt ürünü | %9 | %9 | %6 | %0 (6/6 kota) | %15 |

**İçecek ve süt ürünü kategorilerindeki düşük değerler büyük ölçüde test setinden kaynaklanıyor:**

- 6 içecek görselinin 4'ü maden suyu. Etikette besin tablosu değil mineral analizi var (kalsiyum,
  magnezyum, bikarbonat…), ground truth'taki besin değerleri etikette hiç yazmıyor. Birinde
  ground truth hatalı: maden suyu için "4.2 g karbonhidrat, 1.4 g şeker". Diğer iki görselden
  biri uzaktan çekilmiş, okunamaz bir şişe; öteki besin tablosu içermeyen bir barkod yüzeyi.
- Süt ürünlerinde elle incelenen örnekte ground truth'taki 9 alanın 5'i (kJ, lif, tuz, doymuş
  yağ, şeker) etikette yazmıyor. Ayrıca etiket formatı birimi sayıdan önce yazıyor ("Protéines
  g 8,0"), ayrıştırıcı bu formatı tanımıyor. Bu örnekte OpenAI etiketi kusursuz okuduğu halde 0
  alan çıkarılabildi (bkz. 4.4).

### 4.4 Darboğaz analizi: OCR mı, ayrıştırıcı mı?

Sıfır olmayan 201 ground truth alanı için iki ölçüt birlikte değerlendirildi:

| Motor | Okundu ve ayrıştırıldı | Okundu, ayrıştırıcı kaçırdı | Hiç okunamadı |
|---|---|---|---|
| Tesseract | %8 | %6 | **%83** |
| EasyOCR | %16 | %20 | **%58** |
| Windows OCR | %11 | %17 | **%70** |
| OCR.space* | %45 | %13 | %36 |
| OpenAI | %53 | **%19** | %17 |

(Satır toplamlarının %100'e tamamlanmayan kısmı, sayı metinde birebir geçmediği halde alanın
tolerans ya da kJ↔kcal dönüşümüyle doğru sayıldığı durumlardır.)

- **Yerel motorlarda asıl darboğaz okuma:** Sayıların %58–83'ü hiç okunamıyor. Düşük
  çözünürlüklü görseller bunun başlıca nedeni.
- **OpenAI'da asıl darboğaz ayrıştırma:** Okunamayan oran yalnızca %17, ama doğru okunan sayıların
  %19'u ayrıştırıcıda kayboluyor. Kusursuz bir ayrıştırıcıyla OpenAI'ın sıfır olmayan alanlardaki
  doğruluğu kabaca sayı yakalama oranına (%72.6) yaklaşabilir. Şu anki değer %63.7'dir (128/201).

### 4.5 Windows OCR hakkında not

Windows OCR'ın Türkçe tanıyıcısı, Fransızca aksanlı harfleri sistematik olarak Türkçe harflere
çeviriyor ("Matières" → "Matiğres", "saturés" → "saturös"). Bu yüzden okuma oranı (%28.4)
EasyOCR'a yakın olsa da ayrıştırıcının Fransızca anahtar kelimeleri bulması zorlaşıyor. Test
setinin çoğu Fransızca olduğu için bu, Windows OCR aleyhine bir etki yaratıyor.

---

## 5. Kritik Bulgu: Görsel-Dil Modelinin İçerik Uydurması

OpenAI modeli üç farklı talimat (prompt) sürümüyle çalıştırıldı. Her sürümde, modelin metni
diğer 4 motorun hiçbirinin görmediği kelimelerle dolu olan görseller tek tek elle incelendi:

| Prompt sürümü | Alan doğruluğu | Sayı yakalama | Uydurma tespit edilen görsel |
|---|---|---|---|
| v1: Türkçe talimat | %52.8 | %73.6 | **3** (10 alan uydurma sayesinde "doğru") |
| v2: İngilizce; "çevirme, tahmin etme, okunmuyorsa `[UNREADABLE]` yaz" | %44.5 | %72.6 | 1 (0 alan) |
| v3: v2 + "üst üste duran tablo satırlarını ayrı yaz" (**kullanılan**) | %50.6 | %72.6 | 2 (≥6 alan) |

**Örnekler:**

1. **Okunamayan su şişesi** (uzaktan çekilmiş, etiket yazısı seçilemiyor):
   - v1, eksiksiz bir **Türkçe** besin tablosu üretti: "Su, Doğal Kaynak Suyu… Enerji 0 kJ / 0 kcal…
     Tuz 0 g". Fas ürünü olduğu halde.
   - v2 doğru şekilde `[UNREADABLE]` döndürdü.
   - v3 bu kez **Fransızca** bir tablo uydurdu: "Valeur nutritionnelle pour 100 ml… 0 g".
   - Ground truth'taki değerler de sıfır olduğu için bu görsel v1 ve v3'te **6/6 tam puan** aldı.
2. **Lidl doğranmış domates konservesi** (çok dilli, küçük yazılı etiket):
   - OCR.space, aynı düşük çözünürlükte gerçek tabloyu okuyup **8/9** aldı.
   - OpenAI v1 Türkçe bir "kurutulmuş domates" etiketi uydurdu ("Domatesler Türkiye'den").
   - v2 ve v3 İsveççe bir içindekiler listesi ve tablo uydurdu: soğan, sarımsak, şeker,
     "Energi 100 kJ/24 kcal". Gerçek ürün yalnızca domates, domates suyu ve sitrik asit
     içeriyor, gerçek değer 20 kcal.
3. **Barkod yüzeyi** (besin tablosu yok): v1 baştan sona bir Türkçe süt ürünü etiketi uydurdu
   ("İçindekiler: Süt, Süt Tozu, Maya, Tuz… Enerji 120 kcal"). v2 ve v3 yalnızca barkodu yazdı.

**Değerlendirme:** Uydurma, talimatla azaltılabiliyor ama **yerini ve dilini değiştirerek**
devam ediyor. Sıcaklık 0 olduğu halde talimattaki küçük bir değişiklik farklı bir görselde
uydurmaya yol açtı. Klasik OCR motorlarında böyle bir risk yok: okuyamadıkları metin için
"iranander WMsniien falee 44" gibi anlamsız ve kolayca fark edilen çıktılar üretiyorlar.

**Proje için sonuç:** Bir görsel-dil modeli kullanılacaksa, çıktısı tek başına güvenilir kabul
edilmemeli. Örneğin şu önlemler alınabilir:
- Modelin okuduğu sayıları uydurma riski olmayan bir motorla (OCR.space/EasyOCR) çapraz doğrulamak.
- Düşük çözünürlüklü görsellerde kullanıcıdan daha yakın bir fotoğraf istemek.

Bu bulgu, öneri formundaki "halüsinasyon riskini kural tabanlı katmanla sınırlama" ilkesini
OCR katmanı için de geçerli kılıyor.

---

## 6. Sınırlamalar

1. **Görsel çözünürlüğü:** Görseller OFF'un küçültülmüş (≤400 piksel) sürümleri. Gerçek telefon
   fotoğrafları çok daha yüksek çözünürlüklüdür, yerel motorlar orada belirgin şekilde daha iyi
   sonuç verebilir. Sonuçlar bu koşula özgüdür.
2. **Ground truth kalitesi:** OFF verisi gönüllüler tarafından girilir. Bazı alanlar etikette
   yazmıyor (64/265 alan 0 değerli), bazıları hatalı (maden suyu örneği). Bu yüzden hiçbir motor
   %100'e ulaşamaz. Ulaşılabilir üst sınır bilinmiyor; etikette görünen alanların elle
   işaretlenmesi gerekir.
3. **Dil:** Etiketler ağırlıkla Fransızca ve Arapça-Fransızca. Sistemin hedefi olan Türkçe
   etiketler test setinde yok. Yerel Türk market ürünleriyle ek test planlanmıştır (bkz.
   `docs/local_data_collection_protocol.md`).
4. **Örneklem büyüklüğü:** 30 görsel, 265 alan. Motorlar arasındaki büyük farklar (yerel ve bulut)
   güvenilir. OCR.space ile OpenAI arasındaki 1–2 puanlık fark ise bu örneklemle istatistiksel
   olarak ayırt edilemez.
5. **OCR.space kotası:** Ücretsiz katman, aynı gün yapılan ikinci tam koşuda 10 görseli reddetti.
   Sonuçlar eksik görseller tamamlanınca güncellenecektir.
6. **Tolerans:** Alan doğruluğundaki ±1.0 mutlak tolerans küçük değerler için geniştir. Örneğin
   0.8 g tuz ile 1.3 g tuz "eşleşmiş" sayılıyor. Bu, bütün motorları aynı ölçüde etkiler.
7. **Sayı yakalama iyimserdir:** Küçük sayılar (1, 2, 3) metnin başka bir yerinde tesadüfen
   geçebilir. Bu metrik okuma başarısının bir üst sınırı olarak yorumlanmalıdır.
8. **Süre ölçümü:** Windows OCR süresine her çağrıda yeni bir PowerShell süreci başlatma maliyeti
   dahildir, yani motorun kendisi ölçülenden daha hızlıdır. EasyOCR GPU üzerinde çalıştı (NVIDIA
   RTX 4050 Laptop, CUDA 12.1); CPU'da belirgin şekilde daha yavaş olur. Bulut motorlarının
   süresi ağ gecikmesine bağlıdır.
9. **Tekrarlanabilirlik:** EasyOCR (GPU) ve OpenAI aynı ayarlarla bile koşudan koşuya ±1–3 alan
   farklı sonuç verebiliyor. EasyOCR Ağustos ölçümünde %15.8, bu ölçümde %16.2 verdi.

---

## 7. Sonuç ve Öneriler

| Kullanım senaryosu | Önerilen motor | Gerekçe |
|---|---|---|
| İnternetsiz / tamamen ücretsiz | EasyOCR | Yerel motorların en iyisi; Windows OCR daha hızlı ama Fransızca/Türkçe aksan sorunu var |
| Canlı demo (mevcut durum) | OCR.space → kota/hata olursa EasyOCR | Ücretsiz, uydurma riski yok, ortak alt kümede en yüksek doğruluk |
| En yüksek okuma oranı | OpenAI gpt-4.1-mini **+ çapraz doğrulama** | En yüksek sayı yakalama (%72.6) ve kotaya takılmıyor, ama uydurma riski nedeniyle tek başına kullanılmamalı |

**Sonraki adımlar:**

1. Ayrıştırıcıyı (normalize) "birim sayıdan önce" ve birleşik hücre formatları için genişletmek.
   OpenAI'ın doğru okuyup kaybedilen %19'luk kısmını hedefler.
2. OCR.space'in eksik 10 görselini tamamlamak.
3. Test setini, Türk market ürünlerinin yüksek çözünürlüklü fotoğraflarıyla genişletmek.
4. Görsel-dil modeli için çapraz doğrulama katmanı tasarlamak.

---

## 8. Tekrar Üretilebilirlik

```bash
python -m src.ocr.evaluate                                 # 5 motorun tam ölçümü
python -m src.ocr.evaluate --engines openai_vision --merge  # tek motoru yeniden ölç
python -m src.ocr.evaluate --engines ocrspace --retry-failed  # kotaya takılan görselleri tamamla
```

- Ham sonuçlar, her görsel için OCR metni dahil: `docs/ocr_evaluation_report.json`
- OpenAI prompt v1 çıktıları (uydurma örnekleri): `docs/ocr_openai_vision_prompt_v1.json`
- Kod: `src/ocr/extract.py` (motorlar), `src/ocr/windows_ocr.ps1`, `src/ocr/evaluate.py` (ölçüm)
