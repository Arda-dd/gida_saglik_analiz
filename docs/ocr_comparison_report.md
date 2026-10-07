# OCR Motorlarının Karşılaştırmalı Değerlendirmesi

**Proje:** Gıda Etiketi Görüntülerinden Sağlık Analizi (TÜBİTAK 2209-A)
**Tarih:** 7 Ekim 2026
**Kapsam:** Besin değeri tablosu fotoğraflarından metin çıkarımı için 5 OCR motorunun aynı test
seti ve aynı ölçütlerle karşılaştırılması (öneri formu 2.1 ve 2.6)

---

## 1. Özet

| Motor | Tür | Alan doğruluğu (30 görsel) | Sayı yakalama | Ortak 27 görselde alan doğruluğu | Süre (medyan) | Maliyet (30 görsel) |
|---|---|---|---|---|---|---|
| Tesseract 5.4 | Yerel, klasik OCR | %7.9 | %14.4 | %7.9 | 0.17 sn | 0 |
| EasyOCR 1.7 | Yerel, derin öğrenme | %16.2 | %36.3 | %13.7 | 0.33 sn | 0 |
| Windows OCR | Yerel, işletim sistemine gömülü | %9.4 | %28.4 | %9.5 | 0.32 sn | 0 |
| OCR.space | Bulut API (ücretsiz katman) | %44.2* | %72.1 | **%48.5** | 2.37 sn | 0 |
| OpenAI gpt-4.1-mini (görsel) | Bulut, görsel-dil modeli | **%50.6**† | **%72.6** | **%48.5** | 1.85 sn | $0.011 |

\* OCR.space 3 görselde üç ayrı denemede de hata verdi (ikisinde HTTP 429, birinde 30 sn zaman
aşımı). Bu görsellerin alanları yanlış sayıldı. Hata vermediği 27 görselde alan doğruluğu
%48.5'tir.
† Bu değerin 6 alanı, modelin okunamayan bir görsel için **uydurduğu** besin tablosundan gelir
(bkz. Bölüm 5). Uydurma hariç: 128/265 = %48.3.

**Ana bulgular:**

1. **Bulut tabanlı iki motor, yerel motorların 3–6 katı doğruluk sağlıyor.** Yerel motorların en
   iyisi EasyOCR (%16.2) iken OCR.space ve OpenAI %44–51 aralığında.
2. **OCR.space ile OpenAI eşit.** İkisinin de işleyebildiği 27 görselde ikisi de tam olarak aynı
   sonucu verdi: 117/241 alan, %48.5. OpenAI'ın 30 görsellik toplamdaki 17 alanlık önünün
   tamamı, OCR.space'in hata verdiği 3 görselden geliyor. Bu 17 alanın 6'sı da uydurma bir
   tablodan.
3. **Görsel-dil modeli (OpenAI) okuyamadığı etikette içerik uydurabiliyor.** Bu, sağlık
   uygulaması için en ciddi risk. Klasik OCR motorları okuyamadığında anlamsız karakterler
   üretir, yani hata görünür olur. OpenAI ise makul görünen ama sahte besin değerleri yazdı.
   Talimat iyileştirmesi bunu azaltıyor ama tamamen ortadan kaldırmıyor.
4. **Darboğaz yalnızca OCR değil.** Her iki bulut motorunda da doğru okunan sayıların yaklaşık
   %20'si, kural tabanlı ayrıştırıcı (regex) etiket formatını tanımadığı için kayboluyor.

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
görsel → OCR motoru → ham metin → ortak regex ayrıştırıcı → besin değerleri → ground truth ile karşılaştırma
```

Bütün motorların çıktısı aynı ayrıştırıcıdan (`src/ocr/normalize.py`) geçer. OpenAI modelinden
yalnızca **metni olduğu gibi yazıya dökmesi** istendi, yapılandırılmış besin değeri çıkarması
istenmedi. Böylece karşılaştırma "metni hangi motor daha iyi okuyor" sorusunu ölçer.

### 2.3 Ölçütler

| Ölçüt | Tanım | Neyi ölçer |
|---|---|---|
| **Alan doğruluğu** | Ayrıştırılan değer ground truth'a yakın mı? (tolerans: ±1.0 ya da ±%15, hangisi büyükse) | OCR ve ayrıştırıcı birlikte, yani uçtan uca başarı |
| **Sayı yakalama** | Sıfır olmayan ground truth değeri OCR'ın ham metninde sayı olarak geçiyor mu? (±0.05) | Yalnızca OCR'ın okuma başarısı, ayrıştırıcıdan bağımsız |
| Süre | Görsel başına duvar saati süresi (medyan) | Hız |
| Maliyet | Ücretli API için token tabanlı tahmin | İşletme maliyeti |

**Hata yönetimi:** Bir motor bir görselde hata verirse o görselin bütün alanları **yanlış**
sayılır. Önceki raporlarda hata alan görseller paydadan düşürülüyordu, bu da OCR.space'in
Ağustos'taki %50.6'lık sonucunu 265 yerine 176 alan üzerinden hesaplatıp şişiriyordu.

OCR.space ilk gün 10 görselde ücretsiz kota sınırına takıldı. Ertesi gün bu görseller yeniden
denendi ve 7'si tamamlandı. Kalan 3 görsel üç ayrı denemede de hata verdiği için kota değil,
motorun bu görsellerdeki gerçek başarısızlığı olarak sayıldı.

### 2.4 Karşılaştırılan motorlar

| Motor | Çalışma yeri | İnternet gerekir mi? | Ücret | Dil ayarı | Not |
|---|---|---|---|---|---|
| Tesseract 5.4 | Yerel (CPU) | Hayır | Ücretsiz, açık kaynak | `tur+eng` | Kelime kutuları satırlara gruplanır |
| EasyOCR 1.7 | Yerel (GPU/CPU) | Hayır | Ücretsiz, açık kaynak | `tr, en` | Kelime kutuları satırlara gruplanır |
| Windows OCR | Yerel (Windows 10/11'e gömülü) | Hayır | Ücretsiz | `tr` | Ek kurulum yok; PowerShell üzerinden çağrılır |
| OCR.space | Bulut | Evet | Ücretsiz katman (kotalı) | `tur`, Engine 3, tablo modu | Tablo satırlarını kendisi düzenler |
| OpenAI gpt-4.1-mini | Bulut | Evet | ~$0.0004/görsel | Talimat ile | Görsel-dil modeli (VLM) |

---

## 3. Genel Sonuçlar

Özet tablo Bölüm 1'dedir. Ek gözlemler:

- **Bütün alanları doğru okunan görsel sayısı:** OpenAI 8/30 (1'i uydurma tablo, bkz. Bölüm 5),
  OCR.space 7/30. Yerel motorların hiçbiri tek bir görselde bile bütün alanları doğru okuyamadı.
- **Hiç doğru alan bulunamayan görsel sayısı:** Tesseract 21, Windows 19, EasyOCR 18, OCR.space 12
  (3'ü hata), OpenAI 9.
- **Motorların birbirini tamamlaması:** Herhangi bir motorun doğru bulduğu alanların birleşimi
  155/265'tir (%58.5). Yalnızca yerel motorlar birleştirilse bile en fazla %20.0'a ulaşılıyor.
  Yerel motorları bir arada kullanmak bulut motorlarının seviyesine yaklaştırmıyor.

---

## 4. Ayrıntılı Sonuçlar

### 4.1 Ortak 27 görselde karşılaştırma

OCR.space'in hata verdiği 3 görsel (1 içecek, 2 konserve) dışarıda bırakıldığında bütün
motorlar aynı 27 görsel (241 alan) üzerinde karşılaştırılabilir:

| Motor | Alan doğruluğu | Sayı yakalama |
|---|---|---|
| Tesseract | %7.9 | %15.8 |
| EasyOCR | %13.7 | %38.8 |
| Windows OCR | %9.5 | %30.1 |
| **OCR.space** | **%48.5** | **%79.2** |
| **OpenAI gpt-4.1-mini** | **%48.5** | %78.1 |

İki bulut motoru bu alt kümede aynı sayıda alanı (117) doğru buldu. Sayı yakalamada OCR.space
1 puan önde. Bu örneklemle iki motor arasında anlamlı bir fark yok.

### 4.2 Besin öğesi bazında (doğru / değerlendirilen)

| Alan | Tesseract | EasyOCR | Windows | OCR.space | OpenAI |
|---|---|---|---|---|---|
| Enerji (kcal) | 8/30 | 11/30 | 5/30 | 17/30 | 19/30 |
| Enerji (kJ) | 9/30 | 11/30 | 6/30 | 18/30 | 20/30 |
| Yağ | 0/30 | 5/30 | 0/30 | 13/30 | 16/30 |
| Doymuş yağ | 0/29 | 3/29 | 0/29 | 12/29 | 12/29 |
| Karbonhidrat | 2/30 | 3/30 | 4/30 | 11/30 | 13/30 |
| Şeker | 0/29 | 4/29 | 2/29 | 11/29 | 12/29 |
| Lif | 0/27 | 1/27 | 2/27 | 10/27 | 9/27 |
| Protein | 1/30 | 3/30 | 0/30 | 12/30 | 16/30 |
| Tuz | 1/30 | 2/30 | 6/30 | 13/30 | 17/30 |

Enerji değerleri bütün motorlarda en kolay okunan alan. Büyük ve belirgin yazıldıkları için bu
beklenen bir sonuç. Yerel motorlar yağ ve doymuş yağ satırlarını neredeyse hiç çıkaramıyor.

### 4.3 Kategori bazında alan doğruluğu

| Kategori | Tesseract | EasyOCR | Windows | OCR.space | OpenAI |
|---|---|---|---|---|---|
| Atıştırmalık | %13 | %20 | %15 | %74 | %74 |
| Hazır gıda | %7 | %11 | %7 | %80 | %85 |
| Konserve | %9 | %39 | %19 | %54 | %63 |
| Süt ürünü | %9 | %9 | %6 | %9 | %15 |
| İçecek | %0 | %0 | %0 | %0 | %12 |

**İçecek ve süt ürünü kategorilerindeki düşük değerler büyük ölçüde test setinden kaynaklanıyor:**

- 6 içecek görselinin 4'ü maden suyu. Etikette besin tablosu değil mineral analizi var (kalsiyum,
  magnezyum, bikarbonat…), ground truth'taki besin değerleri etikette hiç yazmıyor. Birinde
  ground truth hatalı: maden suyu için "4.2 g karbonhidrat, 1.4 g şeker". Kalan iki görselden
  biri uzaktan çekilmiş, okunamaz bir şişe; diğeri besin tablosu içermeyen bir barkod yüzeyi.
- Süt ürünlerinde sorun büyük ölçüde ayrıştırıcıda. Bu kategoride OCR.space sayıların %50'sini,
  OpenAI %61'ini okuyor, ama alan doğruluğu %9–15'te kalıyor. Etiketler birimi sayıdan önce
  yazıyor ("Protéines g 8,0") ve ayrıştırıcı bu formatı tanımıyor. Elle incelenen bir örnekte
  OpenAI etiketi kusursuz okuduğu halde 0 alan çıkarılabildi. Aynı örnekte ground truth'taki 9
  alanın 5'i (kJ, lif, tuz, doymuş yağ, şeker) etikette hiç yazmıyor.

### 4.4 Darboğaz analizi: OCR mı, ayrıştırıcı mı?

Sıfır olmayan 201 ground truth alanı için iki ölçüt birlikte değerlendirildi:

| Motor | Okundu ve ayrıştırıldı | Okundu, ayrıştırıcı kaçırdı | Hiç okunamadı |
|---|---|---|---|
| OpenAI gpt-4.1-mini | %53 | **%19** | %17 |
| OCR.space | %52 | **%20** | %22 |
| EasyOCR | %16 | %20 | **%58** |
| Windows OCR | %11 | %17 | **%70** |
| Tesseract | %8 | %6 | **%83** |

(Satır toplamlarının %100'e tamamlanmayan kısmı, sayı metinde birebir geçmediği halde alanın
tolerans ya da kJ–kcal dönüşümüyle doğru sayıldığı durumlardır.)

- **Yerel motorlarda asıl darboğaz okuma:** Sayıların %58–83'ü hiç okunamıyor. Düşük
  çözünürlüklü görseller bunun başlıca nedeni.
- **Bulut motorlarında asıl darboğaz ayrıştırma:** Okunamayan oran %17–22, ama doğru okunan
  sayıların yaklaşık %20'si ayrıştırıcıda kayboluyor. Kusursuz bir ayrıştırıcıyla sıfır olmayan
  alanlardaki doğruluk, şu anki değerlerden (OpenAI %63.7, OCR.space %57.7) sayı yakalama
  oranlarına (~%72) yaklaşabilir.

### 4.5 Windows OCR hakkında not

Windows OCR'ın Türkçe tanıyıcısı, Fransızca aksanlı harfleri sistematik olarak Türkçe harflere
çeviriyor ("Matières" → "Matiğres", "saturés" → "saturös"). Bu yüzden okuma oranı (%28.4)
EasyOCR'a yakın olsa da ayrıştırıcının Fransızca anahtar kelimeleri bulması zorlaşıyor. Test
setinin çoğu Fransızca olduğu için bu, Windows OCR aleyhine bir etki yaratıyor.

---

## 5. Kritik Bulgu: Görsel-Dil Modelinin İçerik Uydurması

OpenAI modeli üç farklı talimat (prompt) sürümüyle çalıştırıldı. Her sürümde, modelin metni
diğer 4 motorun hiçbirinin görmediği kelimelerle dolu olan görseller tek tek elle incelendi:

| Talimat sürümü | Alan doğruluğu | Sayı yakalama | Uydurma tespit edilen görsel |
|---|---|---|---|
| v1: Türkçe talimat | %52.8 | %73.6 | **3** (10 alan uydurma sayesinde "doğru") |
| v2: İngilizce; "çevirme, tahmin etme, okunmuyorsa `[UNREADABLE]` yaz" | %44.5 | %72.6 | 1 (0 alan) |
| v3: v2 + "üst üste duran tablo satırlarını ayrı yaz" (**kullanılan**) | %50.6 | %72.6 | 2 (en az 6 alan) |

**Örnekler:**

1. **Okunamayan su şişesi** (uzaktan çekilmiş, etiket yazısı seçilemiyor; OCR.space bu görselde
   hata verdi):
   - v1 eksiksiz bir **Türkçe** besin tablosu üretti: "Su, Doğal Kaynak Suyu… Enerji 0 kJ /
     0 kcal… Tuz 0 g". Ürün bir Fas ürünü.
   - v2 doğru şekilde `[UNREADABLE]` döndürdü.
   - v3 bu kez **Fransızca** bir tablo uydurdu: "Valeur nutritionnelle pour 100 ml… 0 g".
   - Ground truth'taki değerler de sıfır olduğu için bu görsel v1 ve v3'te **6/6 tam puan** aldı.
2. **Lidl doğranmış domates konservesi** (çok dilli, küçük yazılı etiket):
   - OCR.space, aynı düşük çözünürlükte gerçek tabloyu okuyup **8/9** aldı.
   - OpenAI v1 Türkçe bir "kurutulmuş domates" etiketi uydurdu ("Domatesler Türkiye'den").
   - v2 ve v3 İsveççe bir içindekiler listesi ve tablo uydurdu: soğan, sarımsak, şeker,
     "Energi 100 kJ/24 kcal". Gerçek ürün yalnızca domates, domates suyu ve sitrik asit
     içeriyor, gerçek enerji değeri 20 kcal.
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
   etiketler test setinde yok. Yerel Türk market ürünleriyle ek test planlanmıştır.
4. **Örneklem büyüklüğü:** 30 görsel, 265 alan. Yerel ve bulut motorları arasındaki büyük fark
   güvenilir. OCR.space ile OpenAI arasındaki fark ise bu örneklemle ayırt edilemez.
5. **OCR.space'in güvenilirliği:** Ücretsiz katman ilk gün 10 görseli kota nedeniyle reddetti;
   ertesi gün 7'si tamamlandı. 3 görselde üç denemede de hata verdi. OpenAI'ın toplam
   sonuçtaki farkı tamamen bu 3 görselden geliyor.
6. **Tolerans:** Alan doğruluğundaki ±1.0 mutlak tolerans küçük değerler için geniştir. Örneğin
   0.8 g tuz ile 1.3 g tuz "eşleşmiş" sayılıyor. Bu, bütün motorları aynı ölçüde etkiler.
7. **Sayı yakalama iyimserdir:** Küçük sayılar (1, 2, 3) metnin başka bir yerinde tesadüfen
   geçebilir. Bu ölçüt okuma başarısının bir üst sınırı olarak yorumlanmalıdır.
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
| Canlı demo (mevcut durum) | OCR.space, hata olursa EasyOCR | Ücretsiz, uydurma riski yok, ortak alt kümede OpenAI ile eşit doğruluk; 30 görselin 3'ünde hata verdiği için yedek motor şart |
| Kota/hata olmadan en yüksek okuma | OpenAI gpt-4.1-mini **+ çapraz doğrulama** | Hiç hata vermedi ve en yüksek sayı yakalamayı verdi (%72.6), ama uydurma riski nedeniyle tek başına kullanılmamalı |

**Sonraki adımlar:**

1. Ayrıştırıcıyı (normalize) "birim sayıdan önce" ve birleşik hücre formatları için genişletmek.
   İki bulut motorunda da doğru okunup kaybedilen ~%20'lik kısmı hedefler.
2. Test setini Türk market ürünlerinin yüksek çözünürlüklü fotoğraflarıyla genişletmek.
3. Görsel-dil modeli için çapraz doğrulama katmanı tasarlamak.

---

## 8. Tekrar Üretilebilirlik

```bash
python -m src.ocr.evaluate                                    # 5 motorun tam ölçümü
python -m src.ocr.evaluate --engines openai_vision --merge    # tek motoru yeniden ölç
python -m src.ocr.evaluate --engines ocrspace --retry-failed  # hata veren görselleri yeniden dene
```

- Ham sonuçlar, her görsel için OCR metni dahil: `docs/ocr_evaluation_report.json`
- OpenAI talimat v1 çıktıları (uydurma örnekleri): `docs/ocr_openai_vision_prompt_v1.json`
- Kod: `src/ocr/extract.py` (motorlar), `src/ocr/windows_ocr.ps1`, `src/ocr/evaluate.py` (ölçüm)
