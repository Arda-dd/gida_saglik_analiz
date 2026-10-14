# Doğruluk ve Güvenilirlik İyileştirmeleri (14 Ekim 2026)

Bu turda backend'e beş iyileştirme yapıldı. Her biri gerçek verilerle ölçüldü. En büyük
kazançlar alerjen tespitinde (yakalama oranı %25 → %100) ve kategori tahmininde
(%73.3 → %95.0) oldu.

## Özet

| Alan | Önce | Sonra | Ölçüm |
|---|---|---|---|
| Alerjen tespiti, "içerir" (OpenAI OCR) | %25 (5/20) | **%100 (20/20)** | 30 içindekiler görseli, Open Food Facts alerjen etiketleri |
| Alerjen tespiti, "eser miktarda içerebilir" | %7 (2/27) | **%100 (27/27)** | aynı set |
| Kategori tahmini | CNN %73.3 | **CNN + OpenAI %95.0** | CNN'in eğitimde görmediği 60 test görseli (ürün ön yüzü) |
| Besin değerleri (OpenAI OCR + ayrıştırıcı) | %50.6 | **%66.8** | 30 besin tablosu görseli (iyimser, bkz. Sınırlamalar) |
| RAG açıklaması (FC / GTA) | %96.5 / %89.2 | **%100 / %100** | 4 senaryo × 3 koşu (kısmen metrik düzeltmesi) |
| Barkod ile ürün sorgusu | yok | Open Food Facts'te kayıtlı ürünlerde besin, kategori ve alerjen veritabanından | doğruluk bu sette ölçülemez (aşağıda) |
| OpenAI OCR uydurma koruması | yok | doğru değerlerde %6 yanlış alarm, yanlış değerlerin 3/7'si yakalandı | 27 görsel, kayıtlı OCR metinleri |
| Sessiz hatalar | RAG haftalarca sessizce kapalı kaldı | `/health` + yanıtta `warnings` | birim testleri |

## Yeni analiz akışı

```
fotoğraf
 ├─ paralel (ağ):  OpenAI OCR  |  OCR.space (çapraz doğrulama)  |  OpenAI kategori görüşü
 └─ yerel:         CNN kategori  |  barkod çözme (OpenCV)
      ↓
 barkod var mı? (görselden ya da OCR metninden) → Open Food Facts sorgusu
      ↓
 besin: veritabanı > etiket (OCR)   kategori: veritabanı > CNN+OpenAI oylaması > CNN
 alerjen: etiket ∪ veritabanı ∪ "eser miktarda içerebilir"
      ↓
 risk motoru → RAG açıklaması → kişisel filtre → yanıt + uyarılar
```

Analiz başına maliyet: 3 OpenAI çağrısı (OCR ~$0.0004, kategori ~$0.0001, RAG ~$0.001).
Toplam 0.2 sentin altında. Süre açıklama dahil 4–7 sn; ağ çağrıları paralel çalıştığı için
çapraz doğrulama ve kategori görüşü süreyi uzatmıyor.

## 1. Sessiz hataların görünür yapılması

**Sorun:** HuggingFace'in ücretsiz kredisi bitince RAG açıklaması üretilemiyordu. Pipeline hatayı
bilinçli olarak yutuyordu, `/health` de her zaman "ok" dönüyordu, yani kimse fark etmedi.

**Çözüm:**
- `/health` artık gerçek kontroller yapıyor: API anahtarları, OCR motor sırası, RAG indeksinin
  embedding modeliyle boyut uyumu ve CNN modeli.
- `/health?deep=true` embedding ve LLM API'lerine gerçek, küçük bir çağrı yapıyor. Kredi bitmesi
  gibi sorunları ancak bu yakalar.
- Opsiyonel katman hataları loglanıyor ve yanıtın `warnings` alanına ekleniyor. Bu kapsamda RAG
  hatası, OCR'ın yedek motora düşmesi, metin ya da besin değeri bulunamaması ve düşük görsel
  kalitesi bildiriliyor. Demo bu uyarıları gösteriyor.

## 2. Barkoddan ürün sorgusu

Barkod iki kaynaktan aranıyor:
- **OpenCV'nin barkod çözücüsü.** Sentetik ve gerçek görsellerle test edildi.
- **OCR metni.** OpenAI ve OCR.space barkodun altındaki rakamları okuyabiliyor. Gürültülü
  metinden sahte kod çıkmasın diye sadece sağlama hanesi tutan 12–13 haneli kodlar kabul
  ediliyor. 8 haneli kodlar kabul edilmiyor, çünkü EasyOCR metninden sağlaması tutan ama yanlış
  iki 8 haneli kod çıktı.

Ürün Open Food Facts'te bulunursa:
- **Besin değerleri** veritabanından alınıyor, eksik alanlar etiketten tamamlanıyor.
- Etiket ile veritabanı belirgin şekilde ayrışırsa **uyarı** veriliyor. Tarif değişmiş ya da
  kayıt hatalı olabilir.
- **Kategori** veritabanından geliyor.
- **Alerjenler** etiket, veritabanı ve "eser miktarda" bilgilerinin birleşimi. Alerji açısından
  güvenli taraf bu.

Sorgu 8 sn zaman aşımlı ve tekrarsız. Hata olursa OCR sonucuyla devam ediliyor.

**Ölçüm sınırı:** Test setinin ground truth'u zaten Open Food Facts'ten geldiği için barkod
yolunun doğruluğu bu sette anlamlı ölçülemez. Ayrıca test görselleri besin tablosu ve
içindekiler kırpıntıları, barkod 60 görselin sadece 2'sinde görünüyor (ikisi de doğru çözüldü),
OCR metinlerinden de 2 barkod çıktı. Gerçek etkiyi Türk market fotoğrafları gösterecek. Türk
ürünlerinin Open Food Facts'teki kapsamı orta düzeyde.

## 3. OpenAI OCR uydurma koruması

OCR karşılaştırma raporunda OpenAI'ın okunamayan etiketlerde besin tablosu uydurduğu görülmüştü.
İki koruma eklendi:
- **Çapraz doğrulama:** OpenAI'ın okuduğu her değer, OCR.space'in ham metninde de geçiyor mu?
  Geçmeyen değerler kullanıcıya "doğrulanamadı" diye bildiriliyor. Değerlerin yarısından fazlası
  doğrulanamazsa daha net bir fotoğraf isteniyor. OCR.space hata verirse bu da bildiriliyor.
- **Görsel kalite kontrolü:** Uzun kenar 600 pikselin altındaysa ya da görsel bulanıksa
  (Laplacian varyansı) uyarı veriliyor.

Kayıtlı metinlerdeki ölçüm (OCR.space'in çalıştığı 27 görsel):
- Doğru değerlerin 141'i doğrulandı, 9'u yanlış alarm aldı (%6).
- Yanlış değerlerin 3'ü yakalandı, 4'ü kaçtı. Yakalananlardan biri Lidl görselindeki uydurma
  "24 kcal" değeri. Kaçan 4 değeri iki motor da aynı okudu; büyük ihtimalle etiket ile ground
  truth farklı.

**Sınır:** Asıl uydurma vakasında (su şişesi) OCR.space de hata vermişti. Orada çapraz doğrulama
yapılamaz, kullanıcıya yalnızca "doğrulanamadı" ve düşük çözünürlük uyarısı gider.

**Yeni bir uydurma örneği (alerjen ölçümü sırasında):** Bir Fas süt ürününün içindekiler
görselinde OpenAI, transkripsiyon yerine kendi yorumunu içeren bir tablo üretti: "Analyse: Élevé
pour un produit laitier nature classique…". Talimat "sadece okuduğunu yaz" demesine rağmen oldu.
Bu, görsel-dil modeli çıktısının neden doğrulanması gerektiğini bir kez daha gösteriyor.

## 4. Alerjen tespiti

**İlk ölçüm çok kötüydü:** OpenAI OCR ile "içerir" alerjenlerinin sadece %25'i yakalanıyordu.
İki neden vardı:
1. Sözlük sadece Türkçe ve İngilizceydi; test etiketleri Fransızca ağırlıklı.
2. Kelime ayırıcı yalnız Türkçe harfleri tanıyordu, "BLÉ" kelimesini "bl", "ŒUF"u "uf" diye
   bölüyordu.

**Yapılanlar:**
- Unicode kelime ayırma ve aksan normalizasyonu.
- Fransızca, Almanca, Hollandaca, İtalyanca ve İspanyolca eş anlamlılar.
- Birleşik kelimeler ve Türkçe ekler ("Weizenmehl", "cevizli").
- Çok kelimeli ifadeler ("fruits à coque").
- Bilinen yanlış pozitiflerin hariç tutulması: hindistan cevizi, kakao yağı, "sans gluten",
  "ferments lactiques".
- 4 harf ve altındaki kelimelerde tam eşleşme, 5 harflilerde bulanık eşleştirme yok. Böylece
  "uova"/"uva", "lait"/"laitue", "boter"/"boite" gibi karışıklıklar önleniyor.

| Motor | "İçerir" yakalama | "Eser miktarda" yakalama | Şüpheli yanlış pozitif |
|---|---|---|---|
| OpenAI OCR | %25 → **%100** (20/20) | %7 → **%100** (27/27) | 2 |
| EasyOCR | %15 → %65 (13/20) | %4 → %48 (13/27) | 1 |

İki "yanlış pozitif" de büyük ihtimalle ground truth eksiği:
- Etikette "aynı tesiste balık işlenir" yazıyor, Open Food Facts'in eser listesinde yok.
- Metinde "Mandelpulver" (badem tozu) geçiyor.

**Ground truth sınırı:** Open Food Facts'te içindekiler metni girilmemiş ürünlerde (özellikle Fas
süt ürünleri) alerjen etiketi de boş. Bu ürünlerde yanlış pozitif sayılmadı.

## 5. Kategori: CNN + OpenAI oylaması

CNN, öneri formundaki taahhüt gereği sistemde kalıyor. Yanına OpenAI'dan ikinci bir görüş
ekleniyor (düşük çözünürlüklü görsel, çağrı başına yaklaşık 85 token).

| 60 test görseli (ürün ön yüzü) | Doğruluk |
|---|---|
| CNN tek başına | %73.3 |
| OpenAI tek başına | %95.0 |
| İkisi hemfikir (43 görsel) | %97.7 |
| Ayrıştıklarında (17 görsel) | OpenAI 15'inde, CNN 2'sinde haklı |
| CNN güven eşiğiyle birleştirme (en iyi eşik) | %91.7 |

**Karar kuralı:**
- Barkod kategorisi varsa o kullanılır.
- Yoksa CNN ve OpenAI hemfikirse karar güçlenir.
- Ayrışırlarsa OpenAI'ın kategorisi kullanılır.
- OpenAI'a ulaşılamazsa CNN tek başına karar verir.

Güven değerleri ölçülen doğruluklardan alındı: hemfikirken 0.977, ayrışmada 0.88. Kategori
kaynağı yanıtta ve demoda gösteriliyor.

**Sınır:** %95 ürünün **ön yüz** fotoğraflarında ölçüldü. Yalnızca besin tablosu görünen bir
kırpıntıda iki model de zorlanıyor; canlı testte biri böyle bir görsele "atıştırmalık", diğeri
"süt ürünü" dedi, doğrusu "hazır gıda"ydı. Mobil uygulama barkod taratmalı ya da ön yüz
fotoğrafı da istemeli.

## Genel sınırlamalar

1. **Aynı set hem geliştirme hem ölçüm için kullanıldı.** Ayrıştırıcı ve alerjen sözlüğü bu 30
   görselin hata örnekleri incelenerek genişletildi. Kurallar genel dil ve format bilgisine göre
   yazıldı ve birim testleriyle sabitlendi, ama rakamlar iyimser. Kategori ölçümü bu sorundan
   muaf: CNN'in test bölmesi kullanıldı ve OpenAI'a hiçbir ayar yapılmadı.
2. **Ground truth Open Food Facts'ten geliyor.** Gönüllü verisi; eksik ve hatalı kayıtlar var.
3. **Türk market ürünü yok.** Bütün rakamların gerçek doğrulaması, yerel market fotoğraflarıyla
   yapılacak bağımsız bir testle olacak (`docs/local_data_collection_protocol.md`).
4. **Örneklemler küçük:** 20 alerjen, 60 kategori, 27 doğrulama görseli. Güven aralıkları geniş.

## Tekrar üretilebilirlik

```bash
python -m src.ocr.evaluate --reparse          # besin değerleri (kayıtlı OCR metinleriyle)
python -m src.ocr.evaluate_allergens --reparse # alerjenler (kayıtlı metinlerle)
python -m src.ocr.evaluate_allergens           # alerjenler (OCR'ı yeniden çalıştırarak)
python -m src.vision.evaluate_category         # kategori: CNN / OpenAI / birleşim
python -m src.rag.compare_providers --runs 3 --model openai:gpt-4.1-mini
```

Raporlar: `docs/ocr_evaluation_report.json`, `docs/allergen_evaluation_report.json`,
`docs/category_ensemble_report.json`, `docs/rag_llm_comparison_report.json`.
