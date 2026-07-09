# Gıda & Sağlık Asistanı

Görsel + Metin Analizi ile Besin Bilgilendirme Sistemi — TÜBİTAK 2209-A Araştırma Projesi.

## Bu Proje Ne Yapar?

Paketli bir gıda ürününün etiket fotoğrafını çeker, sisteme yüklersiniz. Sistem sırasıyla:

1. **Kategori tanır** (süt ürünü / atıştırmalık / içecek / hazır gıda / konserve) — bir CNN
   (MobileNetV3/EfficientNet-B3) ile.
2. **Besin tablosunu okur** (OCR ile) ve enerji/yağ/karbonhidrat/şeker/protein/tuz/sodyum
   değerlerini standart bir forma (100g bazında) normalize eder.
3. **Risk analizi yapar**: WHO/Nutri-Score eşiklerine göre "yüksek şeker", "yüksek tuz" gibi
   bayraklar üretir; içindekiler metninden alerjen (laktoz, gluten, fındık, soya, yumurta,
   balık) tespit eder.
4. **Kaynak referanslı açıklama üretir** (RAG): WHO/EFSA/TGK bilgi tabanından ilgili pasajları
   bulup bir dil modeliyle (LLM) anlaşılır, kaynak gösterilmiş bir sağlık yorumu yazar.
5. **Kişiye özelleştirir** (opsiyonel): kullanıcının kronik hastalıkları (diyabet, hipertansiyon
   vb.), alerjileri ve kalori hedefi girilirse, "diyabet hastaları için önerilmez" gibi kişisel
   uyarılar + bir diyet uyum skoru + alternatif ürün önerisi üretir.

Sonuç, **3 katmanlı** bir çıktıdır: (1) Sağlık Riski, (2) Diyet Uyum Skoru, (3) Alerjen Uyarısı
— hepsi kaynak referanslarıyla birlikte.

**Önemli:** Bu bir araştırma prototipidir, tıbbi tavsiye yerine geçmez. Güncel doğruluk
durumu için aşağıdaki "Bilinen Sınırlamalar" bölümüne mutlaka bakın.

## Mimari

```
Foto -> [On isleme] -> [CNN Siniflandirma] -> kategori baglami
                     -> [OCR + Normalizasyon] -> standart besin JSON
                                              -> [Kural Motoru: risk esikleri + alerjen]
     -> [RAG: retriever (FAISS) + LLM] -> [Kisisel Profil Filtresi]
     -> 3 katmanli cikti: (1) Saglik Riski (2) Diyet Uyum Skoru (3) Alerjen Uyarisi + kaynak referans
```

Sayısal hesaplar (eşik kıyaslama, risk bayrakları) **her zaman kural tabanlı Python'dadır**,
asla LLM'e bırakılmaz — LLM sadece kural motorunun ürettiği bilgiyi doğal dile döker ve
WHO/EFSA/TGK kaynaklarından alıntı yapar (halüsinasyon riskine karşı).

## Klasör Yapısı

```
config/             merkezi konfigurasyon (esikler, model yollari, LLM/embedding saglayicisi)
data/                raw/processed/knowledge_base (buyuk/binary dosyalar git'e girmez)
src/common/          ortak sema (Pydantic) ve birim donusumleri
src/data/            veri toplama + on isleme (Faz 1)
src/vision/          gorsel siniflandirma - MobileNetV3/EfficientNet-B3 (Faz 2)
src/ocr/             OCR + normalizasyon + kural motoru + alerjen tespiti (Faz 3)
src/rag/             chunking + FAISS/BM25 retriever + LLMProvider + generation (Faz 4)
src/health/          kisisel saglik profili + risk/diyet skoru + oneri (Faz 5)
api/                 FastAPI backend - pipeline orkestrasyonu (Faz 6)
demo/                Streamlit web demo (Faz 6)
mobile/              Flutter mobil uygulama (Faz 7 - henuz baslanmadi)
models/              egitilmis agirliklar (git'e girmez, bkz. asagi)
tests/               pytest testleri (235+ test)
docs/                her fazin sonuc/durum notlari (dogruluk rakamlari, bilinen sorunlar)
```

## Kurulum (Sıfırdan, Yeni Bir Bilgisayarda)

### 1. Ön koşullar

- **Python 3.10+**
- **Git**
- Windows'ta OCR için **winget** (genelde hazır gelir)

### 2. Depoyu klonla ve sanal ortam kur

```bash
git clone https://github.com/Arda-dd/gida_saglik_analiz.git
cd gida_saglik_analiz

python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

> `requirements.txt` tüm fazların bağımlılıklarını içerir (torch, easyocr, faiss, fastapi,
> streamlit vb.) — kurulum biraz zaman alabilir (~5-10 dk, torch büyük bir paket).

### 3. Tesseract OCR kur (Faz 3 için gerekli, Windows)

```powershell
winget install --id UB-Mannheim.TesseractOCR
```

Türkçe dil verisi proje-içi bir klasöre (admin yetkisi gerektirmeden) indirilmeli:

```powershell
mkdir data\tessdata -Force
# eng.traineddata ve osd.traineddata'yi Tesseract kurulum dizininden kopyala
copy "C:\Program Files\Tesseract-OCR\tessdata\eng.traineddata" data\tessdata\
copy "C:\Program Files\Tesseract-OCR\tessdata\osd.traineddata" data\tessdata\
# Turkce dil dosyasini resmi depodan indir:
curl -L -o data\tessdata\tur.traineddata https://github.com/tesseract-ocr/tessdata_fast/raw/main/tur.traineddata
```

### 4. `.env` dosyasını oluştur ve ücretsiz API anahtarını al

```bash
cp .env.example .env
```

Sistem LLM (metin üretimi) ve embedding (retrieval) için **HuggingFace Inference API**'yi
varsayılan olarak kullanır — **tamamen ücretsiz, kredi kartı gerektirmez**:

1. [huggingface.co](https://huggingface.co) üzerinde ücretsiz bir hesap oluştur.
2. [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → **Create new
   token** → "Fine-grained" seç → sadece **Inference → "Make calls to Inference Providers"**
   kutusunu işaretle (başka hiçbir izin gerekmez) → token oluştur.
3. `.env` dosyasını aç, şu satırı doldur:
   ```
   HUGGINGFACE_API_KEY=hf_...senin_tokenin...
   ```

(İstersen `config/config.yaml` içindeki `llm.provider` / `rag.embedding_provider` değerlerini
`anthropic` veya `openai` yapıp ilgili ödemeli API anahtarını da girebilirsin — arayüz zaten
her ikisini de destekler.)

### 5. RAG bilgi tabanı index'ini kur

```bash
python -m src.rag.index_builder
```

Bu, `data/knowledge_base/docs/*.md` dokümanlarını embed edip bir FAISS index'i oluşturur
(`data/knowledge_base/index/` — git'e girmez, her klonlamada bir kez çalıştırılmalı).

### 6. Eğitilmiş görsel sınıflandırma modelini temin et

`models/vision_best.pt` (Faz 2'de eğitilen CNN ağırlıkları, ~17MB) **git'e dahil değildir**
(büyük binary dosya). İki seçenek:

- **A) Dosya transferi (hızlı):** Arda'dan `models/vision_best.pt` dosyasını (USB/Drive/
  WeTransfer vb.) al, aynı isimle `models/` klasörüne koy.
- **B) Sıfırdan üret (yavaş ama tam bağımsız):**
  ```bash
  python -m src.data.fetch_off_dataset   # Open Food Facts'ten ~400 urun+gorsel ceker (internet gerekir)
  python -m src.vision.train             # CNN'i egitir, models/vision_best.pt'yi uretir
  ```
  Bu adım internet bağlantısı ve birkaç dakika (CPU) sürer.

### 7. Kurulumu doğrula

```bash
pytest tests/ -v
```

Tümü yeşil olmalı (API anahtarı/model dosyası eksikse bazı testler değil, sadece gerçek
uçtan-uca çalıştırma script'leri —`index_builder`/`evaluate`— etkilenir; pytest'teki 235+ test
tamamı mock'lu olduğundan bağımsız çalışır).

## Çalıştırma

**Web demo (en kolay, tarayıcıdan dene):**

```bash
streamlit run demo/app.py
```

`http://localhost:8501` otomatik açılır. Sol menüden opsiyonel bir sağlık profili gir, bir
etiket fotoğrafı yükle, sonucu gör.

**API (FastAPI, geliştirici/entegrasyon amaçlı):**

```bash
uvicorn api.main:app --reload
```

`http://127.0.0.1:8000/docs` → etkileşimli Swagger arayüzü (`/analyze`, `/profile`,
`/recommend` uç noktalarını dosya yükleyerek deneyebilirsin).

## Fazların Durumu ve Doğruluk Rakamları

Her fazın sonucu, hedeflenen metrikle birlikte `docs/` altında **dürüstçe** belgelenmiştir
(hedefin altında kalan sonuçlar da gizlenmeden raporlanmıştır):

| Faz | Konu | Durum | Detay |
|---|---|---|---|
| 0-1 | İskelet + veri toplama | Tamamlandı (395 OFF kaydı) | `docs/attribution_off.md` |
| 2 | Görsel sınıflandırma | **%75 test accuracy** (hedef ≥%85 — veri hacmi kısıtı, bkz. altta) | `docs/vision_results_notes.md` |
| 3 | OCR + normalizasyon | **%16.2 alan doğruluğu** (hedef ≥%90 — çok sütunlu tablo/görsel kalitesi kısıtı) | `docs/ocr_results_notes.md` |
| 4 | RAG | Recall@5 %100, Factual Consistency %100, Ground Truth Alignment %38 | `docs/rag_results_notes.md` |
| 5 | Kişisel profil | Profile Consistency %100, Recommendation Relevance %100 | `docs/health_results_notes.md` |
| 6 | API + Demo | Gerçek fotoğrafla uçtan uca doğrulandı | `docs/faz6_results_notes.md` |

## Bilinen Sınırlamalar (Mobil Aşamadan Önce Öncelik: Doğruluğu Artırmak)

Gerçek market etiketleriyle canlı testte (2026-07-09) doğrulanan, bilinen ve henüz
**çözülmemiş** iki temel doğruluk sorunu var — mobil uygulamaya geçmeden önce bunların
iyileştirilmesi planlanıyor:

1. **OCR doğruluğu düşük (~%16-52 güven).** Kök nedenler: (a) gerçek market fotoğraflarında
   parlama/yansıma ve eğik açı, (b) çok sütunlu besin tablosu düzeni OCR'da düz metne
   dönüşünce etiket-değer eşleşmesi bozuluyor, (c) küçük/yoğun yazı. Etkisi: bazı besin
   değerleri (özellikle şeker/yağ/karbonhidrat) hiç çıkarılamayabiliyor, bu da risk motorunun
   gerçek bir riski (ör. yüksek şeker) kaçırmasına yol açabiliyor. Bkz. `docs/ocr_results_notes.md`.
2. **Görsel sınıflandırma %75'te sınırlı** (394 eğitim görseli — veri hacmi darboğazı,
   hiperparametre sorunu değil, iki ayrı deneyle doğrulandı). Bkz. `docs/vision_results_notes.md`.

**Planlanan iyileştirme yönleri:** (a) yerel Türkiye marketlerinden gerçek, düz açılı,
parlamasız fotoğraf toplama (`docs/local_data_collection_protocol.md`), (b) EasyOCR'ın
bounding box çıktısını kullanarak satır/sütun yapısını koruyan (layout-aware) bir ayrıştırma
katmanı, (c) daha fazla eğitim görseli ile CNN'i yeniden eğitme.

## LLM ve Embedding Sağlayıcısı

Sistem, RAG modülünde hem LLM'i hem embedding'i **API üzerinden** kullanır — hiçbir model
yerel olarak eğitilmez veya diske indirilmez (bkz. öneri formu 2.4). Varsayılan sağlayıcı
**HuggingFace Inference API** (ücretsiz, kart gerektirmez); Anthropic/OpenAI ödemeli
alternatifler olarak `config/config.yaml -> llm.provider` / `rag.embedding_provider` ve `.env`
üzerinden seçilebilir.

## Geliştirme Ekibi

- Arda Tınmazoğlu
- Semih Erdoğan
- Danışman: Ahmet Metin — Bursa Teknik Üniversitesi
