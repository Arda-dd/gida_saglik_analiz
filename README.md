# Gıda & Sağlık Asistanı

Görsel + Metin Analizi ile Besin Bilgilendirme Sistemi — TÜBİTAK 2209-A Araştırma Projesi.

Paketli gıda etiketlerini fotoğraf üzerinden analiz eden; görsel tanıma (CNN), OCR, RAG
(WHO/EFSA/TGK kaynaklı) ve kişisel sağlık profili katmanlarını birleştirerek açıklanabilir,
kaynak referanslı sağlık değerlendirmeleri üreten yapay zekâ destekli sistem.

Detaylı faz faz uygulama planı için proje sahibinin plan dosyasına bakınız
(`bu-bizim-tubitak-projemiz-*.md`).

## Mimari (özet)

```
Foto -> [On isleme] -> [CNN Siniflandirma] -> kategori baglami
                     -> [OCR + Normalizasyon] -> standart besin JSON
                                              -> [Kural Motoru: risk esikleri + alerjen]
     -> [RAG: retriever (FAISS) + LLM] -> [Kisisel Profil Filtresi]
     -> 3 katmanli cikti: (1) Saglik Riski (2) Diyet Uyum Skoru (3) Alerjen Uyarisi + kaynak referans
```

## Klasör Yapısı

```
config/             merkezi konfigurasyon (esikler, model yollari, LLM provider)
data/               raw/processed/images/knowledge_base (git'e girmez, .gitkeep ile takip edilir)
src/common/         ortak sema (Pydantic) ve birim donusumleri
src/data/           veri toplama + on isleme (Faz 1)
src/vision/         gorsel siniflandirma - MobileNetV3/EfficientNet-B3 (Faz 2)
src/ocr/            OCR + normalizasyon + kural motoru + alerjen tespiti (Faz 3)
src/rag/            retriever + LLMProvider + generation (Faz 4)
src/health/         kisisel saglik profili + risk/diyet skoru (Faz 5)
api/                FastAPI backend (Faz 6)
demo/               Streamlit web demo (Faz 6)
mobile/             Flutter mobil uygulama (Faz 7)
models/             egitilmis agirliklar (git'e girmez)
tests/              pytest testleri
docs/               mimari sema, metrik raporlari, bildiri taslagi
```

## Kurulum

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env   # ve .env icine API anahtarlarini girin
```

> Not: `requirements.txt` tum fazlarin bagimliliklarini icerir (torch, easyocr, faiss vb.).
> Gelistirme sirasinda sadece ulasilan faz icin gerekli paketleri kurmak yeterlidir; ilk
> kurulumda sadece `pydantic`, `pydantic-settings`, `pyyaml`, `python-dotenv`, `pytest`
> yuklenmistir.

## Testleri Çalıştırma

```bash
pytest tests/ -v
```

## LLM ve Embedding Sağlayıcısı

Sistem, RAG modülünde hem LLM'i hem embedding'i **API üzerinden** kullanır — hiçbir model
yerel olarak eğitilmez veya diske indirilmez (bkz. öneri formu 2.4). Varsayılan sağlayıcı
**HuggingFace Inference API** (ücretsiz, kart gerektirmez); Anthropic/OpenAI ödemeli
alternatifler olarak `config/config.yaml -> llm.provider` / `rag.embedding_provider` ve `.env`
üzerinden seçilebilir.

## API ve Web Demo Çalıştırma (Faz 6)

RAG index'i ilk kullanımdan önce bir kez kurulmalı (knowledge_base chunk'larını embed eder):

```bash
python -m src.rag.index_builder
```

FastAPI backend:

```bash
uvicorn api.main:app --reload
# http://127.0.0.1:8000/docs -> etkilesimli API dokumantasyonu
```

Streamlit web demo (api katmanini dogrudan cagirir, ayri sunucu gerekmez):

```bash
streamlit run demo/app.py
```

## Geliştirme Ekibi

- Arda Tınmazoğlu
- Semih Erdoğan
- Danışman: Ahmet Metin — Bursa Teknik Üniversitesi
