# Faz 4 Durumu — RAG Tabanlı Bilgi Getirme ve Yorumlama

**Durum:** Modül mimarisi tamamlandı, tüm birimler sentetik/mock veriyle test edildi
(182 pytest testinin 45'i bu faza ait, tamamı yeşil). Ancak **gerçek bilgi tabanı üzerinde
uçtan uca retrieval + generation değerlendirmesi henüz çalıştırılmadı** — bu, API anahtarı
gerektirir ve kullanıcı kararıyla (2026-07-08) bir sonraki adıma bırakıldı.

## Neden API tabanlı embedding'e geçildi (mimari değişiklik #1)

İlk uygulama `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` modelini yerel
olarak indirip kullanıyordu (plan dosyasındaki orijinal tasarım). Kurulum sırasında
bilgisayarın C: sürücüsünün **%100 dolu** olduğu (477 GB'ın sadece ~250 MB'ı boş) ve model
indirmenin buna katkı sağladığı tespit edildi. Kullanıcı açık talimatla ("her şey api tabanlı
olsun yerel llm model falan olmaz yer kaplamamalı") embedding'in de LLM gibi API üzerinden
alınmasını istedi. Bunun üzerine:

- Yarım kalan/indirilmiş model dosyası (`~/.cache/huggingface`) silindi.
- `sentence-transformers`, `transformers`, `tokenizers` paketleri kaldırıldı (`huggingface-hub`
  ve `safetensors` sadece `timm`'in bağımlılığı olduğu için, model indirmeden, geri kuruldu).
- `src/rag/embeddings.py` eklendi: OpenAI `text-embedding-3-small` API'siyle embedding üretir,
  hiçbir model dosyası diske inmez.
- `src/rag/index_builder.py` ve `src/rag/retriever.py` bu yeni modülü kullanacak şekilde
  güncellendi; `config/config.yaml` içindeki `rag.embedding_model` değeri güncellendi.
- Bu, projenin LLM için zaten benimsediği "API tabanlı, yerel kaynağı zorlama" ilkesiyle
  tam tutarlıdır (bkz. plan dosyası: "6 GB ekran kartını yerel LLM ile zorlamamak").

## Ücretsiz HuggingFace Inference API desteği eklendi (mimari değişiklik #2)

Kullanıcı (2026-07-09) hem LLM hem embedding için Anthropic/OpenAI'nin ödemeli olduğunu belirtip
kart gerektirmeyen, tamamen ücretsiz bir alternatif istedi. Bunun üzerine:

- `src/rag/llm_provider.py`'ye `HuggingFaceProvider` eklendi (`huggingface_hub.InferenceClient.
  chat_completion`, OpenAI ile uyumlu bir arayüz sunuyor).
- `src/rag/embeddings.py`, tek bir OpenAI-özel fonksiyon olmaktan çıkıp `embed_texts(texts,
  provider=...)` şeklinde çoklu-sağlayıcı bir dispatch'e dönüştürüldü (`openai` | `huggingface`).
  HuggingFace tarafı, aynı çokdilli `paraphrase-multilingual-MiniLM-L12-v2` modelini yerel
  indirmek yerine HF'nin sunucusunda çalıştırıyor (feature-extraction endpoint + gerekirse
  mean-pooling) — orijinal disk sorununu da dolaylı olarak çözüyor.
- `config/config.yaml`: `llm.provider` ve `rag.embedding_provider` varsayılanı `huggingface`
  yapıldı; `anthropic`/`openai` ödemeli alternatifler olarak korundu (kod değişikliği gerekmeden
  config'ten seçilebilir).
- `.env.example`'a `HUGGINGFACE_API_KEY` eklendi.

**Önemli sınırlama (dürüstçe not edilmeli):** HuggingFace'in ücretsiz serverless Inference
API kotası zaman zaman sıkılaştırılıyor ve hangi modellerin bu kotada aktif olduğu değişebilir;
şu an `config.yaml`'daki `model_huggingface: meta-llama/Llama-3.1-8B-Instruct` değeri gerçek bir
API çağrısıyla doğrulanmadı (kullanıcı henüz `.env`'e token girmedi). İlk gerçek çalıştırmada bu
modelin ücretsiz kotada erişilebilir olmadığı görülürse, `config.yaml`'daki bu değerin başka bir
aktif modelle değiştirilmesi gerekebilir.

## Tamamlanan Bileşenler

| Modül | Sorumluluk | Test |
|---|---|---|
| `src/rag/chunking.py` | `data/knowledge_base/docs/*.md` dokümanlarını `##` bölüm bazında chunk'lara ayırır, doğrulama metadata'sını (`Durum:`/`Kaynak:`/`URL:`) miras bırakır | 8 test |
| `src/rag/embeddings.py` | Çoklu-sağlayıcı embedding dispatch'i (`huggingface` varsayılan/ücretsiz, `openai` ödemeli alternatif), L2-normalize | 7 test |
| `src/rag/index_builder.py` | FAISS `IndexFlatIP` (cosine) index kurma/saklama/yükleme | 4 test |
| `src/rag/retriever.py` | Dense (embedding API) + BM25 (`rank_bm25`) hibrit skor, min-max normalize + ağırlıklı birleşim (0.6/0.4) | 6 test |
| `src/rag/llm_provider.py` | `LLMProvider` soyut arayüzü + `AnthropicProvider`/`OpenAIProvider`/`HuggingFaceProvider`, config.yaml'dan factory seçimi | 10 test |
| `src/rag/generate.py` | Kaynak referanslı ([Kaynak: chunk_id]) üretim + **self-consistency katmanı**: halüsinasyon (uydurma kaynak) tespit edilirse otomatik yeniden üretim | 10 test |
| `src/rag/evaluate.py` | Top-k Recall + MRR (retrieval), Factual Consistency Score + Ground Truth Alignment Ratio (generation) | henüz gerçek veriyle çalıştırılmadı (bkz. aşağı) |

Testlerin tamamı, gerçek API çağrısı yapmadan (network/API key gerektirmeden) çalışır:
LLM testleri `unittest.mock.patch` ile `anthropic.Anthropic`/`openai.OpenAI`/`huggingface_hub.
InferenceClient` SDK'larını, retrieval testleri ise deterministik bir bag-of-words "sahte
embedding" fonksiyonuyla gerçek embedding API çağrısını taklit eder (bkz.
`tests/test_rag_index_builder.py`, `tests/test_rag_retriever.py`).

## Self-Consistency Katmanı Nasıl Çalışır (form 2.4 taahhüdü)

`generate_explanation()`, LLM'in ürettiği `[Kaynak: chunk_id]` etiketlerini regex ile çıkarır
ve bunların gerçekten o sorgu için retrieval'dan dönen chunk'lara ait olup olmadığını kontrol
eder (`valid_citation_ratio`). Oran eşiğin (varsayılan %70) altındaysa, geçerli chunk_id
listesini açıkça hatırlatan daha sıkı bir talimatla **otomatik olarak yeniden üretim** yapılır.
Bu mekanizma, LLM'in kaynaklarda olmayan bir referans "uydurmasını" (halüsinasyon) doğrudan
ölçülebilir ve test edilebilir hale getirir — birim testlerinde hem "geçerli atıf → yeniden
üretim yok" hem de "uydurma atıf → yeniden üretim tetiklenir" senaryoları doğrulanmıştır.

## Eksik Kalan: Gerçek Veriyle Uçtan Uca Değerlendirme

`data/rag_eval/queries.json` içinde bilgi tabanının gerçek 19 chunk'ına karşılık gelen 15 elle
etiketlenmiş (sorgu, ilgili chunk_id) çifti hazırlandı; `src/rag/evaluate.py` bunları
kullanarak Top-k Recall + MRR (retrieval) ve Factual Consistency Score + Ground Truth
Alignment Ratio (generation, 4 örnek ürün senaryosuyla) hesaplayacak şekilde yazıldı. Ancak:

- Retrieval değerlendirmesi bile artık bir **embedding API çağrısı** gerektiriyor (varsayılan:
  HuggingFace, ücretsiz) — kullanıcının `.env` dosyasına `HUGGINGFACE_API_KEY` girmesi gerekiyor.
- Generation değerlendirmesi de aynı token ile `HuggingFaceProvider` üzerinden çalışabilir
  (config.yaml'da `llm.provider: huggingface` zaten varsayılan).

Kullanıcı (2026-07-08) API anahtarı kurulumunu şimdilik ertelemeyi tercih etti ("şu an için
atla, sadece kod+testlerle devam et"). Bu nedenle bu doküman **gerçek Top-k Recall/MRR/Factual
Consistency/Ground Truth Alignment rakamları içermiyor** — Faz 2/3'te olduğu gibi rakamları
olmadan "başarılı" göstermek yerine durumu şeffaf bırakmayı tercih ediyoruz.

## Sıradaki Somut Adım

1. Kullanıcı [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)'dan
   ücretsiz bir token alıp `.env.example`'ı `.env` olarak kopyalayıp `HUGGINGFACE_API_KEY`
   değerini girdiğinde: `python -m src.rag.index_builder` (gerçek FAISS index'i kurar) ve
   ardından `python -m src.rag.evaluate` (gerçek Top-k Recall/MRR + Factual Consistency/Ground
   Truth Alignment rakamlarını üretir, `docs/rag_evaluation_report.json`'a kaydeder) çalıştırılmalı.
2. `config.yaml`'daki `model_huggingface` değerinin ücretsiz kotada gerçekten erişilebilir olup
   olmadığı bu ilk çalıştırmada doğrulanmalı; erişilemezse başka bir aktif modelle değiştirilmeli.
3. Sonuçlar bu dosyaya eklenip hedeflerle (form: kaynak referanslı çıktı, ölçülmüş halüsinasyon
   oranı) kıyaslanmalı.
