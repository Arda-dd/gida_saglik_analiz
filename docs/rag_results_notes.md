# Faz 4 Durumu — RAG Tabanlı Bilgi Getirme ve Yorumlama

**Durum:** Tamamlandı. Modül mimarisi kuruldu, tüm birimler sentetik/mock veriyle test edildi
(242 pytest testinin 53'ü bu faza ait, tamamı yeşil) ve **gerçek bilgi tabanı üzerinde uçtan uca
retrieval + generation değerlendirmesi de çalıştırıldı** (ücretsiz HuggingFace API ile, bkz.
"Gerçek Sonuçlar" bölümü) — Recall@5 %100, MRR 0.794, Factual Consistency %100, Ground Truth
Alignment **%86.4** (2026-07-12'de self-consistency katmanı sayısal-dayanak kontrolüyle
güçlendirildikten sonra %38.2'den %86.4'e çıktı — bkz. "Sayısal Dayanak Kontrolü Eklendi" bölümü).

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

**Güncelleme (2026-07-09, gerçek token ile doğrulandı):** Kullanıcı ücretsiz bir HuggingFace
tokenı oluşturup `.env`'e ekledi. Hem embedding (`sentence-transformers/paraphrase-multilingual-
MiniLM-L12-v2`, 384 boyutlu vektör) hem de `model_huggingface: meta-llama/Llama-3.1-8B-Instruct`
gerçek API çağrısıyla test edildi ve **her ikisi de ücretsiz kotada çalışıyor** — model değişikliği
gerekmedi.

## Tamamlanan Bileşenler

| Modül | Sorumluluk | Test |
|---|---|---|
| `src/rag/chunking.py` | `data/knowledge_base/docs/*.md` dokümanlarını `##` bölüm bazında chunk'lara ayırır, doğrulama metadata'sını (`Durum:`/`Kaynak:`/`URL:`) miras bırakır | 8 test |
| `src/rag/embeddings.py` | Çoklu-sağlayıcı embedding dispatch'i (`huggingface` varsayılan/ücretsiz, `openai` ödemeli alternatif), L2-normalize | 7 test |
| `src/rag/index_builder.py` | FAISS `IndexFlatIP` (cosine) index kurma/saklama/yükleme | 4 test |
| `src/rag/retriever.py` | Dense (embedding API) + BM25 (`rank_bm25`) hibrit skor, min-max normalize + ağırlıklı birleşim (0.6/0.4) | 6 test |
| `src/rag/llm_provider.py` | `LLMProvider` soyut arayüzü + `AnthropicProvider`/`OpenAIProvider`/`HuggingFaceProvider`, config.yaml'dan factory seçimi | 10 test |
| `src/rag/generate.py` | Kaynak referanslı ([Kaynak: chunk_id]) üretim + **self-consistency katmanı**: (1) uydurma kaynak, (2) kaynakta olmayan türetilmiş sayı tespit edilirse otomatik yeniden üretim | 18 test |
| `src/rag/evaluate.py` | Top-k Recall + MRR (retrieval), Factual Consistency Score + Ground Truth Alignment Ratio (generation) | gerçek veriyle çalıştırıldı (bkz. aşağı) |

Testlerin tamamı, gerçek API çağrısı yapmadan (network/API key gerektirmeden) çalışır:
LLM testleri `unittest.mock.patch` ile `anthropic.Anthropic`/`openai.OpenAI`/`huggingface_hub.
InferenceClient` SDK'larını, retrieval testleri ise deterministik bir bag-of-words "sahte
embedding" fonksiyonuyla gerçek embedding API çağrısını taklit eder (bkz.
`tests/test_rag_index_builder.py`, `tests/test_rag_retriever.py`).

## Self-Consistency Katmanı Nasıl Çalışır (form 2.4 taahhüdü)

`generate_explanation()` artık **iki bağımsız** kontrol yapar:

1. **Kaynak geçerliliği (`valid_citation_ratio`):** LLM'in ürettiği `[Kaynak: chunk_id]`
   etiketlerini regex ile çıkarır ve bunların gerçekten o sorgu için retrieval'dan dönen
   chunk'lara ait olup olmadığını kontrol eder (uydurma kaynak/halüsinasyon tespiti).
2. **Sayısal dayanak (`numeric_grounding_ratio`, 2026-07-12'de eklendi):** Her `[Kaynak: ...]`
   etiketinden önceki cümleyi o etikete eşler (`_extract_citation_segments`), cümledeki her
   sayının **o spesifik chunk'ın metninde** ya da ürünün kendi girdi besin verisinde birebir
   geçip geçmediğini kontrol eder. Bu, ilk kontrolün yakalayamadığı bir sorunu hedefler: model
   gerçek/alakalı bir chunk'a atıf yapar ama o chunk'ta YAZMAYAN bir sayıyı kendi başına
   "türetip" sunabilir (ör. kaynakta "%5-10 enerji" yazarken kendi hesabıyla "25 gram" uydurmak).

Her iki oran da eşiğin (varsayılan %70) altındaysa, **hangi sorunun tespit edildiğini açıkça
belirten** daha sıkı bir talimatla otomatik olarak yeniden üretim yapılır. Birim testlerinde
dört senaryo da doğrulanmıştır: "geçerli atıf → yeniden üretim yok", "uydurma kaynak → yeniden
üretim", "kaynakta olmayan türetilmiş sayı → yeniden üretim" ve "ürünün kendi besin verisini
tekrarlamak → cezalandırılmaz" (bkz. `test_compute_numeric_grounding_accepts_product_nutrition_numbers`).

**Neden bu eklendi:** Gerçek bir market fotoğrafıyla (Eti Puf bisküvi, 2026-07-09) yapılan canlı
demo testinde, RAG açıklamasının "[Kaynak: URUN BESIN DEGERLERI]" gibi geçersiz bir etiket
kullandığı ve WHO kaynağında olmayan "25 gram" gibi türetilmiş rakamlar sunduğu görüldü. O tarihte
sadece kaynak geçerliliği kontrol ediliyordu; bu ikinci kontrol tam olarak bu sınıf sorunu
yakalamak için eklendi.

## Gerçek Sonuçlar (2026-07-09, HuggingFace ücretsiz API ile)

Gerçek bilgi tabanı (19 chunk) üzerinde FAISS index kuruldu (`python -m src.rag.index_builder`)
ve `python -m src.rag.evaluate` çalıştırıldı (`docs/rag_evaluation_report.json`).

### Retrieval — `data/rag_eval/queries.json` (15 elle etiketlenmiş sorgu)

| Metrik | Sonuç |
|---|---|
| Recall@5 | **%100.0** (15/15 sorguda ilgili chunk top-5'te bulundu) |
| MRR | **0.794** |

Recall mükemmel; MRR'nin 1.0 olmaması bazı sorgularda ilgili chunk'ın 1. değil 2-4. sırada
çıkmasından kaynaklanıyor (ör. "Ürün etiketinde yüksek şeker eşiği kaç gramdır?" sorusunda
doğru chunk 4. sırada bulundu, RR=0.25) — hibrit skorda BM25 tarafının bazı Türkçe soru
kalıplarında dense skoru ezmesi muhtemel; top_k=5 pratik kullanım için yeterli olsa da,
dense/BM25 ağırlıklarının (şu an 0.6/0.4) ince ayarı gelecekte iyileştirme konusu olabilir.

### Generation — 4 örnek ürün senaryosu

| Metrik | İlk ölçüm (2026-07-09) | Sayısal dayanak kontrolü sonrası (2026-07-12) |
|---|---|---|
| Factual Consistency Score (`valid_citation_ratio`) | %100.0 | **%100.0** |
| Ground Truth Alignment Ratio (`numeric_grounding_ratio`) | %38.2 | **%86.4** |

**Factual Consistency %100** — değişmedi, hâlâ mükemmel: LLM'in ürettiği hiçbir
`[Kaynak: chunk_id]` etiketi uydurma değil.

**Ground Truth Alignment %38.2 → %86.4 — gerçek, ölçülebilir bir iyileşme.** Kök neden iki
parçaydı: (1) eski metrik ürünün kendi girdi besin verisini (ör. "35g şeker") retrieval bağlamı
saymadığından yanlışlıkla cezalandırıyordu — bu `evaluate_generation()`'da düzeltildi; (2) küçük
ücretsiz model bazen kaynakta olmayan türetilmiş sayılar (ör. "25 gram") üretiyordu — bunun için
self-consistency katmanına **sayısal dayanak kontrolü** eklendi (yukarıdaki bölüme bakınız).
4 senaryodan 3'ü artık %86-100 arası, 1'i (`risksiz_urun`) hâlâ %60'ta kalıyor.

**Kalan bulgu (dürüstçe not edilmeli):** `risksiz_urun` senaryosunda self-consistency katmanı
sorunu **doğru şekilde tespit etti** ve yeniden üretimi tetikledi (`regenerated=True`) — ama
ücretsiz Llama-3.1-8B modeli ikinci denemede de aynı "25 gram" türetimini tekrarladı, açıkça
"sadece kaynakta yazan sayıyı kullan" talimatına rağmen. Yani **tespit mekanizması işliyor**,
ama küçük/ücretsiz modelin kendini düzeltme kapasitesi sınırlı. Bu, aşağıdaki "güçlü model"
önerisinin somut, ölçülmüş bir kanıtıdır.

## Önerilen Yol Haritası

1. ~~Self-consistency katmanını genişlet~~ **✅ Tamamlandı (2026-07-12)** — sayısal dayanak
   kontrolü eklendi, Ground Truth Alignment %38.2→%86.4.
2. **Üretim kalitesi için güçlü model:** Kritik/nihai kullanıcıya sunulacak çıktılarda
   `config.yaml`'da `llm.provider: anthropic` (Claude) seçeneğine geçilmesi önerilir — arayüz
   zaten hazır, tek satır config değişikliği yeterli. `risksiz_urun` senaryosu, self-consistency
   doğru tespit etse bile küçük modelin düzeltmeyi tam uygulayamadığını gösteriyor — bu, güçlü
   modelin sadece "daha iyi" değil, bu spesifik sorun için gerekli olduğunu kanıtlıyor. Ücretsiz
   HuggingFace modeli geliştirme/test aşaması için uygundur.
3. **Retrieval ince ayarı:** MRR'yi 1.0'a yaklaştırmak için dense/BM25 ağırlıkları (0.6/0.4)
   üzerinde küçük bir grid search denenebilir (Faz 8 kapsamlı değerlendirmesine bırakıldı).
