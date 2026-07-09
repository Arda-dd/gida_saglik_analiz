# Faz 4 Durumu — RAG Tabanlı Bilgi Getirme ve Yorumlama

**Durum:** Tamamlandı. Modül mimarisi kuruldu, tüm birimler sentetik/mock veriyle test edildi
(182 pytest testinin 45'i bu faza ait, tamamı yeşil) ve **gerçek bilgi tabanı üzerinde uçtan uca
retrieval + generation değerlendirmesi de çalıştırıldı** (ücretsiz HuggingFace API ile, bkz.
"Gerçek Sonuçlar" bölümü) — Recall@5 %100, MRR 0.794, Factual Consistency %100, Ground Truth
Alignment %38.2 (düşük skorun kök nedeni ve önerilen çözüm aşağıda dürüstçe belgelenmiştir).

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
| `src/rag/generate.py` | Kaynak referanslı ([Kaynak: chunk_id]) üretim + **self-consistency katmanı**: halüsinasyon (uydurma kaynak) tespit edilirse otomatik yeniden üretim | 10 test |
| `src/rag/evaluate.py` | Top-k Recall + MRR (retrieval), Factual Consistency Score + Ground Truth Alignment Ratio (generation) | gerçek veriyle çalıştırıldı (bkz. aşağı) |

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

| Metrik | Sonuç |
|---|---|
| Factual Consistency Score (`valid_citation_ratio`) | **%100.0** |
| Ground Truth Alignment Ratio | **%38.2** |

**Factual Consistency %100** — self-consistency katmanı beklendiği gibi çalıştı: LLM'in
ürettiği hiçbir `[Kaynak: chunk_id]` etiketi uydurma değildi, hepsi gerçekten retrieval'dan
dönen chunk'lara aitti (4 senaryonun 1'inde ilk denemede geçersiz bir referans üretildi, sistem
otomatik olarak yeniden üretim yaptı ve ikinci denemede düzeldi).

**Ground Truth Alignment %38.2 — düşük, ve kök nedeni tek bir şey değil, iki ayrı bulgu var:**

1. **Metrik tanımı ilk baştaki haliyle eksikti (düzeltildi):** İlk çalıştırmada bu oran %48.5
   çıkmıştı ve üretilen metni okuyunca görüldü ki LLM, ürünün kendi girdi besin değerlerini
   (ör. "35g şeker", "450 kcal" — bunlar prompt'ta doğrudan verilen gerçek ürün verisi, retrieval
   sonucu değil) doğru şekilde tekrarlıyordu; ama `ground_truth_alignment_ratio` fonksiyonu
   sadece retrieval bağlamına karşı kontrol ediyordu, prompt'a verilen ürün verisine karşı değil.
   `evaluate_generation()` düzeltildi (artık ürünün `NutritionFacts` verisi de "grounding"
   kaynağına dahil ediliyor) — bu, metriği daha doğru hale getirdi ama tek başına sorunu çözmedi.
2. **Gerçek, düzeltilmemiş bulgu:** Ücretsiz 8B model (Llama-3.1-8B-Instruct), sistem promptunda
   açıkça yasaklanmasına rağmen ("Hiçbir sayısal eşik veya oran UYDURMA") **kendi başına türetilmiş
   yüzde/oran hesapları** üretiyor — ör. "doymuş yağın %280'ine denk gelmektedir", "günlük şeker
   tüketimini 25 gram'a indirmeyi önerir" (kaynak metinde sadece "%5-10 enerji" yazıyor, "25 gram"
   ifadesi hiçbir kaynakta geçmiyor — model bunu kendi hesaplamış, referans aldığı 2000 kcal'lik
   varsayımı da belirtmeden). Bu değerler **uydurma kaynak değil** (`valid_citation_ratio` bunları
   yakalayamaz çünkü atıf edilen chunk_id gerçekten var ve alakalı) ama **kaynakta birebir
   bulunmayan, modelin kendi türettiği sayısal ifadeler** — self-consistency katmanımızın şu anki
   tasarımı (sadece chunk_id geçerliliğini kontrol eder) bu inceliği yakalamıyor.

**Sonuç olarak:** Retrieval katmanı üretim-kalitesinde (Recall %100, MRR 0.794). Kaynak
referanslama da güvenilir (halüsinasyon/uydurma kaynak oranı %0). Ancak **ücretsiz 8B modelin
kendi başına sayısal türetim yapma eğilimi**, projenin "sayısal hesaplar LLM'de değil kural
motorunda yapılır" ilkesini generation metninde tam olarak sağlamıyor — bu, ücretsiz/küçük model
seçiminin somut bir bedeli olarak dürüstçe not edilmelidir.

## Önerilen Yol Haritası

1. **Self-consistency katmanını genişlet:** Şu anki kontrol sadece `[Kaynak: chunk_id]`
   etiketinin var olan bir chunk'a ait olup olmadığını doğruluyor. Bir sonraki adım, atıf yapılan
   her cümledeki sayısal değerlerin (regex ile) o **spesifik chunk'ın metninde birebir** geçip
   geçmediğini de kontrol etmek olmalı — bu, "gerçek ama alakasız kaynağa yanlış sayı iliştirme"
   durumunu yakalar.
2. **Üretim kalitesi için güçlü model:** Kritik/nihai kullanıcıya sunulacak çıktılarda
   `config.yaml`'da `llm.provider: anthropic` (Claude) seçeneğine geçilmesi önerilir — arayüz
   zaten hazır, tek satır config değişikliği yeterli. Ücretsiz HuggingFace modeli geliştirme/test
   aşaması için uygundur.
3. **Retrieval ince ayarı:** MRR'yi 1.0'a yaklaştırmak için dense/BM25 ağırlıkları (0.6/0.4)
   üzerinde küçük bir grid search denenebilir (Faz 8 kapsamlı değerlendirmesine bırakıldı).
