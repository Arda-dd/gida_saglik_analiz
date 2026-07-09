# Faz 6 Sonuçları — Backend API + Web Demo

**Durum:** Tamamlandı ve **gerçek bir uçtan uca akışla doğrulandı** — bu, projenin ilk kez
fiilen "denenebilir" hale geldiği aşama (foto yükle → 3 katmanlı sonuç).

## Mimari

- `api/pipeline.py` — tüm fazları (2-5) tek bir `analyze_label_image()` fonksiyonunda birleştiren
  orkestrasyon katmanı: ön işleme (Faz 1, resize yok) → CNN kategori tahmini (Faz 2) → OCR +
  besin normalizasyonu (Faz 3) → risk motoru + alerjen tespiti (Faz 3) → RAG açıklaması (Faz 4,
  **opsiyonel**) → kişisel profil filtresi + alternatif öneri (Faz 5, **opsiyonel**, sadece
  profil verilirse). `api/main.py` (FastAPI) ve `demo/app.py` (Streamlit) bu tek fonksiyonu
  ORTAK kullanır — iş mantığı tek yerde yaşar, HTTP/UI katmanları ince birer sarmalayıcıdır.
- `api/main.py` — `POST /profile`, `POST /analyze` (multipart dosya + opsiyonel `profile_id`),
  `GET /recommend`, `GET /health`. Global exception handler (form Risk Yönetimi B-planı):
  beklenmeyen hatalar ham stack trace yerine yapılandırılmış `500` JSON'ı olarak döner;
  `ValueError` (ör. okunamayan görsel) `422`'ye, bilinmeyen profil/ürün `404`'e çevrilir.
- `demo/app.py` — Streamlit: görsel yükle, opsiyonel profil (kronik durum/alerjen/kalori
  hedefi) gir, 3 katmanlı sonucu + RAG kaynak referanslarını (dogrulanmış/taslak etiketiyle)
  gör.

**Tasarım kararı — RAG ve kişisel filtre opsiyoneldir:** RAG bir API çağrısı gerektirdiğinden
(rate limit, ağ hatası, anahtar eksikliği mümkün), başarısız olursa `explanation=None` ile
sessizce devam edilir — **tüm istek çökmez**, kural tabanlı (Faz 3) sonuçlar yine de döner.
Bu, `tests/test_api_pipeline.py::test_rag_failure_gracefully_degrades_instead_of_crashing` ile
doğrulanmıştır.

## Test Kapsamı (28 yeni test, toplam 235)

| Dosya | İçerik |
|---|---|
| `tests/test_api_pipeline.py` | Aday ürün havuzu CSV ayrıştırma, orkestrasyon mantığı (profil dallanması, RAG hata toleransı) — mock'lu, gerçek model/görsel gerektirmez — 12 test |
| `tests/test_api_main.py` | FastAPI `TestClient` ile uç nokta sözleşmeleri (400/404/422 hata kodları, başarılı serileştirme) — 16 test |

## Gerçek Uçtan Uca Doğrulama (2026-07-09)

Gerçek bir OFF besin tablosu görseliyle (`atistirmalik_3046920022651.jpg`) hem doğrudan
pipeline hem de canlı `uvicorn` sunucusu üzerinden test edildi:

```
Kategori: atistirmalik (güven %69.6)
OCR güven: %83.6
Besin (100g): enerji 566 kcal, yağ 41g, karbonhidrat 35g, şeker 30g, protein 9.5g, tuz 0.02g
Risk bayrakları: yuksek_seker
Diyabet + fındık alerjisi profiliyle: "Diyabet hastaları için önerilmez" uyarısı doğru tetiklendi,
diyet uyum skoru 68.4/100, alternatif olarak 3 gerçek OFF ürünü önerildi.
RAG açıklaması (HuggingFace API): geçerli atıf oranı %100, 5 kaynak.
```

`POST /profile`, `POST /analyze` (multipart + profile_id), `GET /recommend` gerçek HTTP
istekleriyle (`curl`) doğrulandı — hepsi beklenen sonucu verdi. Streamlit demo başlatıldı,
temiz şekilde ayağa kalktığı (hatasız import/render) doğrulandı.

## Değerlendirme Sürecinde Bulunan ve Düzeltilen Bir Hata

`get_candidate_products()` ilk halinde, `dataset.csv`'deki **boş** `allergens` hücrelerini
yanlış işliyordu: pandas boş bir hücreyi `NaN` (float) olarak okur, ve Python'da `NaN` truthy
olduğundan (`bool(float('nan')) == True`) `row.get("allergens") or ""` deyimi bunu
YAKALAYAMIYORDU — sonuç `str(nan) = "nan"` string'ine dönüşüp `Allergen("nan")` çağrısında
`ValueError` fırlatıyordu. Bu, gerçek `dataset.csv` üzerinde çalışan bir testte (`GET
/recommend` uç nokta testi) ortaya çıktı. Kök nedeni tespit edilip `pd.isna()` ile açık kontrol
eklendi, ayrıca bu senaryoyu kilitleyen bir regresyon testi yazıldı
(`test_handles_blank_allergens_cell_read_as_nan_by_pandas`).

## Bilinen Sınırlamalar

- Profil deposu (`_PROFILE_STORE`) bellek-içidir — sunucu yeniden başlatıldığında silinir.
  Kalıcı bir veritabanı, formun kapsamı dışındadır (Faz 6 hedefi uçtan uca akışı göstermek).
- `/analyze` senkron çalışır; RAG API çağrısı birkaç saniye sürebilir (ücretsiz HuggingFace
  kotasında bazen daha da yavaş) — üretim ortamında arka plan görevi/kuyruk düşünülebilir.
- Streamlit'in varsayılan yapılandırması ağ üzerinden erişilebilir (`0.0.0.0`) olabilir; yerel
  geliştirme dışında çalıştırılırken güvenlik duvarı/kimlik doğrulama eklenmelidir.
