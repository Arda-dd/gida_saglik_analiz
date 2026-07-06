# Türk Gıda Kodeksi — Beslenme ve Sağlık Beyanları Yönetmeliği (Özet)

**Durum:** TASLAK — DOĞRULANMALI. Bu oturumda mevzuat.gov.tr ve ilgili resmi sayfalara erişim
denendi (WebFetch), sayfalar 404 döndürdü / içerik alınamadı. Aşağıdaki metin, öneri formunun
kendi kaynakçasında (EK-1, madde 9) atıf yapılan "Türk Gıda Kodeksi Beslenme ve Sağlık Beyanları
Yönetmeliği, Resmî Gazete (2020)" düzenlemesine dair **kamuya bilinen, genel kabul görmüş**
bilgilerdir — birebir madde/fıkra numarasıyla alıntı DEĞİLDİR.

**TODO:** Arda/Semih tarafından Resmî Gazete'nin resmi arşivinden ("Türk Gıda Kodeksi Beslenme ve
Sağlık Beyanları Yönetmeliği") tam metin indirilip bu doküman gerçek madde numaralarıyla
güncellenmelidir. Faz 4 (RAG) öncesi tamamlanması gerekli bir ön koşuldur.

## Bilinen Genel Çerçeve (doğrulama gerektirir)

- Besin değerleri etikette **100 g veya 100 mL** başına (ve opsiyonel olarak porsiyon başına)
  beyan edilir.
- **Sodyum → Tuz dönüşümü:** `Tuz (g) = Sodyum (g) × 2.5` (WHO'nun `1g tuz = 400mg sodyum`
  oranıyla matematiksel olarak tutarlıdır — bkz. `who_salt_reduction.md`). Bu proje kodunda
  `src/common/units.py` içinde bu formül zaten uygulanmaktadır.
- "Yüksek/düşük" ibareli sağlık beyanları için niceliksel eşikler yönetmelikte tanımlanır
  (AB mevzuatıyla büyük ölçüde uyumludur); bu projede kullanılan `sugar_high_g_per_100g` vb.
  eşikler şu an için TGK'nin kendi rakamlarıyla DEĞİL, yaygın kullanılan Nutri-Score/UK FSA
  front-of-pack konvansiyonuyla doldurulmuştur (bkz. `nutriscore_thresholds_overview.md`) —
  bu, doğrulanana kadar geçici bir yaklaşımdır.
- Alerjen beyanı zorunlu 14 majör alerjen listesi AB Gıda Bilgilendirme Yönetmeliği (1169/2011)
  ile büyük ölçüde uyumludur (bkz. `allergen_labeling_overview.md`).

## Bu Projede Kullanımı

RAG modülü bu dokümanı **düşük güvenilirlikli, doğrulanmamış** kaynak olarak işaretlemeli; sistem
çıktılarında TGK'ye atıf yapılan bir yorum üretildiğinde, kullanıcıya bu doğrulamanın henüz resmi
metinle tamamlanmadığı gösterilmelidir (Faz 4 açıklanabilirlik katmanı — kaynak güven skoru).
