# WHO — Tuz/Sodyum Alımı ve Azaltımı

**Durum:** DOĞRULANDI (resmi kaynaktan gerçek metin çekildi)
**Kaynak:** World Health Organization, "Salt reduction" Fact Sheet
**URL:** https://www.who.int/news-room/fact-sheets/detail/salt-reduction
**Erişim tarihi:** 2026-07-06 (WebFetch ile)

## Özet

- WHO, yetişkinler için günlük **2000 mg sodyum (yaklaşık 5 gramdan az tuz)** alımını önerir
  ("yaklaşık bir çay kaşığı" seviyesinde).
- 2-15 yaş çocuklar için doz, enerji gereksinimlerine göre aşağıya doğru ayarlanmalıdır.
- Dönüşüm oranı: **1 gram tuz = 400 mg sodyum**, dolayısıyla **5 gram tuz = 2000 mg sodyum**.
  (Not: Bu proje kodunda kullanılan `Tuz(g) = Sodyum(g) × 2.5` — yani `Sodyum(g) = Tuz(g) / 2.5` —
  formülü ile matematiksel olarak tutarlıdır: `5 / 2.5 = 2` → 2000 mg. Bkz. `src/common/units.py`.)
- Küresel ortalama günlük sodyum tüketimi (2021): **4278 mg** — WHO tavsiyesinin ikiden fazla katı.
- Yüksek sodyumlu diyetle ilişkilendirilen başlıca sağlık riskleri: yüksek kan basıncı, kardiyovasküler
  hastalıklar, mide kanseri, obezite, osteoporoz, böbrek hastalığı. 2023'te aşırı sodyum tüketimi ile
  **1,7 milyon ölüm** ilişkilendirilmiştir.

## Bu Projede Kullanımı

`who_salt_daily_intake_g` eşiği (bkz. `thresholds.json`) doğrudan bu kaynaktan alınmıştır (değer: 5).
Bu, **günlük toplam alım** önerisidir — ürün etiketindeki "100g'da yüksek tuz" eşiği (`salt_high_g_per_100g`)
ile karıştırılmamalıdır; o eşik ayrı bir kaynaktan (UK FSA/Nutri-Score front-of-pack konvansiyonu) gelir.
