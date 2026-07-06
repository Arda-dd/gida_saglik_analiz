# Front-of-Pack "Yüksek/Düşük" Eşikleri (Nutri-Score / UK FSA Konvansiyonu)

**Durum:** TASLAK — DOĞRULANMALI. Bu oturumda resmi Nutri-Score/Santé publique France algoritma
belgesine WebFetch ile erişim denenmedi (URL tahmin etme riskinden kaçınıldı). Aşağıdaki rakamlar,
öneri formunun kaynakçasında atıf yapılan "Aune ve ark. (2017), Nutri-Score" makalesi ve yaygın
literatürde sıkça tekrarlanan, İngiltere Gıda Standartları Ajansı (FSA) kaynaklı "traffic-light"
eşikleridir.

**TODO:** Resmi Nutri-Score algoritma dokümanı (Santé publique France) veya UK FSA "traffic light
labelling" resmi rehberinden birebir doğrulanmalı.

## Yaygın Kullanılan Eşikler (100 g katı gıda için, doğrulama bekliyor)

- **Şeker:** >22.5 g/100g → "yüksek"; <5 g/100g → "düşük"
- **Tuz:** >1.5 g/100g (≈600 mg sodyum) → "yüksek"; <0.3 g/100g → "düşük"
- **Doymuş yağ:** >5 g/100g → "yüksek"; <1.5 g/100g → "düşük"
- **Toplam yağ:** >17.5 g/100g → "yüksek"; <3 g/100g → "düşük"

(İçecekler için eşikler genelde farklı ve daha düşüktür — bu proje henüz içecek-özel eşikleri
ayrıştırmamıştır; Faz 3'te `src/ocr/risk_engine.py` yazılırken kategori bazlı eşik ayrımı
[bkz. öneri formu 2.2: "içecek sınıfındaki ürünlerde sodyum oranının farklı eşiklerle
değerlendirilmesi"] bu doküman doğrulandıktan sonra netleştirilmelidir.)

## Bu Projede Kullanımı

`config/config.yaml` ve `thresholds.json` içindeki `sugar_high_g_per_100g` (22.5),
`salt_high_g_per_100g` (1.5), `saturated_fat_high_g_per_100g` (5.0), `sodium_high_mg_per_100g`
(600) değerleri buradan gelir. Bunlar **WHO'nun günlük alım önerileri DEĞİLDİR** — ürün etiketi
seviyesinde "bu ürün 100g'da fazla mı" sorusuna cevap veren ayrı bir konvansiyondur. İki kaynak
türü RAG çıktısında karıştırılmamalı, ayrı ayrı atıflandırılmalıdır.
