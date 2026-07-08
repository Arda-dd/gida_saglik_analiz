# Faz 2 Sonuçları ve Sınırlamalar — Görsel Sınıflandırma

**Durum:** Pipeline uçtan uca çalışıyor (veri → augmentation → transfer learning → k-fold CV →
metrikler → checkpoint), ancak doğruluk henüz öneri formunun hedeflerine (**≥%85**, esnek **≥%90**)
ulaşmadı. Bu doküman sonuçları ve kök nedeni şeffaf şekilde raporlar.

## Sonuçlar (2026-07-08, 394 gerçek OFF etiket görseli, 5 kategori)

| Backbone         | CV ortalama acc | CV ortalama f1 | Held-out test acc | Held-out test f1 |
|------------------|------------------|-----------------|--------------------|--------------------|
| MobileNetV3      | %69.5 ± %6.1     | %67.6 ± %7.4    | **%75.0**          | **%73.2**          |
| EfficientNet-B3  | %74.3 ± %4.7     | %73.2 ± %5.0    | %68.3              | %68.1              |

**Seçilen model:** MobileNetV3 (`models/vision_best.pt`) — held-out test f1'i daha yüksek.
Not: İki model arası CV/test sıralaması tutarsız (EfficientNet-B3 CV'de daha iyi ama testte
geride) — bu, held-out test setinin küçüklüğünden (60 görsel) kaynaklanan örnekleme
gürültüsüdür, mimari bir üstünlük iddiası değildir.

## Kök Neden Analizi (neden ≥%85 değil?)

İki tanı denemesi yapıldı:

1. **Daha fazla epoch (8→20):** `train_loss` epoch 9'da ~0.03-0.07'ye düşüyor (neredeyse sıfır)
   while val/test accuracy %65-73 civarında düz kalıyor. Bu **klasik aşırı öğrenme (overfitting)**
   imzasıdır — model eğitim setini ezberliyor ama genelleyemiyor. Daha fazla epoch, kayda değer bir
   iyileşme sağlamadı (20 epoch sonunda hâlâ %70-75 bandında).
2. **Backbone dondurma (yalnızca son katmanı eğitme):** Aşırı öğrenmeyi azaltmak için denendi,
   ancak 15 epoch sonunda sadece %58 doğruluğa ulaştı — tam ince ayardan (full fine-tuning) daha
   kötü. Demek ki sorun "model çok esnek/ezberliyor" değil, **domain gap + yetersiz örnek sayısı**.

**Sonuç:** Asıl darboğaz **veri hacmi**. 394 görsel (kategori başına 64-90), transfer learning ile
bile 5 sınıflı sağlam bir sınıflandırıcı için azdır. Bu, Faz 1'de tasarlanan veri stratejisiyle
(OFF ağırlıklı + ~150 yerel fotoğraf) tam örtüşüyor — **yerel market fotoğrafları henüz
çekilmedi** (fiziksel insan görevi, `docs/local_data_collection_protocol.md`).

## Önerilen Yol Haritası

1. **Öncelik:** Arda/Semih'in `docs/local_data_collection_protocol.md`'ye göre ~150 yerel etiket
   fotoğrafını çekip `src/data/local_intake.py` ile organize etmesi. Veri seti ~544 görsele
   çıkınca (394 OFF + ~150 yerel), model bu genişletilmiş veriyle **yeniden eğitilmelidir**
   (`python -m src.vision.train` — script idempotent önbellekleme sayesinde OFF verisini tekrar
   çekmez, sadece yeni veriyle eğitir).
2. Doğruluk hâlâ hedefin altında kalırsa, önerinin kendi **Risk Yönetimi B-Planı** devreye girer:
   sınıf sayısını 3 ana kategoriye indirmek (Katı Atıştırmalık / Sıvı İçecek / Süt Ürünü) veya
   CLIP tabanlı zero-shot sınıflandırmaya geçmek.
3. Bu ilk sonuçlar, projenin bilimsel raporunda "veri hacminin sınıflandırma başarımına etkisi"
   şeklinde dürüst bir bulgu olarak da sunulabilir — sistemin çalıştığını ama veri ölçeğiyle
   doğrudan ilişkili olduğunu gösteren nicel bir kanıttır.

## Pipeline'ın Kendisi Hakkında

Aşırı öğrenme/veri hacmi sorunundan bağımsız olarak, şu bileşenler **doğru ve test edilmiş**
şekilde çalışıyor: stratified train/test split, 5-fold stratified CV, iki backbone'un (timm
üzerinden ImageNet ön-eğitimli) transfer learning ile eğitimi, accuracy/F1-macro metrikleri,
en iyi modelin otomatik seçimi ve `models/vision_best.pt` olarak kaydı, `src/vision/infer.py`
ile tekil görsel üzerinde tahmin. Faz 6'da (web demo) bu model doğrudan kullanılabilir; sadece
doğruluğu düşük olacaktır ta ki veri seti büyüyene kadar.
