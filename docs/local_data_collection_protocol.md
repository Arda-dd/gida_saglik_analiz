# Yerel Market Veri Toplama Protokolü

Bu doküman, Arda ve Semih'in Migros/A101/Şok/CarrefourSA ve yerli üreticilerden gıda etiketi
fotoğrafı toplarken izleyeceği pratik protokoldür. **Bu adım fizikseldir — kod bunu otomatikleştiremez.**

Hedef: **~150 fotoğraf** (OFF ağırlıklı veri stratejisi kararı gereği; kategori başına ~30 foto × 5
kategori). Toplam veri seti hedefi (OFF + yerel) ≥500'dür (form başarı ölçütü).

## Ekipman

- Akıllı telefon kamerası, **minimum 12 MP**.
- Sabit veya doğal, yeterli ışık kaynağı (loş ortamlardan kaçının).
- Opsiyonel: tripod (el titremesinden kaynaklı bulanıklığı azaltmak için).

## Çekim Kuralları

- Etikete **paralel açı** ile çekin — perspektif bozulması OCR'ı (Faz 3) zorlaştırır.
- **Yansıma yok**: parlak ambalajlarda flaş kullanmayın, ışık kaynağının doğrudan yansımasından kaçının.
- **Gölgesiz**: etiketin tamamı okunaklı, gölgesiz olmalı.
- Etiketin **tamamı kadrajda** olsun (kırpılmamış), mümkünse besin değerleri tablosu net odakta.
- Aynı üründen birden fazla açı/tekrar çekmeyin — çeşitlilik için **farklı ürünler** tercih edin.

## Kategori Başına Hedef (~150 toplam)

| Kategori       | Hedef foto sayısı |
|----------------|--------------------|
| süt_urunu      | ~30                |
| atıştırmalık   | ~30                |
| içecek         | ~30                |
| hazır_gıda     | ~30                |
| konserve       | ~30                |

Migros/A101/Şok/CarrefourSA ve yerli üreticiler arasında dengeli dağıtım hedeflenir (tek bir
marketten tüm fotoğrafları çekmeyin — veri çeşitliliği için).

## Dosya Adlandırma (ham dosyalar, `inbox/` klasörüne atılır)

Format: `{market}_{kategori}_{sekans}.jpg` (örn. `migros_icecek_01.jpg`).
Bu ham adlandırma yalnızca sizin organizasyonunuz içindir — `src/data/local_intake.py` bu dosyaları
okuyup **anonim** `local_{kategori}_{seq:04d}.jpg` ID'sine çevirecektir.

## Anonimleştirme Adımları (kod tarafından otomatik yapılır)

1. **EXIF/GPS/cihaz metadata'sı silinir** (`anonymize_image_exif` — konum, cihaz modeli, çekim
   zamanı gibi kişisel/cihaz bilgisi kaldırılır).
2. **Marka/logo GÖRSELDEN KIRPILMAZ** — ürünün besin bilgisi/etiketi olarak kalması gereklidir;
   sadece kişisel/cihaz metadata'sı silinir, görsel içerik değişmez.
3. Çekimi yapan kişinin adı/bilgisi **hiçbir dosya adında veya metadata'da** yer almaz.

## Kullanım (intake script)

```bash
python -m src.data.local_intake --inbox path/to/inbox --category icecek --raw-dir data/raw/local
```

Script her kategori için ayrı ayrı çalıştırılır (`--category` argümanı ile). Çalıştırma sonunda
`data/raw/local/<kategori>/` altında anonim, organize edilmiş dosyalar oluşur ve konsola kaç
görselin kalite kontrolünden geçtiği/geçmediği (düşük çözünürlük veya bulanıklık) raporlanır.

## Kalite Eşikleri (otomatik kontrol, `check_image_quality`)

- Minimum çözünürlük: **1600×1600 piksel**.
- Bulanıklık: Laplacian varyansı **≥120** olmalı (bulanık fotoğraflar reddedilir, tekrar çekilmelidir).

Kalite kontrolünden geçmeyen fotoğraflar veri setine dahil edilmez — `build_intake_manifest`
çıktısında `is_valid=False` olarak işaretlenir ve nedeni (`reasons` sütunu) raporlanır.
