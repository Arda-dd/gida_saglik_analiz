# Open Food Facts — Atıf

Bu projede kullanılan ürün verileri ve görselleri, [Open Food Facts](https://world.openfoodfacts.org/)
topluluk veritabanından **Open Food Facts API v2** (`https://world.openfoodfacts.org/api/v2/search`)
aracılığıyla çekilmiştir.

**Lisans:** Open Food Facts verileri ve görselleri **Open Database License (ODbL)** / **CC-BY-SA 4.0**
lisansı altındadır.

**Atıf:** *Open Food Facts contributors, [https://world.openfoodfacts.org](https://world.openfoodfacts.org),
CC-BY-SA 4.0.*

## Bu Projede Kullanım Özeti

- **Kaynak:** `src/data/off_client.py`, `src/data/fetch_off_dataset.py`
- **Çekilen kategori/ürün sayısı:** 5 kategori × 90 ürün = 450 ham kayıt (2026-07-06 tarihinde çekildi)
- **Doğrulama sonrası geçerli kayıt:** 395 (`data/processed/dataset.csv`)
- **İndirilen ürün görseli:** 430 (`data/raw/openfoodfacts/images/`)
- **Ham JSON verisi:** `data/raw/openfoodfacts/{kategori}.json`

Bu veriler TÜBİTAK 2209-A "Gıda & Sağlık Asistanı" projesinin akademik/eğitim amaçlı araştırma
kapsamında kullanılmaktadır; ticari bir ürün olarak dağıtılması durumunda ODbL/CC-BY-SA
koşullarına yeniden uyum sağlanmalıdır.
