# Faz 5 Sonuçları — Kişisel Sağlık Profili ve Karar Destek

**Durum:** Tamamlandı. Tüm hedef metrikler formun ≥%90 eşiğinin üzerinde: **Profile
Consistency Score %100**, **Recommendation Relevance %100** (52 pytest testi + 13 sentetik
değerlendirme senaryosu, tamamı yeşil/geçti).

## Mimari

Bu katman **tamamen kural tabanlıdır** — Faz 4'teki RAG/LLM katmanına bağımlı değildir,
API çağrısı gerektirmez, bu yüzden değerlendirme anında ve deterministik çalışır:

- `src/health/profile.py` — `HealthProfile` (Pydantic): anonim `profile_id`, kronik durumlar
  (`ChronicCondition`: diyabet/hipertansiyon/böbrek hastalığı/kalp hastalığı), alerjenler
  (mevcut `Allergen` enum'ı yeniden kullanılır), günlük kalori/makro hedefleri. İsim/e-posta/
  telefon gibi kişisel kimlik alanları **yok** (bkz. `test_health_profile_does_not_contain_pii_fields`).
- `src/health/personal_filter.py` — `condition_based_warnings()` (kronik durum → ilgili risk
  bayrağı → kişiselleştirilmiş uyarı eşlemesi), `check_allergen_conflict()`, `compute_diet_
  compliance_score()` (0-100, kalori payı aşımı + kronik durum çelişkisi cezası).
- `src/health/recommend.py` — `build_health_assessment()`: Faz 3'ün genel risk motorunu
  (`risk_engine.py`) Faz 5'in kişisel filtresiyle birleştirip **3 katmanlı çıktı** üretir
  (Sağlık Riski, Diyet Uyum Skoru, Alerjen Uyarısı). `recommend_alternatives()`: aynı
  kategoride, profille çelişmeyen ve daha az risk bayraklı alternatif ürünleri sıralar; hiçbir
  aday daha iyi değilse bilinçli olarak **boş liste döner** (yanlış/anlamsız bir "alternatif"
  önerip kullanıcıyı yanıltmamak için).
- `src/health/evaluate.py` — iki form metriğini sentetik ama elle etiketlenmiş senaryolarla ölçer.

## Test Kapsamı (52 test)

| Dosya | İçerik |
|---|---|
| `tests/test_health_profile.py` | Pydantic doğrulama (pozitif kalori hedefi, PII alanı yok) — 5 test |
| `tests/test_personal_filter.py` | Koşul-bayrak eşleşmesi (çapraz bulaşma yok), alerjen çelişkisi, diyet uyum skoru — 22 test |
| `tests/test_health_recommend.py` | 3 katmanlı çıktı entegrasyonu, alternatif öneri filtreleme — 13 test |

**Risk Yönetimi senaryoları (form 2.5 taahhüdü) özellikle test edildi:** "yanlış alerjen"
(ürün fındık içeriyor ama kullanıcının alerjisi soya — çelişme YOK, `test_no_conflict_when_
allergens_differ`) ve "yanlış eşik/kondisyon eşleşmesi" (diyabet koşulu, tuz bayrağıyla
tetiklenmemeli, `test_diabetes_does_not_warn_on_unrelated_flag`) açıkça doğrulandı.

## Gerçek Değerlendirme Sonuçları

`python -m src.health.evaluate` → `docs/health_evaluation_report.json`

| Metrik | Sonuç | Hedef |
|---|---|---|
| Profile Consistency Score | **%100** (9/9 senaryo) | ≥%90 |
| Recommendation Relevance | **%100** (4/4 senaryo) | ≥%90 |

**Profile Consistency** senaryoları, aynı ürünün farklı profiller için doğru şekilde
farklılaştığını doğrular (ör. yüksek şekerli bir üründe diyabetik profil "Diyabet hastaları
için önerilmez" uyarısı alırken sağlıklı profil almaz; fındık alerjisi olan bir profil sadece
fındık içeren üründe uyarı alır, içermeyende almaz).

**Recommendation Relevance** senaryoları, kasıtlı olarak "çeldirici" adaylar (yanlış kategori,
alerjen çelişkisi, eşit/daha yüksek risk) içeren aday havuzlarında algoritmanın doğru filtreleme
yaptığını doğrular.

## Değerlendirme Sürecinde Bulunan ve Düzeltilen Bir Kalibrasyon Hatası

İlk çalıştırmada Recommendation Relevance %75 çıktı (4 senaryodan 1'i başarısız). Kök nedeni
incelendiğinde, bunun `recommend_alternatives()` fonksiyonundaki bir hata değil, **test
senaryosunun kendisindeki bir kalibrasyon hatası** olduğu görüldü: "risky1" adlı çeldirici ürün
sadece yüksek şeker içeriyordu (1 risk bayrağı), oysa mevcut ürün hem yüksek şeker hem yüksek
tuz içeriyordu (2 risk bayrağı) — algoritma, risky1'i haklı olarak "daha az riskli" (1 < 2) kabul
edip önerdi. Beklenti güncellenip risky1'e mevcut ürünle **eşit** risk sayısı verildi (hem şeker
hem tuz), bu da doğru şekilde elenmesini sağladı. Bu, Faz 2/3/4'te kurulan "gerçek sonucu
incele, kök nedeni bul, doğru tarafı düzelt" alışkanlığının bir örneğidir — burada hatalı olan
kod değil, testin kendisiydi.

## Bilinen Sınırlama

`compute_diet_compliance_score()`'daki ceza ağırlıkları (kalori payı için %25 eşik, kronik
durum başına 25 puan) şu an **makul varsayılan değerlerdir**, klinik/bilimsel bir kaynaktan
kalibre edilmemiştir — bu, gelecekte bir beslenme uzmanı/literatür kaynağıyla doğrulanabilir
bir TODO'dur (tıpkı Faz 1'deki `nutriscore_thresholds_overview.md` gibi, doğrulanana kadar
"taslak" olarak işaretlenir).
