# WHO — Serbest Şeker ve Yağ Alımı

**Durum:** DOĞRULANDI (resmi kaynaktan gerçek metin çekildi)
**Kaynak:** World Health Organization, "Healthy diet" Fact Sheet
**URL:** https://www.who.int/news-room/fact-sheets/detail/healthy-diet
**Erişim tarihi:** 2026-07-06 (WebFetch ile)
**Not:** Aynı bilgiyi içeren özel "Sugars intake for adults and children" fact sheet sayfası bu
oturumda 404 döndürdü; içerik bu genel "healthy diet" sayfasından teyit edilmiştir.

## Özet

- **Serbest şeker (free sugars):** Günlük toplam enerjinin **%10'undan az** olmalı. Ek sağlık
  yararı için **%5 veya altına** indirilmesi önerilir.
- **Toplam yağ:** Yetişkinler için günlük enerjinin **%30 veya daha azı**, en az **%15'i** yağdan
  gelmelidir.
- **Doymuş yağ:** Toplam enerjinin **%10'undan fazla olmamalı**.
- **Trans yağ:** Toplam enerjinin **%1'inden fazla olmamalı**.
- Çocuklar için farklı ihtiyaçlar söz konusudur; yukarıdaki oranlar yetişkinlere yöneliktir.

## Bu Projede Kullanımı

`who_free_sugars_energy_pct` (10), `who_fat_energy_pct_max` (30), `who_saturated_fat_energy_pct_max` (10)
eşikleri doğrudan bu kaynaktan alınmıştır. Bunlar **günlük toplam enerji yüzdesi** cinsindendir —
ürün etiketindeki "100g'da yüksek şeker/yağ" eşikleri (`sugar_high_g_per_100g`,
`saturated_fat_high_g_per_100g`) ayrı bir front-of-pack etiketleme konvansiyonundan gelir, doğrudan
WHO rakamı değildir (bkz. `nutriscore_thresholds_overview.md`).
