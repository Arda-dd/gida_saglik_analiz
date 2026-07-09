"""Streamlit web demo (Faz 6): etiket fotografi yukle, 3 katmanli sonucu kaynak
referanslariyla goster.

Bu demo api/pipeline.py'yi DOGRUDAN cagirir (api/main.py'nin HTTP katmanini atlar) - boylece
ayri bir FastAPI sunucusu calistirmadan tek komutla ("streamlit run demo/app.py") uctan uca
akis denenebilir. Calistirmadan once proje kokunden `streamlit run demo/app.py` calistirin.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from api.pipeline import analyze_label_image
from src.common.schema import Allergen
from src.health.profile import ChronicCondition, HealthProfile

st.set_page_config(page_title="Gida & Saglik Asistani", page_icon="🥗")
st.title("Gida & Saglik Asistani")
st.caption(
    "TUBITAK 2209-A prototipi — etiket fotografi yukleyin, kategori + besin degerleri + "
    "saglik riski + (opsiyonel) kisisel degerlendirme alin."
)

with st.sidebar:
    st.header("Kisisel Profil (opsiyonel)")
    st.caption("Bos birakilirsa sadece genel (kisisellestirilmemis) sonuc gosterilir.")

    condition_labels = st.multiselect(
        "Kronik durumlar", options=[c.value for c in ChronicCondition]
    )
    allergen_labels = st.multiselect("Alerjiler", options=[a.value for a in Allergen])
    calorie_target = st.number_input(
        "Gunluk kalori hedefi (kcal, 0 = belirtilmedi)", min_value=0, value=0, step=100
    )

    st.divider()
    generate_explanation = st.checkbox(
        "Kaynak referansli LLM aciklamasi uret (API cagrisi gerektirir)", value=True
    )

uploaded_file = st.file_uploader("Etiket fotografi", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        image_path = Path(tmp_dir) / uploaded_file.name
        image_path.write_bytes(uploaded_file.getvalue())

        st.image(str(image_path), caption="Yuklenen gorsel", use_container_width=True)

        profile = None
        if condition_labels or allergen_labels or calorie_target:
            profile = HealthProfile(
                profile_id="demo_session",
                chronic_conditions=[ChronicCondition(c) for c in condition_labels],
                allergens=[Allergen(a) for a in allergen_labels],
                daily_calorie_target_kcal=calorie_target or None,
            )

        with st.spinner("Analiz ediliyor (kategori + OCR + risk motoru + RAG)..."):
            try:
                result = analyze_label_image(
                    image_path, profile=profile, generate_llm_explanation=generate_explanation
                )
            except Exception as exc:  # noqa: BLE001 - demo katmaninda kullaniciya guvenli mesaj
                st.error(f"Analiz sirasinda bir hata olustu: {exc}")
                st.stop()

        st.subheader("Kategori")
        st.write(f"**{result.category}**  (guven: %{result.category_confidence * 100:.1f})")
        st.caption(f"OCR ortalama guven skoru: %{result.ocr_confidence:.1f}")

        st.subheader("Besin Degerleri (100g bazinda)")
        nutrition_dict = result.nutrition.model_dump(exclude_none=True)
        if nutrition_dict:
            st.json(nutrition_dict)
        else:
            st.info("OCR'dan besin degeri cikarilamadi (gorsel kalitesi/tablo duzeni etkileyebilir).")

        st.subheader("1️⃣ Saglik Riski")
        if result.risk_messages:
            for msg in result.risk_messages:
                st.warning(msg)
        else:
            st.success("Kural motoruna gore belirgin bir risk bayragi tespit edilmedi.")

        if result.health_assessment is not None:
            st.subheader("2️⃣ Diyet Uyum Skoru")
            st.metric("Uyum Skoru", f"{result.health_assessment.diet_compliance_score:.0f} / 100")

            st.subheader("3️⃣ Alerjen Uyarisi")
            if result.health_assessment.allergen_warning:
                conflict_names = ", ".join(a.value for a in result.health_assessment.allergen_conflicts)
                st.error(f"Dikkat: profilinizdeki alerjenlerle celisiyor ({conflict_names})!")
            else:
                st.success("Profilinizdeki alerjenlerle bilinen bir celisme yok.")
        elif result.detected_allergens:
            st.caption(
                "Tespit edilen alerjenler (profil girilmedigi icin kisisel uyari uretilmedi): "
                + ", ".join(a.value for a in result.detected_allergens)
            )

        if result.explanation is not None:
            st.subheader("Kaynak Referansli Aciklama (RAG)")
            st.write(result.explanation.text)
            with st.expander(f"Kaynaklar ({len(result.explanation.retrieved)})"):
                for r in result.explanation.retrieved:
                    verified_label = "✅ dogrulanmis" if r.chunk.verified else "⚠️ taslak"
                    st.caption(f"`{r.chunk.chunk_id}` — {r.chunk.title} / {r.chunk.section} ({verified_label})")
        elif generate_explanation:
            st.info(
                "RAG aciklamasi uretilemedi (API anahtari eksik olabilir veya FAISS index'i "
                "henuz kurulmamis olabilir — bkz. `python -m src.rag.index_builder`)."
            )

        if result.alternatives:
            st.subheader("Alternatif Urun Onerileri")
            for alt in result.alternatives:
                st.write(f"- `{alt.product_id}` ({alt.category.value})")
