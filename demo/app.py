"""Streamlit web demo (Faz 6): etiket fotografi yukle, 3 katmanli sonucu kaynak
referanslariyla goster.

Bu demo api/pipeline.py'yi DOGRUDAN cagirir (api/main.py'nin HTTP katmanini atlar) - boylece
ayri bir FastAPI sunucusu calistirmadan tek komutla ("streamlit run demo/app.py") uctan uca
akis denenebilir. Calistirmadan once proje kokunden `streamlit run demo/app.py` calistirin.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st

# `streamlit run demo/app.py` calistirilan dizinden bagimsiz olarak proje kokunu sys.path'e
# ekler - Streamlit, pytest'in aksine proje kokunu otomatik path'e eklemez, bu yuzden
# `from api...`/`from src...` importlari bu satir olmadan ModuleNotFoundError verir.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hashlib
import json
import uuid

from api.database import init_db, SessionLocal, User as DBUser, Profile as DBProfile, ScanHistory as DBScanHistory
from api.auth import hash_password, verify_password
from api.pipeline import analyze_label_image, get_candidate_products
from src.common.schema import Allergen, NutritionFacts, ProductRecord, UserObjective
from src.health.profile import ChronicCondition, HealthProfile
from src.health.recommend import build_health_assessment, recommend_alternatives
from src.ocr.risk_engine import describe_risks
from src.rag.chunking import chunk_knowledge_base
from src.rag.generate import CITATION_PATTERN

# Veritabanını ilklendir
init_db()

# Besin degerlerini kullaniciya okunakli Turkce etiket + birimle gostermek icin siralama ve
# (alan_adi -> (Turkce etiket, birim)) eslemesi - NutritionFacts'in alan sirasiyla ayni.
NUTRITION_DISPLAY_ORDER: list[tuple[str, str, str]] = [
    ("energy_kcal", "Enerji", "kcal"),
    ("energy_kj", "Enerji", "kJ"),
    ("fat_g", "Yağ", "g"),
    ("saturated_fat_g", "Doymuş Yağ", "g"),
    ("carbohydrate_g", "Karbonhidrat", "g"),
    ("sugar_g", "Şeker", "g"),
    ("fiber_g", "Lif", "g"),
    ("protein_g", "Protein", "g"),
    ("salt_g", "Tuz", "g"),
    ("sodium_mg", "Sodyum", "mg"),
]

CATEGORY_TR_LABELS: dict[str, str] = {
    "sut_urunu": "Süt Ürünü",
    "atistirmalik": "Atıştırmalık",
    "icecek": "İçecek",
    "hazir_gida": "Hazır Gıda",
    "konserve": "Konserve",
    "bilinmiyor": "Bilinmiyor",
}

# Secim kutusundaki (selectbox) ham enum degerlerini ("kilo_verme" vb.) kullaniciya gosterirken
# okunakli Turkce etikete cevirmek icin - format_func ile: secilen deger degismez, sadece
# ekranda gorunen metin degisir.
OBJECTIVE_TR_LABELS: dict[str, str] = {
    "Belirtilmedi": "Belirtilmedi",
    "kilo_verme": "Kilo Verme",
    "protein_agirlikli": "Protein Ağırlıklı Beslenme",
    "alerji_takibi": "Alerji Takibi",
}


def _load_kb_doc_info() -> dict[str, tuple[str, bool, str | None]]:
    """Bilgi tabanindaki her dokuman icin (baslik, dogrulanmis-mi, kaynak-adi) haritasi cikarir.

    Amac: LLM ciktisindaki ham "[Kaynak: who_sugars_intake::2]" gibi chunk_id'leri, kullaniciya
    "WHO — Serbest Seker ve Yag Alimi" gibi okunakli bir isimle gostermek (bkz. render_sources_legend).
    """
    info: dict[str, tuple[str, bool, str | None]] = {}
    for chunk in chunk_knowledge_base():
        info.setdefault(chunk.doc_id, (chunk.title, chunk.verified, chunk.source))
    return info


KB_DOC_INFO = _load_kb_doc_info()


def format_citations_as_footnotes(text: str) -> tuple[str, list[str]]:
    """Metindeki her "[Kaynak: chunk_id]" etiketini kucuk bir "[n]" dipnot isaretine cevirir.

    Kullanicinin ham chunk_id'lerle (ornegin "who_sugars_intake::2") karsilasmasini onler -
    okunakli kaynak adlari ayri bir listede render_sources_legend ile gosterilir.
    """
    seen: list[str] = []

    def _replace(match) -> str:
        chunk_id = match.group(1).strip()
        if chunk_id not in seen:
            seen.append(chunk_id)
        return f"[{seen.index(chunk_id) + 1}]"

    cleaned = CITATION_PATTERN.sub(_replace, text)
    return cleaned, seen


def render_sources_legend(chunk_ids: list[str]) -> None:
    """format_citations_as_footnotes'un urettigi [n] dipnotlarina karsilik gelen okunakli
    kaynak listesini gosterir (dogrulanmis/taslak rozetiyle birlikte)."""
    if not chunk_ids:
        return
    st.markdown("**Kaynaklar:**")
    for i, chunk_id in enumerate(chunk_ids, start=1):
        doc_id = chunk_id.split("::")[0]
        title, verified, _source = KB_DOC_INFO.get(doc_id, (doc_id, False, None))
        badge = "✅ Doğrulanmış kaynak" if verified else "⚠️ Taslak/doğrulanmamış kaynak"
        st.caption(f"[{i}] {title} — {badge}")


def clean_explanation_text(text: str) -> str:
    """Kucuk/ucretsiz modelin bazen ekledigi "Kullanıcıya 3-5 cümlelik bir değerlendirme
    yazabiliriz:" turu meta-yorum satirini temizler - kullanicinin ilk okudugu seyin
    dogrudan saglik degerlendirmesi olmasi icin (sistemin kendi kendine konusmasi degil)."""
    lines = text.strip().splitlines()
    if lines and lines[0].strip().lower().startswith("kullanıcıya"):
        lines = lines[1:]
    return "\n".join(lines).strip()


def render_nutrition_table(nutrition_dict: dict) -> None:
    """Ham NutritionFacts sozlugunu (ornegin {'sugar_g': 5.0}) okunakli bir Turkce
    tabloya cevirir - kullaniciya ham JSON/alan-adi/ondalik-basamak karmasasi gostermemek icin."""
    rows = [
        {"Besin Öğesi": f"{label} ({unit})", "Değer": round(nutrition_dict[key], 1)}
        for key, label, unit in NUTRITION_DISPLAY_ORDER
        if key in nutrition_dict
    ]
    st.table(rows)


st.set_page_config(page_title="Gıda & Sağlık Asistanı", page_icon="🥗")
st.title("Gıda & Sağlık Asistanı")
st.caption(
    "TÜBİTAK 2209-A prototipi — etiket fotoğrafı yükleyin, kategori + besin değerleri + "
    "sağlık riski + kişisel değerlendirme alın."
)

db = SessionLocal()

# Oturum durumlarını tanımla
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "user_email" not in st.session_state:
    st.session_state.user_email = None

with st.sidebar:
    # 1) Kullanıcı Giriş / Kayıt Bölümü
    st.header("👤 Kullanıcı Hesabı")
    if st.session_state.user_id is None:
        auth_mode = st.radio("İşlem", ["Giriş Yap", "Kayıt Ol"])
        email = st.text_input("E-posta")
        password = st.text_input("Şifre", type="password")
        remember_me = st.checkbox("Beni Hatırla (30 gün)")

        if auth_mode == "Giriş Yap":
            if st.button("Giriş"):
                user = db.query(DBUser).filter(DBUser.email == email).first()
                if user and verify_password(password, user.hashed_password):
                    st.session_state.user_id = user.id
                    st.session_state.user_email = user.email
                    st.success("Giriş başarılı!")
                    st.rerun()
                else:
                    st.error("Hatalı e-posta veya şifre.")
        else:
            if st.button("Kayıt Ol"):
                if not email or not password:
                    st.error("E-posta ve şifre boş bırakılamaz.")
                else:
                    existing = db.query(DBUser).filter(DBUser.email == email).first()
                    if existing:
                        st.error("Bu e-posta zaten kayıtlı.")
                    else:
                        hashed = hash_password(password)
                        new_user = DBUser(email=email, hashed_password=hashed)
                        db.add(new_user)
                        db.commit()
                        db.refresh(new_user)

                        profile = DBProfile(user_id=new_user.id, objective=None)
                        db.add(profile)
                        db.commit()
                        st.success("Kayıt başarılı! Şimdi giriş yapabilirsiniz.")
    else:
        st.write(f"Giriş yapıldı: **{st.session_state.user_email}**")
        if st.button("Çıkış Yap"):
            st.session_state.user_id = None
            st.session_state.user_email = None
            st.rerun()

    st.divider()

    # 2) Profil Yönetimi Bölümü
    profile = None
    if st.session_state.user_id is not None:
        st.header("⚙️ Kişisel Sağlık Profili")
        user = db.query(DBUser).filter(DBUser.id == st.session_state.user_id).first()
        db_profile = user.profile
        if not db_profile:
            db_profile = DBProfile(user_id=user.id)
            db.add(db_profile)
            db.commit()

        saved_conditions = [ChronicCondition(c) for c in db_profile.chronic_conditions.split(",") if c]
        saved_allergens = [Allergen(a) for a in db_profile.allergens.split(",") if a]
        saved_cal = db_profile.daily_calorie_target_kcal or 0
        saved_obj_str = db_profile.objective if db_profile.objective else "Belirtilmedi"

        condition_labels = st.multiselect(
            "Kronik durumlar",
            options=[c.value for c in ChronicCondition],
            default=[c.value for c in saved_conditions]
        )
        allergen_labels = st.multiselect(
            "Alerjiler",
            options=[a.value for a in Allergen],
            default=[a.value for a in saved_allergens]
        )
        calorie_target = st.number_input(
            "Günlük kalori hedefi (kcal, 0 = belirtilmedi)",
            min_value=0,
            value=int(saved_cal),
            step=100
        )
        objective_options = ["Belirtilmedi", "kilo_verme", "protein_agirlikli", "alerji_takibi"]
        objective_label = st.selectbox(
            "Beslenme Amacınız",
            options=objective_options,
            index=objective_options.index(saved_obj_str),
            format_func=lambda x: OBJECTIVE_TR_LABELS.get(x, x)
        )

        if st.button("Ayarları Kaydet"):
            db_profile.chronic_conditions = ",".join(condition_labels)
            db_profile.allergens = ",".join(allergen_labels)
            db_profile.daily_calorie_target_kcal = calorie_target if calorie_target > 0 else None
            db_profile.objective = objective_label if objective_label != "Belirtilmedi" else None
            db.commit()
            st.success("Ayarlar başarıyla güncellendi!")
            st.rerun()

        profile = HealthProfile(
            profile_id=f"user_{user.id}",
            chronic_conditions=[ChronicCondition(c) for c in condition_labels],
            allergens=[Allergen(a) for a in allergen_labels],
            daily_calorie_target_kcal=calorie_target if calorie_target > 0 else None,
            objective=UserObjective(objective_label) if objective_label != "Belirtilmedi" else None
        )
    else:
        st.header("Kişisel Profil (Anonim)")
        st.caption("Giriş yapmadan yapılan değişiklikler kaydedilmez.")

        condition_labels = st.multiselect(
            "Kronik durumlar", options=[c.value for c in ChronicCondition]
        )
        allergen_labels = st.multiselect("Alerjiler", options=[a.value for a in Allergen])
        calorie_target = st.number_input(
            "Günlük kalori hedefi (kcal, 0 = belirtilmedi)", min_value=0, value=0, step=100
        )
        objective_options = ["Belirtilmedi", "kilo_verme", "protein_agirlikli", "alerji_takibi"]
        objective_label = st.selectbox(
            "Beslenme Amacınız",
            options=objective_options,
            index=0,
            format_func=lambda x: OBJECTIVE_TR_LABELS.get(x, x)
        )

        if condition_labels or allergen_labels or calorie_target or objective_label != "Belirtilmedi":
            profile = HealthProfile(
                profile_id="demo_session",
                chronic_conditions=[ChronicCondition(c) for c in condition_labels],
                allergens=[Allergen(a) for a in allergen_labels],
                daily_calorie_target_kcal=calorie_target if calorie_target > 0 else None,
                objective=UserObjective(objective_label) if objective_label != "Belirtilmedi" else None
            )

    st.divider()
    generate_explanation = st.checkbox(
        "Kaynak referanslı LLM açıklaması üret (API çağrısı)", value=True
    )

tab_scan, tab_dash = st.tabs(["🔍 Yeni Analiz & Tarama", "📊 Sağlık Raporu & Tarihçe Analizi"])

with tab_scan:
    # 3) Geçmiş Tarama Seçici (Tarihçe)
    selected_scan = None
    if st.session_state.user_id is not None:
        scans = db.query(DBScanHistory).filter(DBScanHistory.user_id == st.session_state.user_id).order_by(DBScanHistory.scanned_at.desc()).all()
        if scans:
            st.subheader("📜 Geçmiş Taramalarınız")
            scan_options = {f"{s.scanned_at.strftime('%Y-%m-%d %H:%M')} — {s.category.upper()}": s for s in scans}
            selected_scan_key = st.selectbox("Önceki taramalarınızı hızlıca inceleyin:", ["— Seçin —"] + list(scan_options.keys()))
            if selected_scan_key != "— Seçin —":
                selected_scan = scan_options[selected_scan_key]

    uploaded_file = st.file_uploader("Yeni bir etiket fotoğrafı yükleyin", type=["jpg", "jpeg", "png"])

    result_data = None
    is_cached = False

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        # Önbellek kontrolü
        cached_scan = None
        if st.session_state.user_id is not None:
            cached_scan = db.query(DBScanHistory).filter(
                DBScanHistory.user_id == st.session_state.user_id,
                DBScanHistory.file_hash == file_hash
            ).first()

        if cached_scan:
            selected_scan = cached_scan
            is_cached = True
            st.info("Bu görsel daha önce analiz edilmiş. Sonuçlar önbellekten (cache) yüklendi.")
        else:
            with tempfile.TemporaryDirectory() as tmp_dir:
                image_path = Path(tmp_dir) / uploaded_file.name
                image_path.write_bytes(file_bytes)

                st.image(str(image_path), caption="Yüklenen görsel", use_container_width=True)

                with st.spinner("Analiz ediliyor (kategori + OCR + risk motoru + RAG)..."):
                    try:
                        result = analyze_label_image(
                            image_path, profile=profile, generate_llm_explanation=generate_explanation
                        )
                        result_data = {
                            "category": result.category,
                            "category_confidence": result.category_confidence,
                            "nutrition": result.nutrition.model_dump(exclude_none=True),
                            "detected_allergens": [a.value for a in result.detected_allergens],
                            "risk_flags": result.risk_flags,
                            "risk_messages": result.risk_messages,
                            "ocr_confidence": result.ocr_confidence,
                            "explanation_text": result.explanation.text if result.explanation else None,
                        }

                        # Veritabanına kaydet
                        if st.session_state.user_id is not None:
                            new_scan = DBScanHistory(
                                user_id=st.session_state.user_id,
                                product_id=f"scan_{uuid.uuid4().hex[:8]}",
                                category=result.category,
                                category_confidence=result.category_confidence,
                                nutrition_json=json.dumps(result_data["nutrition"]),
                                detected_allergens=",".join(result_data["detected_allergens"]),
                                risk_flags=",".join(result_data["risk_flags"]),
                                ocr_confidence=result.ocr_confidence,
                                file_hash=file_hash,
                                explanation_text=result_data["explanation_text"]
                            )
                            db.add(new_scan)
                            db.commit()
                            st.success("Yeni analiz veritabanı geçmişinize kaydedildi.")
                    except Exception as exc:
                        st.error(f"Analiz sırasında bir hata oluştu: {exc}")
                        st.stop()

    if selected_scan is not None and result_data is None:
        nutrition_data = json.loads(selected_scan.nutrition_json)
        result_data = {
            "category": selected_scan.category,
            "category_confidence": selected_scan.category_confidence,
            "nutrition": nutrition_data,
            "detected_allergens": [a for a in selected_scan.detected_allergens.split(",") if a],
            "risk_flags": [rf for rf in selected_scan.risk_flags.split(",") if rf],
            "risk_messages": describe_risks([rf for rf in selected_scan.risk_flags.split(",") if rf]),
            "ocr_confidence": selected_scan.ocr_confidence,
            "explanation_text": selected_scan.explanation_text,
        }
        is_cached = True

    if result_data is not None:
        # --- Genel Özet (en ust, tek bakista anlasilir kisa ozet) ---
        category_label = CATEGORY_TR_LABELS.get(result_data["category"], result_data["category"])
        if result_data["risk_messages"]:
            st.warning(
                f"**{category_label}** kategorisinde bir ürün — {len(result_data['risk_messages'])} "
                "sağlık uyarısı bulundu (aşağıda detaylı)."
            )
        else:
            st.success(f"**{category_label}** kategorisinde bir ürün — belirgin bir sağlık riski tespit edilmedi.")

        if result_data["ocr_confidence"] < 50:
            st.caption(
                f"⚠️ OCR (metin okuma) güveni düşük (%{result_data['ocr_confidence']:.0f}) — bazı besin "
                "değerleri yanlış veya eksik çıkmış olabilir. Etiketi düz açıyla, parlamasız ve yakından "
                "çekmek doğruluğu artırabilir."
            )

        st.subheader("Kategori ve Besin Değerleri")
        st.write(f"Tahmin edilen kategori: **{category_label}** (model güveni: %{result_data['category_confidence'] * 100:.0f})")

        if result_data["nutrition"]:
            st.caption("Aşağıdaki değerler 100 gram/100 mL bazına normalize edilmiştir.")
            render_nutrition_table(result_data["nutrition"])
        else:
            st.info(
                "OCR, etiketten besin değeri çıkaramadı. Bu genelde görsel kalitesi (parlama, açı, "
                "çözünürlük) veya alışılmadık bir tablo düzeninden kaynaklanır."
            )

        st.subheader("1️⃣ Sağlık Riski")
        if result_data["risk_messages"]:
            for msg in result_data["risk_messages"]:
                st.warning(msg)
        else:
            st.success("Kural motoruna göre (şeker/tuz/doymuş yağ/sodyum eşikleri) belirgin bir risk tespit edilmedi.")

        # Kişisel profil uyarısı ve diyet uyumu
        if profile is not None:
            nutrition = NutritionFacts(**result_data["nutrition"])
            detected_allergens_list = [Allergen(a) for a in result_data["detected_allergens"]]
            health_assessment = build_health_assessment(nutrition, detected_allergens_list, profile)

            st.subheader("2️⃣ Diyet Uyum Skoru")
            st.metric("Uyum Skoru", f"{health_assessment.diet_compliance_score:.0f} / 100")
            objective_display = (
                OBJECTIVE_TR_LABELS.get(profile.objective.value, profile.objective.value)
                if profile.objective
                else "Belirtilmedi"
            )
            st.caption(f"Aktif beslenme amacı: **{objective_display}**")

            st.subheader("3️⃣ Alerjen Uyarısı")
            if health_assessment.allergen_warning:
                conflict_names = ", ".join(a.value for a in health_assessment.allergen_conflicts)
                st.error(f"Dikkat: profilinizdeki alerjenlerle çelişiyor ({conflict_names})!")
            else:
                st.success("Profilinizdeki alerjenlerle bilinen bir çelişki yok.")

            # Alternatif öneriler
            candidates = get_candidate_products()
            try:
                from src.common.schema import ProductCategory
                current_cat = ProductCategory(result_data["category"])
            except ValueError:
                current_cat = ProductCategory.BILINMIYOR

            current_prod = ProductRecord(
                product_id="uploaded_image",
                category=current_cat,
                nutrition=nutrition,
                allergens=detected_allergens_list,
                source="upload"
            )
            alternatives = recommend_alternatives(current_prod, list(candidates), profile)
            if alternatives:
                st.subheader("Alternatif Ürün Önerileri")
                for alt in alternatives:
                    st.write(f"- `{alt.product_id}` ({alt.category.value})")
        elif result_data["detected_allergens"]:
            st.caption(
                "Tespit edilen alerjenler (profil girilmediği için kişisel uyarı üretilmedi): "
                + ", ".join(result_data["detected_allergens"])
            )

        if result_data["explanation_text"]:
            st.subheader("Uzman Kaynaklara Dayalı Açıklama")
            st.caption("WHO/EFSA/TGK kaynaklarından alınan bilgiyle üretilmiştir.")
            cleaned_text, cited_chunk_ids = format_citations_as_footnotes(
                clean_explanation_text(result_data["explanation_text"])
            )
            st.write(cleaned_text)
            render_sources_legend(cited_chunk_ids)

with tab_dash:
    st.header("📊 Sağlık Raporu & Tarihçe Analizi")
    if st.session_state.user_id is None:
        st.warning("Lütfen kişisel sağlık raporunuzu ve geçmiş analizlerinizi görüntülemek için sol menüden Giriş Yapın.")
    else:
        from datetime import datetime, timedelta
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go

        cutoff_date = datetime.utcnow() - timedelta(days=30)
        scans_30d = db.query(DBScanHistory).filter(
            DBScanHistory.user_id == st.session_state.user_id,
            DBScanHistory.scanned_at >= cutoff_date
        ).order_by(DBScanHistory.scanned_at.asc()).all()

        if not scans_30d:
            st.info("Son 30 gün içinde taranmış ürün geçmişiniz bulunmuyor. Yeni analizler yaptıktan sonra bu panel güncellenecektir.")
        else:
            dates = []
            calories = []
            sugar = []
            salt = []
            categories = []
            product_names = []

            for scan in scans_30d:
                try:
                    nut = json.loads(scan.nutrition_json)
                    dates.append(scan.scanned_at.date())
                    calories.append(nut.get("energy_kcal", 0) or 0)
                    sugar.append(nut.get("sugar_g", 0) or 0)
                    salt.append(nut.get("salt_g", 0) or 0)
                    categories.append(scan.category)
                    product_names.append(scan.product_id)
                except Exception:
                    continue

            df = pd.DataFrame({
                "Tarih": dates,
                "Kalori (kcal)": calories,
                "Şeker (g)": sugar,
                "Tuz (g)": salt,
                "Kategori": categories,
                "Urun": product_names
            })

            df_daily = df.groupby("Tarih").sum().reset_index()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Toplam Tarama", len(df))
            col2.metric("Günlük Ort. Kalori", f"{df_daily['Kalori (kcal)'].mean():.0f} kcal")
            col3.metric("Günlük Ort. Şeker", f"{df_daily['Şeker (g)'].mean():.1f} g")
            col4.metric("Günlük Ort. Tuz", f"{df_daily['Tuz (g)'].mean():.2f} g")

            st.markdown("---")

            st.subheader("🔥 Günlük Kalori Alımı ve Hedef Durumu")
            user = db.query(DBUser).filter(DBUser.id == st.session_state.user_id).first()
            cal_target = (user.profile.daily_calorie_target_kcal if user.profile else 2000.0) or 2000.0

            fig_cal = px.bar(df_daily, x="Tarih", y="Kalori (kcal)", title="Günlük Toplam Kalori (Taranan Ürünler)", color_discrete_sequence=["#FF4B4B"])
            fig_cal.add_hline(y=cal_target, line_dash="dash", line_color="red", annotation_text=f"Günlük Limit ({cal_target} kcal)", annotation_position="top left")
            st.plotly_chart(fig_cal, use_container_width=True)

            excess_cal_days = df_daily[df_daily["Kalori (kcal)"] > cal_target]
            if not excess_cal_days.empty:
                st.warning(f"⚠️ Son 30 günde günlük kalori limitinizi aştığınız **{len(excess_cal_days)}** gün tespit edildi!")

            st.subheader("🧂 Günlük Şeker ve Tuz Tüketim Analizi")
            sugar_limit = 50.0
            salt_limit = 5.0

            col_sug, col_salt = st.columns(2)
            with col_sug:
                fig_sug = px.bar(df_daily, x="Tarih", y="Şeker (g)", title="Günlük Şeker Tüketimi (g)", color_discrete_sequence=["#FFA07A"])
                fig_sug.add_hline(y=sugar_limit, line_dash="dash", line_color="red", annotation_text=f"WHO Limiti ({sugar_limit}g)", annotation_position="top left")
                st.plotly_chart(fig_sug, use_container_width=True)

                excess_sug = df_daily[df_daily["Şeker (g)"] > sugar_limit]
                if not excess_sug.empty:
                    st.error(f"🚨 Şeker limitinin aşıldığı gün sayısı: {len(excess_sug)}")

            with col_salt:
                fig_salt = px.bar(df_daily, x="Tarih", y="Tuz (g)", title="Günlük Tuz Tüketimi (g)", color_discrete_sequence=["#20B2AA"])
                fig_salt.add_hline(y=salt_limit, line_dash="dash", line_color="red", annotation_text=f"WHO Limiti ({salt_limit}g)", annotation_position="top left")
                st.plotly_chart(fig_salt, use_container_width=True)

                excess_salt = df_daily[df_daily["Tuz (g)"] > salt_limit]
                if not excess_salt.empty:
                    st.error(f"🚨 Tuz limitinin aşıldığı gün sayısı: {len(excess_salt)}")

            st.markdown("---")

            st.subheader("🥗 Tüketim Çeşitliliği ve Dağılımları")
            col_pie, col_box = st.columns(2)
            with col_pie:
                fig_pie = px.pie(df, names="Kategori", title="Taranan Ürünlerin Kategorilere Göre Dağılımı", hole=0.3)
                st.plotly_chart(fig_pie, use_container_width=True)
            with col_box:
                fig_box = go.Figure()
                fig_box.add_trace(go.Box(y=df["Şeker (g)"], name="Şeker (g)", marker_color="#FFA07A"))
                fig_box.add_trace(go.Box(y=df["Tuz (g)"], name="Tuz (g)", marker_color="#20B2AA"))
                fig_box.update_layout(title="Ürün Bazında Şeker ve Tuz Değerlerinin Dağılımı")
                st.plotly_chart(fig_box, use_container_width=True)
