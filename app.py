# IMPORT LIBRARY & DEPENDENCIES
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# KONFIGURASI TAMPILAN HALAMAN
st.set_page_config(
    page_title="Early Warning System - Jaya Jaya Institut",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# PEMUATAN ARTEFAK MODEL MACHINE LEARNING & PREPROCESSOR
@st.cache_resource
def load_artifacts():
    possible_models = [
        os.path.join("model", "model_random_forest.joblib"),
        os.path.join("model", "random_forest_model.joblib"),
        "model_random_forest.joblib",
        "random_forest_model.joblib"
    ]
    
    model_path = None
    for p in possible_models:
        if os.path.exists(p):
            model_path = p
            break
            
    if model_path is None:
        if os.path.exists("model") and len(os.listdir("model")) > 0:
            for f in os.listdir("model"):
                if f.endswith(".joblib") and "forest" in f:
                    model_path = os.path.join("model", f)
                    break

    if model_path is None:
        raise FileNotFoundError("Berkas model Random Forest tidak ditemukan di folder model/ atau root.")

    model = joblib.load(model_path)

    scaler_path = os.path.join("model", "scaler.joblib") if os.path.exists(os.path.join("model", "scaler.joblib")) else "scaler.joblib"
    encoder_path = os.path.join("model", "label_encoder.joblib") if os.path.exists(os.path.join("model", "label_encoder.joblib")) else "label_encoder.joblib"

    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    encoder = joblib.load(encoder_path) if os.path.exists(encoder_path) else None

    return model, scaler, encoder

try:
    model, scaler, encoder = load_artifacts()
except Exception as e:
    st.error(f"Gagal memuat artefak model: {e}")
    st.stop()

# PANEL SIDEBAR (INFORMASI INSTITUSI & PROFIL MODEL)
with st.sidebar:
    st.title("ℹ️ Info Institusi")
    st.markdown("**Jaya Jaya Institut**  \nSistem Deteksi Dini & Retensi Mahasiswa")
    st.divider()
    
    st.markdown("### ⚙️ Profil Model")
    st.write("- **Algoritma:** Random Forest Classifier")
    st.write("- **Target:** Status Kelulusan Mahasiswa")
    
    # Visualisasi Top 5 Feature Importance yang dipelajari oleh model
    if hasattr(model, "feature_importances_") and hasattr(model, "feature_names_in_"):
        st.markdown("### 📌 5 Faktor Penentu Utama")
        fi_df = pd.DataFrame({
            'Fitur': model.feature_names_in_,
            'Bobot': model.feature_importances_
        }).sort_values(by='Bobot', ascending=False).head(5)
        st.bar_chart(fi_df.set_index('Fitur'))
        
    st.divider()
    st.caption("Dikembangkan untuk Proyek Data Analytics & Machine Learning")
    st.caption("© 2026 Galang Dava Ramadhan. All rights reserved.")

# HEADER UTAMA APLIKASI
st.title("Sistem Prediksi Retensi & Intervensi Mahasiswa")
st.markdown("Identifikasi risiko putus studi sedini mungkin dan lakukan simulasi perbaikan secara terukur.")
st.divider()

# FORMULIR INPUT DATA PROFIL & AKADEMIK MAHASISWA
st.subheader("📋 Data Profil & Capaian Akademik Mahasiswa")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### 📌 Demografi & Administrasi")
    age = st.number_input("Usia Saat Mendaftar", min_value=15, max_value=70, value=20)
    gender = st.selectbox("Jenis Kelamin", options=[1, 0], format_func=lambda x: "Laki-laki" if x == 1 else "Perempuan")
    debtor = st.selectbox("Memiliki Tunggakan Utang?", options=[0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    tuition = st.selectbox("Status Pembayaran SPP", options=[1, 0], format_func=lambda x: "Lancar / Lunas" if x == 1 else "Menunggak")
    scholarship = st.selectbox("Penerima Beasiswa?", options=[1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak")

with col2:
    st.markdown("##### 📚 Evaluasi Semester 1")
    admission_grade = st.number_input("Nilai Seleksi Masuk (0 - 200)", min_value=0.0, max_value=200.0, value=125.0, step=0.5)
    sem1_enrolled = st.number_input("SKS Diambil Sem 1", min_value=0, max_value=30, value=6)
    sem1_eval = st.number_input("Jumlah Evaluasi Sem 1", min_value=0, max_value=30, value=6)
    sem1_approved = st.number_input("SKS Lulus Sem 1", min_value=0, max_value=30, value=5)
    sem1_grade = st.number_input("Rata-rata Nilai Sem 1 (0 - 20)", min_value=0.0, max_value=20.0, value=13.0, step=0.1)

with col3:
    st.markdown("##### 📚 Evaluasi Semester 2")
    sem2_enrolled = st.number_input("SKS Diambil Sem 2", min_value=0, max_value=30, value=6)
    sem2_eval = st.number_input("Jumlah Evaluasi Sem 2", min_value=0, max_value=30, value=6)
    sem2_approved = st.number_input("SKS Lulus Sem 2", min_value=0, max_value=30, value=5)
    sem2_grade = st.number_input("Rata-rata Nilai Sem 2 (0 - 20)", min_value=0.0, max_value=20.0, value=13.0, step=0.1)

st.write("")

# PIPELINE PREPROCESSING & INFERENSI DATA
def run_pipeline(custom_input):
    df_clean = pd.read_csv('clean_students_performance.csv')
    df_template = df_clean.drop(columns=['Status']) if 'Status' in df_clean.columns else df_clean
    sample = df_template.iloc[[0]].copy()

    for k, v in custom_input.items():
        sample[k] = v

    expected_cols = getattr(scaler, "feature_names_in_", getattr(model, "feature_names_in_", None))
    if expected_cols is not None:
        for c in expected_cols:
            if c not in sample.columns:
                sample[c] = 0
        sample = sample[expected_cols]

    processed = scaler.transform(sample) if scaler is not None else sample
    pred = model.predict(processed)[0]
    prob = model.predict_proba(processed)[0] if hasattr(model, "predict_proba") else None
    label = encoder.inverse_transform([pred])[0] if encoder is not None else str(pred)
    return label, prob

# EKSEKUSI PREDIKSI & VISUALISASI HASIL
if st.button("🔍 Jalankan Analisis Risiko", type="primary", use_container_width=True):
    current_data = {
        'Admission_grade': float(admission_grade),
        'Age_at_enrollment': int(age),
        'Curricular_units_1st_sem_enrolled': int(sem1_enrolled),
        'Curricular_units_1st_sem_evaluations': int(sem1_eval),
        'Curricular_units_1st_sem_approved': int(sem1_approved),
        'Curricular_units_1st_sem_grade': float(sem1_grade),
        'Curricular_units_2nd_sem_enrolled': int(sem2_enrolled),
        'Curricular_units_2nd_sem_evaluations': int(sem2_eval),
        'Curricular_units_2nd_sem_approved': int(sem2_approved),
        'Curricular_units_2nd_sem_grade': float(sem2_grade),
        'Tuition_fees_up_to_date': int(tuition),
        'Scholarship_holder': int(scholarship),
        'Debtor': int(debtor),
        'Gender': int(gender)
    }

    status_result, proba = run_pipeline(current_data)
    
    # Menyimpan hasil ke session_state agar tidak ter-reset saat widget interaktif digeser
    st.session_state['has_predicted'] = True
    st.session_state['current_data'] = current_data
    st.session_state['status_result'] = status_result
    st.session_state['proba'] = proba

# Menampilkan hasil jika analisis sudah pernah dijalankan
if st.session_state.get('has_predicted', False):
    current_data = st.session_state['current_data']
    status_result = st.session_state['status_result']
    proba = st.session_state['proba']
    
    st.divider()
    st.subheader("📊 Hasil Inferensi Kondisi Saat Ini")
    
    res1, res2 = st.columns([1, 1])

    with res1:
        if status_result == "Dropout":
            st.error(f"### Status Prediksi: {status_result}")
            st.markdown("**Tingkat Risiko: 🔴 TINGGI (Rawan Putus Studi)**")
        elif status_result == "Enrolled":
            st.info(f"### Status Prediksi: {status_result}")
            st.markdown("**Tingkat Risiko: 🟡 SEDANG (Perlu Supervisi Akademik)**")
        else:
            st.success(f"### Status Prediksi: {status_result}")
            st.markdown("**Tingkat Risiko: 🟢 RENDAH (Prospek Kelulusan Baik)**")

        if proba is not None and encoder is not None and "Dropout" in encoder.classes_:
            dropout_idx = list(encoder.classes_).index("Dropout")
            risk_pct = float(proba[dropout_idx])
            st.metric(label="Skor Probabilitas Dropout", value=f"{risk_pct * 100:.1f}%")
            st.progress(risk_pct)

    with res2:
        if proba is not None and encoder is not None:
            st.markdown("**Distribusi Probabilitas:**")
            prob_df = pd.DataFrame({
                'Kategori Status': encoder.classes_,
                'Probabilitas': [f"{p*100:.2f}%" for p in proba]
            })
            st.dataframe(prob_df, use_container_width=True, hide_index=True)

    # SIMULASI SKENARIO INTERVENSI (WHAT-IF ANALYSIS)
    st.divider()
    st.subheader("🧪 Simulasi Tindakan Intervensi (What-If Analysis)")
    st.caption("Lihat perubahan probabilitas kelulusan jika pihak kampus memberikan bantuan intervensi pada mahasiswa ini.")
    
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        sim_tuition = st.selectbox(
            "Skenario SPP:", 
            options=[1, 0], 
            index=0 if current_data['Tuition_fees_up_to_date'] == 1 else 0, 
            format_func=lambda x: "SPP Dilunasi / Diberi Keringanan" if x == 1 else "SPP Tetap Menunggak"
        )
        sim_debtor = st.selectbox(
            "Skenario Utang:", 
            options=[0, 1], 
            index=0 if current_data['Debtor'] == 0 else 0, 
            format_func=lambda x: "Bebas Tunggakan Utang" if x == 0 else "Masih Memiliki Utang"
        )
    with sim_col2:
        base_approved = int(current_data['Curricular_units_2nd_sem_approved'])
        max_enrolled = int(max(current_data['Curricular_units_2nd_sem_enrolled'], 10))
        sim_sem2_approved = st.slider(
            "Target Tambahan SKS Lulus Sem 2:", 
            min_value=base_approved, 
            max_value=max_enrolled, 
            value=base_approved
        )
        base_grade = float(current_data['Curricular_units_2nd_sem_grade'])
        sim_sem2_grade = st.slider(
            "Target Nilai Rata-rata Sem 2:", 
            min_value=base_grade, 
            max_value=20.0, 
            value=float(max(base_grade, 12.0)), 
            step=0.5
        )

    sim_data = current_data.copy()
    sim_data['Tuition_fees_up_to_date'] = int(sim_tuition)
    sim_data['Debtor'] = int(sim_debtor)
    sim_data['Curricular_units_2nd_sem_approved'] = int(sim_sem2_approved)
    sim_data['Curricular_units_2nd_sem_grade'] = float(sim_sem2_grade)
    
    sim_status, sim_proba = run_pipeline(sim_data)

    if sim_proba is not None and encoder is not None and "Dropout" in encoder.classes_:
        dropout_idx = list(encoder.classes_).index("Dropout")
        grad_idx = list(encoder.classes_).index("Graduate") if "Graduate" in encoder.classes_ else None
        
        old_risk = float(proba[dropout_idx]) * 100
        new_risk = float(sim_proba[dropout_idx]) * 100
        delta_risk = new_risk - old_risk
        
        m1, m2 = st.columns(2)
        m1.metric("Status Pasca-Intervensi", sim_status)
        m2.metric("Peluang Risiko Dropout", f"{new_risk:.1f}%", f"{delta_risk:.1f}% (Penurunan)", delta_color="inverse")

    # SECTION 10: REKOMENDASI TINDAKAN BISNIS / STRATEGIS
    st.markdown("### 💡 Rekomendasi Intervensi Kampus")
    if status_result == "Dropout" or current_data['Tuition_fees_up_to_date'] == 0 or current_data['Curricular_units_2nd_sem_approved'] < 3:
        st.warning("""
        * **Intervensi Keuangan Cepat:** Hubungkan mahasiswa dengan program beasiswa darurat atau restrukturisasi cicilan pembayaran SPP.
        * **Pendampingan Akademik Intensif:** Wajibkan sesi pembimbingan berkala dengan Dosen Pembimbing Akademik (DPA) untuk merancang ulang rencana studi.
        * **Layanan Konseling Mahasiswa:** Lakukan penelusuran motivasi belajar serta adaptasi perkuliahan via bimbingan konseling kampus.
        """)
    else:
        st.success("""
        * **Akselerasi Prestasi:** Dorong keikutsertaan mahasiswa dalam kegiatan riset, kompetisi ilmiah, atau program magang industri.
        * **Pemantauan Berkala:** Lanjutkan sistem monitoring rutin kartu hasil studi (KHS) tiap akhir semester.
        """)