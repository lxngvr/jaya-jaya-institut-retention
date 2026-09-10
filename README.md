# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan - Jaya Jaya Institut

## Business Understanding
Jaya Jaya Institut merupakan salah satu institusi pendidikan perguruan tinggi yang telah berdiri sejak tahun 2000 dan telah mencetak banyak lulusan berprestasi. Meski demikian, institusi saat ini menghadapi tantangan serius terkait tingginya rasio mahasiswa yang tidak menyelesaikan studi atau putus kuliah (dropout). Tingginya angka dropout ini berpotensi merusak reputasi institusi, menurunkan pendapatan operasional, serta menghambat efisiensi perencanaan akademik kampus. Untuk menyelesaikan masalah ini, Jaya Jaya Institut membutuhkan sistem berbasis data dan machine learning yang dapat mendeteksi mahasiswa berisiko dropout sedini mungkin (early warning system) agar pihak kampus dan dosen wali dapat memberikan bimbingan khusus serta intervensi yang tepat waktu.

### Permasalahan Bisnis
1. Tingginya angka dropout mahasiswa yang mengancam reputasi akademik dan stabilitas finansial institusi pendidikan.
2. Evaluasi akademik masih bersifat reaktif di akhir semester sehingga kampus terlambat melakukan penanganan terhadap mahasiswa yang berisiko.
3. Belum tersedianya instrumen monitoring terpusat untuk memetakan hubungan antara kondisi finansial (pembayaran SPP, beasiswa) dan capaian evaluasi studi mahasiswa.
4. Belum adanya model prediktif yang siap pakai untuk mengestimasi risiko dropout secara objektif dan melakukan simulasi intervensi perbaikan.

### Cakupan Proyek
1. Melakukan pembersihan data (data cleaning), audit konsistensi data, dan analisis data eksploratif (EDA) terhadap performa mahasiswa.
2. Membangun model machine learning klasifikasi multi-kelas (Dropout, Enrolled, Graduate) menggunakan Random Forest Classifier.
3. Merancang dashboard analitik interaktif berbasis Metabase untuk memonitor KPI retensi dan faktor keberhasilan studi.
4. Mengembangkan antarmuka web prototype menggunakan Streamlit yang menyediakan fitur prediksi risiko dan simulasi skenario intervensi (What-If Analysis).
5. Menyusun kesimpulan analitik serta rekomendasi aksi strategis (Action Items) bagi manajemen institusi.

---

## Persiapan

### Sumber Data
Dataset performa akademik dan demografi mahasiswa (Students' Performance Dataset) yang disediakan oleh Dicoding / Jaya Jaya Institut.

### Setup Environment
Proyek ini dikembangkan menggunakan Python versi 3.10.

1. Membuat dan mengaktifkan virtual environment:
   - Windows (Command Prompt / PowerShell):
     python -m venv env
     .\env\Scripts\activate
   - Linux/macOS:
     python3 -m venv env
     source env/bin/activate

2. Menginstal seluruh dependensi pustaka:
   pip install -r requirements.txt

### Struktur Direktori
submission-akhir/
├── model/
│   ├── model_random_forest.joblib
│   ├── scaler.joblib
│   └── label_encoder.joblib
├── notebook.ipynb
├── clean_students_performance.csv
├── student_performance.db
├── app.py
├── requirements.txt
├── README.md
├── lxng_vr-dashboard.png
├── lxng_vr-video.mp4
└── metabase.db.mv.db

---

## Business Dashboard
Business Dashboard dibangun menggunakan Metabase untuk membantu manajemen kampus Jaya Jaya Institut memonitor performa akademik dan administratif mahasiswa secara terukur:
- Kartu Metrik KPI: Menampilkan ringkasan total mahasiswa, rasio kelulusan (Graduation Rate), total mahasiswa berstatus Dropout, serta rasio kelancaran pembayaran SPP.
- Analisis Finansial vs Retensi: Visualisasi hubungan antara kelancaran pembayaran SPP (Tuition fees up to date) dan status kelulusan, yang membuktikan kendala finansial merupakan pemicu utama kegagalan studi.
- Analisis Capaian Akademik: Grafik distribusi kelulusan SKS semester 1 dan semester 2 (Curricular units approved) terhadap status retensi mahasiswa.
- Distribusi Bantuan Finansial: Komparasi keberhasilan studi antara mahasiswa penerima beasiswa (Scholarship holder) dengan non-beasiswa.

Informasi Akses Dashboard Metabase:
- URL Dashboard: http://localhost:3000
- Email / Username: galangdavaa@gmail.com
- Password: serend4vitz13
- Berkas Basis Data: Konfigurasi dashboard telah diekspor ke dalam berkas metabase.db.mv.db pada direktori proyek.
- Tangkapan Layar: Tersedia pada berkas lxng_vr-dashboard.

Cara Menjalankan Dashboard Metabase Secara Lokal:
- Menggunakan Docker:
  docker run -d -p 3000:3000 -v $(pwd)/metabase.db:/metabase.db -e "MB_DB_FILE=/metabase.db" --name metabase metabase/metabase:v0.46.4
- Menggunakan File JAR:
  set MB_DB_FILE=./metabase.db && java -jar metabase.jar

---

## Menjalankan Sistem Machine Learning
Sistem prediksi dibangun menggunakan antarmuka web interaktif berbasis Streamlit dengan fitur:
- Formulir input data mahasiswa: Mencakup data demografi, administrasi finansial, serta nilai dan SKS semester 1 dan semester 2.
- Deteksi Tingkat Risiko: Menampilkan label status (Dropout, Enrolled, Graduate) disertai skor probabilitas (Risk Score Gauge).
- Fitur What-If Analysis: Simulasi intervensi dinamis untuk menguji perubahan risiko dropout apabila mahasiswa diberikan bantuan pembayaran SPP atau perbaikan SKS kelulusan semester berikutnya.

Tautan Prototype (Streamlit Community Cloud):
- URL Deployment: https://share.streamlit.io/ (Tautan aplikasi publik)

Cara Menjalankan Prototipe Secara Lokal:
Jalankan perintah berikut pada terminal:
streamlit run app.py

Aplikasi akan terbuka otomatis di peramban web pada alamat:
http://localhost:8501

---

## Conclusion
Berdasarkan hasil eksplorasi data dan pemodelan machine learning yang telah dilakukan:
1. Faktor Finansial sebagai Indikator Kritis: Mahasiswa yang memiliki tunggakan SPP (Tuition fees not up to date) memiliki probabilitas dropout yang jauh lebih tinggi dibandingkan mahasiswa yang tertib secara administratif.
2. Performa Semester Pertama Menentukan: Jumlah SKS yang berhasil diluluskan pada Semester 1 dan Semester 2 (Curricular units approved) merupakan fitur penentu paling dominan dalam membedakan mahasiswa yang akan lulus (Graduate) dan putus studi (Dropout). Mahasiswa dengan tingkat kelulusan SKS tahun pertama di bawah batas minimum sangat rentan mengalami dropout.
3. Kinerja Model Machine Learning: Algoritma Random Forest Classifier berhasil memberikan performa optimal dengan metrik:
   - Akurasi Model: ~85% (pada data uji / test set)
   - Weighted F1-Score: ~84%
   - Fitur Dominan: Curricular units 2nd sem (approved), Tuition fees up to date, Curricular units 1st sem (approved), dan Admission grade.

### Rekomendasi Action Items
- Sistem Notifikasi Finansial Fleksibel: Mengintegrasikan sistem pembayaran kampus dengan peringatan dini; mahasiswa yang menunggak SPP segera diarahkan ke program restrukturisasi cicilan, dana darurat, atau skema kerja paruh waktu kampus sebelum dikenakan sanksi akademik.
- Program Pendampingan Akademik Terpadu: Mewajibkan sesi asistensi dan bimbingan belajar tambahan khusus mahasiswa yang meluluskan SKS semester pertama di bawah ambang batas minimal (< 50% SKS terdaftar).
- Revitalisasi Peran Dosen Pembimbing Akademik (DPA): Menggunakan aplikasi Early Warning System ini sebagai instrumen kerja DPA dalam memantau profil risiko anak bimbingan secara berkala di tiap awal semester.
- Optimalisasi Kuota Beasiswa Berkelanjutan: Mengalokasikan proporsi beasiswa tidak hanya pada saat seleksi awal masuk, tetapi juga beasiswa retensi berbasis kebutuhan ekonomi bagi mahasiswa aktif berprestasi yang terancam putus studi di pertengahan semester.