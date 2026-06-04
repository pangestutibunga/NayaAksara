# NayaAksara - Analytics Dashboard

**Coding Camp 2026 | CC26-PSU374 | Data Scientist Track**

Dashboard Streamlit untuk menyajikan analisis data dan hasil evaluasi model CNN
proyek NayaAksara: Smart AI for Cultural Literacy and Adaptive Learning.

---

## Research Questions yang Dijawab

**RQ1:** Bagaimana arsitektur CNN dapat dioptimasi untuk memberikan penilaian
akurasi yang tinggi pada variasi tulisan tangan aksara Jawa?

**RQ2:** Sejauh mana teknik Data Augmentation ekstrem mampu meningkatkan
ketahanan model terhadap variasi kualitas input foto di dunia nyata, seperti
pencahayaan buruk, distorsi kamera, dan proporsi tulisan yang tidak standar?

---

## Halaman Dashboard

| Halaman | Konten |
|---|---|
| Overview | Ringkasan metrik dataset dan model |
| Distribusi Dataset | Jumlah gambar per kelas, train/val/test |
| Kualitas Gambar | Analisis brightness dan contrast (fondasi RQ2) |
| Performa Model CNN | Kurva training, F1, confusion matrix (RQ1) |
| Robustness dan Augmentasi | Hubungan kualitas-performa, korelasi (RQ2) |
| Kesimpulan | Jawaban kedua RQ dan rekomendasi |

---

## Struktur File

Semua file berikut harus berada dalam **satu folder yang sama** dengan `dashboard.py`:

```
dashboard.py                       <- kode utama
requirements.txt                   <- dependensi Python
README.md                          <- dokumentasi ini

# Dari hasil notebook (file_dari_notebook.zip):
distribusi_kelas.csv
metadata_dataset.csv
label_mapping.csv
quality_analysis.csv
quality_performance_merged.csv
rq_summary.csv
distribusi_dataset.png
kualitas_gambar.png
sampel_kelas.png
augmentasi_preview.png
training_performance_curves.png
f1_per_kelas.png

# Dari AI Engineer (opsional, ditampilkan jika tersedia):
confusion_matrix.png
```

---

## Cara Menjalankan Secara Lokal

### 1. Persiapan lingkungan

Pastikan Python 3.9 atau lebih baru sudah terpasang.

```bash
python --version
```

### 2. Buat virtual environment (disarankan)

```bash
python -m venv venv
```

Aktifkan virtual environment:

- **Windows:**
  ```bash
  venv\Scripts\activate
  ```
- **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```

### 3. Pasang dependensi

```bash
pip install -r requirements.txt
```

### 4. Pindah ke folder dashboard

```bash
cd /path/ke/folder/dashboard
```

### 5. Jalankan dashboard

```bash
streamlit run dashboard.py
```

Browser akan terbuka otomatis di `http://localhost:8501`.
Jika tidak terbuka, buka manual di browser dan ketik alamat tersebut.

---

## Cara Deploy ke Streamlit Cloud

### 1. Buat repository GitHub

Buat repository baru di GitHub (dapat bersifat publik atau privat).

### 2. Upload semua file ke repository

Upload seluruh file yang ada di folder ini ke repository GitHub:
- `dashboard.py`
- `requirements.txt`
- Semua file CSV dan PNG

Pastikan semua file berada di **root (akar) repository**, bukan di dalam subfolder.

### 3. Buat akun Streamlit Cloud

Buka [share.streamlit.io](https://share.streamlit.io) dan login menggunakan akun GitHub.

### 4. Deploy aplikasi

1. Klik **"New app"**
2. Pilih repository yang sudah dibuat
3. Pada kolom **"Main file path"**, isi dengan: `dashboard.py`
4. Klik **"Deploy"**

Proses deployment berlangsung 1-3 menit. Setelah selesai, dashboard dapat diakses
melalui URL publik yang diberikan Streamlit Cloud.

---

## Pembaruan Dashboard

Setiap perubahan yang di-push ke branch utama repository GitHub akan
secara otomatis memperbarui dashboard yang sudah di-deploy.

---

## Menambahkan Confusion Matrix

Jika file `confusion_matrix.png` tersedia dari AI Engineer:

1. Salin file tersebut ke folder yang sama dengan `dashboard.py`
2. Confusion matrix akan otomatis tampil di halaman **Performa Model CNN**, tab **Confusion Matrix**

---

## Catatan

- Dashboard membaca data dari file CSV dan PNG secara lokal
- Semua data sudah di-cache (`@st.cache_data`) untuk performa yang lebih baik
- Jika ada file CSV yang tidak ditemukan, dashboard akan menampilkan pesan error
  beserta nama file yang kurang
- Dashboard tidak memerlukan koneksi internet setelah dependensi terpasang,
  kecuali saat di-deploy ke Streamlit Cloud

---

*NayaAksara — Coding Camp 2026 | CC26-PSU374*
