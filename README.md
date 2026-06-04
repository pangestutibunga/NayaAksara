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

```
dashboard.py                       <- kode utama
requirements.txt                   <- dependensi Python
README.md                          <- dokumentasi ini

# Dari hasil notebook:
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

# Dari AI Engineer:
confusion_matrix.png
classification.csv
training_history_fixed.csv
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

*NayaAksara — Coding Camp 2026 | CC26-PSU374*
