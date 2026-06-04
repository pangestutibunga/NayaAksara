"""
NayaAksara - Analytics Dashboard
Coding Camp 2026 | CC26-PSU374

Menjawab dua Research Questions:
RQ1: Bagaimana arsitektur CNN dapat dioptimasi untuk akurasi tinggi pada
     variasi tulisan tangan aksara Jawa?
RQ2: Sejauh mana Data Augmentation ekstrem mampu meningkatkan ketahanan
     model terhadap variasi kualitas input foto di dunia nyata?
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ── Konfigurasi halaman ──────────────────────────────────────
st.set_page_config(
    page_title="NayaAksara Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS sederhana ────────────────────────────────────────────
st.markdown("""
<style>
    footer {visibility: hidden;}
    .block-container {padding-top: 1.5rem;}
    [data-testid="metric-container"] {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 12px 16px;
    }
    .kotak-info {
        background: #e8f4fd;
        border-left: 4px solid #1a73e8;
        border-radius: 0 6px 6px 0;
        padding: 12px 16px;
        margin: 8px 0 16px 0;
        font-size: 14px;
        line-height: 1.6;
    }
    .kotak-peringatan {
        background: #fff8e1;
        border-left: 4px solid #f9a825;
        border-radius: 0 6px 6px 0;
        padding: 12px 16px;
        margin: 8px 0 16px 0;
        font-size: 14px;
        line-height: 1.6;
    }
    .kotak-hasil {
        background: #e8f5e9;
        border-left: 4px solid #2e7d32;
        border-radius: 0 6px 6px 0;
        padding: 12px 16px;
        margin: 8px 0 16px 0;
        font-size: 14px;
        line-height: 1.6;
    }
    h3 {margin-top: 1.5rem;}
</style>
""", unsafe_allow_html=True)


# ── Loader data ──────────────────────────────────────────────

from pathlib import Path

BASE_DIR = Path(__file__).parent

@st.cache_data
def load_distribusi():
    return pd.read_csv(BASE_DIR / "distribusi_kelas.csv")

@st.cache_data
def load_metadata():
    df = pd.read_csv(BASE_DIR / "metadata_dataset.csv")
    return df.iloc[0].to_dict()

@st.cache_data
def load_quality():
    return pd.read_csv(BASE_DIR / "quality_analysis.csv")

@st.cache_data
def load_quality_performance():
    return pd.read_csv(BASE_DIR / "quality_performance_merged.csv")

@st.cache_data
def load_rq_summary():
    return pd.read_csv(BASE_DIR / "rq_summary.csv")

@st.cache_data
def load_label_mapping():
    return pd.read_csv(BASE_DIR / "label_mapping.csv")

# Load semua data
try:
    dist_df     = load_distribusi()
    meta        = load_metadata()
    quality_df  = load_quality()
    qp_df       = load_quality_performance()
    rq_df       = load_rq_summary()
    label_df    = load_label_mapping()
except FileNotFoundError as e:
    st.error(f"File tidak ditemukan: {e}")
    st.info("Pastikan semua file CSV ada di folder yang sama dengan dashboard.py")
    st.stop()

# Nilai ringkas dari rq_summary
rq1 = rq_df[rq_df["research_question"] == "RQ1"].iloc[0]
rq2 = rq_df[rq_df["research_question"] == "RQ2"].iloc[0]

best_val_acc   = float(rq1["best_val_accuracy"])
best_epoch     = int(rq1["best_epoch"])
total_epoch    = int(rq1["total_epoch"])
avg_f1         = float(rq1["avg_f1_test"])
avg_precision  = float(rq1["avg_precision_test"])
avg_recall     = float(rq1["avg_recall_test"])
kelas_terlemah = str(rq1["kelas_terlemah"])
kelas_terbaik  = str(rq1["kelas_terbaik"])
pct_bright     = float(rq2["pct_high_brightness"])
corr_bright_f1 = float(rq2["corr_brightness_f1"])
corr_cont_f1   = float(rq2["corr_contrast_f1"])


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### NayaAksara")
    st.caption("Smart AI for Cultural Literacy\nand Adaptive Learning")
    st.divider()

    halaman = st.radio(
        "Navigasi",
        options=[
            "Overview",
            "Distribusi Dataset",
            "Kualitas Gambar",
            "Performa Model CNN",
            "Robustness dan Augmentasi",
            "Kesimpulan",
        ],
        label_visibility="collapsed"
    )
    st.divider()

    # Ringkasan metrik di sidebar
    st.markdown("**Metrik Utama**")
    st.metric("Best Val Accuracy",  f"{best_val_acc*100:.2f}%", f"Epoch {best_epoch}")
    st.metric("Avg F1-Score",       f"{avg_f1*100:.2f}%",       "20 kelas")
    st.metric("Kelas Terlemah",     kelas_terlemah.upper())
    st.metric("Gambar Corrupt",     int(meta["gambar_corrupt"]))
    st.divider()
    st.caption("Coding Camp 2026 | CC26-PSU374")


# ════════════════════════════════════════════════════════════
# HALAMAN: OVERVIEW
# ════════════════════════════════════════════════════════════
if halaman == "Overview":
    st.title("NayaAksara - Analytics Dashboard")
    st.markdown(
        "Dashboard ini menyajikan analisis data dan hasil evaluasi model CNN "
        "untuk proyek NayaAksara, yaitu aplikasi pembelajaran aksara Jawa berbasis AI."
    )

    st.markdown("""
    <div class='kotak-info'>
    <b>Research Question 1</b><br>
    Bagaimana arsitektur CNN dapat dioptimasi untuk memberikan penilaian akurasi yang
    tinggi pada variasi tulisan tangan aksara Jawa?
    </div>
    <div class='kotak-peringatan'>
    <b>Research Question 2</b><br>
    Sejauh mana teknik Data Augmentation ekstrem mampu meningkatkan ketahanan model
    terhadap variasi kualitas input foto di dunia nyata, seperti pencahayaan buruk,
    distorsi kamera, dan proporsi tulisan yang tidak standar?
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Baris 1: Statistik dataset
    st.markdown("**Statistik Dataset**")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Kelas Aksara",         int(meta["jumlah_kelas"]))
    c2.metric("Total Gambar Train",   f"{int(meta['total_train']):,}")
    c3.metric("Total Gambar Val",     f"{int(meta['total_val']):,}")
    c4.metric("Total Gambar Test",    f"{int(meta['total_test']):,}")
    c5.metric("Total Keseluruhan",    f"{int(meta['total_keseluruhan']):,}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Baris 2: Statistik model
    st.markdown("**Hasil Model CNN**")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Best Val Accuracy",    f"{best_val_acc*100:.2f}%",   f"Epoch {best_epoch}")
    m2.metric("Avg Precision",        f"{avg_precision*100:.2f}%")
    m3.metric("Avg Recall",           f"{avg_recall*100:.2f}%")
    m4.metric("Avg F1-Score",         f"{avg_f1*100:.2f}%")
    m5.metric("Total Epoch Training", total_epoch)

    st.divider()

    # Dua chart ringkas
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Distribusi Gambar per Split**")
        pivot = dist_df.groupby("Split")["Jumlah"].sum().reset_index()
        pivot.columns = ["Split", "Jumlah"]
        fig = px.pie(
            pivot, values="Jumlah", names="Split",
            color_discrete_map={"train": "#3498db", "val": "#2ecc71", "test": "#e74c3c"},
            hole=0.4
        )
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**F1-Score per Kelas Aksara**")
        qp_sorted = qp_df.sort_values("f1-score")
        colors    = ["#c0392b" if v < 0.9 else "#e67e22" if v < 0.95 else "#27ae60"
                     for v in qp_sorted["f1-score"]]
        fig2 = go.Figure(go.Bar(
            x=qp_sorted["f1-score"] * 100,
            y=qp_sorted["class"],
            orientation="h",
            marker_color=colors,
        ))
        fig2.add_vline(x=avg_f1 * 100, line_dash="dash", line_color="navy",
                       annotation_text=f"Avg {avg_f1*100:.1f}%")
        fig2.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(range=[75, 103], ticksuffix="%"),
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig2, use_container_width=True)


# ════════════════════════════════════════════════════════════
# HALAMAN: DISTRIBUSI DATASET
# ════════════════════════════════════════════════════════════
elif halaman == "Distribusi Dataset":
    st.title("Distribusi Dataset")
    st.markdown(
        "Analisis jumlah dan sebaran gambar pada setiap kelas aksara Jawa "
        "untuk tiga split: train, val, dan test."
    )

    # Pivot table
    pivot = dist_df.pivot(index="Label", columns="Split", values="Jumlah")
    for col in ["train", "val", "test"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["train", "val", "test"]]
    pivot["total"] = pivot.sum(axis=1)

    # Metrik ringkas
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Gambar",         f"{int(pivot['total'].sum()):,}")
    c2.metric("Kelas Terbanyak (Train)", f"{pivot['train'].idxmax()} ({pivot['train'].max()})")
    c3.metric("Kelas Tersedikit (Train)",f"{pivot['train'].idxmin()} ({pivot['train'].min()})")
    c4.metric("Rasio Imbalance",       f"{meta['rasio_imbalance']:.4f}x")

    st.divider()

    tab1, tab2 = st.tabs(["Grafik Distribusi", "Tabel Detail"])

    with tab1:
        # Stacked bar
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=pivot.index, y=pivot["train"],
            name="Train", marker_color="#3498db",
            text=pivot["train"], textposition="inside", textfont=dict(size=9)
        ))
        fig.add_trace(go.Bar(
            x=pivot.index, y=pivot["val"],
            name="Val", marker_color="#2ecc71",
            text=pivot["val"], textposition="inside", textfont=dict(size=9)
        ))
        fig.add_trace(go.Bar(
            x=pivot.index, y=pivot["test"],
            name="Test", marker_color="#e74c3c",
            text=pivot["test"], textposition="inside", textfont=dict(size=9)
        ))
        fig.update_layout(
            barmode="stack", height=420,
            xaxis_title="Kelas Aksara",
            yaxis_title="Jumlah Gambar",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.2),
            margin=dict(l=0, r=0, t=20, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class='kotak-info'>
        <b>Insight Distribusi:</b><br>
        Dataset terdiri dari 4.196 gambar yang terbagi ke tiga split dengan proporsi
        train 69,8% / val 14,8% / test 15,5%. Distribusi antar kelas sangat seimbang
        dengan rasio imbalance hanya 1.0634x (kisaran 142-151 gambar per kelas pada
        train set), sehingga model tidak berisiko bias terhadap kelas tertentu.
        </div>
        """, unsafe_allow_html=True)

        # Tampilkan gambar distribusi dari notebook jika ada
        if os.path.exists("distribusi_dataset.png"):
            st.markdown("**Visualisasi dari Notebook (per Split)**")
            st.image("distribusi_dataset.png", use_column_width=True)

    with tab2:
        st.markdown("**Tabel Jumlah Gambar per Kelas**")
        tbl = pivot.reset_index()
        tbl.columns = ["Kelas", "Train", "Val", "Test", "Total"]
        st.dataframe(
            tbl.style.highlight_max(subset=["Train"], color="#d4edda")
                     .highlight_min(subset=["Train"], color="#f8d7da"),
            use_container_width=True, hide_index=True
        )

        st.markdown("**Tampilan Sampel per Kelas**")
        if os.path.exists("sampel_kelas.png"):
            st.image("sampel_kelas.png", use_column_width=True,
                     caption="Satu sampel gambar dari setiap kelas aksara Jawa (128x128 grayscale)")


# ════════════════════════════════════════════════════════════
# HALAMAN: KUALITAS GAMBAR
# ════════════════════════════════════════════════════════════
elif halaman == "Kualitas Gambar":
    st.title("Analisis Kualitas Gambar")
    st.markdown(
        "Analisis brightness dan contrast gambar per kelas aksara sebagai "
        "fondasi untuk menjawab Research Question 2."
    )

    st.markdown("""
    <div class='kotak-info'>
    <b>Relevansi ke RQ2:</b> Analisis ini mengukur kondisi awal dataset sebelum augmentasi.
    Jika dataset hanya berisi gambar berkualitas tinggi (brightness seragam, kontras rendah),
    maka augmentasi ekstrem sangat diperlukan agar model tahan terhadap kondisi foto dunia nyata.
    </div>
    """, unsafe_allow_html=True)

    # Ringkasan kualitas
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rata-rata Brightness",  f"{quality_df['brightness'].mean():.1f} / 255")
    c2.metric("Rata-rata Contrast",    f"{quality_df['contrast'].mean():.1f}")
    c3.metric("Gambar Brightness Tinggi", f"{pct_bright:.1f}%", "> 200 dari 255")
    c4.metric("Total Gambar Dianalisis", f"{len(quality_df):,}")

    st.divider()
    tab1, tab2 = st.tabs(["Visualisasi Kualitas", "Detail per Kelas"])

    with tab1:
        if os.path.exists("kualitas_gambar.png"):
            st.image("kualitas_gambar.png", use_column_width=True,
                     caption="Distribusi brightness dan contrast seluruh gambar train set")

        st.markdown("""
        <div class='kotak-peringatan'>
        <b>Temuan Penting untuk RQ2:</b><br>
        Sebesar 98.4% gambar memiliki brightness tinggi (di atas 200 dari 255) karena
        tulisan tangan di atas kertas putih. Kontras relatif rendah (rata-rata 38.3).
        Dataset belum merepresentasikan kondisi dunia nyata seperti pencahayaan buruk,
        blur kamera, atau distorsi. Hal ini menjadikan Data Augmentation ekstrem sangat
        krusial untuk meningkatkan ketahanan model.
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        # Tabel kualitas per kelas
        q_summary = quality_df.groupby("aksara").agg(
            avg_brightness=("brightness", "mean"),
            std_brightness=("brightness", "std"),
            avg_contrast=("contrast", "mean"),
            std_contrast=("contrast", "std"),
            n_gambar=("filename", "count")
        ).round(2).reset_index()
        q_summary.columns = ["Kelas", "Avg Brightness", "Std Brightness",
                              "Avg Contrast", "Std Contrast", "N Gambar"]
        st.dataframe(q_summary, use_container_width=True, hide_index=True)

        # Bar chart brightness per kelas
        qs = quality_df.groupby("aksara")["brightness"].mean().sort_values().reset_index()
        qs.columns = ["Kelas", "Avg Brightness"]
        fig = px.bar(
            qs, x="Avg Brightness", y="Kelas", orientation="h",
            color="Avg Brightness",
            color_continuous_scale=["#e74c3c", "#f39c12", "#27ae60"],
            title="Rata-Rata Brightness per Kelas Aksara"
        )
        fig.add_vline(x=qs["Avg Brightness"].mean(), line_dash="dash", line_color="navy",
                      annotation_text=f"Rata-rata: {qs['Avg Brightness'].mean():.1f}")
        fig.update_layout(
            height=420, margin=dict(l=0, r=0, t=40, b=0),
            plot_bgcolor="rgba(0,0,0,0)", showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════
# HALAMAN: PERFORMA MODEL CNN (RQ1)
# ════════════════════════════════════════════════════════════
elif halaman == "Performa Model CNN":
    st.title("Performa Model CNN")
    st.markdown("""
    <div class='kotak-info'>
    <b>Research Question 1:</b> Bagaimana arsitektur CNN dapat dioptimasi untuk
    memberikan penilaian akurasi yang tinggi pada variasi tulisan tangan aksara Jawa?
    </div>
    """, unsafe_allow_html=True)

    # Metrik utama
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Val Accuracy",  f"{best_val_acc*100:.2f}%",   f"Epoch {best_epoch}")
    c2.metric("Avg Precision",      f"{avg_precision*100:.2f}%")
    c3.metric("Avg Recall",         f"{avg_recall*100:.2f}%")
    c4.metric("Avg F1-Score",       f"{avg_f1*100:.2f}%")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["Kurva Training", "F1 per Kelas", "Confusion Matrix"])

    # ── Tab 1: Kurva Training
    with tab1:
        if os.path.exists("training_performance_curves.png"):
            st.image("training_performance_curves.png", use_column_width=True,
                     caption="Kurva accuracy dan loss selama 48 epoch training")

        st.markdown("""
        <div class='kotak-hasil'>
        <b>Interpretasi Kurva Training (RQ1):</b><br>
        Val accuracy secara konsisten lebih tinggi dari train accuracy sepanjang
        48 epoch — gap rata-rata 12.6%. Ini terjadi karena augmentasi aktif
        hanya saat training sehingga train set lebih sulit. Val loss terus turun
        tanpa tanda overfitting. Model belum konvergen di epoch 48, sehingga ada
        potensi peningkatan dengan training lanjutan.
        </div>
        """, unsafe_allow_html=True)

        # Tabel milestone
        milestone_data = {
            "Epoch"          : [1, 10, 20, 30, 40, 47, 48],
            "Val Accuracy (%)" : [77.54, 81.26, 82.23, 84.98, 87.56, 87.88, 86.59],
            "Val Loss"         : [1.6244, 1.3133, 1.2283, 1.1417, 1.0978, 1.0813, 1.0892],
        }
        st.markdown("**Milestone Training**")
        st.dataframe(pd.DataFrame(milestone_data), use_container_width=True, hide_index=True)

    # ── Tab 2: F1 per Kelas
    with tab2:
        if os.path.exists("f1_per_kelas.png"):
            st.image("f1_per_kelas.png", use_column_width=True,
                     caption="F1-Score per kelas aksara — merah <90%, kuning 90-95%, hijau >95%")

        # Tabel classification report
        st.markdown("**Tabel Classification Report**")
        tbl = qp_df[["class","precision","recall","f1-score","support"]].copy()
        tbl = tbl.sort_values("f1-score")
        tbl["precision"] = (tbl["precision"] * 100).round(1).astype(str) + "%"
        tbl["recall"]    = (tbl["recall"]    * 100).round(1).astype(str) + "%"
        tbl["f1-score"]  = (tbl["f1-score"]  * 100).round(1).astype(str) + "%"
        tbl.columns      = ["Kelas", "Precision", "Recall", "F1-Score", "Support"]
        st.dataframe(tbl, use_container_width=True, hide_index=True)

        # Kelas bermasalah
        weak = qp_df[qp_df["f1-score"] < 0.90].sort_values("f1-score")
        if len(weak) > 0:
            st.markdown("**Kelas dengan F1-Score di Bawah 90%**")
            for _, row in weak.iterrows():
                f1   = round(row["f1-score"]  * 100, 1)
                prec = round(row["precision"] * 100, 1)
                rec  = round(row["recall"]    * 100, 1)
                issue = (
                    "Precision rendah — model sering salah mengklasifikasikan kelas lain sebagai kelas ini."
                    if prec < rec else
                    "Recall rendah — model sering melewatkan kelas ini dan memprediksinya sebagai kelas lain."
                )
                st.markdown(f"""
                <div class='kotak-peringatan'>
                <b>Aksara '{row['class'].upper()}'</b> — F1: {f1}% | Precision: {prec}% | Recall: {rec}%<br>
                {issue}
                </div>
                """, unsafe_allow_html=True)

    # ── Tab 3: Confusion Matrix
    with tab3:
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", use_column_width=True,
                     caption="Confusion Matrix — NayaAksara Fine-Tuned (sumber: AI Engineer)")
        else:
            st.info(
                "File confusion_matrix.png tidak ditemukan di folder dashboard. "
                "Salin file tersebut ke folder yang sama dengan dashboard.py."
            )

        st.markdown("""
        <div class='kotak-hasil'>
        <b>Interpretasi Confusion Matrix (RQ1):</b><br>
        Tiga kelas mencapai akurasi sempurna: ba, ma, dan ya (0 kesalahan).
        Kelas 'na' memiliki misklasifikasi terbanyak — 3 gambar diprediksi sebagai 'ka'
        karena kemiripan struktur visual. Kelas 'dha' memiliki 5 total kesalahan yang
        tersebar ke 4 kelas berbeda, mengindikasikan bentuk visual yang ambigu.
        Secara keseluruhan, overall accuracy mencapai 94.6% dari 650 sampel test.
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# HALAMAN: ROBUSTNESS DAN AUGMENTASI (RQ2)
# ════════════════════════════════════════════════════════════
elif halaman == "Robustness dan Augmentasi":
    st.title("Robustness dan Data Augmentasi")
    st.markdown("""
    <div class='kotak-peringatan'>
    <b>Research Question 2:</b> Sejauh mana teknik Data Augmentation ekstrem mampu
    meningkatkan ketahanan model terhadap variasi kualitas input foto di dunia nyata?
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "Preview Augmentasi",
        "Kualitas vs Performa",
        "Analisis Korelasi"
    ])

    # ── Tab 1: Preview Augmentasi
    with tab1:
        st.markdown("**Perbandingan Augmentasi Ringan vs Ekstrem**")

        col1, col2 = st.columns(2)
        with col1:
            aug_desc_ringan = pd.DataFrame([
                {"Transformasi": "Rotasi",    "Parameter": "10 derajat"},
                {"Transformasi": "Shift",     "Parameter": "10%"},
                {"Transformasi": "Zoom",      "Parameter": "10%"},
            ])
            st.markdown("**Augmentasi Ringan (Baseline)**")
            st.dataframe(aug_desc_ringan, use_container_width=True, hide_index=True)

        with col2:
            aug_desc_ekstrem = pd.DataFrame([
                {"Transformasi": "Rotasi",      "Parameter": "30 derajat",    "Kondisi Nyata": "Proporsi tulisan tidak standar"},
                {"Transformasi": "Shift",        "Parameter": "20%",           "Kondisi Nyata": "Framing tidak ideal"},
                {"Transformasi": "Zoom",         "Parameter": "30%",           "Kondisi Nyata": "Distorsi kamera"},
                {"Transformasi": "Brightness",   "Parameter": "0.3x - 1.8x",  "Kondisi Nyata": "Pencahayaan buruk / overexposure"},
                {"Transformasi": "Shear",        "Parameter": "0.2",           "Kondisi Nyata": "Kemiringan perspektif"},
            ])
            st.markdown("**Augmentasi Ekstrem (Robustness)**")
            st.dataframe(aug_desc_ekstrem, use_container_width=True, hide_index=True)

        if os.path.exists("augmentasi_preview.png"):
            st.image("augmentasi_preview.png", use_column_width=True,
                     caption="Baris atas: Augmentasi ringan. Baris bawah: Augmentasi ekstrem.")

        st.markdown("""
        <div class='kotak-info'>
        <b>Bukti Efektivitas Augmentasi:</b><br>
        Augmentasi aktif hanya saat training menghasilkan gap val-train accuracy sebesar
        12.6%. Train accuracy lebih rendah karena model belajar dari kondisi yang lebih
        sulit. Val accuracy lebih tinggi membuktikan model berhasil menggeneralisasi
        ke data bersih yang tidak pernah dilihat sebelumnya.
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 2: Kualitas vs Performa
    with tab2:
        st.markdown("**Hubungan Kualitas Gambar dengan Performa Model per Kelas**")

        # Scatter: brightness vs f1-score
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                "Brightness vs F1-Score",
                "Contrast vs F1-Score"
            )
        )

        colors_dot = [
            "#c0392b" if v < 0.9 else "#e67e22" if v < 0.95 else "#27ae60"
            for v in qp_df["f1-score"]
        ]

        # Brightness
        fig.add_trace(go.Scatter(
            x=qp_df["avg_brightness"],
            y=qp_df["f1-score"] * 100,
            mode="markers+text",
            text=qp_df["class"],
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(size=10, color=colors_dot),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Avg Brightness: %{x:.1f}<br>"
                "F1-Score: %{y:.1f}%<extra></extra>"
            )
        ), row=1, col=1)

        # Contrast
        fig.add_trace(go.Scatter(
            x=qp_df["avg_contrast"],
            y=qp_df["f1-score"] * 100,
            mode="markers+text",
            text=qp_df["class"],
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(size=10, color=colors_dot),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Avg Contrast: %{x:.1f}<br>"
                "F1-Score: %{y:.1f}%<extra></extra>"
            )
        ), row=1, col=2)

        fig.update_layout(
            height=420, showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=40, b=0)
        )
        fig.update_xaxes(gridcolor="rgba(128,128,128,0.15)")
        fig.update_yaxes(
            gridcolor="rgba(128,128,128,0.15)",
            ticksuffix="%",
            range=[75, 102]
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class='kotak-peringatan'>
        <b>Batas Efektivitas Augmentasi:</b><br>
        Kelas 'na' dan 'dha' tetap menjadi kelas terlemah meskipun sudah diaugmentasi.
        Keduanya memiliki tingkat kontras sedikit lebih rendah dibanding kelas lain,
        namun faktor utama kelemahannya adalah kemiripan struktur visual dengan kelas
        lain — bukan semata-mata kualitas gambar. Ini menunjukkan batas kemampuan
        augmentasi: efektif meningkatkan robustness terhadap variasi kualitas foto,
        namun tidak dapat mengatasi ambiguitas visual antar karakter.
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 3: Analisis Korelasi
    with tab3:
        st.markdown("**Korelasi Kualitas Gambar dengan Performa Model**")

        c1, c2 = st.columns(2)
        c1.metric(
            "Korelasi Brightness - F1",
            f"{corr_bright_f1:.4f}",
            "Positif lemah"
        )
        c2.metric(
            "Korelasi Contrast - F1",
            f"{corr_cont_f1:.4f}",
            "Negatif lemah"
        )

        st.markdown("""
        <div class='kotak-hasil'>
        <b>Interpretasi Korelasi untuk RQ2:</b><br>
        Korelasi brightness terhadap F1-score sebesar 0.2156 (positif lemah) menunjukkan
        kelas dengan gambar sedikit lebih gelap cenderung memiliki F1-score lebih rendah,
        namun hubungannya tidak kuat. Korelasi contrast terhadap F1-score sebesar -0.148
        (negatif lemah) menunjukkan kontras tinggi tidak secara otomatis meningkatkan
        performa. Kesimpulan: kualitas gambar bukan faktor dominan penentu performa —
        kompleksitas visual aksara lebih berpengaruh daripada brightness atau contrast.
        </div>
        """, unsafe_allow_html=True)

        # Tabel lengkap merged
        st.markdown("**Data Kualitas dan Performa per Kelas**")
        tbl = qp_df[["class","avg_brightness","avg_contrast","f1-score","support"]].copy()
        tbl["f1-score"]      = (tbl["f1-score"] * 100).round(1)
        tbl["avg_brightness"] = tbl["avg_brightness"].round(1)
        tbl["avg_contrast"]   = tbl["avg_contrast"].round(1)
        tbl = tbl.sort_values("f1-score")
        tbl.columns = ["Kelas", "Avg Brightness", "Avg Contrast", "F1-Score (%)", "Support"]
        st.dataframe(tbl, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════
# HALAMAN: KESIMPULAN
# ════════════════════════════════════════════════════════════
elif halaman == "Kesimpulan":
    st.title("Kesimpulan dan Jawaban Research Questions")

    # RQ1
    st.subheader("Jawaban Research Question 1")
    st.markdown(
        "*Bagaimana arsitektur CNN dapat dioptimasi untuk memberikan penilaian "
        "akurasi yang tinggi pada variasi tulisan tangan aksara Jawa?*"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='kotak-hasil'>
        <b>Hasil yang Dicapai</b><br>
        Model CNN berhasil mencapai val accuracy 87.88% pada epoch 47 dan overall accuracy
        94.6% berdasarkan confusion matrix terhadap 650 gambar test. Sebanyak 17 dari
        20 kelas aksara mencapai F1-Score di atas 93%, dengan 3 kelas sempurna (ba, ma, ya).
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='kotak-peringatan'>
        <b>Ruang untuk Optimasi</b><br>
        Model belum konvergen di epoch 48. Potensi peningkatan tersedia dengan melanjutkan
        training ke 80-100 epoch, menambahkan Learning Rate Scheduler, serta
        menambahkan BatchNormalization untuk mempercepat konvergensi di awal training.
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # RQ2
    st.subheader("Jawaban Research Question 2")
    st.markdown(
        "*Sejauh mana teknik Data Augmentation ekstrem mampu meningkatkan ketahanan "
        "model terhadap variasi kualitas input foto di dunia nyata?*"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='kotak-hasil'>
        <b>Efektivitas yang Terbukti</b><br>
        Augmentasi ekstrem terbukti efektif: val accuracy konsisten lebih tinggi dari
        train accuracy (gap 12.6%) karena model belajar dari kondisi yang lebih sulit.
        Tidak ada overfitting — val loss turun konsisten sepanjang 48 epoch. Overall
        accuracy 94.6% pada test set membuktikan model cukup robust terhadap variasi.
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='kotak-peringatan'>
        <b>Batas Efektivitas</b><br>
        Kelas dha dan na tetap menjadi kelas terlemah dengan F1-Score di bawah 91%.
        Augmentasi efektif mengatasi variasi foto namun tidak dapat mengatasi ambiguitas
        visual antar karakter. Untuk kondisi ekstrem seperti foto gelap atau buram,
        diperlukan augmentasi brightness dan blur yang belum diterapkan.
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Tabel ringkasan kedua RQ
    st.subheader("Ringkasan Metrik")
    ringkasan = pd.DataFrame([
        {
            "Aspek"         : "Best Val Accuracy",
            "Nilai"         : f"{best_val_acc*100:.2f}%",
            "Relevansi"     : "RQ1"
        },
        {
            "Aspek"         : "Overall Accuracy (test)",
            "Nilai"         : "94.6%",
            "Relevansi"     : "RQ1"
        },
        {
            "Aspek"         : "Avg F1-Score (20 kelas)",
            "Nilai"         : f"{avg_f1*100:.2f}%",
            "Relevansi"     : "RQ1"
        },
        {
            "Aspek"         : "Kelas Terbaik",
            "Nilai"         : "ba, ma, ya (100%)",
            "Relevansi"     : "RQ1"
        },
        {
            "Aspek"         : "Kelas Terlemah",
            "Nilai"         : f"{kelas_terlemah.upper()} (84%)",
            "Relevansi"     : "RQ1"
        },
        {
            "Aspek"         : "Gap Train-Val Accuracy",
            "Nilai"         : "12.6%",
            "Relevansi"     : "RQ2"
        },
        {
            "Aspek"         : "Korelasi Brightness - F1",
            "Nilai"         : f"{corr_bright_f1:.4f} (positif lemah)",
            "Relevansi"     : "RQ2"
        },
        {
            "Aspek"         : "Gambar Brightness Tinggi",
            "Nilai"         : f"{pct_bright:.1f}%",
            "Relevansi"     : "RQ2"
        },
        {
            "Aspek"         : "Status Overfitting",
            "Nilai"         : str(rq1.get("status_overfitting", "Tidak ada")),
            "Relevansi"     : "RQ1 & RQ2"
        },
    ])
    st.dataframe(ringkasan, use_container_width=True, hide_index=True)

    st.divider()

    # Rekomendasi
    st.subheader("Rekomendasi untuk Tim")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**AI Engineer**")
        st.markdown("""
        - Lanjutkan training ke 80-100 epoch
        - Implementasikan ReduceLROnPlateau
        - Evaluasi model pada gambar real-world yang baru
        - Tambahkan BatchNormalization pada arsitektur
        """)
    with col2:
        st.markdown("**Data Scientist**")
        st.markdown("""
        - Tambah data asli untuk kelas dha dan na
        - Tambahkan augmentasi brightness dan blur
        - Analisis error pattern kelas ka lebih dalam
        - Monitor distribusi test set secara berkala
        """)
    with col3:
        st.markdown("**Fullstack**")
        st.markdown("""
        - Preprocessing gambar di client side sebelum API
        - Tampilkan confidence score ke pengguna
        - Tambahkan fallback jika confidence di bawah 70%
        - Optimalkan ukuran gambar sebelum dikirim ke server
        """)

    st.divider()
    st.caption("NayaAksara — Coding Camp 2026 | CC26-PSU374 | Data Scientist Track")
