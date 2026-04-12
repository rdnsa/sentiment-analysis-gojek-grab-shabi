# Shabi V2 - Submission Ready Sentiment Analysis

Folder ini adalah versi submission-ready dari flow sebelumnya, disesuaikan untuk memenuhi kriteria utama dan saran penilaian tinggi.

## Ringkasan Kriteria dan Implementasi

### Kriteria 1 - Scraping mandiri (minimal 3.000 sampel)
- File scraping: `scripts/scrape_playstore_reviews.py`
- Sumber data: Google Play Store (independen, bukan dataset publik siap pakai)
- Target default: 10.000 sampel (5.000 per aplikasi)
- Batas minimum tetap terpenuhi saat target diturunkan (misalnya 1.500 per aplikasi = 3.000)

### Kriteria 2 - Ekstraksi fitur dan pelabelan
- File preprocessing + labeling: `scripts/prepare_dataset.py`
- Pembersihan teks: lowercase, hapus URL, mention, hashtag, karakter non-alfanumerik
- Pelabelan 3 kelas:
  - `negative` untuk skor 1-2
  - `neutral` untuk skor 3
  - `positive` untuk skor 4-5

### Kriteria 3 - Algoritma machine learning
- File training: `scripts/train_experiments.py`
- Terdapat 3 skema pelatihan berbeda:
  1. SVM + TF-IDF + split 80/20
  2. Logistic Regression + TF-IDF + split 70/30
  3. BiLSTM + sequence tokenizer + split 80/20 (deep learning)

### Kriteria 4 - Akurasi testing minimal 85%
- Script training otomatis menyimpan metrik akurasi train/test di:
  - `reports/metrics_experiments.csv`
  - `reports/metrics_experiments.json`
- Script juga menandai eksperimen PASS/FAIL untuk ambang 85%.

## Struktur Folder

```text
shabi_v2/
  data/
    raw/
    processed/
  models/
  notebooks/
    01_training_submission.ipynb
  reports/
    figures/
  scripts/
    scrape_playstore_reviews.py
    prepare_dataset.py
    train_experiments.py
    run_inference.py
    run_pipeline.py
  requirements.txt
  README.md
```

## Setup Environment

Dari folder `shabi_v2`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Catatan kompatibilitas:
- Untuk menjalankan eksperimen deep learning (BiLSTM), disarankan Python 3.10-3.12.
- Jika memakai Python 3.13, gunakan mode cepat (`--skip-deep-learning`) atau jalankan deep learning di environment Python 3.12 terpisah.

## Menjalankan Pipeline

### Opsi 1 - Full pipeline otomatis

```powershell
python .\scripts\run_pipeline.py --per-app-target 5000
```

### Opsi 2 - Jalankan per tahap

```powershell
python .\scripts\scrape_playstore_reviews.py --per-app-target 5000
python .\scripts\prepare_dataset.py
python .\scripts\train_experiments.py
python .\scripts\run_inference.py
```

### Opsi cepat (tanpa deep learning)

```powershell
python .\scripts\train_experiments.py --skip-deep-learning
```

## Notebook Submission

Notebook utama:
- `notebooks/01_training_submission.ipynb`

Notebook ini berisi:
- scraping
- preprocessing + labeling
- 3 eksperimen training
- inference dengan output kelas kategorikal

## Output Penting untuk Submission

1. Kode scraping (`.py`):
- `scripts/scrape_playstore_reviews.py`

2. Notebook training (`.ipynb`):
- `notebooks/01_training_submission.ipynb`

3. Requirements:
- `requirements.txt`

4. Dataset hasil scraping (`.csv`):
- `data/raw/playstore_reviews_raw_latest.csv`
- `data/processed/playstore_reviews_processed_latest.csv`

5. Bukti inference (opsional bernilai tambah):
- Output cell inference pada notebook
- Atau output terminal dari `scripts/run_inference.py`

## Checklist Sebelum Zip Submission

- [ ] Notebook sudah dijalankan penuh dari atas ke bawah.
- [ ] Output cell inference terlihat (label kategorikal).
- [ ] File CSV raw dan processed tersedia.
- [ ] `reports/metrics_experiments.csv` sudah terisi hasil pelatihan.
- [ ] Semua file berada dalam satu folder `shabi_v2`, lalu di-zip.
