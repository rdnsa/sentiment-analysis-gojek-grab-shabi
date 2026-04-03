# Analisis Sentimen Gojek vs Grab (Data Google Play)

Project ini dipakai untuk membandingkan opini publik tentang Gojek dan Grab dari review aplikasi di Google Play Store.

Script utama ada di file gojek_grab_sentiment.py.

## Ringkasan Alur

1. Ambil review aplikasi dari Google Play:
   - Gojek: com.gojek.app
   - Grab: com.grabtaxi.passenger
2. Target data:
   - 1500 review Gojek
   - 1500 review Grab
   - Total target 3000 data
3. Kolom data utama:
   - date
   - username
   - content
   - brand
   - source
4. Bersihkan teks:
   - lowercase
   - hapus URL
   - hapus mention
   - hapus hashtag
5. Jalankan analisis sentimen dengan model HuggingFace.
6. Tampilkan perbandingan sentimen per brand.
7. Simpan hasil ke CSV dan buat grafik bar.
8. Bonus: tampilkan 10 kata paling sering per sentimen per brand.

## Kebutuhan

- Python 3.10+
- Internet aktif

## Install Dependency

Jalankan di folder project:

```powershell
.\.venv\Scripts\python.exe -m pip install pandas matplotlib transformers torch google-play-scraper
```

Kalau belum pakai virtual environment:

```powershell
python -m pip install pandas matplotlib transformers torch google-play-scraper
```

## Cara Menjalankan

### Opsi 1 (pakai venv)

```powershell
.\.venv\Scripts\python.exe .\gojek_grab_sentiment.py
```

### Opsi 2 (pakai Python global)

```powershell
python .\gojek_grab_sentiment.py
```

## Output yang Dihasilkan

Setelah script selesai, file berikut akan dibuat:

- playstore_reviews_raw_gojek_grab.csv
- playstore_reviews_processed_sentiment.csv
- playstore_sentiment_distribution_gojek_vs_grab.png

Di console juga akan tampil:

- Tabel perbandingan sentimen (jumlah dan persentase)
- Ringkasan insight
- Top 10 kata paling sering per sentimen per brand

## Catatan Penting

- Run pertama biasanya lebih lama karena model HuggingFace diunduh dulu.
- Jumlah data bisa kurang dari target jika review baru dari suatu aplikasi tidak cukup.
- Script sudah menambahkan fallback ke review Most Relevant jika data Newest belum cukup.

## Struktur Project

- gojek_grab_sentiment.py: script utama end-to-end
- README.md: panduan menjalankan project