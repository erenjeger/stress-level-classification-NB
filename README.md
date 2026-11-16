# Stress Level Classification — Naive Bayes

Aplikasi Flask untuk pengolahan data stres kerja karyawan dengan enam faktor, Categorical Naive Bayes, dashboard admin, evaluasi model, riwayat submission, dan ekspor Excel.

> Aplikasi ini adalah alat bantu pengolahan data penelitian, bukan alat diagnosis medis atau psikologis.

## Fitur

- Form publik tanpa login untuk 30 pernyataan skala 1–7.
- Import workbook rekap skor agregat enam faktor dari perusahaan, tanpa membutuhkan jawaban per butir.
- Deteksi otomatis baris header workbook pada beberapa baris pertama sheet Excel.
- Pembuatan kategori target dari skor faktor dengan ambang yang dikonfigurasi pada aplikasi.
- Training enam model Categorical Naive Bayes; setiap model memprediksi kategori satu faktor dari kategori lima faktor lainnya.
- Prediksi pada kuesioner publik menggunakan kategori skor lima faktor lain sebagai fitur jika model agregat tersedia.
- Kompatibilitas dengan dataset lama yang berisi jawaban_1–jawaban_30 dan enam kolom label.
- Login admin, dashboard, riwayat, confusion matrix, classification report, dan ekspor Excel.
- Analisis submission dengan filter sumber, batch, dan departemen.
- Label target, prediksi Naive Bayes, dan fallback aturan skor ditampilkan sebagai sumber hasil yang berbeda.
- MySQL/MariaDB melalui PyMySQL.

## Teknologi

Python, Flask, Flask-WTF (CSRF), MySQL/MariaDB, PyMySQL, pandas, NumPy, scikit-learn, joblib, openpyxl, dan xlrd.

## Persyaratan

- Python 3.10 atau lebih baru.
- XAMPP dengan MySQL/MariaDB aktif.
- MySQL 8+ atau MariaDB yang mendukung tipe JSON dan fungsi SQL yang digunakan.
- DBeaver opsional untuk mengelola database.

## Instalasi Windows + XAMPP

### 1. Buat database

Jalankan MySQL pada XAMPP. Di DBeaver, phpMyAdmin, atau MySQL console jalankan:

```sql
CREATE DATABASE db_stres_karyawan
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;
```

Aplikasi membuat tabel saat pertama dijalankan. Database harus sudah ada.

### 2. Siapkan Python environment

Buka terminal pada folder repository:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Jika PowerShell memblokir aktivasi environment, gunakan terminal CMD atau jalankan Python dari environment sesuai kebijakan komputer Anda.

### 3. Konfigurasi environment

Salin file contoh:

```powershell
Copy-Item .env.example .env
```

Buka `.env` lalu atur:
- `DB_HOST=localhost`
- `DB_PORT=3306`
- `DB_USER=root`
- `DB_PASSWORD=` (kosong jika root XAMPP Anda memang tanpa password)
- `DB_NAME=db_stres_karyawan`
- Ganti `FLASK_SECRET_KEY` dengan nilai acak yang panjang.
- Atur `ADMIN_USERNAME` dan `ADMIN_PASSWORD` yang kuat.

Jangan commit file `.env` ke GitHub.

### 4. Jalankan aplikasi

```powershell
python app.py
```

Buka:
- Kuesioner: http://127.0.0.1:5000/
- Login admin: http://127.0.0.1:5000/admin/login
- Health check: http://127.0.0.1:5000/health

Tabel dibuat otomatis. Akun admin pertama dibuat dari `ADMIN_USERNAME` dan `ADMIN_PASSWORD` saat aplikasi mulai, jika username tersebut belum ada.

Jika variabel admin diubah setelah akun sudah dibuat, password lama tidak otomatis diganti.

## Alur import workbook rekap skor faktor

Menu **Training Model** kini memprioritaskan workbook Excel rekap dari perusahaan, misalnya `Psikologi 1(1).xlsx`. Parser mencari header pada 15 baris pertama di setiap sheet. Workbook sumber yang sudah diperiksa memiliki header pada baris ke-4 di sheet `Yang dikirim`.

Kolom yang dikenali:
- `ID NUMBER`: kode karyawan.
- `BAGIAN`: departemen/bagian.
- `Cek`: nama karyawan bila diisi nama.
- `Ketaksan Peran`
- `Konflik Peran`
- `Beban Berlebih Kuantitatif`
- `Beban Berlebih Kualitatif`
- `Pengembangan Karir`
- `Tanggung Jawab Kepada Orang Lain`
- `Total` (opsional; jika tersedia dibandingkan dengan penjumlahan enam skor faktor).

Proses import:
1. Membaca enam skor numerik agregat dari setiap baris responden.
2. Memeriksa bahwa skor berupa bilangan bulat dalam rentang 0–35. Nilai di bawah 5 tetap disimpan apa adanya dan diberi peringatan, tidak dikoreksi otomatis.
3. Menghasilkan target kategori untuk masing-masing faktor berdasarkan aturan skor aplikasi.
4. Melatih enam model Categorical Naive Bayes. Untuk model faktor tertentu, fitur yang digunakan adalah kategori dari lima faktor lainnya; kategori faktor yang menjadi target tidak dimasukkan sebagai fitur model itu.
5. Menyimpan satu submission dan enam hasil faktor per baris workbook. Tidak ada baris yang dimasukkan ke tabel `answers`, karena jawaban item mentah memang tidak tersedia.
6. Menyimpan hasil evaluasi holdout stratified dan file model.

Setiap file dengan SHA-256 yang sama hanya dapat diimpor sekali. Duplikasi kode karyawan tidak otomatis dihapus; baris sumber dipertahankan dan diberi peringatan.

### Catatan metodologis penting

Workbook rekap yang diperiksa berisi **skor agregat**, bukan kolom kategori teks. Karena itu, kategori target dibentuk oleh aplikasi menggunakan ambang yang saat ini dipakai:
- Skor ≤ 9: Stres Ringan
- Skor 10–24: Stres Sedang
- Skor > 24: Stres Berat

Ambang ini harus diverifikasi terhadap instrumen dan pedoman resmi penelitian sebelum hasil digunakan. Karena label target diturunkan dari skor menggunakan aturan tersebut, akurasi holdout bukan validasi independen atas kategori atau bukti bahwa Naive Bayes lebih unggul daripada aturan skor. Untuk workbook rekap, model mempelajari pola hubungan kategori antarfaktor. Kategori "Berat" yang tidak ada pada faktor tertentu juga tidak dapat dipelajari oleh model untuk faktor tersebut.

Saat kuesioner publik diisi, aplikasi menghitung enam skor dari 30 jawaban, mengubah lima faktor lain menjadi kategori fitur, lalu meminta model memprediksi kategori faktor target. Kategori aturan skor tetap ditampilkan sebagai pembanding. Jika model agregat belum tersedia, aplikasi menggunakan fallback aturan skor.

## Kompatibilitas dataset jawaban item lama

Aplikasi masih menerima CSV/Excel lama yang berisi:
- `answer_1` sampai `answer_30`, masing-masing bilangan bulat 1–7.
- Enam label target: `label_ketaksaan_peran`, `label_konflik_peran`, `label_beban_kuantitatif`, `label_beban_kualitatif`, `label_pengembangan_karir`, dan `label_tanggung_jawab`.
- Kolom identitas opsional: `employee_code`, `employee_name`, dan `department`.

Dataset jawaban item disimpan dengan `source_type='dataset'`; workbook rekap skor faktor disimpan sebagai `source_type='aggregate'`; kuesioner publik memakai `source_type='form'`.

## Analisis Submission

Menu **Analisis Submission** membaca data dari `submissions` dan `factor_results`, dengan:
- Filter sumber (semua, rekap skor faktor, dataset jawaban item, atau form), batch, dan departemen.
- Jumlah submission, identitas responden, rerata skor faktor, dan jumlah departemen.
- Rerata, minimum, maksimum, dan distribusi kategori enam faktor.
- Perbandingan departemen, tabel detail dengan pagination, dan ekspor Excel.
- Keterangan apakah kategori berasal dari skor agregat, label dataset, prediksi Naive Bayes, atau fallback aturan skor.

Tabel `aggregate_import_batches` dan `aggregate_factor_scores` dari versi impor lama tetap dipertahankan untuk menjaga data historis. Dashboard saat ini berpusat pada tabel submission.

## Struktur utama

```text
app.py
templates/
  base.html
  index.html
  result.html
  admin_login.html
  admin_dashboard.html
  training.html
  training_result.html
  history.html
  confusion.html
static/
  css/style.css
docs/PRD.md
requirements.txt
.env.example
```

## Database tables

- `employees`
- `dataset_import_batches`
- `submissions`
- `answers` (hanya jika jawaban per butir tersedia)
- `factor_results`
- `model_metrics`
- `admin_users`

## Security notes

- Admin routes dilindungi session login.
- Password admin disimpan sebagai password hash.
- Form POST dilindungi CSRF.
- Ukuran upload dibatasi 10 MB dan nama file diamankan.
- Untuk deployment publik, gunakan HTTPS, secret key unik, password DB yang kuat, backup teratur, dan server WSGI production. Jangan aktifkan debug mode di server publik.

## Status

Perubahan alur rekap skor faktor berada pada branch pengembangan `feature/aggregate-score-dashboard`. Lakukan pengujian fungsional di lingkungan lokal XAMPP sebelum digunakan untuk analisis penelitian.
