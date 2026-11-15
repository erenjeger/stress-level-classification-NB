# Stress Level Classification — Naive Bayes

Aplikasi Flask untuk pengisian kuesioner stres kerja karyawan, klasifikasi enam faktor menggunakan Categorical Naive Bayes, dashboard admin, evaluasi model, riwayat, dan ekspor Excel.

> Aplikasi ini adalah alat bantu pengolahan kuesioner/penelitian, bukan alat diagnosis medis atau psikologis.

## Fitur

- Form publik tanpa login untuk 30 pernyataan skala 1–7.
- Perhitungan skor enam faktor.
- Prediksi Naive Bayes per faktor jika model sudah dilatih; fallback aturan skor ditandai secara eksplisit.
- Login admin, dashboard, riwayat, dan ekspor Excel.
- Upload dataset CSV/Excel, training enam model CategoricalNB, holdout evaluation, confusion matrix, dan classification report.
- Import terpisah untuk workbook skor agregat historis, dashboard statistik per faktor/departemen, filter batch/departemen, tabel detail, dan ekspor analisis Excel.
- MySQL/MariaDB melalui PyMySQL.

## Teknologi

Python, Flask, Flask-WTF (CSRF), MySQL/MariaDB, PyMySQL, pandas, NumPy, scikit-learn, joblib, dan openpyxl.

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

Jika variabel admin diubah setelah akun sudah dibuat, password lama tidak otomatis diganti. Perubahan password perlu dilakukan melalui mekanisme pengelolaan akun yang aman atau pembaruan hash di database.

## Import Skor Agregat Historis

Menu **Import Skor** menerima workbook rekap enam skor faktor (header pada baris ke-4) seperti file Psikologi. Data disimpan terpisah pada tabel `aggregate_import_batches` dan `aggregate_factor_scores`.

Halaman ini menampilkan:
- Ringkasan jumlah baris, ID unik, dan rata-rata total skor.
- Rata-rata, minimum, dan maksimum skor masing-masing faktor.
- Distribusi kategori yang dihitung dari aturan skor aplikasi (bukan label ground-truth atau prediksi Naive Bayes).
- Perbandingan rata-rata skor per departemen.
- Tabel detail responden dengan pagination.
- Filter batch/departemen dan ekspor ringkasan/detail ke Excel.

Import ini menyimpan enam skor agregat sebagaimana ada di sumber. Ia tidak merekonstruksi jawaban 30 item dan tidak memasukkan data agregat ke training Naive Bayes item-level.

## Dataset Training

Buka menu **Training Model** setelah login admin, lalu unduh template CSV. Isi:
- `answer_1` sampai `answer_30`: bilangan bulat 1–7.
- Enam kolom label:
  - `label_ketaksaan_peran`
  - `label_konflik_peran`
  - `label_beban_kuantitatif`
  - `label_beban_kualitatif`
  - `label_pengembangan_karir`
  - `label_tanggung_jawab`

Nilai label harus tepat salah satu dari:
- `Stres Ringan`
- `Stres Sedang`
- `Stres Berat`

Satu baris mewakili satu responden dan harus memiliki jawaban serta keenam label. Aplikasi melatih satu model untuk setiap faktor. Evaluasi holdout stratified dilakukan hanya ketika ukuran dan distribusi kelas memadai.

### Catatan metodologi

Supervised Naive Bayes membutuhkan label yang valid untuk setiap faktor. Jika label training dihasilkan dari aturan ambang skor yang sama dengan pembanding, model pada dasarnya mempelajari aturan tersebut. Akurasi terhadap label turunan seperti ini bukan bukti kemampuan prediktif independen. Jelaskan sumber label, prosedur pembagian data, dan keterbatasan ukuran dataset dalam laporan penelitian.

Aturan skor pada aplikasi saat ini mengikuti implementasi awal:
- Skor ≤ 9: Stres Ringan
- Skor 10–24: Stres Sedang
- Skor > 24: Stres Berat

Aturan ini diterapkan pada skor **masing-masing faktor** dan harus diverifikasi terhadap instrumen/metodologi penelitian sebelum digunakan sebagai kriteria final.

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
- `submissions`
- `answers`
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

Implementasi awal pada branch pengembangan. Lakukan pengujian fungsional di lingkungan XAMPP lokal sebelum digunakan untuk pengumpulan data penelitian.
