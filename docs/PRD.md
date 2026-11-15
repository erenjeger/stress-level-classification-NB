# Product Requirements Document (PRD)
## Sistem Klasifikasi Tingkat Stres Kerja Karyawan Menggunakan Naive Bayes

- **Repository:** [stress-level-classification-NB](https://github.com/erenjeger/stress-level-classification-NB)
- **Versi dokumen:** 1.0
- **Status:** Draft
- **Bahasa:** Indonesia
- **Platform:** Web application

---

## 1. Ringkasan Produk

Aplikasi web untuk mengumpulkan jawaban kuesioner stres kerja karyawan, menghitung skor pada enam faktor stres kerja, dan mengklasifikasikan tingkat stres menggunakan algoritma Categorical Naive Bayes yang dilatih dari dataset berlabel.

Karyawan mengisi kuesioner tanpa perlu membuat akun. Admin mengelola dataset pelatihan, memantau pengisian dan hasil klasifikasi, melihat evaluasi model, serta mengekspor data ke Excel.

Aplikasi merupakan alat bantu pengolahan kuesioner dan penelitian, bukan alat diagnosis medis atau psikologis.

## 2. Tujuan

1. Mempermudah pengumpulan data kuesioner stres kerja.
2. Menghitung skor untuk enam faktor stres kerja secara konsisten.
3. Menerapkan algoritma Naive Bayes untuk klasifikasi tiap faktor.
4. Menyediakan dashboard dan riwayat hasil bagi admin.
5. Menyediakan pelatihan model, evaluasi, dan ekspor data.
6. Menyimpan data secara terstruktur pada MySQL/MariaDB.

## 3. Pengguna dan Hak Akses

### 3.1 Karyawan

- Mengakses halaman kuesioner tanpa login.
- Mengisi nama dan departemen.
- Menjawab 30 pertanyaan dengan skala 1–7.
- Mengirim jawaban dan melihat hasil klasifikasi.

### 3.2 Admin

Admin memerlukan autentikasi untuk mengakses halaman administrasi. Admin dapat:

- Melihat dashboard dan seluruh pengisian.
- Membuka detail riwayat jawaban dan hasil.
- Mengunggah dataset training.
- Melihat metrik evaluasi model.
- Mengunduh data dalam format Excel.
- Logout dari aplikasi.

**Status implementasi:** Pada kode aplikasi yang diperiksa sebelumnya, halaman admin belum dilindungi login. Autentikasi admin adalah kebutuhan yang masih harus dikembangkan.

## 4. Ruang Lingkup

### Termasuk

- Form kuesioner 30 pertanyaan.
- Validasi identitas dan jawaban.
- Perhitungan skor enam faktor.
- Klasifikasi dengan Naive Bayes.
- Penyimpanan data di MySQL/MariaDB.
- Dashboard, distribusi klasifikasi, dan riwayat.
- Unggah dataset Excel/CSV.
- Pelatihan dan penyimpanan model.
- Evaluasi accuracy, confusion matrix, dan classification report.
- Ekspor Excel.
- Autentikasi dan otorisasi admin.

### Tidak termasuk

- Payroll dan absensi.
- Sistem HRIS menyeluruh.
- Diagnosis atau konsultasi psikologis.
- Integrasi dengan sistem eksternal perusahaan pada tahap awal.

## 5. Kuesioner dan Faktor

Kuesioner memiliki 30 item. Setiap item dijawab menggunakan nilai 1 sampai 7. Setiap faktor terdiri dari lima item.

| No. | Faktor | Nomor item |
|---|---|---|
| 1 | Ketaksaan Peran | 1, 7, 13, 19, 25 |
| 2 | Konflik Peran | 2, 8, 14, 20, 26 |
| 3 | Beban Berlebih Kuantitatif | 3, 9, 15, 21, 27 |
| 4 | Beban Berlebih Kualitatif | 4, 10, 16, 22, 28 |
| 5 | Pengembangan Karir | 5, 11, 17, 23, 29 |
| 6 | Tanggung Jawab terhadap Orang Lain | 6, 12, 18, 24, 30 |

Label skala jawaban yang ditampilkan kepada pengguna harus mengikuti instrumen kuesioner penelitian yang disetujui.

### 5.1 Skor faktor

Skor faktor adalah penjumlahan jawaban lima item pada faktor tersebut. Rentang matematisnya 5–35.

### 5.2 Kategori berbasis aturan skor

Kode aplikasi yang ditinjau menggunakan aturan berikut untuk masing-masing faktor:

| Skor faktor | Kategori |
|---|---|
| ≤ 9 | Stres Ringan |
| 10–24 | Stres Sedang |
| > 24 | Stres Berat |

**Perlu verifikasi:** Ambang ini harus dikonfirmasi terhadap instrumen, pedoman, dan metodologi penelitian sebelum dianggap sebagai standar final. Aturan skor dan prediksi model juga harus diberi nama serta dijelaskan secara terpisah di UI dan laporan.

## 6. Kebutuhan Fungsional

### FR-01 — Form kuesioner karyawan

Sistem menyediakan form publik untuk nama, departemen, dan 30 jawaban.

**Acceptance criteria:**
- Nama wajib diisi.
- Semua 30 item wajib dijawab.
- Setiap jawaban harus berupa bilangan bulat 1–7.
- Input yang tidak valid ditolak dengan pesan yang dapat dipahami.
- Pengiriman yang valid diproses dan disimpan.

### FR-02 — Penghitungan skor

Sistem menjumlahkan lima jawaban yang termasuk dalam setiap faktor dan menampilkan skor per faktor.

**Acceptance criteria:**
- Keenam faktor dihitung menggunakan pemetaan item yang ditetapkan.
- Skor yang ditampilkan dapat ditelusuri ke jawaban sumbernya.
- Skor total seluruh item, bila ditampilkan, tidak menggantikan skor per faktor.

### FR-03 — Klasifikasi Naive Bayes

Sistem melatih dan menggunakan model Categorical Naive Bayes secara terpisah untuk masing-masing faktor.

**Input model:** Lima jawaban pada faktor terkait, dengan encoding kategori yang konsisten antara training dan prediksi.

**Output:** Kelas prediksi (Stres Ringan, Stres Sedang, atau Stres Berat) dan probabilitas kelas bila model mendukungnya.

**Acceptance criteria:**
- Dataset training memiliki label untuk masing-masing faktor.
- Data prediksi menggunakan preprocessing yang sama dengan data training.
- Sistem tidak menyatakan model sudah dilatih ketika file model belum tersedia.
- Fallback berbasis aturan skor, jika digunakan, ditandai jelas sebagai hasil aturan dan bukan prediksi ML.

### FR-04 — Dashboard admin

Dashboard menampilkan jumlah pengisian, data terbaru, distribusi hasil klasifikasi, dan ringkasan pelatihan model.

### FR-05 — Riwayat dan detail pengisian

Admin dapat melihat identitas yang disimpan, departemen, waktu pengisian, skor, hasil enam faktor, serta jawaban untuk satu submission.

### FR-06 — Unggah dataset training

Sistem menerima Excel (.xlsx/.xls) atau CSV.

Kolom wajib:
- `answer_1` sampai `answer_30`
- `label_ketaksaan_peran`
- `label_konflik_peran`
- `label_beban_kuantitatif`
- `label_beban_kualitatif`
- `label_pengembangan_karir`
- `label_tanggung_jawab`

Validasi:
- Semua kolom wajib tersedia.
- Jawaban numerik dan berada pada rentang 1–7.
- Label hanya berisi Stres Ringan, Stres Sedang, atau Stres Berat.
- Setiap faktor memiliki variasi kelas yang cukup untuk pelatihan.
- Kesalahan validasi dilaporkan secara spesifik.
- Nama file dan konten upload ditangani secara aman.

### FR-07 — Evaluasi model

Jika jumlah data dan distribusi kelas mencukupi, sistem melakukan holdout evaluation dengan stratifikasi.

Metrik:
- Accuracy.
- Confusion matrix dengan urutan kelas yang konsisten.
- Classification report (precision, recall, F1-score, support).

Simpan faktor, waktu training, jumlah sampel, metrik, dan lokasi artifact model. Bila evaluasi tidak layak dilakukan, UI harus menyatakan keterbatasannya, bukan menampilkan angka seolah-olah valid.

### FR-08 — Ekspor Excel

Admin dapat mengunduh workbook `.xlsx` yang berisi:
1. **Pengisian** — identitas, waktu, dan skor total.
2. **Hasil 6 Faktor** — skor, kategori aturan, prediksi model, dan probabilitas.
3. **Jawaban Mentah** — jawaban tiap item.

### FR-09 — Autentikasi admin

- Login dengan kredensial admin.
- Password disimpan menggunakan password hashing yang aman.
- Session admin dikelola secara aman.
- Semua endpoint admin dilindungi.
- Tersedia logout.
- Form dan endpoint publik tidak boleh membuka akses ke data admin.

## 7. Alur Utama

### 7.1 Karyawan

1. Membuka halaman kuesioner.
2. Mengisi nama dan departemen.
3. Menjawab seluruh 30 item.
4. Mengirim form.
5. Sistem memvalidasi input.
6. Sistem menghitung skor enam faktor.
7. Sistem menjalankan model yang tersedia atau fallback yang diberi label dengan jelas.
8. Sistem menyimpan submission, jawaban, dan hasil dalam satu transaksi database.
9. Sistem menampilkan hasil.

### 7.2 Admin

1. Login.
2. Membuka dashboard.
3. Memeriksa pengisian dan hasil.
4. Mengakses detail riwayat.
5. Mengunggah dataset berlabel bila perlu.
6. Sistem memvalidasi data dan melatih model.
7. Admin memeriksa evaluasi model.
8. Admin mengunduh workbook Excel.
9. Logout.

## 8. Model Data

Nama database: `db_stres_karyawan`.

Tabel inti:

| Tabel | Fungsi | Kolom utama |
|---|---|---|
| `employees` | Identitas karyawan | id, employee_name, department, created_at |
| `submissions` | Pengisian kuesioner | id, employee_id, submitted_at, total_score |
| `answers` | Jawaban per item | id, submission_id, question_no, answer |
| `factor_results` | Hasil per faktor | submission_id, factor_key, factor_score, rule_class, predicted_class, probabilities |
| `model_metrics` | Riwayat evaluasi/training | factor_key, trained_at, sample_count, accuracy, confusion_json, report_json, model_file |

Relasi submission, jawaban, dan hasil faktor harus menggunakan foreign key. Penyimpanan satu submission beserta jawabannya dan hasil faktor harus bersifat transaksional.

## 9. Kebutuhan Nonfungsional

### Keamanan dan privasi
- Hash password admin; jangan menyimpan password plaintext.
- Gunakan secret key dari environment variable.
- Lindungi semua route admin dengan autentikasi.
- Validasi input dan file upload.
- Batasi ukuran file dan tipe file yang diterima.
- Hindari mengekspos nama dan jawaban karyawan ke pengunjung publik.
- Tetapkan tujuan penggunaan, retensi, dan akses data karyawan.
- Jangan memasukkan kredensial database atau file rahasia ke Git.

### Keandalan
- Gunakan transaksi database dan rollback ketika penyimpanan gagal.
- Tangani error upload dan training.
- Pastikan model yang disimpan dapat dimuat kembali.
- Hindari status sukses palsu ketika proses training gagal.

### Kegunaan
- Antarmuka berbahasa Indonesia.
- Formulir mudah dipahami dan responsif.
- Pesan validasi dan status proses jelas.
- Kategori aturan dan prediksi model dibedakan.

### Kompatibilitas
- Python dan Flask.
- MySQL/MariaDB XAMPP.
- PyMySQL.
- Browser modern.
- DBeaver untuk administrasi database.

## 10. Teknologi

| Lapisan | Teknologi |
|---|---|
| Backend | Python, Flask |
| Database | MySQL/MariaDB |
| Driver database | PyMySQL |
| Machine learning | scikit-learn, CategoricalNB |
| Pengolahan data | pandas, NumPy |
| Penyimpanan model | joblib |
| Ekspor Excel | openpyxl |
| Frontend | HTML, CSS, JavaScript |
| Database client | DBeaver |

## 11. Indikator Keberhasilan

- Karyawan dapat mengirim 30 jawaban yang valid.
- Skor keenam faktor sesuai pemetaan item.
- Data tersimpan konsisten di MySQL.
- Hasil klasifikasi ditampilkan dengan asal metode yang jelas.
- Admin dapat memantau pengisian dan riwayat.
- Dataset training tervalidasi sebelum digunakan.
- Model enam faktor dapat dilatih jika data memenuhi persyaratan.
- Metrik evaluasi dihitung dari data evaluasi yang terpisah dari data training.
- Ekspor Excel dapat dibuka dan berisi tiga sheet yang ditetapkan.
- Endpoint administrasi tidak dapat diakses tanpa login.

## 12. Risiko dan Batasan Metodologis

1. **Kualitas label:** supervised Naive Bayes membutuhkan label ground-truth yang dapat dipertanggungjawabkan untuk setiap faktor.
2. **Label berbasis ambang:** jika label training dibuat dari aturan skor yang sama, model terutama belajar mereproduksi aturan tersebut. Akurasi tinggi terhadap label semacam itu bukan bukti validitas klinis atau prediktif independen.
3. **Ketidakseimbangan kelas:** distribusi kelas yang tidak seimbang dapat memengaruhi metrik. Laporkan confusion matrix, precision, recall, dan F1, bukan accuracy saja.
4. **Ukuran data:** evaluasi holdout tidak andal bila sampel per kelas terlalu sedikit.
5. **Instrumen:** ambang skor dan pemetaan skala harus sesuai sumber instrumen penelitian.
6. **Privasi:** nama dan jawaban karyawan merupakan data yang perlu dilindungi dan hanya boleh diakses sesuai kewenangan.

## 13. Rencana Pengujian

| ID | Skenario | Hasil yang diharapkan |
|---|---|---|
| TC-01 | Buka halaman form | Form tampil |
| TC-02 | Kirim tanpa nama | Ditolak dengan validasi |
| TC-03 | Jawaban kurang dari 30 | Ditolak |
| TC-04 | Jawaban di luar 1–7 | Ditolak |
| TC-05 | Kirim jawaban valid | Submission tersimpan |
| TC-06 | Periksa enam skor | Sesuai pemetaan item |
| TC-07 | Model tersedia | Prediksi model ditampilkan |
| TC-08 | Model belum tersedia | Fallback ditandai sebagai aturan |
| TC-09 | Buka dashboard tanpa login | Ditolak setelah auth diterapkan |
| TC-10 | Login admin valid/tidak valid | Session sesuai hasil autentikasi |
| TC-11 | Upload dataset valid | Validasi dan training berjalan |
| TC-12 | Upload dataset tidak valid | Ditolak dengan alasan |
| TC-13 | Evaluasi data cukup | Metrik dihitung dan disimpan |
| TC-14 | Evaluasi data tidak cukup | Keterbatasan dijelaskan |
| TC-15 | Ekspor Excel | Workbook tiga sheet terunduh |
| TC-16 | Simulasi kegagalan penyimpanan | Transaksi rollback; tidak ada data parsial |

## 14. Prioritas Pengembangan

| Prioritas | Pekerjaan | Status berdasarkan kode yang pernah ditinjau |
|---|---|---|
| P0 | Koneksi dan skema MySQL | File konversi telah disiapkan; perlu diuji di lingkungan lokal |
| P0 | Form 30 item dan validasi | Ada di kode; perlu pengujian |
| P0 | Penyimpanan submission/jawaban | Ada di kode; perlu pengujian MySQL |
| P0 | Skor dan hasil enam faktor | Ada di kode; aturan ambang perlu diverifikasi |
| P0 | Autentikasi admin | Belum tersedia |
| P1 | Dashboard dan distribusi | Ada di kode; perlu pengujian |
| P1 | Riwayat pengisian | Ada di kode; perlu pengujian |
| P1 | Training dan validasi dataset | Ada di kode; perlu pengujian |
| P1 | Evaluasi model | Ada di kode; perlu verifikasi metodologis |
| P1 | Ekspor Excel | Ada di kode; perlu pengujian |
| P1 | Pengujian end-to-end | Belum dilakukan |

## 15. Definisi Selesai

Rilis awal dianggap selesai ketika seluruh kebutuhan P0 dan P1 telah diimplementasikan dan diuji, data tersimpan dengan benar, akses admin terlindungi, alur training dan evaluasi dapat dijelaskan secara metodologis, dan ekspor Excel berfungsi.

Dokumen ini merupakan baseline kebutuhan produk. Perubahan ruang lingkup, ambang skor, instrumen, atau metode pelabelan perlu dicatat sebagai revisi PRD.
