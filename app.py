from __future__ import annotations

import io
import json
import logging
import math
import os
import uuid
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pymysql
from dotenv import load_dotenv
from flask import (
    Flask, abort, flash, redirect, render_template, request,
    send_file, session, url_for
)
from flask_wtf.csrf import CSRFProtect
from pymysql.cursors import DictCursor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
INSTANCE_DIR = BASE_DIR / "instance"
UPLOAD_DIR = INSTANCE_DIR / "uploads"
MODEL_DIR = INSTANCE_DIR / "models"
for folder in (INSTANCE_DIR, UPLOAD_DIR, MODEL_DIR):
    folder.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE_DIR / ".env")

CLASSES = ["Stres Ringan", "Stres Sedang", "Stres Berat"]
FACTORS = [
    {"key": "ketaksaan_peran", "name": "Ketaksaan Peran", "questions": [1, 7, 13, 19, 25], "label": "label_ketaksaan_peran"},
    {"key": "konflik_peran", "name": "Konflik Peran", "questions": [2, 8, 14, 20, 26], "label": "label_konflik_peran"},
    {"key": "beban_kuantitatif", "name": "Beban Berlebih Kuantitatif", "questions": [3, 9, 15, 21, 27], "label": "label_beban_kuantitatif"},
    {"key": "beban_kualitatif", "name": "Beban Berlebih Kualitatif", "questions": [4, 10, 16, 22, 28], "label": "label_beban_kualitatif"},
    {"key": "pengembangan_karir", "name": "Pengembangan Karir", "questions": [5, 11, 17, 23, 29], "label": "label_pengembangan_karir"},
    {"key": "tanggung_jawab", "name": "Tanggung Jawab terhadap Orang Lain", "questions": [6, 12, 18, 24, 30], "label": "label_tanggung_jawab"},
]

QUESTIONS = [
    "Tujuan tugas-tugas dan pekerjaan saya tidak jelas",
    "Saya mengerjakan tugas-tugas atau proyek-proyek tidak perlu",
    "Saya harus membawa pulang pekerjaan agar dapat mengejar waktu",
    "Tuntutan mutu pekerjaan terhadap saya terasa keterlaluan",
    "Saya tidak mempunyai kesempatan memadai untuk maju dalam organisasi",
    "Saya bertanggung jawab untuk pengembangan karyawan lain",
    "Saya tidak jelas kepada siapa harus melapor dan/atau siapa yang melapor kepada saya",
    "Saya terjepit di tengah-tengah antara atasan dan bawahan",
    "Saya menghabiskan terlalu banyak waktu untuk pertemuan yang tidak penting",
    "Tugas-tugas yang diberikan terlalu sulit dan/atau kompleks",
    "Kalau ingin naik pangkat, saya harus mencari pekerjaan pada satuan kerja lain",
    "Saya bertanggung jawab membimbing dan/atau membantu bawahan",
    "Saya tidak mempunyai wewenang untuk melaksanakan tanggung jawab pekerjaan",
    "Jalur perintah formal tidak dipatuhi",
    "Saya bertanggung jawab atas banyak proyek dalam waktu bersamaan",
    "Tugas-tugas tampaknya makin hari menjadi makin kompleks",
    "Saya merugikan kemajuan karir dengan menetap pada organisasi ini",
    "Saya membuat keputusan yang mempengaruhi keselamatan dan kesejahteraan orang lain",
    "Saya tidak mengerti sepenuhnya apa yang diharapkan dari saya",
    "Saya melakukan pekerjaan yang diterima satu orang tetapi tidak diterima orang lain",
    "Saya mempunyai pekerjaan lebih banyak daripada yang biasanya dapat dikerjakan sehari",
    "Organisasi mengharapkan saya melebihi keterampilan dan/atau kemampuan",
    "Saya hanya mempunyai sedikit kesempatan untuk berkembang dan belajar",
    "Tanggung jawab saya lebih mengenai orang daripada barang",
    "Saya tidak mengerti bagian pekerjaan saya dalam memenuhi tujuan organisasi",
    "Saya menerima permintaan yang saling bertentangan",
    "Saya tidak punya waktu untuk istirahat berkala",
    "Saya kurang terlatih dan/atau kurang pengalaman untuk tugas mandiri",
    "Saya merasa karir saya tidak berkembang",
    "Saya bertanggung jawab atas hari depan (karir) orang lain",
]
QUESTION_FACTOR = {
    question_no: factor["name"]
    for factor in FACTORS
    for question_no in factor["questions"]
}

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.getenv("FLASK_SECRET_KEY", "dev-only-change-this-secret-key"),
    MAX_CONTENT_LENGTH=10 * 1024 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)
csrf = CSRFProtect(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@contextmanager
def db_connection():
    """Buka koneksi MySQL/MariaDB dan selalu tutup setelah digunakan."""
    conn = pymysql.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "db_stres_karyawan"),
        charset="utf8mb4",
        cursorclass=DictCursor,
        autocommit=False,
        connect_timeout=10,
    )
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    """Buat tabel aplikasi. Database db_stres_karyawan harus sudah dibuat."""
    statements = [
        """
        CREATE TABLE IF NOT EXISTS employees (
            id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            employee_name VARCHAR(150) NOT NULL,
            department VARCHAR(150) NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """,
        """
        CREATE TABLE IF NOT EXISTS submissions (
            id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            employee_id BIGINT NOT NULL,
            submitted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            total_score INTEGER NOT NULL,
            CONSTRAINT fk_submissions_employee
                FOREIGN KEY (employee_id) REFERENCES employees(id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """,
        """
        CREATE TABLE IF NOT EXISTS answers (
            id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            submission_id BIGINT NOT NULL,
            question_no SMALLINT NOT NULL,
            answer SMALLINT NOT NULL,
            CONSTRAINT chk_answer_range CHECK (answer BETWEEN 1 AND 7),
            CONSTRAINT uq_submission_question UNIQUE (submission_id, question_no),
            CONSTRAINT fk_answers_submission
                FOREIGN KEY (submission_id) REFERENCES submissions(id) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """,
        """
        CREATE TABLE IF NOT EXISTS factor_results (
            id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            submission_id BIGINT NOT NULL,
            factor_key VARCHAR(60) NOT NULL,
            factor_name VARCHAR(120) NOT NULL,
            factor_score SMALLINT NOT NULL,
            rule_class VARCHAR(30) NOT NULL,
            predicted_class VARCHAR(30) NOT NULL,
            probabilities JSON NULL,
            model_version VARCHAR(100) NULL,
            CONSTRAINT uq_submission_factor UNIQUE (submission_id, factor_key),
            CONSTRAINT fk_factor_submission
                FOREIGN KEY (submission_id) REFERENCES submissions(id) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """,
        """
        CREATE TABLE IF NOT EXISTS model_metrics (
            id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            factor_key VARCHAR(60) NOT NULL,
            trained_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            sample_count INTEGER NOT NULL,
            accuracy DECIMAL(7,4) NULL,
            confusion_json JSON NULL,
            report_json JSON NULL,
            model_file VARCHAR(255) NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """,
        """
        CREATE TABLE IF NOT EXISTS admin_users (
            id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(80) NOT NULL UNIQUE,
            password_hash VARCHAR(255) NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """,
        """
        CREATE TABLE IF NOT EXISTS aggregate_import_batches (
            id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            source_filename VARCHAR(255) NOT NULL,
            total_records INTEGER NOT NULL,
            duplicate_code_rows INTEGER NOT NULL DEFAULT 0,
            below_minimum_score_cells INTEGER NOT NULL DEFAULT 0,
            imported_by VARCHAR(80) NULL,
            imported_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """,
        """
        CREATE TABLE IF NOT EXISTS aggregate_factor_scores (
            id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            batch_id BIGINT NOT NULL,
            source_no INTEGER NULL,
            source_excel_row INTEGER NOT NULL,
            employee_code VARCHAR(100) NOT NULL,
            department VARCHAR(150) NULL,
            ketaksaan_peran SMALLINT NOT NULL,
            konflik_peran SMALLINT NOT NULL,
            beban_kuantitatif SMALLINT NOT NULL,
            beban_kualitatif SMALLINT NOT NULL,
            pengembangan_karir SMALLINT NOT NULL,
            tanggung_jawab SMALLINT NOT NULL,
            total_score SMALLINT NOT NULL,
            CONSTRAINT uq_aggregate_batch_row UNIQUE (batch_id, source_excel_row),
            CONSTRAINT fk_aggregate_batch FOREIGN KEY (batch_id)
                REFERENCES aggregate_import_batches(id) ON DELETE CASCADE,
            INDEX idx_aggregate_employee_code (employee_code)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """,
    ]
    with db_connection() as conn:
        try:
            with conn.cursor() as cur:
                for statement in statements:
                    cur.execute(statement)
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def ensure_admin_user():
    """Buat akun admin pertama dari environment variable bila belum ada."""
    username = os.getenv("ADMIN_USERNAME", "").strip()
    password = os.getenv("ADMIN_PASSWORD", "")
    if not username or not password:
        logger.warning(
            "ADMIN_USERNAME/ADMIN_PASSWORD belum diatur. "
            "Tambahkan keduanya ke file .env sebelum login admin."
        )
        return
    with db_connection() as conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM admin_users WHERE username=%s", (username,))
                exists = cur.fetchone()
                if not exists:
                    cur.execute(
                        "INSERT INTO admin_users(username, password_hash) VALUES(%s, %s)",
                        (username, generate_password_hash(password)),
                    )
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("admin_id"):
            flash("Silakan login sebagai admin terlebih dahulu.", "warning")
            return redirect(url_for("admin_login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


def rule_class(score: int) -> str:
    """Aturan skor per faktor sesuai implementasi awal; validasi ambang di metodologi."""
    if score <= 9:
        return CLASSES[0]
    if score <= 24:
        return CLASSES[1]
    return CLASSES[2]


def model_path(factor: dict) -> Path:
    return MODEL_DIR / f"{factor['key']}.joblib"


def load_model(factor: dict):
    path = model_path(factor)
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        logger.exception("Model %s gagal dimuat.", factor["key"])
        return None


def predict_factor(factor: dict, answers: dict):
    score = sum(answers[q] for q in factor["questions"])
    model = load_model(factor)
    if model is None:
        # Fallback ini adalah aturan skor, bukan prediksi Naive Bayes.
        return score, rule_class(score), rule_class(score), None, "Aturan skor (model belum tersedia)"

    features = np.array(
        [[answers[q] - 1 for q in factor["questions"]]],
        dtype=int,
    )
    predicted = str(model.predict(features)[0])
    probabilities = {
        str(label): round(float(probability) * 100, 2)
        for label, probability in zip(model.classes_, model.predict_proba(features)[0])
    }
    return score, rule_class(score), predicted, probabilities, "Categorical Naive Bayes"


@app.context_processor
def inject_globals():
    return {"classes": CLASSES}


@app.errorhandler(413)
def file_too_large(_error):
    flash("Ukuran file terlalu besar. Maksimal 10 MB.", "danger")
    return redirect(url_for("training"))


@app.route("/")
def home():
    return render_template(
        "index.html",
        factors=FACTORS,
        questions=QUESTIONS,
        question_factor=QUESTION_FACTOR,
    )


@app.route("/submit", methods=["POST"])
def submit():
    name = request.form.get("employee_name", "").strip()
    department = request.form.get("department", "").strip()
    if not name:
        flash("Nama karyawan wajib diisi.", "danger")
        return redirect(url_for("home"))

    try:
        answers = {
            number: int(request.form.get(f"q{number}", ""))
            for number in range(1, 31)
        }
        if any(value < 1 or value > 7 for value in answers.values()):
            raise ValueError
    except (TypeError, ValueError):
        flash("Semua 30 pertanyaan wajib dijawab dengan angka 1 sampai 7.", "danger")
        return redirect(url_for("home"))

    outputs = []
    for factor in FACTORS:
        score, rule, predicted, probabilities, version = predict_factor(factor, answers)
        outputs.append((factor, score, rule, predicted, probabilities, version))

    conn = None
    try:
        conn = pymysql.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "3306")),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", ""),
            database=os.getenv("DB_NAME", "db_stres_karyawan"),
            charset="utf8mb4",
            cursorclass=DictCursor,
            autocommit=False,
            connect_timeout=10,
        )
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO employees(employee_name, department) VALUES(%s, %s)",
                (name, department or None),
            )
            employee_id = cur.lastrowid
            cur.execute(
                "INSERT INTO submissions(employee_id, total_score) VALUES(%s, %s)",
                (employee_id, sum(answers.values())),
            )
            submission_id = cur.lastrowid
            cur.executemany(
                "INSERT INTO answers(submission_id, question_no, answer) VALUES(%s, %s, %s)",
                [(submission_id, number, answers[number]) for number in range(1, 31)],
            )
            for factor, score, rule, predicted, probabilities, version in outputs:
                cur.execute(
                    """
                    INSERT INTO factor_results
                    (submission_id, factor_key, factor_name, factor_score,
                     rule_class, predicted_class, probabilities, model_version)
                    VALUES(%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        submission_id, factor["key"], factor["name"], score,
                        rule, predicted,
                        json.dumps(probabilities, ensure_ascii=False) if probabilities else None,
                        version,
                    ),
                )
        conn.commit()
    except Exception:
        if conn:
            conn.rollback()
        logger.exception("Gagal menyimpan pengisian kuesioner.")
        flash("Data belum berhasil disimpan. Periksa koneksi database atau hubungi admin.", "danger")
        return redirect(url_for("home"))
    finally:
        if conn:
            conn.close()

    results = [
        {
            "name": factor["name"],
            "score": score,
            "rule": rule,
            "pred": predicted,
            "probs": probabilities,
            "version": version,
        }
        for factor, score, rule, predicted, probabilities, version in outputs
    ]
    return render_template(
        "result.html",
        name=name,
        department=department,
        results=results,
        submission_id=submission_id,
    )


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if session.get("admin_id"):
        return redirect(url_for("admin_dashboard"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        try:
            with db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id, username, password_hash FROM admin_users WHERE username=%s",
                        (username,),
                    )
                    admin = cur.fetchone()
        except Exception:
            logger.exception("Tidak dapat memeriksa akun admin.")
            flash("Database belum siap. Periksa konfigurasi dan jalankan ulang aplikasi.", "danger")
            return render_template("admin_login.html")

        if admin and check_password_hash(admin["password_hash"], password):
            session.clear()
            session["admin_id"] = admin["id"]
            session["admin_username"] = admin["username"]
            session.permanent = True
            next_url = request.args.get("next", "")
            if not next_url.startswith("/") or next_url.startswith("//"):
                next_url = url_for("admin_dashboard")
            return redirect(next_url)

        flash("Username atau password tidak sesuai.", "danger")
    return render_template("admin_login.html")


@app.route("/admin/logout", methods=["POST"])
@admin_required
def admin_logout():
    session.clear()
    flash("Anda telah logout.", "success")
    return redirect(url_for("admin_login"))


@app.route("/admin")
@admin_required
def admin_dashboard():
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.id, e.employee_name, e.department, s.submitted_at, s.total_score
                FROM submissions s
                JOIN employees e ON e.id=s.employee_id
                ORDER BY s.id DESC LIMIT 25
                """
            )
            rows = cur.fetchall()
            cur.execute("SELECT COUNT(*) AS total FROM submissions")
            total = cur.fetchone()["total"]
            cur.execute(
                """
                SELECT factor_name, predicted_class, COUNT(*) AS jumlah
                FROM factor_results
                GROUP BY factor_name, predicted_class
                ORDER BY factor_name, predicted_class
                """
            )
            distribution = cur.fetchall()
            cur.execute(
                """
                SELECT mm.factor_key, mm.trained_at, mm.sample_count, mm.accuracy
                FROM model_metrics mm
                WHERE NOT EXISTS (
                    SELECT 1 FROM model_metrics newer
                    WHERE newer.factor_key=mm.factor_key
                      AND (newer.trained_at > mm.trained_at
                           OR (newer.trained_at = mm.trained_at AND newer.id > mm.id))
                )
                ORDER BY mm.factor_key
                """
            )
            metrics = cur.fetchall()

    return render_template(
        "admin_dashboard.html",
        rows=rows,
        total=total,
        distribution=distribution,
        metrics=metrics,
        factors=FACTORS,
    )


@app.route("/admin/training/template.csv")
@admin_required
def training_template():
    headers = [f"answer_{number}" for number in range(1, 31)]
    headers += [factor["label"] for factor in FACTORS]
    output = io.StringIO()
    output.write(",".join(headers) + "\n")
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode("utf-8-sig")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="template_dataset_training.csv",
    )


@app.route("/admin/training", methods=["GET", "POST"])
@admin_required
def training():
    if request.method == "GET":
        return render_template("training.html", factors=FACTORS, classes=CLASSES)

    uploaded = request.files.get("dataset")
    if not uploaded or not uploaded.filename:
        flash("Pilih file dataset terlebih dahulu.", "danger")
        return redirect(url_for("training"))

    original_name = secure_filename(uploaded.filename)
    extension = Path(original_name).suffix.lower()
    if extension not in {".xlsx", ".xls", ".csv"}:
        flash("Format file harus .xlsx, .xls, atau .csv.", "danger")
        return redirect(url_for("training"))

    saved_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{original_name}"
    uploaded.save(saved_path)

    try:
        if extension == ".csv":
            dataframe = pd.read_csv(saved_path)
        else:
            dataframe = pd.read_excel(saved_path)

        required = [f"answer_{number}" for number in range(1, 31)]
        required += [factor["label"] for factor in FACTORS]
        missing = [column for column in required if column not in dataframe.columns]
        if missing:
            raise ValueError("Kolom belum lengkap: " + ", ".join(missing))

        dataframe = dataframe[required].copy()
        if dataframe.empty:
            raise ValueError("Dataset tidak memiliki baris data.")

        for column in [f"answer_{number}" for number in range(1, 31)]:
            numeric = pd.to_numeric(dataframe[column], errors="coerce")
            if numeric.isna().any():
                raise ValueError(f"Kolom {column} berisi nilai kosong atau bukan angka.")
            if not np.equal(numeric.to_numpy(), np.floor(numeric.to_numpy())).all():
                raise ValueError(f"Kolom {column} harus berisi bilangan bulat.")
            dataframe[column] = numeric.astype(int)
            if not dataframe[column].between(1, 7).all():
                raise ValueError(f"Nilai pada {column} harus berada pada rentang 1 sampai 7.")

        if dataframe.isna().any().any():
            raise ValueError("Dataset masih memiliki sel kosong. Lengkapi atau hapus baris kosong.")

        trained_models = []
        summaries = []
        for factor in FACTORS:
            label_column = factor["label"]
            labels = dataframe[label_column].astype(str).str.strip()
            invalid = sorted(set(labels) - set(CLASSES))
            if invalid:
                raise ValueError(
                    f"Label {label_column} tidak valid: {invalid}. "
                    f"Gunakan salah satu dari: {', '.join(CLASSES)}."
                )
            if labels.nunique() < 2:
                raise ValueError(f"{factor['name']}: minimal harus ada dua kelas label.")

            feature_columns = [f"answer_{number}" for number in factor["questions"]]
            features = dataframe[feature_columns].to_numpy(dtype=int) - 1
            class_counts = labels.value_counts()
            accuracy = None
            matrix = None
            report = {
                "note": "Jumlah sampel per kelas belum mencukupi untuk evaluasi holdout yang layak."
            }

            number_of_classes = labels.nunique()
            test_size = math.ceil(len(dataframe) * 0.25)
            train_size = len(dataframe) - test_size
            can_evaluate = (
                class_counts.min() >= 2
                and test_size >= number_of_classes
                and train_size >= number_of_classes
            )
            if can_evaluate:
                x_train, x_test, y_train, y_test = train_test_split(
                    features,
                    labels,
                    test_size=0.25,
                    random_state=42,
                    stratify=labels,
                )
                evaluation_model = CategoricalNB(min_categories=7, alpha=1.0)
                evaluation_model.fit(x_train, y_train)
                predicted_test = evaluation_model.predict(x_test)
                accuracy = float(accuracy_score(y_test, predicted_test))
                matrix = confusion_matrix(y_test, predicted_test, labels=CLASSES).tolist()
                report = classification_report(
                    y_test,
                    predicted_test,
                    labels=CLASSES,
                    target_names=CLASSES,
                    output_dict=True,
                    zero_division=0,
                )

            final_model = CategoricalNB(min_categories=7, alpha=1.0)
            final_model.fit(features, labels)
            final_path = model_path(factor)
            temporary_path = final_path.with_suffix(".joblib.tmp")
            joblib.dump(final_model, temporary_path)
            trained_models.append((factor, temporary_path, final_path, accuracy, matrix, report, len(dataframe)))
            summaries.append(
                {
                    "factor": factor["name"],
                    "samples": len(dataframe),
                    "accuracy": round(accuracy * 100, 2) if accuracy is not None else None,
                    "confusion": matrix,
                }
            )

        # Ganti file model setelah seluruh model berhasil dilatih.
        for _factor, temporary_path, final_path, *_rest in trained_models:
            os.replace(temporary_path, final_path)

        with db_connection() as conn:
            try:
                with conn.cursor() as cur:
                    for factor, _tmp, final_path, accuracy, matrix, report, sample_count in trained_models:
                        cur.execute(
                            """
                            INSERT INTO model_metrics
                            (factor_key, sample_count, accuracy, confusion_json, report_json, model_file)
                            VALUES(%s, %s, %s, %s, %s, %s)
                            """,
                            (
                                factor["key"],
                                sample_count,
                                accuracy,
                                json.dumps(matrix) if matrix is not None else None,
                                json.dumps(report, ensure_ascii=False),
                                str(final_path.relative_to(BASE_DIR)),
                            ),
                        )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        flash(
            f"Training berhasil: {len(dataframe)} baris data diproses untuk enam faktor.",
            "success",
        )
        return render_template("training_result.html", summaries=summaries, classes=CLASSES)

    except Exception as error:
        logger.exception("Training dataset gagal.")
        flash(f"Training gagal: {error}", "danger")
        return redirect(url_for("training"))
    finally:
        try:
            saved_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Tidak dapat menghapus file upload sementara: %s", saved_path)


@app.route("/admin/history/<int:submission_id>")
@admin_required
def history(submission_id):
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT e.employee_name, e.department, s.id, s.submitted_at, s.total_score
                FROM submissions s
                JOIN employees e ON e.id=s.employee_id
                WHERE s.id=%s
                """,
                (submission_id,),
            )
            person = cur.fetchone()
            if not person:
                abort(404)
            cur.execute(
                "SELECT * FROM factor_results WHERE submission_id=%s ORDER BY id",
                (submission_id,),
            )
            results = cur.fetchall()
            cur.execute(
                "SELECT question_no, answer FROM answers WHERE submission_id=%s ORDER BY question_no",
                (submission_id,),
            )
            answers = cur.fetchall()
    return render_template(
        "history.html",
        person=person,
        results=results,
        answers=answers,
        questions=QUESTIONS,
        question_factor=QUESTION_FACTOR,
    )


@app.route("/admin/export.xlsx")
@admin_required
def export_excel():
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.id AS submission_id, e.employee_name, e.department,
                       s.submitted_at, s.total_score
                FROM submissions s
                JOIN employees e ON e.id=s.employee_id
                ORDER BY s.id
                """
            )
            submissions = cur.fetchall()
            cur.execute(
                """
                SELECT submission_id, factor_name, factor_score, rule_class,
                       predicted_class, probabilities, model_version
                FROM factor_results ORDER BY submission_id, id
                """
            )
            factors = cur.fetchall()
            cur.execute(
                """
                SELECT submission_id, question_no, answer
                FROM answers ORDER BY submission_id, question_no
                """
            )
            answers = cur.fetchall()

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(submissions).to_excel(writer, index=False, sheet_name="Pengisian")
        pd.DataFrame(factors).to_excel(writer, index=False, sheet_name="Hasil 6 Faktor")
        pd.DataFrame(answers).to_excel(writer, index=False, sheet_name="Jawaban Mentah")
    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name="data_stres_karyawan.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.route("/admin/import-aggregate", methods=["GET", "POST"])
@admin_required
def import_aggregate():
    """Impor workbook skor agregat historis; data ini bukan data training item-level."""
    if request.method == "GET":
        with db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, source_filename, total_records, duplicate_code_rows,
                           below_minimum_score_cells, imported_by, imported_at
                    FROM aggregate_import_batches
                    ORDER BY id DESC LIMIT 20
                    """
                )
                batches = cur.fetchall()
        return render_template("aggregate_import.html", batches=batches)

    uploaded = request.files.get("dataset")
    if not uploaded or not uploaded.filename:
        flash("Pilih file Excel terlebih dahulu.", "danger")
        return redirect(url_for("import_aggregate"))

    original_name = secure_filename(uploaded.filename)
    extension = Path(original_name).suffix.lower()
    if extension not in {".xlsx", ".xls"}:
        flash("Impor skor agregat saat ini menerima file Excel .xlsx atau .xls.", "danger")
        return redirect(url_for("import_aggregate"))

    saved_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{original_name}"
    uploaded.save(saved_path)

    try:
        # File Psikologi 1.xlsx menempatkan header pada baris ke-4 (header=3).
        dataframe = pd.read_excel(saved_path, header=3)
        dataframe.columns = [str(column).strip() for column in dataframe.columns]

        required = [
            "No", "ID NUMBER", "BAGIAN",
            "Ketaksan Peran", "Konflik Peran",
            "Beban Berlebih Kuantitatif", "Beban Berlebih Kualitatif",
            "Pengembangan Karir", "Tanggung Jawab Kepada Orang Lain",
            "Total",
        ]
        missing = [column for column in required if column not in dataframe.columns]
        if missing:
            raise ValueError(
                "Kolom tidak sesuai dengan workbook Psikologi. Kolom yang belum ditemukan: "
                + ", ".join(missing)
            )

        dataframe = dataframe[required].dropna(
            subset=["ID NUMBER", "Ketaksan Peran", "Konflik Peran",
                    "Beban Berlebih Kuantitatif", "Beban Berlebih Kualitatif",
                    "Pengembangan Karir", "Tanggung Jawab Kepada Orang Lain", "Total"]
        ).copy()
        if dataframe.empty:
            raise ValueError("Tidak ada baris data yang dapat diimpor.")

        numeric_columns = [
            "No", "Ketaksan Peran", "Konflik Peran",
            "Beban Berlebih Kuantitatif", "Beban Berlebih Kualitatif",
            "Pengembangan Karir", "Tanggung Jawab Kepada Orang Lain", "Total",
        ]
        for column in numeric_columns:
            numeric = pd.to_numeric(dataframe[column], errors="coerce")
            if numeric.isna().any() or not np.equal(
                numeric.to_numpy(), np.floor(numeric.to_numpy())
            ).all():
                raise ValueError(f"Kolom {column} harus berisi bilangan bulat dan tidak kosong.")
            dataframe[column] = numeric.astype(int)

        score_columns = [
            "Ketaksan Peran", "Konflik Peran",
            "Beban Berlebih Kuantitatif", "Beban Berlebih Kualitatif",
            "Pengembangan Karir", "Tanggung Jawab Kepada Orang Lain",
        ]
        for column in score_columns:
            if not dataframe[column].between(0, 35).all():
                raise ValueError(f"Skor {column} harus berada pada rentang 0 sampai 35.")

        calculated_total = dataframe[score_columns].sum(axis=1)
        mismatch_count = int((calculated_total != dataframe["Total"]).sum())
        if mismatch_count:
            raise ValueError(
                f"Ada {mismatch_count} baris dengan Total yang tidak sama dengan jumlah enam skor. "
                "Perbaiki workbook sebelum impor."
            )

        def normalize_employee_code(value):
            if pd.isna(value):
                return ""
            if isinstance(value, (int, np.integer)):
                return str(int(value))
            if isinstance(value, (float, np.floating)) and float(value).is_integer():
                return str(int(value))
            return str(value).strip()

        dataframe["employee_code_normalized"] = dataframe["ID NUMBER"].apply(
            normalize_employee_code
        )
        if (dataframe["employee_code_normalized"] == "").any():
            raise ValueError("Terdapat ID NUMBER kosong.")
        duplicate_code_rows = int(
            dataframe["employee_code_normalized"].duplicated(keep=False).sum()
        )
        below_minimum_score_cells = int(
            (dataframe[score_columns] < 5).sum().sum()
        )

        records = []
        for dataframe_index, row in dataframe.iterrows():
            excel_row = int(dataframe_index) + 5
            source_no = int(row["No"]) if pd.notna(row["No"]) else None
            department = None if pd.isna(row["BAGIAN"]) else str(row["BAGIAN"]).strip()
            records.append((
                excel_row,
                source_no,
                row["employee_code_normalized"],
                department or None,
                int(row["Ketaksan Peran"]),
                int(row["Konflik Peran"]),
                int(row["Beban Berlebih Kuantitatif"]),
                int(row["Beban Berlebih Kualitatif"]),
                int(row["Pengembangan Karir"]),
                int(row["Tanggung Jawab Kepada Orang Lain"]),
                int(row["Total"]),
            ))

        with db_connection() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO aggregate_import_batches
                        (source_filename, total_records, duplicate_code_rows,
                         below_minimum_score_cells, imported_by)
                        VALUES(%s, %s, %s, %s, %s)
                        """,
                        (
                            original_name, len(records), duplicate_code_rows,
                            below_minimum_score_cells, session.get("admin_username"),
                        ),
                    )
                    batch_id = cur.lastrowid
                    cur.executemany(
                        """
                        INSERT INTO aggregate_factor_scores
                        (batch_id, source_excel_row, source_no, employee_code, department,
                         ketaksaan_peran, konflik_peran, beban_kuantitatif,
                         beban_kualitatif, pengembangan_karir, tanggung_jawab, total_score)
                        VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        [
                            (batch_id, excel_row, source_no, employee_code, department,
                             ketaksaan, konflik, beban_kuantitatif, beban_kualitatif,
                             karir, tanggung_jawab, total)
                            for (excel_row, source_no, employee_code, department,
                                 ketaksaan, konflik, beban_kuantitatif, beban_kualitatif,
                                 karir, tanggung_jawab, total) in records
                        ],
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        message = f"Berhasil mengimpor {len(records)} baris skor agregat ke batch #{batch_id}."
        flash(message, "success")
        if duplicate_code_rows:
            flash(
                f"Perhatian: {duplicate_code_rows} baris memiliki ID NUMBER yang duplikat. "
                "Semua baris tetap disimpan dan tidak digabung otomatis.",
                "warning",
            )
        if below_minimum_score_cells:
            flash(
                f"Perhatian: ditemukan {below_minimum_score_cells} nilai skor di bawah 5. "
                "Nilai asli dipertahankan; verifikasi terhadap sumber kuesioner.",
                "warning",
            )
        return redirect(url_for("import_aggregate"))
    except Exception as error:
        logger.exception("Impor skor agregat gagal.")
        flash(f"Impor gagal: {error}", "danger")
        return redirect(url_for("import_aggregate"))
    finally:
        try:
            saved_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Tidak dapat menghapus file upload sementara: %s", saved_path)


@app.route("/admin/confusion")
@admin_required
def confusion():
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT mm.factor_key, mm.trained_at, mm.sample_count, mm.accuracy,
                       mm.confusion_json, mm.report_json
                FROM model_metrics mm
                WHERE NOT EXISTS (
                    SELECT 1 FROM model_metrics newer
                    WHERE newer.factor_key=mm.factor_key
                      AND (newer.trained_at > mm.trained_at
                           OR (newer.trained_at = mm.trained_at AND newer.id > mm.id))
                )
                ORDER BY mm.factor_key
                """
            )
            rows = cur.fetchall()

    for row in rows:
        for column in ("confusion_json", "report_json"):
            value = row.get(column)
            if isinstance(value, str):
                try:
                    row[column] = json.loads(value)
                except json.JSONDecodeError:
                    row[column] = None
    return render_template("confusion.html", rows=rows, factors=FACTORS, classes=CLASSES)


@app.route("/health")
def health():
    try:
        with db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 AS ok")
                cur.fetchone()
        return {"status": "ok", "database": "connected"}, 200
    except Exception:
        logger.exception("Health check database gagal.")
        return {"status": "error", "database": "unavailable"}, 503


if __name__ == "__main__":
    try:
        init_db()
        ensure_admin_user()
    except Exception as exc:
        logger.exception("Inisialisasi gagal. Pastikan database sudah dibuat: %s", exc)
        raise SystemExit(
            "Gagal menginisialisasi database. Buat db_stres_karyawan di MySQL "
            "dan periksa pengaturan pada file .env."
        )
    app.run(
        host=os.getenv("FLASK_HOST", "127.0.0.1"),
        port=int(os.getenv("FLASK_PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
    )
