from __future__ import annotations

import io
import json
import hashlib
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
    """Create application tables and migrate existing installations safely."""
    statements = [
        """
        CREATE TABLE IF NOT EXISTS employees (
            id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            employee_name VARCHAR(150) NOT NULL,
            employee_code VARCHAR(100) NULL,
            department VARCHAR(150) NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_employees_code (employee_code)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """,
        """
        CREATE TABLE IF NOT EXISTS dataset_import_batches (
            id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            source_filename VARCHAR(255) NOT NULL,
            source_sha256 CHAR(64) NOT NULL,
            total_records INTEGER NOT NULL,
            imported_by VARCHAR(80) NULL,
            imported_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uq_dataset_sha256 (source_sha256)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """,
        """
        CREATE TABLE IF NOT EXISTS submissions (
            id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            employee_id BIGINT NOT NULL,
            submitted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            total_score INTEGER NOT NULL,
            source_type VARCHAR(20) NOT NULL DEFAULT 'form',
            source_batch_id BIGINT NULL,
            INDEX idx_submissions_source (source_type, source_batch_id),
            CONSTRAINT fk_submissions_employee
                FOREIGN KEY (employee_id) REFERENCES employees(id),
            CONSTRAINT fk_submissions_dataset_batch
                FOREIGN KEY (source_batch_id) REFERENCES dataset_import_batches(id)
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
            training_label VARCHAR(30) NULL,
            predicted_class VARCHAR(30) NULL,
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
        # Legacy aggregate tables are retained so existing imported data is not deleted.
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

                # Migrate databases created by earlier application versions.
                migrations = [
                    ("employees", "employee_code", "ALTER TABLE employees ADD COLUMN employee_code VARCHAR(100) NULL"),
                    ("submissions", "source_type", "ALTER TABLE submissions ADD COLUMN source_type VARCHAR(20) NOT NULL DEFAULT 'form'"),
                    ("submissions", "source_batch_id", "ALTER TABLE submissions ADD COLUMN source_batch_id BIGINT NULL"),
                    ("factor_results", "training_label", "ALTER TABLE factor_results ADD COLUMN training_label VARCHAR(30) NULL"),
                ]
                for table_name, column_name, alter_sql in migrations:
                    cur.execute(f"SHOW COLUMNS FROM {table_name} LIKE %s", (column_name,))
                    if not cur.fetchone():
                        cur.execute(alter_sql)

                # Imported training rows have a target label but no model prediction.
                cur.execute("ALTER TABLE factor_results MODIFY predicted_class VARCHAR(30) NULL")

                cur.execute("SHOW INDEX FROM submissions WHERE Key_name='idx_submissions_source'")
                if not cur.fetchone():
                    cur.execute(
                        "ALTER TABLE submissions ADD INDEX idx_submissions_source (source_type, source_batch_id)"
                    )
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
    headers = ["employee_code", "employee_name", "department"]
    headers += [f"answer_{number}" for number in range(1, 31)]
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
        with db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, source_filename, total_records, imported_by, imported_at
                    FROM dataset_import_batches ORDER BY id DESC LIMIT 10
                """)
                batches = cur.fetchall()
        return render_template("training.html", factors=FACTORS, classes=CLASSES, batches=batches)

    uploaded = request.files.get("dataset")
    if not uploaded or not uploaded.filename:
        flash("Pilih dataset CSV atau Excel terlebih dahulu.", "danger")
        return redirect(url_for("training"))

    original_name = secure_filename(uploaded.filename)
    extension = Path(original_name).suffix.lower()
    if extension not in {".xlsx", ".xls", ".csv"}:
        flash("Format file harus .xlsx, .xls, atau .csv.", "danger")
        return redirect(url_for("training"))

    saved_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{original_name}"
    uploaded.save(saved_path)
    staged_models = []

    try:
        source_digest = hashlib.sha256(saved_path.read_bytes()).hexdigest()
        if extension == ".csv":
            dataframe = pd.read_csv(saved_path, encoding="utf-8-sig")
        else:
            dataframe = pd.read_excel(saved_path)

        dataframe.columns = [str(column).strip() for column in dataframe.columns]
        answer_columns = [f"answer_{number}" for number in range(1, 31)]
        label_columns = [factor["label"] for factor in FACTORS]
        required = answer_columns + label_columns
        missing = [column for column in required if column not in dataframe.columns]
        if missing:
            raise ValueError("Kolom wajib belum lengkap: " + ", ".join(missing))
        if dataframe.empty:
            raise ValueError("Dataset tidak memiliki baris data.")

        optional_columns = [
            column for column in ("employee_code", "employee_name", "department")
            if column in dataframe.columns
        ]
        dataframe = dataframe[optional_columns + required].copy()

        if dataframe[required].isna().any().any():
            missing_columns = dataframe[required].columns[
                dataframe[required].isna().any()
            ].tolist()
            raise ValueError(
                "Ada nilai kosong pada kolom wajib: " + ", ".join(missing_columns)
            )

        for column in answer_columns:
            numeric = pd.to_numeric(dataframe[column], errors="coerce")
            if numeric.isna().any() or not np.equal(
                numeric.to_numpy(), np.floor(numeric.to_numpy())
            ).all():
                raise ValueError(f"Kolom {column} harus berisi bilangan bulat.")
            dataframe[column] = numeric.astype(int)
            if not dataframe[column].between(1, 7).all():
                raise ValueError(f"Nilai {column} harus berada pada rentang 1 sampai 7.")

        for column in optional_columns:
            if column in ("employee_code", "employee_name", "department"):
                dataframe[column] = dataframe[column].apply(
                    lambda value: None if pd.isna(value) or str(value).strip() == ""
                    else str(value).strip()
                )

        labels_by_factor = {}
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
                raise ValueError(f"{factor['name']}: dataset harus memiliki minimal dua kelas.")
            labels_by_factor[factor["key"]] = labels

        # Prepare and evaluate six models before writing any submissions.
        summaries = []
        for factor in FACTORS:
            labels = labels_by_factor[factor["key"]]
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
                    features, labels, test_size=0.25, random_state=42, stratify=labels
                )
                evaluation_model = CategoricalNB(min_categories=7, alpha=1.0)
                evaluation_model.fit(x_train, y_train)
                predicted_test = evaluation_model.predict(x_test)
                accuracy = float(accuracy_score(y_test, predicted_test))
                matrix = confusion_matrix(y_test, predicted_test, labels=CLASSES).tolist()
                report = classification_report(
                    y_test, predicted_test, labels=CLASSES, target_names=CLASSES,
                    output_dict=True, zero_division=0,
                )

            final_model = CategoricalNB(min_categories=7, alpha=1.0)
            final_model.fit(features, labels)
            final_path = model_path(factor)
            temporary_path = MODEL_DIR / f"{factor['key']}.{uuid.uuid4().hex}.joblib.tmp"
            joblib.dump(final_model, temporary_path)
            staged_models.append({
                "factor": factor,
                "temporary_path": temporary_path,
                "final_path": final_path,
                "accuracy": accuracy,
                "matrix": matrix,
                "report": report,
                "sample_count": len(dataframe),
            })
            summaries.append({
                "factor": factor["name"],
                "samples": len(dataframe),
                "accuracy": round(accuracy * 100, 2) if accuracy is not None else None,
                "confusion": matrix,
            })

        # Persist batch, respondents, answers, six target labels and metrics atomically.
        with db_connection() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id FROM dataset_import_batches WHERE source_sha256=%s",
                        (source_digest,),
                    )
                    if cur.fetchone():
                        raise ValueError(
                            "File dataset yang sama sudah pernah diimpor. "
                            "Periksa riwayat training untuk menghindari submission duplikat."
                        )

                    cur.execute("""
                        INSERT INTO dataset_import_batches
                        (source_filename, source_sha256, total_records, imported_by)
                        VALUES(%s, %s, %s, %s)
                    """, (
                        original_name, len(dataframe), source_digest,
                        session.get("admin_username"),
                    ))
                    batch_id = cur.lastrowid

                    for row_index, row in dataframe.reset_index(drop=True).iterrows():
                        employee_name = (
                            row.get("employee_name")
                            or f"Responden {row_index + 1:03d}"
                        )
                        employee_code = row.get("employee_code")
                        department = row.get("department")

                        cur.execute(
                            """
                            INSERT INTO employees(employee_name, employee_code, department)
                            VALUES(%s, %s, %s)
                            """,
                            (employee_name, employee_code, department),
                        )
                        employee_id = cur.lastrowid
                        answers = {
                            number: int(row[f"answer_{number}"])
                            for number in range(1, 31)
                        }
                        cur.execute(
                            """
                            INSERT INTO submissions
                            (employee_id, total_score, source_type, source_batch_id)
                            VALUES(%s, %s, 'dataset', %s)
                            """,
                            (employee_id, sum(answers.values()), batch_id),
                        )
                        submission_id = cur.lastrowid

                        cur.executemany(
                            """
                            INSERT INTO answers(submission_id, question_no, answer)
                            VALUES(%s, %s, %s)
                            """,
                            [
                                (submission_id, question_no, answers[question_no])
                                for question_no in range(1, 31)
                            ],
                        )

                        factor_result_rows = []
                        for factor in FACTORS:
                            factor_score = sum(answers[q] for q in factor["questions"])
                            training_label = labels_by_factor[factor["key"]].iloc[row_index]
                            factor_result_rows.append((
                                submission_id,
                                factor["key"],
                                factor["name"],
                                factor_score,
                                rule_class(factor_score),
                                training_label,
                                None,   # Imported training rows have a target label, not a prediction.
                                None,
                                "Label dataset (belum merupakan prediksi model)",
                            ))
                        cur.executemany(
                            """
                            INSERT INTO factor_results
                            (submission_id, factor_key, factor_name, factor_score,
                             rule_class, training_label, predicted_class, probabilities, model_version)
                            VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            factor_result_rows,
                        )

                    for item in staged_models:
                        factor = item["factor"]
                        cur.execute(
                            """
                            INSERT INTO model_metrics
                            (factor_key, sample_count, accuracy, confusion_json, report_json, model_file)
                            VALUES(%s, %s, %s, %s, %s, %s)
                            """,
                            (
                                factor["key"],
                                item["sample_count"],
                                item["accuracy"],
                                json.dumps(item["matrix"]) if item["matrix"] is not None else None,
                                json.dumps(item["report"], ensure_ascii=False),
                                str(item["final_path"].relative_to(BASE_DIR)),
                            ),
                        )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        # Publish the model artifacts only after the database transaction succeeds.
        for item in staged_models:
            os.replace(item["temporary_path"], item["final_path"])

        flash(
            f"Berhasil mengimpor {len(dataframe)} responden sebagai submission batch #{batch_id} "
            "dan melatih enam model Naive Bayes.",
            "success",
        )
        return render_template(
            "training_result.html",
            summaries=summaries,
            classes=CLASSES,
            imported_count=len(dataframe),
            batch_id=batch_id,
        )

    except Exception as error:
        logger.exception("Import dataset/training gagal.")
        flash(f"Import dan training gagal: {error}", "danger")
        return redirect(url_for("training"))
    finally:
        for item in staged_models:
            try:
                item["temporary_path"].unlink(missing_ok=True)
            except OSError:
                logger.warning("Tidak dapat menghapus model sementara: %s", item["temporary_path"])
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


AGGREGATE_FACTOR_FIELDS = [
    {"key": "ketaksaan_peran", "name": "Ketaksaan Peran", "column": "ketaksaan_peran"},
    {"key": "konflik_peran", "name": "Konflik Peran", "column": "konflik_peran"},
    {"key": "beban_kuantitatif", "name": "Beban Berlebih Kuantitatif", "column": "beban_kuantitatif"},
    {"key": "beban_kualitatif", "name": "Beban Berlebih Kualitatif", "column": "beban_kualitatif"},
    {"key": "pengembangan_karir", "name": "Pengembangan Karir", "column": "pengembangan_karir"},
    {"key": "tanggung_jawab", "name": "Tanggung Jawab terhadap Orang Lain", "column": "tanggung_jawab"},
]


def get_submission_factor_rows(source_filter="all", department=None, batch_id=None):
    query = """
        SELECT s.id AS submission_id, s.submitted_at, s.total_score,
               s.source_type, s.source_batch_id,
               e.employee_name, e.employee_code, e.department,
               fr.factor_key, fr.factor_name, fr.factor_score,
               fr.rule_class, fr.training_label, fr.predicted_class, fr.model_version,
               COALESCE(fr.training_label, fr.predicted_class, fr.rule_class) AS category_label
        FROM submissions s
        JOIN employees e ON e.id=s.employee_id
        JOIN factor_results fr ON fr.submission_id=s.id
        WHERE 1=1
    """
    params = []
    if source_filter in ("dataset", "form"):
        query += " AND s.source_type=%s"
        params.append(source_filter)
    if department:
        query += " AND e.department=%s"
        params.append(department)
    if batch_id:
        query += " AND s.source_batch_id=%s"
        params.append(batch_id)
    query += " ORDER BY s.id DESC, fr.factor_key"

    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()


def build_submission_aggregate_report(factor_rows):
    grouped_submissions = {}
    factor_summaries = []
    for factor in AGGREGATE_FACTOR_FIELDS:
        subset = [row for row in factor_rows if row["factor_key"] == factor["key"]]
        scores = [int(row["factor_score"]) for row in subset]
        class_counts = {label: 0 for label in CLASSES}
        for row in subset:
            label = row.get("category_label") or "Tidak tersedia"
            if label in class_counts:
                class_counts[label] += 1
        factor_summaries.append({
            "key": factor["key"],
            "name": factor["name"],
            "column": factor["column"],
            "average": round(sum(scores) / len(scores), 2) if scores else 0,
            "minimum": min(scores) if scores else 0,
            "maximum": max(scores) if scores else 0,
            "counts": class_counts,
            "count": len(scores),
        })

    for row in factor_rows:
        submission_id = row["submission_id"]
        if submission_id not in grouped_submissions:
            grouped_submissions[submission_id] = {
                "submission_id": submission_id,
                "employee_name": row["employee_name"],
                "employee_code": row.get("employee_code"),
                "department": row.get("department"),
                "submitted_at": row["submitted_at"],
                "total_score": row["total_score"],
                "source_type": row["source_type"],
                "source_batch_id": row.get("source_batch_id"),
                "scores": {},
                "categories": {},
            }
        grouped_submissions[submission_id]["scores"][row["factor_key"]] = row["factor_score"]
        grouped_submissions[submission_id]["categories"][row["factor_key"]] = row.get("category_label")

    submissions = list(grouped_submissions.values())
    by_department = {}
    for item in submissions:
        department_name = (item.get("department") or "Tidak diisi").strip() or "Tidak diisi"
        by_department.setdefault(department_name, []).append(item)

    department_summaries = []
    for name, department_submissions in sorted(by_department.items(), key=lambda pair: pair[0].lower()):
        department_summaries.append({
            "name": name,
            "count": len(department_submissions),
            "average_total": round(
                sum(int(item["total_score"]) for item in department_submissions)
                / len(department_submissions), 2
            ),
            "averages": {
                factor["key"]: round(
                    sum(int(item["scores"].get(factor["key"]) or 0)
                        for item in department_submissions)
                    / len(department_submissions), 2
                )
                for factor in AGGREGATE_FACTOR_FIELDS
            },
        })

    total_scores = [int(item["total_score"]) for item in submissions]
    unique_employees = len({item["employee_name"] for item in submissions})
    return {
        "factors": factor_summaries,
        "departments": department_summaries,
        "respondents": len(submissions),
        "unique_employees": unique_employees,
        "average_total": round(sum(total_scores) / len(total_scores), 2) if total_scores else 0,
    }, submissions


@app.route("/admin/import-aggregate")
@admin_required
def import_aggregate():
    """Dashboard statistik berdasarkan submissions dan factor_results."""
    source_filter = request.args.get("source", "all").strip().lower()
    if source_filter not in ("all", "dataset", "form"):
        source_filter = "all"
    department_filter = request.args.get("department", "").strip()
    batch_id = request.args.get("batch_id", type=int)

    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT e.department
                FROM submissions s JOIN employees e ON e.id=s.employee_id
                WHERE e.department IS NOT NULL AND e.department <> ''
                ORDER BY e.department
            """)
            departments = [row["department"] for row in cur.fetchall()]
            cur.execute("""
                SELECT id, source_filename, total_records, imported_by, imported_at
                FROM dataset_import_batches ORDER BY id DESC LIMIT 50
            """)
            batches = cur.fetchall()

    if source_filter == "form":
        batch_id = None
    elif source_filter == "dataset" and not batch_id and batches:
        batch_id = batches[0]["id"]

    factor_rows = get_submission_factor_rows(
        source_filter=source_filter,
        department=department_filter or None,
        batch_id=batch_id,
    )
    report, submission_rows = build_submission_aggregate_report(factor_rows)

    page_size = 25
    total_filtered = len(submission_rows)
    total_pages = max(1, math.ceil(total_filtered / page_size))
    page_number = request.args.get("page", default=1, type=int) or 1
    page_number = min(max(page_number, 1), total_pages)
    offset = (page_number - 1) * page_size
    page_rows = submission_rows[offset:offset + page_size]

    selected_batch = next((item for item in batches if item["id"] == batch_id), None)
    return render_template(
        "aggregate_import.html",
        batches=batches,
        selected_batch=selected_batch,
        source_filter=source_filter,
        batch_id=batch_id,
        departments=departments,
        department_filter=department_filter,
        report=report,
        rows=page_rows,
        page=page_number,
        total_pages=total_pages,
        total_filtered=total_filtered,
    )


@app.route("/admin/import-aggregate/export.xlsx")
@admin_required
def export_aggregate_excel():
    source_filter = request.args.get("source", "all").strip().lower()
    if source_filter not in ("all", "dataset", "form"):
        source_filter = "all"
    department = request.args.get("department", "").strip()
    batch_id = request.args.get("batch_id", type=int)
    factor_rows = get_submission_factor_rows(
        source_filter=source_filter,
        department=department or None,
        batch_id=batch_id,
    )
    report, submissions = build_submission_aggregate_report(factor_rows)

    detail_records = []
    for item in submissions:
        record = {
            "Submission ID": item["submission_id"],
            "Nama/identitas responden": item["employee_name"],
            "Kode karyawan": item.get("employee_code"),
            "Departemen": item.get("department"),
            "Sumber data": item["source_type"],
            "Batch import": item.get("source_batch_id"),
            "Waktu pengisian": item["submitted_at"],
            "Total 30 jawaban": item["total_score"],
        }
        for factor in AGGREGATE_FACTOR_FIELDS:
            record[f"Skor {factor['name']}"] = item["scores"].get(factor["key"])
            record[f"Kategori {factor['name']}"] = item["categories"].get(factor["key"])
        detail_records.append(record)

    summary = pd.DataFrame([
        {
            "Faktor": item["name"],
            "Rata-rata skor": item["average"],
            "Skor minimum": item["minimum"],
            "Skor maksimum": item["maximum"],
            "Jumlah submission": item["count"],
        }
        for item in report["factors"]
    ])
    distribution = pd.DataFrame([
        {
            "Faktor": item["name"],
            "Ringan": item["counts"][CLASSES[0]],
            "Sedang": item["counts"][CLASSES[1]],
            "Berat": item["counts"][CLASSES[2]],
        }
        for item in report["factors"]
    ])
    department_rows = []
    for item in report["departments"]:
        record = {
            "Departemen": item["name"],
            "Jumlah submission": item["count"],
            "Rata-rata total": item["average_total"],
        }
        for factor in AGGREGATE_FACTOR_FIELDS:
            record[f"Rata-rata {factor['name']}"] = item["averages"][factor["key"]]
        department_rows.append(record)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="Ringkasan")
        distribution.to_excel(writer, index=False, sheet_name="Distribusi")
        pd.DataFrame(department_rows).to_excel(writer, index=False, sheet_name="Per Departemen")
        pd.DataFrame(detail_records).to_excel(writer, index=False, sheet_name="Detail Submission")
    output.seek(0)
    return send_file(
        output,
        as_attachment=True,
        download_name="analisis_submission_stres_karyawan.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


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
