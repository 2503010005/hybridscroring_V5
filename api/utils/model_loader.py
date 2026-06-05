# ============================================================
# BERKAS PERBAIKAN: /var/apps/hybridscoring_V5/api/utils/model_loader.py
# ============================================================
import os
import sys
import json
import pickle
import tensorflow as tf
from sentence_transformers import SentenceTransformer

print("ML-1 START model_loader", flush=True)

ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../.."
    )
)

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from config import Config
config = Config()

# Path dasar output dir dari config
MODEL_DIR = os.path.join(ROOT_DIR, config.OUTPUT_DIR)

def init_models(version: str) -> dict:
    """
    Fungsi untuk memuat komponen model secara dinamis berdasarkan 
    parameter versi yang disuntikkan langsung dari app.py
    """
    print(f"\n[LOADER] Memulai inisialisasi komponen model versi: {version}", flush=True)
    
    # 1. LOAD KERAS MODEL (Dinamis sesuai versi)
    MODEL_PATH = os.path.join(MODEL_DIR, f"hybrid_model_{version}.keras")
    print(f"[LOADER] Loading Keras Model dari: {MODEL_PATH}", flush=True)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # 2. TOKENIZER (Dinamis sesuai versi)
    TOKENIZER_PATH = os.path.join(MODEL_DIR, f"tokenizer_{version}.pkl")
    print(f"[LOADER] Loading Tokenizer dari: {TOKENIZER_PATH}", flush=True)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    # 3. SCALER (Dinamis sesuai versi)
    SCALER_PATH = os.path.join(MODEL_DIR, f"scaler_{version}.pkl")
    print(f"ML-2 BEFORE SCALER LOAD: {SCALER_PATH}", flush=True)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    print("ML-3 AFTER SCALER LOAD", flush=True)

    # Blok Debugging Log Scaler milik Pak Made tetap dipertahankan murni
    print("\n" + "="*60, flush=True)
    print("SCALER CHECK", flush=True)
    print("mean shape:", scaler.mean_.shape, flush=True)
    print("scale shape:", scaler.scale_.shape, flush=True)
    print("mean[0:10]=", scaler.mean_[0:10], flush=True)
    print("scale[0:10]=", scaler.scale_[0:10], flush=True)
    print("="*60 + "\n", flush=True)

    # 4. COURSE ENCODER
    with open(os.path.join(MODEL_DIR, f"course_encoder_{version}.pkl"), "rb") as f:
        course_encoder = pickle.load(f)

    # 5. EVALUATION METRICS (QWK & R2 JSON)
    with open(os.path.join(MODEL_DIR, f"evaluation_{version}.json"), "r") as f:
        metrics = json.load(f)

    # 6. SBERT
    sbert_model = SentenceTransformer("firqaaa/indo-sentence-bert-base")

    # 7. COURSE LIST
    TRAINED_COURSES = list(course_encoder.classes_)

    print("="*60, flush=True)
    print("SCALER INFO", flush=True)
    print("mean shape:", scaler.mean_.shape, flush=True)
    print("scale shape:", scaler.scale_.shape, flush=True)
    print("="*60, flush=True)
    
    # Kembalikan seluruh objek komponen sebagai sebuah dictionary pool
    return {
        "model": model,
        "tokenizer": tokenizer,
        "scaler": scaler,
        "course_encoder": course_encoder,
        "metrics": metrics,
        "sbert_model": sbert_model,
        "TRAINED_COURSES": TRAINED_COURSES
    }