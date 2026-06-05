# api/app.py

# =====================================================
# PROJECT ROOT PATH
# =====================================================
import os
import sys
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional
import numpy as np
from ollama import chat
from fastapi import HTTPException
from rubric_engine import calculate_rubric_score, get_rubric_details
from utils.model_loader import init_models

ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


from config import Config
config = Config()
VERSION = "v5000"

API_PREFIX = "/v1"

print("ML-0 model_loader from app.py", flush=True)
models = init_models(VERSION)
model = models["model"]
tokenizer = models["tokenizer"]
scaler = models["scaler"]
course_encoder = models["course_encoder"]
metrics = models["metrics"]
sbert_model = models["sbert_model"]
TRAINED_COURSES = models["TRAINED_COURSES"]

import utils.model_loader as ml

print("\n" + "="*60)
print("MODEL LOADER FILE:")
print(ml.__file__)
print("="*60)


from utils.preprocessing import (
    prepare_inference,
    repetition_ratio
)

from utils.validators import (
    validate_course
)


# =====================================================
# APP
# =====================================================
app = FastAPI(

    title="Hybrid AES API",
    description=(

        "Context-Aware Hybrid "
        "Spatio-Sequential "
        "Automated Essay Scoring"

    ),

    version=VERSION

)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],
    allow_credentials=True,
    allow_methods=[
        "*"
    ],
    allow_headers=[
        "*"
    ],
)

# =====================================================
# REQUEST MODEL
# =====================================================
class PredictRequest(
    BaseModel
):

    student_answer: str
    reference_answer: str
    course: str
    soal_no: int
    rubric_breakdown: Optional[Dict[str, float]] = None

class FeedbackRequest(BaseModel):
    student_answer: str
    reference_answer: str
    ai_score: float
    course: str

# =====================================================
# ROOT
# =====================================================
@app.get(f"{API_PREFIX}/")
def root():

    return {

        "status":
            "ok",

        "model":
            f"hybrid_model_{VERSION}",

        "modelversion":VERSION,

        "metrics": {

            "qwk":
                round(
                    metrics["qwk"],
                    3
                ),

            "r2":
                round(
                    metrics["r2"],
                    3
                ),

            "rmse":
                round(
                    metrics["rmse"],
                    3
                ),

            "mae":
                round(
                    metrics["mae"],
                    3
                ),

            "mse":
                round(
                    metrics["mse"],
                    3
                )

        },

        "numeric_dim":
            model.input[1].shape[1],

        "max_sequence_length":
            config.MAX_LEN

    }


# =====================================================
# COURSES
# =====================================================
@app.get(f"{API_PREFIX}/courses")
def get_courses():

    return {

        "total_courses":

            len(
                TRAINED_COURSES
            ),

        "courses":

            TRAINED_COURSES

    }

# =====================================================
# STATIC FEEDBACK GENERATION
# =====================================================
def generate_static_feedback(ai_score):
    """
    Menghasilkan umpan balik statis berbasis aturan rentang nilai
    menggunakan Bahasa Indonesia yang baku dan akademis.
    """
    if ai_score >= 85:
        performance = (
            "Jawaban mahasiswa menunjukkan pemahaman konsep yang sangat kuat, "
            "terstruktur dengan baik, serta memiliki keselarasan kontekstual "
            "yang tinggi terhadap kunci acuan."
        )
    elif ai_score >= 70:
        performance = (
            "Jawaban mahasiswa menunjukkan pemahaman konsep yang cukup memadai. "
            "Struktur kalimat dan kelengkapan argumen sudah berada pada tingkat moderat, "
            "namun masih bisa dioptimalkan."
        )
    else:
        performance = (
            "Jawaban mahasiswa memerlukan perbaikan lebih lanjut pada kejelasan "
            "konsep dasar serta kelengkapan elaborasi ide/topik bahasan esai."
        )

    feedback = f"""[Ringkasan Analisis Statistik]
{performance}

Sistem rekomendasi nilai otomatis ini berfungsi sebagai instrumen pendukung keputusan (DSS) bagi Dosen, dengan tetap menempatkan penilaian subjektif penguji manusia (Dosen) sebagai keputusan final yang otoritatif."""
    
    return feedback.strip()

# =====================================================
# PREDICT
# =====================================================
@app.post(f"{API_PREFIX}/predict")
def predict(req: PredictRequest):
    # 1. Validasi Kursus
    validate_course(req.course, TRAINED_COURSES)

    # 2. Pillar 1 & 2: DL (CNN-BiLSTM) & Semantic (SBERT)
    X_text, X_numeric = prepare_inference(
        student_answer=req.student_answer,
        reference_answer=req.reference_answer,
        course=req.course,
        tokenizer=tokenizer,
        scaler=scaler,
        course_encoder=course_encoder,
        sbert_model=sbert_model,
        max_len=config.MAX_LEN
    )

    # Prediksi Model Deep Learning (Spatio-Sequential)
    pred_raw = model.predict([X_text, X_numeric], verbose=0)[0][0]
    x_dl = float(pred_raw * 100.0)

    # Catatan: Pastikan indeks [0][1] benar-benar mengisolasi skor SBERT pada fungsi prepare_inference Anda
    x_sem = float(X_numeric[0][1] * 100.0)

    # 3. Pillar 3: Rubrik Dinamis (Memproses input kustom dari React atau fallback ke database)
    x_rubric = calculate_rubric_score(req.soal_no, req.rubric_breakdown)

    # =========================================================================
    # FUSION LOGIC: SKENARIO 4 (OPTIMAL) -> Rasio Emas Bab IV
    # =========================================================================
    w_dl, w_sem, w_rub = 0.4, 0.3, 0.3
    final_score = (w_dl * x_dl) + (w_sem * x_sem) + (w_rub * x_rubric)

    # 4. Repetition Penalty
    rep_ratio = repetition_ratio(req.student_answer)
    if rep_ratio > 0.55:
        penalty = min(25, rep_ratio * 30)
        final_score -= penalty

    # Safety Clip
    final_score = float(np.clip(final_score, 0, 100))

    # Tampilkan kriteria yang terdeteksi untuk kebutuhan visualisasi UI React
    active_rubric = req.rubric_breakdown if req.rubric_breakdown else get_rubric_details(req.soal_no)

    return {
        "score": round(final_score, 2),
        "breakdown": {
            "dl_score": round(x_dl, 2),
            "semantic_score": round(x_sem, 2),
            "rubric_score": round(x_rubric, 2),
            "weights": {"dl": w_dl, "semantic": w_sem, "rubric": w_rub}
        },
        "rubric_criteria": active_rubric,
        "repetition_ratio": round(rep_ratio, 4),
        "static_feedback": generate_static_feedback(final_score),
        "model_metrics": {
            "qwk": round(metrics["qwk"], 3),
            "r2": round(metrics["r2"], 3)
        }
    }

# =====================================================
# FEEDBACK GENERATION ENDPOINT
# =====================================================
@app.post(f"{API_PREFIX}/feedback")
def generate_llm_feedback(req: FeedbackRequest):
    prompt = f"""
Anda adalah seorang dosen dan evaluator akademik profesional.
Tugas Anda adalah memberikan umpan balik (feedback) edukatif yang konstruktif dan padat menggunakan Bahasa Indonesia yang baik dan benar.

Konteks Evaluasi:
- Mata Kuliah: {req.course}
- Jawaban Referensi Acuan (Dosen): {req.reference_answer}
- Jawaban Teks Mahasiswa: {req.student_answer}
- Skor Akhir Prediksi AI: {req.ai_score}

Instruksi Penulisan Feedback:
1. Jelaskan kekuatan dari jawaban mahasiswa tersebut.
2. Jelaskan kelemahan atau poin yang kurang dari jawaban mahasiswa tersebut.
3. Sebutkan tingkat pemahaman konsep dan kelengkapan penjelasannya.
4. Gunakan nada bicara yang akademis, santun, dan memotivasi mahasiswa.
5. Maksimal penulisan adalah 150 kata. Tuliskan secara langsung, padat, dan ringkas. Do not preamble.
"""
    target_model = "qwen3-coder-next:cloud"  # Default model, bisa diubah melalui config jika diperlukan (gemma4:31b-cloud,qwen3-coder-next:cloud,gemma4:31b-cloud,qwen2.5:3b) )
    sys.stdout.write(f"[OLLAMA RUN] check prompt: {prompt}\n")
    sys.stdout.flush()
    try:
        if 'config' in globals() and hasattr(config, 'AI_MODEL'):
            target_model = config.AI_MODEL
    except Exception:
        pass

    

    try:
        # Menjalankan komputasi token Ollama secara terisolasi
        response = chat(
            model=target_model,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.message.content.strip()
        
        return {
            "status": "success",
            "ai_feedback": text,
            "model_used": target_model
        }
    except Exception as e:
        print(f"Gagal melakukan inferensi ke Ollama SDK: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"Gagal memproses pipa inferensi LLM lokal: {str(e)}")

# =====================================================
# HEALTH
# =====================================================
@app.get(f"{API_PREFIX}/health")
def health():

    return {

        "status":
            "healthy"

    }