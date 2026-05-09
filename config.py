import os

class Config:
    # =========================
    # DATA
    # =========================
    DATA_DIR = "data"
    DATA_SAMPLING_DIR = "sampling"
    LLM_CACHE_FILE = os.path.join(DATA_SAMPLING_DIR, "llm_scoring.csv")
    OUTPUT_DIR = "model_v5"

    # =========================
    # MODEL
    # =========================
    MAX_WORDS = 20000
    MAX_LEN = 250
    MAX_LEN_ESSAY = 250
    MAX_LEN_REF = 150
    EMBED_DIM = 128
    INITIAL_LR = 0.001

    # =========================
    # LLM SAMPLING
    # =========================
    N_SAMPLES_AI = 0.2   # 20% dari dataset (BEST PRACTICE)
    MAX_WORKERS = 2      # Tambahkan ini (FIX ERROR)

    # =========================
    # TRAINING
    # =========================
    EPOCHS = 30
    BATCH_SIZE = 32

    # =========================
    # AI Model
    # =========================
    AI_MODEL = 'gemma:7b'  # Ganti dengan model yang Anda gunakan

    