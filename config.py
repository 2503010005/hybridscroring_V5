# config.py
import os
import tensorflow as tf

class Config:
    # =========================
    # DATA
    # =========================
    DATA_DIR = "data"
    CACHE_DIR = "cache"
    DATA_SAMPLING_DIR = "sampling"
    SAMPLING_CACHE_FILE = os.path.join(
        DATA_SAMPLING_DIR,
        "sampling_indices_v5.pkl"
    )

    LLM_CACHE_FILE = os.path.join(
        CACHE_DIR,
        "llm_scoring_v5.csv"
    )
    SAVE_EVERY = 5  # Simpan cache setiap 5 data baru
    OUTPUT_DIR = "model_v5"

    # =========================
    # PARAMETER DATA (TEXT)
    # =========================
    MAX_WORDS = 10000
    MAX_LEN = 250
    MAX_LEN_ESSAY = 250
    #MAX_LEN_REF = 150

    # =========================
    # ARSITEKTUR MODEL (SPATIO-SEQUENTIAL)
    # =========================
    EMBED_DIM = 128 # Sesuaikan dengan kebutuhan Anda, biasanya antara 128-512
    CNN_FILTERS = 64
    CNN_KERNEL_SIZE = 5
    LSTM_UNITS = 128 # 64 sudah cukup untuk dataset kecil, tapi bisa ditingkatkan jika dataset lebih besar atau kompleksitas lebih tinggi
    DENSE_UNITS = 64
    DROPOUT_RATE = 0.2
    ACTIVATION = 'relu'
    #LOSS = 'huber'  # huber | mse
    LOSS = tf.keras.losses.Huber(delta=1.0)
    METRICS = ['mae']

    # =========================
    # CALLBACKS (STANDAR JURNAL)
    # =========================
    ES_PATIENCE = 5        # Berhenti jika 5 epoch tidak ada kemajuan
    RLRP_FACTOR = 0.2      # Potong LR sebesar 80% jika stagnan
    RLRP_PATIENCE = 3      # Tunggu 3 epoch sebelum potong LR
    MIN_LR = 0.00001       # Lantai terendah LR (Jangan tertukar dengan INITIAL_LR)

    # =========================
    # LLM SAMPLING
    # =========================
    # N_SAMPLES_AI = 0.2   # 20% dari dataset (BEST PRACTICE)
    SET_RATIO = 0.85
    MEDIUM_DATASET_RATIO = 0.65
    LARGE_DATASET_RATIO = 0.75
    MAX_WORKERS = 2      # Tambahkan ini (FIX ERROR)

    # =========================
    # TRAINING
    # =========================
    INITIAL_LR = 0.003 # Sesuaikan dengan kebutuhan Anda pergeseran dari 0.001 ke 0.005 untuk mempercepat konvergensi
    EPOCHS = 30
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2

    # =========================
    # AI Model
    # target_model = gemma4:31b-cloud,qwen3-coder-next:cloud,qwen2.5:3b
    # =========================
    AI_MODEL = 'qwen3-coder-next:cloud'  # Ganti dengan model yang Anda gunakan