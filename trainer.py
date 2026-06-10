# trainer.py
from model_factory import build_hybrid_spatio_sequential
from train_module import run_training # Mengimpor logika baru

def train_model(
    X_text,
    X_numeric,
    y,
    vocab_size,
    max_len
):

    print("Memulai pelatihan dengan Arsitektur Hybrid Spatio-Sequential...")

    model, history = run_training(
        X_text,
        X_numeric,
        y,
        vocab_size,
        max_len
    )

    return model, history