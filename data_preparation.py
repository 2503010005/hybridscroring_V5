from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def prepare_data(df, config):
    # =========================
    # Tokenizer (shared vocab)
    # =========================
    tok = Tokenizer(num_words=config.MAX_WORDS, oov_token="<OOV>")
    
    all_text = df["essay_text"].tolist() + df["reference_answer"].tolist()
    tok.fit_on_texts(all_text)

    # =========================
    # TEXT INPUTS
    # =========================
    X_essay = pad_sequences(
        tok.texts_to_sequences(df["essay_text"]),
        maxlen=config.MAX_LEN_ESSAY,
        padding="post",
        truncating="post"
    )

    X_ref = pad_sequences(
        tok.texts_to_sequences(df["reference_answer"]),
        maxlen=config.MAX_LEN_REF,
        padding="post",
        truncating="post"
    )

    # =========================
    # NUMERICAL FEATURES
    # =========================
    X_sim = df["sim_feat"].values.reshape(-1,1).astype("float32")
    X_rub = (df["rubric_feat"].values / 100).reshape(-1,1).astype("float32")
    X_len = df["length_feat"].values.reshape(-1,1).astype("float32")
    # X_gen = (df["gen_score"].values / 100).reshape(-1,1).astype("float32")

    # =========================
    # TARGET
    # =========================
    y = (df["gen_score"].values / 100).astype("float32")

    return (
        X_essay,
        X_ref,
        X_sim,
        X_rub,
        X_len,
        y,
        tok
    )