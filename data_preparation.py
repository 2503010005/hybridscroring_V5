# data_preparation.py

import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder
)


# =========================================================
# PREPARE DATA
# =========================================================
def prepare_data(
    df,
    config,
    tokenizer=None,
    scaler=None,
    label_encoder=None,
    fit=True
):

    # =====================================================
    # VALIDATION
    # =====================================================
    required_cols = [

        "student_answer",
        "final_score",
        "sim_feat",
        "sbert_sim_feat",
        "length_feat",
        "course"

    ]

    for col in required_cols:

        if col not in df.columns:

            raise ValueError(
                f"Kolom '{col}' tidak ditemukan"
            )

    # =====================================================
    # TEXT
    # =====================================================
    texts = (
        df["student_answer"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    # =====================================================
    # TOKENIZER
    # =====================================================
    if fit:

        tokenizer = Tokenizer(
            num_words=config.MAX_WORDS,
            oov_token="<OOV>"
        )

        tokenizer.fit_on_texts(texts)

    if tokenizer is None:

        raise ValueError(
            "Tokenizer belum tersedia"
        )

    sequences = tokenizer.texts_to_sequences(
        texts
    )

    X_text = pad_sequences(
        sequences,
        maxlen=config.MAX_LEN,
        padding="post",
        truncating="post"
    )

    # =====================================================
    # COURSE FEATURE
    # =====================================================
    label_encoder = LabelEncoder()

    course_feat = label_encoder.fit_transform(
        df["course"]
    ).reshape(-1,1)

    # =====================================================
    # COURSE ENCODING
    # =====================================================
    if "course" not in df.columns:
        raise ValueError("Kolom 'course' tidak ditemukan")

    if fit:
        label_encoder = LabelEncoder()

        course_codes = (
            label_encoder
            .fit_transform(df["course"])
            .astype("float32")
            .reshape(-1,1)
        )
    else:
        course_codes = (
            label_encoder
            .transform(df["course"])
            .astype("float32")
            .reshape(-1,1)
        )

    # =====================================================
    # BASE NUMERIC FEATURES
    # =====================================================
    base_numeric = np.concatenate([

        df["sim_feat"]
            .astype("float32")
            .values
            .reshape(-1,1),

        df["sbert_sim_feat"]
            .astype("float32")
            .values
            .reshape(-1,1),

        df["length_feat"]
            .astype("float32")
            .values
            .reshape(-1,1),

        course_codes

    ], axis=1)

    # =====================================================
    # OPTIONAL SBERT EMBEDDING
    # =====================================================
    if "sbert_embedding" in df.columns:

        sbert_embeddings = np.stack(
            df["sbert_embedding"].values
        ).astype("float32")

        X_numeric = np.concatenate([

            base_numeric,
            sbert_embeddings

        ], axis=1)

    else:

        X_numeric = base_numeric

    # =====================================================
    # SCALER
    # =====================================================
    if fit:

        scaler = StandardScaler()

        X_numeric = scaler.fit_transform(
            X_numeric
        )

    else:

        if scaler is None:

            raise ValueError(
                "Scaler belum tersedia"
            )

        X_numeric = scaler.transform(
            X_numeric
        )

    X_numeric = X_numeric.astype("float32")

    # =====================================================
    # TARGET
    # =====================================================
    y = (
        df["final_score"]
        .astype("float32")
        .values
    )

    # =====================================================
    # NORMALIZATION TARGET
    # =====================================================
    y = y / 100.0

    # =====================================================
    # REPORT
    # =====================================================
    print("\n📊 PREPARE DATA REPORT")
    print("-" * 40)

    print(f"Total Data     : {len(df)}")
    print(f"Text Shape     : {X_text.shape}")
    print(f"Numeric Shape  : {X_numeric.shape}")
    print(f"Label Shape    : {y.shape}")

    print("-" * 40)

    return (

        X_text,
        X_numeric,
        y,
        tokenizer,
        scaler,
        label_encoder

    )