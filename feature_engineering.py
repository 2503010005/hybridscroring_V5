# feature_engineering.py

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from sbert_loader import sbert_model


# =============================================================================
# TEXT CLEANING
# =============================================================================
def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    text = text.strip().lower()

    return text


# =============================================================================
# BUILD FEATURES
# =============================================================================
def build_features(df, artifacts=None):

    print("Build Features....")

    df = df.copy()

    # =========================================================================
    # VALIDATION
    # =========================================================================
    required_cols = [
        "student_answer",
        "reference_answer",
        "instructor_score"
    ]

    for col in required_cols:

        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan")

    # =========================================================================
    # CLEAN TEXT
    # =========================================================================
    df["student_answer"] = df["student_answer"].fillna("").astype(str)
    df["reference_answer"] = df["reference_answer"].fillna("").astype(str)

    df["student_answer_clean"] = (
        df["student_answer"]
        .apply(clean_text)
    )

    df["reference_answer_clean"] = (
        df["reference_answer"]
        .apply(clean_text)
    )

    # =========================================================================
    # TFIDF VECTORIZER
    # =========================================================================
    if artifacts is None:

        tfidf_vectorizer = TfidfVectorizer(
            max_features=5000
        )

        tfidf_vectorizer.fit(
            pd.concat([
                df["student_answer_clean"],
                df["reference_answer_clean"]
            ])
        )

    else:

        tfidf_vectorizer = artifacts["tfidf_vectorizer"]

    # =========================================================================
    # TFIDF TRANSFORM
    # =========================================================================
    student_tfidf = tfidf_vectorizer.transform(
        df["student_answer_clean"]
    )

    reference_tfidf = tfidf_vectorizer.transform(
        df["reference_answer_clean"]
    )

    # =========================================================================
    # COSINE SIMILARITY FEATURE
    # =========================================================================
    similarities = []

    for i in range(len(df)):

        sim = cosine_similarity(
            student_tfidf[i],
            reference_tfidf[i]
        )[0][0]

        similarities.append(sim)

    df["sim_feat"] = similarities

    # =========================================================================
    # RUBRIC FEATURE
    # =========================================================================
    #df["rubric_feat"] = (
    #    df["instructor_score"]
    #    .astype(float)
    #)
    # =========================================================================
    # HEURISTIC RUBRIC FEATURE
    # =========================================================================

    rubric_scores = []

    for idx, row in df.iterrows():

        student = row["student_answer_clean"]
        reference = row["reference_answer_clean"]

        student_words = set(student.split())
        ref_words = set(reference.split())

        # keyword overlap
        overlap = len(
            student_words & ref_words
        )

        # coverage ratio
        coverage = overlap / max(len(ref_words), 1)

        # scaled rubric score
        rubric = min(coverage * 100, 100)

        rubric_scores.append(rubric)

    df["rubric_feat"] = rubric_scores

    # =========================================================================
    # LENGTH FEATURE
    # =========================================================================
    df["length_feat"] = (
        df["student_answer_clean"]
        .apply(lambda x: len(x.split()))
    )

    # =========================================================================
    # SBERT EMBEDDING
    # =========================================================================
    student_embeddings = sbert_model.encode(
        df["student_answer_clean"].tolist(),
        show_progress_bar=True,
        convert_to_numpy=True
    )

    reference_embeddings = sbert_model.encode(
        df["reference_answer_clean"].tolist(),
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # =========================================================================
    # SBERT COSINE FEATURE
    # =========================================================================
    sbert_similarities = []

    for i in range(len(df)):

        emb1 = student_embeddings[i].reshape(1, -1)
        emb2 = reference_embeddings[i].reshape(1, -1)

        sim = cosine_similarity(
            emb1,
            emb2
        )[0][0]

        sbert_similarities.append(sim)

    df["sbert_sim_feat"] = sbert_similarities

    # =========================================================================
    # STORE SBERT EMBEDDINGS
    # =========================================================================
    df["sbert_embedding"] = list(student_embeddings)

    # =========================================================================
    # ARTIFACTS
    # =========================================================================
    artifacts = {

        "tfidf_vectorizer":
            tfidf_vectorizer

    }

    return df, artifacts


# =============================================================================
# NORMALIZATION
# =============================================================================
def normalize_features(df, scaler=None):

    df = df.copy()

    feature_cols = [

        "sim_feat",
        "rubric_feat",
        "length_feat",
        "sbert_sim_feat"

    ]

    # =========================================================================
    # VALIDATION
    # =========================================================================
    for col in feature_cols:

        if col not in df.columns:
            raise ValueError(
                f"Kolom '{col}' tidak ditemukan"
            )

    # =========================================================================
    # TRAIN MODE
    # =========================================================================
    if scaler is None:

        scaler = MinMaxScaler()

        df[feature_cols] = scaler.fit_transform(
            df[feature_cols]
        )

        return df, scaler

    # =========================================================================
    # TEST / INFERENCE MODE
    # =========================================================================
    else:

        df[feature_cols] = scaler.transform(
            df[feature_cols]
        )

        return df