# api/utils/semantic_features.py

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# TFIDF COSINE
# =====================================================
def compute_tfidf_similarity(
    student_answer,
    reference_answer
):

    tfidf = TfidfVectorizer()

    tfidf_matrix = tfidf.fit_transform([

        student_answer,
        reference_answer

    ])

    sim = cosine_similarity(

        tfidf_matrix[0],
        tfidf_matrix[1]

    )[0][0]

    return float(sim)


# =====================================================
# SBERT FEATURE
# =====================================================
def compute_sbert_features(
    student_answer,
    reference_answer,
    sbert_model
):

    student_emb = sbert_model.encode(
        student_answer
    )

    reference_emb = sbert_model.encode(
        reference_answer
    )

    sbert_sim = cosine_similarity(

        student_emb.reshape(1,-1),
        reference_emb.reshape(1,-1)

    )[0][0]

    return (

        student_emb.astype("float32"),
        float(sbert_sim)

    )


# =====================================================
# LENGTH
# =====================================================
def compute_length_feature(
    text
):

    return float(

        len(
            text.split()
        )

    )