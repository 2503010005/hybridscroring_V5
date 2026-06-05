# api/utils/preprocessing.py

import numpy as np

from tensorflow.keras.preprocessing.sequence import (
    pad_sequences
)

from .semantic_features import (

    compute_tfidf_similarity,
    compute_sbert_features,
    compute_length_feature

)


# =====================================================
# PREPARE INFERENCE
# =====================================================
def prepare_inference(

    student_answer,
    reference_answer,
    course,

    tokenizer,
    scaler,
    course_encoder,
    sbert_model,
    max_len

):

    # =========================================
    # TEXT
    # =========================================
    seq = tokenizer.texts_to_sequences([

        student_answer

    ])

    X_text = pad_sequences(

        seq,
        maxlen=max_len,
        padding="post",
        truncating="post"

    )


    # =========================================
    # TFIDF SIM
    # =========================================
    sim_feat = compute_tfidf_similarity(

        student_answer,
        reference_answer

    )


    # =========================================
    # SBERT
    # =========================================
    student_emb, sbert_sim = (

        compute_sbert_features(

            student_answer,
            reference_answer,
            sbert_model

        )

    )


    # =========================================
    # LENGTH
    # =========================================
    length_feat = compute_length_feature(

        student_answer

    )


    # =========================================
    # COURSE
    # =========================================
    course_code = (

        course_encoder
        .transform([course])[0]

    )


    # =========================================
    # LENGTH
    # =========================================
    length_feat = compute_length_feature(
        student_answer
    )

    # =========================================
    # MATCH TRAINING NORMALIZATION
    # =========================================
    sim_feat = sim_feat / 1.0
    sbert_sim = sbert_sim / 1.0
    length_feat = length_feat / 100.0

    base_numeric = np.array([[

        sim_feat,
        sbert_sim,
        length_feat,
        float(course_code)

    ]])

    X_numeric = np.concatenate([

        base_numeric,
        student_emb.reshape(1,-1)

    ], axis=1)

    # =========================================
    # DEBUG RAW FEATURE
    # =========================================
    print("=" * 50)
    print("RAW FEATURE")

    print("TFIDF RAW :", sim_feat)
    print("SBERT RAW :", sbert_sim)
    print("LENGTH RAW:", length_feat)
    print("COURSE RAW:", course_code)

    print("=" * 50)


    X_numeric = scaler.transform(
        X_numeric
    ).astype("float32")

    # =========================================
    # DEBUG SCALED
    # =========================================
    print("=" * 50)
    print("SCALED FEATURE")

    print("TFIDF :", X_numeric[0][0])
    print("SBERT :", X_numeric[0][1])
    print("LENGTH:", X_numeric[0][2])
    print("COURSE:", X_numeric[0][3])

    print("=" * 50)

    return (

        X_text,
        X_numeric

    )

# =====================================================
# REPETITION RATIO
# =====================================================
def repetition_ratio(text):

    words = text.lower().split()

    if len(words) == 0:
        return 0

    unique_words = len(set(words))

    ratio = 1 - (
        unique_words / len(words)
    )

    return ratio