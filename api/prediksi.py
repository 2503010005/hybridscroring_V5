# predictor.py
import re
import pickle
import numpy as np
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_DIR = "model_v5"

# Load artifacts
print("Loading model artifacts...")

model = tf.keras.models.load_model(
    f"{MODEL_DIR}/hybrid_model_v5000.keras"
)

with open(f"{MODEL_DIR}/tokenizer_v5000.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open(f"{MODEL_DIR}/proxy_model_v5000.pkl", "rb") as f:
    proxy_model = pickle.load(f)

# SBERT
sbert = SentenceTransformer(
    "firqaaa/indo-sentence-bert-base"
)

MAX_LEN = 250


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def vocab_diversity(text):
    words = text.split()
    if len(words) == 0:
        return 0
    return len(set(words)) / len(words)


def sentence_count(text):
    return max(1, text.count(".") + text.count("!") + text.count("?"))


def predict_score(
    student_answer,
    reference_answer,
    course_code=0
):
    student_answer = clean_text(student_answer)
    reference_answer = clean_text(reference_answer)

    # Tokenization
    seq = tokenizer.texts_to_sequences(
        [student_answer]
    )
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        seq,
        maxlen=MAX_LEN
    )

    # SBERT embedding
    stu_emb = sbert.encode(
        [student_answer]
    )

    ref_emb = sbert.encode(
        [reference_answer]
    )

    sim = cosine_similarity(
        stu_emb,
        ref_emb
    )[0][0]

    # Additional features
    vocab = vocab_diversity(student_answer)
    sent_count = sentence_count(student_answer)

    # Proxy score
    proxy_score = proxy_model.predict(
        [[sim, vocab, sent_count]]
    )[0]

    numeric_features = np.array([
        [
            sim,
            vocab,
            sent_count,
            course_code,
            proxy_score
        ]
    ])

    # Predict
    pred = model.predict(
        [padded, numeric_features],
        verbose=0
    )[0][0]

    return {
        "predicted_score": float(pred),
        "semantic_similarity": float(sim),
        "proxy_score": float(proxy_score)
    }