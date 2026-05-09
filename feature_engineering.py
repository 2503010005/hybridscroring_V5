from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def rubric_score_logic(question, reference, essay):
    essay_words = str(essay).lower().split()
    ref_words = str(reference).lower().split()
    q_words = str(question).lower().split()

    overlap_ref = len(set(essay_words) & set(ref_words))
    overlap_q = len(set(essay_words) & set(q_words))

    length_factor = min(len(essay_words) / 20, 1.0)

    score = (overlap_ref * 8) + (overlap_q * 5) + (length_factor * 20)
    return min(score, 100)


def build_features(df):
    print("Build Features....")

    # =========================
    # TF-IDF Similarity
    # =========================
    vectorizer = TfidfVectorizer(max_features=5000)

    all_text = df["reference_answer"].tolist() + df["student_answer"].tolist()
    tfidf = vectorizer.fit_transform(all_text)

    ref_vec = tfidf[:len(df)]
    ess_vec = tfidf[len(df):]

    df['sim_feat'] = [
        cosine_similarity(ref_vec[i], ess_vec[i])[0][0]
        for i in range(len(df))
    ]

    # =========================
    # Rubric Feature (WAJIB)
    # =========================
    df['rubric_feat'] = df.apply(
        lambda r: rubric_score_logic(
            r.get('question', ''),
            r['reference_answer'],
            r['student_answer']
        ),
        axis=1
    )

    # =========================
    # Length Feature
    # =========================
    df['length_feat'] = df['student_answer'].apply(
        lambda x: min(len(str(x).split()) / 50, 1.0)
    )

    artifacts = {
        "vectorizer": vectorizer
    }

    return df, artifacts

def normalize_features(df):

    df['sim_feat'] = df['sim_feat'].astype("float32")

    df['rubric_feat'] = df['rubric_feat'] / 100.0
    df['length_feat'] = df['length_feat'].astype("float32")

    if 'gen_score' in df.columns:
        df['gen_score'] = df['gen_score'] / 100.0

    if 'final_score' in df.columns:
        df['final_score'] = df['final_score'] / 100.0

    return df