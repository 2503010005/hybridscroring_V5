import pandas as pd
import numpy as np

def hybrid_sampling(df, n_samples, n_bins=5, random_state=42):
    """
    Hybrid Sampling Strategy:
    - Stratified (berdasarkan distribusi skor)
    - Uncertainty-based (active learning)
    - Exploration (random sampling)

    Parameters:
    - df : DataFrame
    - n_samples : jumlah sample yang diambil
    - n_bins : jumlah stratifikasi
    """

    # =========================
    # 0. VALIDASI
    # =========================
    required_cols = ['sim_feat', 'rubric_feat', 'instructor_score']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan di DataFrame")

    if len(df) == 0:
        raise ValueError("DataFrame kosong")

    if n_samples <= 0:
        raise ValueError("n_samples harus > 0")

    temp = df.copy()

    # =========================
    # 1. UNCERTAINTY (ACTIVE LEARNING)
    # =========================
    temp['uncertainty'] = abs(temp['sim_feat'] - temp['rubric_feat'] / 100)

    # =========================
    # 2. STRATIFIED BINNING
    # =========================
    temp['score_bin'] = pd.cut(
        temp['instructor_score'],
        bins=n_bins,
        labels=False,
        duplicates='drop'
    )

    bins = temp['score_bin'].dropna().unique()

    if len(bins) == 0:
        raise ValueError("Gagal membuat bin (score_bin kosong)")

    samples = []

    # =========================
    # 3. DISTRIBUSI SAMPLE PER BIN
    # =========================
    per_bin = max(1, n_samples // len(bins))

    for b in bins:
        subset = temp[temp['score_bin'] == b]

        if len(subset) == 0:
            continue

        # =========================
        # 4. SORT BY UNCERTAINTY
        # =========================
        subset = subset.sort_values('uncertainty', ascending=False)

        # =========================
        # 5. EXPLOITATION + EXPLORATION
        # =========================
        top_k = int(per_bin * 0.8)
        rand_k = per_bin - top_k

        top_samples = subset.head(top_k)

        rand_samples = subset.sample(
            n=min(rand_k, len(subset)),
            random_state=random_state
        )

        samples.append(pd.concat([top_samples, rand_samples]))

    # =========================
    # 6. GABUNGKAN SEMUA SAMPLE
    # =========================
    result = pd.concat(samples).drop_duplicates()

    # =========================
    # 7. FILL JIKA KURANG
    # =========================
    if len(result) < n_samples:
        remaining = temp.drop(result.index)

        if len(remaining) > 0:
            extra = remaining.sample(
                n=min(n_samples - len(result), len(remaining)),
                random_state=random_state
            )
            result = pd.concat([result, extra])

    # =========================
    # 8. FINAL TRIM
    # =========================
    result = result.sample(
        n=min(n_samples, len(result)),
        random_state=random_state
    )

    return result.index