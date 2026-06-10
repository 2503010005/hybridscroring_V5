# sampling.py

import pandas as pd
import numpy as np


def hybrid_sampling(
    df,
    n_samples,
    stratify_col="course",
    score_col="instructor_score",
    n_bins=5,
    exploration_ratio=0.2,
    random_state=42
):

    """
    =========================================================
    CONTEXT-AWARE HYBRID SAMPLING
    =========================================================

    Strategy:
    1. Stratified per course
    2. Stratified per score distribution
    3. Uncertainty-based sampling
    4. Exploration random sampling

    Parameters
    ----------
    df : pandas.DataFrame

    n_samples : int
        jumlah sample total

    stratify_col : str
        kolom context/course

    score_col : str
        kolom target score

    n_bins : int
        jumlah bin score

    exploration_ratio : float
        rasio random exploration

    random_state : int
        reproducibility seed
    """

    # =========================================================
    # VALIDATION
    # =========================================================
    if len(df) == 0:
        raise ValueError("DataFrame kosong")

    if n_samples <= 0:
        raise ValueError("n_samples harus > 0")

    required_cols = [
        stratify_col,
        score_col,
        'sim_feat',
        'rubric_feat'
    ]

    for col in required_cols:

        if col not in df.columns:
            raise ValueError(
                f"Kolom '{col}' tidak ditemukan"
            )

    # =========================================================
    # COPY
    # =========================================================
    temp = df.copy()

    # =========================================================
    # UNCERTAINTY SCORE
    # =========================================================
    temp["uncertainty"] = np.abs(
        temp["sim_feat"] -
        (temp["rubric_feat"] / 100.0)
    )

    # =========================================================
    # SCORE BINNING
    # =========================================================
    temp["score_bin"] = pd.cut(
        temp[score_col],
        bins=n_bins,
        labels=False,
        duplicates='drop'
    )

    # =========================================================
    # RANDOM SEED
    # =========================================================
    np.random.seed(random_state)

    # =========================================================
    # FINAL CONTAINER
    # =========================================================
    final_indices = []

    # =========================================================
    # COURSE DISTRIBUTION
    # =========================================================
    unique_courses = (
        temp[stratify_col]
        .dropna()
        .unique()
    )

    if len(unique_courses) == 0:
        raise ValueError(
            f"Kolom '{stratify_col}' kosong"
        )

    # =========================================================
    # SAMPLE PER COURSE
    # =========================================================
    samples_per_course = max(
        1,
        n_samples // len(unique_courses)
    )

    # =========================================================
    # LOOP PER COURSE
    # =========================================================
    for course_name in unique_courses:

        course_df = temp[
            temp[stratify_col] == course_name
        ]

        if len(course_df) == 0:
            continue

        # =====================================================
        # BIN DALAM COURSE
        # =====================================================
        bins = (
            course_df["score_bin"]
            .dropna()
            .unique()
        )

        if len(bins) == 0:
            continue

        samples_per_bin = max(
            1,
            samples_per_course // len(bins)
        )

        # =====================================================
        # LOOP BIN
        # =====================================================
        for b in bins:

            subset = course_df[
                course_df["score_bin"] == b
            ]

            if len(subset) == 0:
                continue

            # =================================================
            # SORT BY UNCERTAINTY
            # =================================================
            subset = subset.sort_values(
                by="uncertainty",
                ascending=False
            )

            # =================================================
            # EXPLOITATION
            # =================================================
            top_k = int(
                samples_per_bin *
                (1 - exploration_ratio)
            )

            top_samples = subset.head(top_k)

            # =================================================
            # EXPLORATION
            # =================================================
            remaining_subset = subset.drop(
                top_samples.index,
                errors='ignore'
            )

            random_k = (
                samples_per_bin - len(top_samples)
            )

            if (
                random_k > 0 and
                len(remaining_subset) > 0
            ):

                random_samples = (
                    remaining_subset.sample(
                        n=min(
                            random_k,
                            len(remaining_subset)
                        ),
                        random_state=random_state
                    )
                )

                combined = pd.concat([
                    top_samples,
                    random_samples
                ])

            else:
                combined = top_samples

            # =================================================
            # SAVE INDEX
            # =================================================
            final_indices.extend(
                combined.index.tolist()
            )

    # =========================================================
    # REMOVE DUPLICATES
    # =========================================================
    final_indices = list(
        dict.fromkeys(final_indices)
    )

    # =========================================================
    # FILL IF LESS
    # =========================================================
    if len(final_indices) < n_samples:

        remaining = temp.drop(
            final_indices,
            errors='ignore'
        )

        if len(remaining) > 0:

            additional = remaining.sample(
                n=min(
                    n_samples - len(final_indices),
                    len(remaining)
                ),
                random_state=random_state
            )

            final_indices.extend(
                additional.index.tolist()
            )

    # =========================================================
    # FINAL TRIM
    # =========================================================
    if len(final_indices) > n_samples:

        np.random.shuffle(final_indices)

        final_indices = final_indices[:n_samples]

    # =========================================================
    # FINAL REPORT
    # =========================================================
    print("\n" + "=" * 50)
    print("📊 HYBRID SAMPLING REPORT")
    print("=" * 50)

    print(f"Total Data           : {len(df)}")
    print(f"Requested Samples    : {n_samples}")
    print(f"Final Samples        : {len(final_indices)}")
    print(f"Total Courses        : {len(unique_courses)}")
    print(f"Score Bins           : {n_bins}")
    print(f"Exploration Ratio    : {exploration_ratio}")

    print("=" * 50)

    return final_indices