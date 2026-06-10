# llm_cache.py

import os
import pandas as pd


# =====================================================
# LOAD CACHE
# =====================================================
def load_llm_cache(cache_file):

    if not os.path.exists(cache_file):

        print("✓ No cache file found")

        return None

    try:

        cache_df = pd.read_csv(cache_file)

        # =============================================
        # VALIDATION
        # =============================================
        required_cols = [
            "essay_id",
            "gen_score"
        ]

        for col in required_cols:

            if col not in cache_df.columns:

                raise ValueError(
                    f"Cache missing column: {col}"
                )

        # =============================================
        # DROP DUPLICATES
        # =============================================
        before = len(cache_df)

        cache_df = cache_df.drop_duplicates(
            subset=["essay_id"]
        )

        after = len(cache_df)

        if before != after:

            print(
                f"⚠ Duplicate cache removed: "
                f"{before-after}"
            )

        print(
            f"✓ Load LLM cache: "
            f"{cache_file}"
        )

        print(
            f"✓ Cache entries: "
            f"{len(cache_df)}"
        )

        return cache_df

    except Exception as e:

        print(
            f"❌ Failed loading cache: {e}"
        )

        return None


# =====================================================
# SAVE CACHE
# =====================================================
def save_llm_cache(df_cache, cache_file):

    try:

        # =============================================
        # VALIDATION
        # =============================================
        required_cols = [
            "essay_id",
            "gen_score"
        ]

        for col in required_cols:

            if col not in df_cache.columns:

                raise ValueError(
                    f"Missing cache column: {col}"
                )

        # =============================================
        # REMOVE DUPLICATE
        # =============================================
        df_cache = df_cache.drop_duplicates(
            subset=["essay_id"]
        )

        # =============================================
        # SORT FOR REPRODUCIBILITY
        # =============================================
        df_cache = df_cache.sort_values(
            by="essay_id"
        )

        # =============================================
        # CREATE DIRECTORY
        # =============================================
        os.makedirs(
            os.path.dirname(cache_file),
            exist_ok=True
        )

        # =============================================
        # SAVE
        # =============================================
        df_cache.to_csv(
            cache_file,
            index=False
        )

        print(
            f"✓ LLM cache saved: "
            f"{cache_file}"
        )

        print(
            f"✓ Total cache saved: "
            f"{len(df_cache)}"
        )

    except Exception as e:

        print(
            f"❌ Failed saving cache: {e}"
        )