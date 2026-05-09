import os
import pandas as pd

def load_llm_cache(cache_file):
    if os.path.exists(cache_file):
        print(f"✓ Load LLM cache: {cache_file}")
        return pd.read_csv(cache_file)
    return None


def save_llm_cache(df_cache, cache_file):
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    df_cache.to_csv(cache_file, index=False)
    print(f"✓ LLM cache saved: {cache_file}")