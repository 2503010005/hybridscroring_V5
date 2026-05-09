import re
import os
import time
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from ollama import chat
from config import Config
import pandas as pd

# ==========================
# 1. CORE LLM SCORING
# ==========================
def get_llm_score(essay, reference, course="Umum", idx=None):
    """
    Scoring menggunakan LLM (Gemma via Ollama)
    """
    config = Config()
    prompt = f"""
Anda adalah sistem penilai otomatis.

Beri skor 10-100 berdasarkan:
1. Kesesuaian konsep (40%)
2. Kelengkapan jawaban (30%)
3. Relevansi dengan kunci (20%)
4. Struktur penjelasan (10%)

Kunci:
{reference}

Jawaban:
{essay}

Balas hanya SATU angka (10-100).
"""

    try:
        response = chat(
            model=config.AI_MODEL,
            messages=[{'role': 'user', 'content': prompt}]
        )

        text = response.message.content

        match = re.search(r'\d+', text)

        if match:
            score = int(match.group())
            return max(0, min(score, 100))

        return None

    except Exception as e:
        print(f"❌ LLM error idx={idx}: {e}")
        return None


# ==========================
# 2. SMART SAMPLING ENGINE
# ==========================
def smart_sampling_llm(df, sample_indices, max_workers=1, save_every=5):
    """
    Jalankan LLM scoring dengan progress bar + statistik
    """

    print("\n🚀 LLM Smart Sampling Started...\n")
    config = Config()

    # =========================
    # LOAD CACHE JIKA ADA
    # =========================
    results = {}
    success = 0
    failed = 0

    if os.path.exists(config.LLM_CACHE_FILE):
        cache_df = pd.read_csv(config.LLM_CACHE_FILE)
        cache_dict = dict(zip(cache_df["index"], cache_df["gen_score"]))
        results = dict(cache_dict)
        print(f"✓ Resume from cache: {len(cache_dict)} data")
    else:
        cache_dict = {}


    start_time = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        futures = {
            executor.submit(
                get_llm_score,
                df.loc[idx, 'student_answer'],
                df.loc[idx, 'reference_answer'],
                df.loc[idx, 'course'] if 'course' in df.columns else "Umum",
                idx
            ): idx for idx in sample_indices
        }

        # Progress bar realtime
        for future in tqdm(as_completed(futures), total=len(futures), desc="LLM Scoring"):

            idx = futures[future]

            try:
                score = future.result()

                if score is not None:
                    results[idx] = score
                    success += 1
                else:
                    failed += 1

            except Exception as e:
                print(f"\n❌ Exception idx={idx}: {e}")
                failed += 1

    total_time = time.time() - start_time

    # ==========================
    # SUMMARY REPORT
    # ==========================
    print("\n" + "=" * 50)
    print("📊 LLM SAMPLING REPORT")
    print("=" * 50)
    print(f"Total Sample   : {len(sample_indices)}")
    print(f"Success        : {success}")
    print(f"Failed         : {failed}")
    print(f"Success Rate   : {success/len(sample_indices)*100:.2f}%")
    print(f"Total Time     : {total_time:.2f} sec")
    print(f"Avg/sample     : {total_time/len(sample_indices):.2f} sec")
    print("=" * 50)

    return results

# ==========================
# 3. CACHE MANAGEMENT
# ========================== 
def save_cache():
    pickle.dump(cache, open(CACHE_FILE,"wb"))