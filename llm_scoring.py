# llm_scoring.py

import re
import time
import pandas as pd

from tqdm import tqdm
from ollama import chat
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed
)

from config import Config

config = Config()


# =========================================================
# 1. SINGLE LLM SCORING
# =========================================================
def get_llm_score(
    question,
    essay,
    reference,
    course="Umum",
    instructor_score=None,
    idx=None
):

    prompt = f"""
Anda adalah dosen ahli bidang Teknik Informatika.

Mata Kuliah:
{course}

Tugas:
Beri nilai essay mahasiswa dari 0-100.

Kriteria:
1. Kesesuaian konsep (40%)
2. Kelengkapan jawaban (30%)
3. Relevansi dengan kunci (20%)
4. Struktur penjelasan (10%)

SOAL:
{question}

JAWABAN MAHASISWA:
{essay}

KUNCI JAWABAN:
{reference}

Nilai Referensi Dosen:
{instructor_score}

ATURAN:
- Balas SATU angka saja
- Jangan beri penjelasan
- Nilai 0-100
"""

    try:

        response = chat(
            model=config.AI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        text = response.message.content.strip()

        match = re.search(r"\d+", text)

        if match:

            score = float(match.group())

            score = max(0, min(score, 100))

            return score

        return None

    except Exception as e:

        print(f"❌ LLM Error idx={idx}: {e}")

        return None


# =========================================================
# 2. SMART SAMPLING LLM
# =========================================================
def smart_sampling_llm(
    df,
    sample_indices,
    max_workers=1
):

    print("\n🚀 LLM Smart Sampling Started...\n")

    results = {}

    success = 0
    failed = 0

    start_time = time.time()

    # =====================================================
    # THREAD EXECUTOR
    # =====================================================
    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:

        futures = {}

        for idx in sample_indices:

            try:

                future = executor.submit(

                    get_llm_score,

                    df.loc[idx, "question"],

                    df.loc[idx, "student_answer"],

                    df.loc[idx, "reference_answer"],

                    df.loc[idx, "course"]
                    if "course" in df.columns
                    else "Umum",

                    df.loc[idx, "instructor_score"]
                    if "instructor_score" in df.columns
                    else None,

                    idx
                )

                futures[future] = idx

            except Exception as e:

                print(f"❌ Prepare Error idx={idx}: {e}")

        # =================================================
        # PROCESS RESULT
        # =================================================
        for future in tqdm(

            as_completed(futures),

            total=len(futures),

            desc="LLM Scoring"

        ):

            idx = futures[future]

            try:

                score = future.result()

                if score is not None:

                    results[idx] = float(score)

                    success += 1

                else:

                    failed += 1

            except Exception as e:

                print(f"❌ Future Error idx={idx}: {e}")

                failed += 1

    total_time = time.time() - start_time

    # =====================================================
    # REPORT
    # =====================================================
    print("\n" + "=" * 50)
    print("📊 LLM SAMPLING REPORT")
    print("=" * 50)

    print(f"Total Sample : {len(sample_indices)}")
    print(f"Success      : {success}")
    print(f"Failed       : {failed}")

    if len(sample_indices) > 0:

        print(
            f"Success Rate : "
            f"{(success/len(sample_indices))*100:.2f}%"
        )

        print(
            f"Avg/sample   : "
            f"{total_time/len(sample_indices):.2f} sec"
        )

    print(f"Total Time   : {total_time:.2f} sec")

    print("=" * 50)

    return results