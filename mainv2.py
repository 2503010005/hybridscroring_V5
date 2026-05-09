# =============================================================================
# HYBRID AUTOMATED ESSAY SCORING SYSTEM (FINAL PIPELINE v5)
# =============================================================================

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Internal modules (PASTIKAN NAMA SUDAH SESUAI FILE)
from config import Config
from data_loader import load_dataset
from feature_engineering import build_features, normalize_features
from sampling import hybrid_sampling
from llm_scoring import smart_sampling_llm
from proxy import proxy_fill
from data_preparation import prepare_data
from model_builder import build_model
from trainer import train_model, evaluate_model
from llm_cache import load_llm_cache, save_llm_cache

# =============================================================================
# 1. CONFIG
# =============================================================================
config = Config()
 
if not os.path.exists(config.OUTPUT_DIR):
    os.makedirs(config.OUTPUT_DIR)

print("=" * 60)
print("STEP 1: LOAD DATA")
print("=" * 60)

# =============================================================================
# 2. LOAD DATA
# =============================================================================
df = load_dataset(config.DATA_DIR)
print(f"✓ Total data: {len(df)}")

# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================
print("\n" + "=" * 60)
print("STEP 2: FEATURE ENGINEERING")
print("=" * 60)

df, artifacts = build_features(df)
df = normalize_features(df)
print("✓ Features created: sim_feat, rubric_feat, length_feat")

# =============================================================================
# 4. HYBRID SAMPLING
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3: HYBRID SAMPLING")
print("=" * 60)

if len(df) < 1000:
    ratio = 0.4
elif len(df) < 10000:
    ratio = 0.2
else:
    ratio = 0.1

n_samples = int(len(df) * ratio)

sample_indices = hybrid_sampling(df, n_samples)

print(f"✓ Selected {len(sample_indices)} samples for LLM scoring")

# =============================================================================
# 5. LLM SCORING
# =============================================================================
print("\n" + "=" * 60)
print("STEP 4: LLM SCORING")
print("=" * 60)

cache = load_llm_cache(config.LLM_CACHE_FILE)

if cache is not None:
    cache_dict = dict(zip(cache["index"], cache["gen_score"]))
else:
    cache_dict = {}

# sample_indices = hybrid_sampling(df, n_samples)

# Ambil yang BELUM ada di cache
remaining = [idx for idx in sample_indices if idx not in cache_dict]

print(f"✓ Cached: {len(cache_dict)} | Remaining: {len(remaining)}")

if(len(remaining) > 0):
    new_results = smart_sampling_llm(
        df,
        remaining,
        max_workers=config.MAX_WORKERS
    )

    # Merge hasil lama + baru
    cache_dict.update(new_results)

df["gen_score"] = df.index.map(cache_dict)


# Save ulang
cache_df = pd.DataFrame({
    "index": list(cache_dict.keys()),
    "gen_score": list(cache_dict.values())
})

save_llm_cache(cache_df, config.LLM_CACHE_FILE)

valid_count = df['gen_score'].notna().sum()
print(f"✓ LLM scored: {valid_count} samples")

# =============================================================================
# 6. PROXY FILLING
# =============================================================================
print("\n" + "=" * 60)
print("STEP 5: PROXY FILLING")
print("=" * 60)

# =========================
# SAFETY CHECK
# =========================
print("\n⚠ Checking 'gen_score' column before proxy filling...")
if "gen_score" not in df.columns:
    raise ValueError("❌ Kolom 'gen_score' tidak ditemukan sebelum proxy filling")

total_data = len(df)
available = df['gen_score'].notna().sum()
missing = df['gen_score'].isna().sum()

print(f"✓ Total data          : {total_data}")
print(f"✓ Available gen_score : {available}")
print(f"⚠ Missing gen_score   : {missing}")

# =========================
# JALANKAN PROXY MODEL
# =========================
df, proxy_model, method = proxy_fill(df)

print(f"✓ Proxy method used   : {method}")

# =========================
# CEK HASIL PROXY
# =========================
remaining_nan = df['gen_score'].isna().sum()

if remaining_nan > 0:
    print(f"⚠ Remaining missing after proxy: {remaining_nan}")
else:
    print("✓ All missing values filled by proxy")

# =========================
# FAIL-SAFE FALLBACK
# =========================
if remaining_nan > 0:

    df['gen_score'] = df['gen_score'].fillna(df['sim_feat'] * 100)

    print(f"⚠ {remaining_nan} values filled using similarity fallback")

# =========================
# FINAL VALIDATION
# =========================
final_missing = df['gen_score'].isna().sum()

if final_missing > 0:
    raise ValueError(f"❌ Masih ada {final_missing} NaN setelah proxy + fallback")

# =========================
# CLIPPING (SAFETY)
# =========================
df['gen_score'] = df['gen_score'].clip(0, 100)

# =========================
# SUMMARY
# =========================
print("\n📊 PROXY FILLING SUMMARY")
print("-" * 40)
print(f"Total Data     : {total_data}")
print(f"LLM Filled     : {available}")
print(f"Proxy Filled   : {missing - remaining_nan if missing > 0 else 0}")
print(f"Final Missing  : {final_missing}")
print(f"Score Range    : {df['gen_score'].min():.2f} - {df['gen_score'].max():.2f}")
print("=" * 60)

# =========================
# LOGGING YANG BENAR
# =========================
if method == "proxy":
    print("✓ Missing values predicted using RandomForest proxy model")
else:
    print("⚠ Proxy not used, fallback to final_score")

# =============================================================================
# 7. PREPARE DATA
# =============================================================================
print("\n" + "=" * 60)
print("STEP 6: DATA PREPARATION")
print("=" * 60)

(
    X_essay,
    X_ref,
    X_sim,
    X_rub,
    X_len,
    X_gen,
    y,
    tokenizer
) = prepare_data(df, config)

# =============================================================================
# 8. TRAIN-TEST SPLIT
# =============================================================================
print("\n" + "=" * 60)
print("STEP 7: TRAIN-TEST SPLIT")
print("=" * 60)

(
    X_essay_train, X_essay_test,
    X_ref_train, X_ref_test,
    X_sim_train, X_sim_test,
    X_rub_train, X_rub_test,
    X_len_train, X_len_test,
    X_gen_train, X_gen_test,
    y_train, y_test
) = train_test_split(
    X_essay, X_ref, X_sim, X_rub, X_len, X_gen, y,
    test_size=0.2,
    random_state=42,
    stratify=pd.cut(y, bins=5, labels=False, duplicates='drop')
)

print(f"✓ Train: {len(y_train)} | Test: {len(y_test)}")

# =============================================================================
# 9. BUILD MODEL
# =============================================================================
print("\n" + "=" * 60)
print("STEP 8: BUILD MODEL")
print("=" * 60)

# =========================
# VOCAB SIZE
# =========================
vocab_size = min(
    len(tokenizer.word_index) + 1,
    config.MAX_WORDS
)

model = build_model(
    config,
    vocab=vocab_size
)
model.summary()

# =============================================================================
# 10. TRAIN MODEL
# =============================================================================
print("\n" + "=" * 60)
print("STEP 9: TRAINING")
print("=" * 60)

history = train_model(
    model,
    train_data=(
        X_essay_train,
        X_ref_train,
        X_sim_train,
        X_rub_train,
        X_len_train,
        X_gen_train,
        y_train
    ),
    val_data=(
        X_essay_test,
        X_ref_test,
        X_sim_test,
        X_rub_test,
        X_len_test,
        X_gen_test,
        y_test
    ),
    config=config
)

# =============================================================================
# 11. EVALUATION
# =============================================================================
print("\n" + "=" * 60)
print("STEP 10: EVALUATION")
print("=" * 60)

results = evaluate_model(
    model,
    test_data=(
        X_essay_test,
        X_ref_test,
        X_sim_test,
        X_rub_test,
        X_len_test,
        X_gen_test,
        y_test
    )
)

# =============================================================================
# 12. SAVE ARTIFACTS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 11: SAVING MODEL & ARTIFACTS")
print("=" * 60)

# Save model
model.save(f"{config.OUTPUT_DIR}/hybrid_model_v5.keras")

# Save tokenizer
import pickle
with open(f"{config.OUTPUT_DIR}/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Save proxy model
if proxy_model:
    with open(f"{config.OUTPUT_DIR}/proxy_model.pkl", "wb") as f:
        pickle.dump(proxy_model, f)

# Save report
import json
with open(f"{config.OUTPUT_DIR}/evaluation.json", "w") as f:
    json.dump(results, f, indent=2)

print("✓ All artifacts saved successfully")

print("\n" + "=" * 60)
print("PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 60)