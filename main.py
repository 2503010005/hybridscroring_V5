# =============================================================================
# HYBRID AUTOMATED ESSAY SCORING SYSTEM (FINAL PIPELINE v5)
# =============================================================================
import os
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import json
import pickle

# Internal modules (PASTIKAN NAMA SUDAH SESUAI FILE)
from config import Config
from data_loader import load_dataset
from feature_engineering import build_features, normalize_features
from sampling import hybrid_sampling
from llm_scoring import smart_sampling_llm
from proxy import proxy_fill
from data_preparation import prepare_data
# from model_builder import build_model
from trainer import train_model
from llm_cache import load_llm_cache, save_llm_cache
from evaluate import evaluate_model

import matplotlib.pyplot as plt
from scipy.stats import pearsonr



# =============================================================================
# 1. CONFIG
# =============================================================================
print("\n" + "=" * 60)
print("STEP 1: LOAD CONFIGURATION")
print("=" * 60)

config = Config()

totaldata = 0
SEED = 42

# =============================================================================
# CPU ONLY + REPRODUCIBILITY
# =============================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

random.seed(SEED)
np.random.seed(SEED)

tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================
if not os.path.exists(config.OUTPUT_DIR):
    os.makedirs(config.OUTPUT_DIR)

# =============================================================================
# 2. LOAD DATA
# =============================================================================
print("\n" + "=" * 60)
print("STEP 2: LOAD DATASET")
print("=" * 60)

df = load_dataset(config.DATA_DIR)

totaldata = len(df)

print(f"✓ Total data: {len(df)}")

# =============================================================================
# 3. TRAIN TEST SPLIT
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3: TRAIN TEST SPLIT")
print("=" * 60)

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=SEED,
    stratify=df["course"]
)

# =====================================================
# RESET INDEX (WAJIB)
# =====================================================
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"✓ Train data : {len(train_df)}")
print(f"✓ Test data  : {len(test_df)}")

# =============================================================================
# 4. FEATURE ENGINEERING
# =============================================================================
print("\n" + "=" * 60)
print("STEP 4: FEATURE ENGINEERING")
print("=" * 60)

# =========================
# TRAIN FEATURES
# =========================
train_df, artifacts = build_features(train_df)

# =========================
# TEST FEATURES
# =========================
test_df, _ = build_features(
    test_df,
    artifacts=artifacts
)

# =========================
# NORMALIZATION
# =========================
train_df, feature_scaler = normalize_features(
    train_df
)

# =============================================================================
# NORMALIZE TEST
# =============================================================================
test_df = normalize_features(
    test_df,
    scaler=feature_scaler
)

# =====================================
# SAVE FEATURE SCALER
# =====================================
os.makedirs(
    config.OUTPUT_DIR,
    exist_ok=True
)

with open(
    os.path.join(
        config.OUTPUT_DIR,
        "feature_scaler_v4500.pkl"
    ),
    "wb"
) as f:

    pickle.dump(
        feature_scaler,
        f
    )

print("✓ Feature scaler saved")
print("✓ Features created successfully")


# =============================================================================
# 5. HYBRID SAMPLING
# =============================================================================
print("\n" + "=" * 60)
print("STEP 5: HYBRID SAMPLING")
print("=" * 60)

import pickle

if len(train_df) < 1000:
    ratio = Config.SET_RATIO
elif len(train_df) < 5000:
    ratio = Config.MEDIUM_DATASET_RATIO
else:
    ratio = Config.LARGE_DATASET_RATIO

n_samples = int(len(train_df) * ratio)

# =========================================
# LOAD CACHE SAMPLING
# =========================================
sample_indices = None

if os.path.exists(config.LLM_CACHE_FILE):

    try:

        with open(config.LLM_CACHE_FILE, "rb") as f:
            sample_indices = pickle.load(f)

        print(f"✓ Sampling cache loaded: {len(sample_indices)} samples")

    except Exception as e:

        print(f"⚠ Sampling cache rusak: {e}")
        print("⚠ Regenerating sampling...")

        sample_indices = None

# =========================================
# GENERATE SAMPLING JIKA BELUM ADA
# =========================================
if sample_indices is None:

    sample_indices = hybrid_sampling(
        train_df,
        n_samples=n_samples
    )

    os.makedirs(
        os.path.dirname(config.LLM_CACHE_FILE),
        exist_ok=True
    )

    with open(config.LLM_CACHE_FILE, "wb") as f:
        pickle.dump(sample_indices, f)

    print(f"✓ Sampling cache saved")

print(f"✓ Selected {len(sample_indices)} train samples")

# =============================================================================
# 6. LLM SCORING (TRAIN ONLY)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 6: LLM SCORING")
print("=" * 60)

# =====================================================
# UNIQUE ESSAY ID
# =====================================================
if "essay_id" not in train_df.columns:

    train_df["essay_id"] = (
        train_df["course"].astype(str)
        + "_"
        + train_df.index.astype(str)
    )

# =====================================================
# VERSIONED CACHE FILE
# =====================================================
cache_file = (
    f"{config.OUTPUT_DIR}/"
    f"llm_cache_v{totaldata}.csv"
)

cache = load_llm_cache(cache_file)

# =====================================================
# LOAD CACHE
# =====================================================
if cache is not None:

    print(f"✓ LLM cache loaded: {len(cache)}")

    cache_dict = dict(
        zip(
            cache["essay_id"],
            cache["gen_score"]
        )
    )

else:

    print("✓ No cache found")

    cache_dict = {}

# =====================================================
# REMAINING SAMPLE
# =====================================================
remaining = []

sample_indices = [
    int(idx)
    for idx in sample_indices
]

for idx in sample_indices:

    if idx >= len(train_df):
        continue

    essay_id = str(
        train_df.at[idx, "essay_id"]
    )

    if essay_id not in cache_dict:
        remaining.append(idx)

print(f"✓ Remaining LLM samples: {len(remaining)}")

# =====================================================
# RUN LLM
# =====================================================
if len(remaining) > 0:

    new_results = smart_sampling_llm(
        train_df,
        remaining,
        max_workers=config.MAX_WORKERS
    )

    # =========================================
    # UPDATE CACHE
    # =========================================
    for idx, score in new_results.items():

        essay_id = train_df.loc[idx, "essay_id"]

        cache_dict[essay_id] = score

    # =========================================
    # SAVE CACHE
    # =========================================
    cache_df = pd.DataFrame({

        "essay_id": list(cache_dict.keys()),
        "gen_score": list(cache_dict.values())

    })

    save_llm_cache(
        cache_df,
        cache_file
    )

# =====================================================
# MAP SCORE
# =====================================================
train_df["gen_score"] = (
    train_df["essay_id"]
    .map(cache_dict)
)

# =====================================================
# VALIDATION
# =====================================================
train_df["gen_score"] = (
    train_df["gen_score"]
    .clip(0, 100)
)

print(
    f"✓ LLM scored samples: "
    f"{train_df['gen_score'].notna().sum()}"
)

print(
    f"✓ Missing gen_score: "
    f"{train_df['gen_score'].isna().sum()}"
)

print("\n📊 LLM SCORE DISTRIBUTION")
print(train_df["gen_score"].describe())

# =============================================================================
# 7. PROXY FILL TRAIN ONLY
# =============================================================================
print("\n" + "=" * 60)
print("STEP 7: PROXY LABEL COMPLETION")
print("=" * 60)

# =====================================================
# SAVE RAW LLM SCORE
# =====================================================
train_df["llm_score"] = train_df["gen_score"]

# =====================================================
# PROXY FILL
# =====================================================
train_df, proxy_model, method = proxy_fill(train_df)

# =====================================================
# RENAME FINAL LABEL
# =====================================================
train_df["final_score"] = (
    train_df["gen_score"]
    .astype("float32")
)

# =====================================================
# SAFETY CLIP
# =====================================================
train_df["final_score"] = (
    train_df["final_score"]
    .clip(0, 100)
)

# =====================================================
# TEST LABEL (REAL HUMAN SCORE)
# =====================================================
test_df["final_score"] = (
    test_df["instructor_score"]
    .astype("float32")
)

# =====================================================
# VALIDATION
# =====================================================
missing = (
    train_df["final_score"]
    .isna()
    .sum()
)

print(f"✓ Proxy fill method : {method}")
print(f"✓ Missing values    : {missing}")

print("\n📊 FINAL SCORE DISTRIBUTION")
print(
    train_df["final_score"]
    .describe()
)

# =============================================================================
# 8. PREPARE TRAIN DATA
# =============================================================================
print("\n" + "=" * 60)
print("STEP 8: PREPARE TRAIN DATA")
print("=" * 60)

(
    X_text_train,
    X_num_train,
    y_train,
    tokenizer,
    scaler,
    label_encoder
) = prepare_data(
    train_df,
    config,
    fit=True
)

print(f"✓ Train tensor : {X_text_train.shape}")
print(f"✓ Train numeric: {X_num_train.shape}")

print(f"✓ Train labels : {y_train.shape}")

# =====================================================
# SAVE PREPROCESSING ARTIFACTS
# =====================================================


os.makedirs(
    config.OUTPUT_DIR,
    exist_ok=True
)

# tokenizer
with open(
    os.path.join(
        config.OUTPUT_DIR,
        f"tokenizer_v{totaldata}.pkl"
    ),
    "wb"
) as f:

    pickle.dump(
        tokenizer,
        f
    )

# scaler
with open(
    os.path.join(
        config.OUTPUT_DIR,
        f"scaler_v{totaldata}.pkl"
    ),
    "wb"
) as f:

    pickle.dump(
        scaler,
        f
    )

# course encoder
with open(
    os.path.join(
        config.OUTPUT_DIR,
        f"course_encoder_v{totaldata}.pkl"
    ),
    "wb"
) as f:

    pickle.dump(
        label_encoder,
        f
    )

print(
    "✓ Preprocessing artifacts saved"
)

# =============================================================================
# 9. PREPARE TEST DATA
# =============================================================================
print("\n" + "=" * 60)
print("STEP 9: PREPARE TEST DATA")
print("=" * 60)

(
    X_text_test,
    X_num_test,
    y_test,
    _,
    _,
    _
) = prepare_data(
    test_df,
    config,
    tokenizer=tokenizer,
    scaler=scaler,
    label_encoder=label_encoder,
    fit=False
)

print(f"✓ Test tensor : {X_text_test.shape}")
print(f"✓ Test numeric: {X_num_test.shape}")
print(f"✓ Test labels : {y_test.shape}")

# =============================================================================
# 10. BUILD MODEL
# =============================================================================
print("\n" + "=" * 60)
print("STEP 10: BUILD MODEL")
print("=" * 60)

# =============================================================================
# VOCAB SIZE
# =============================================================================
vocab_size = min(
    len(tokenizer.word_index) + 1,
    config.MAX_WORDS
)

# =============================================================================
# MAX SEQUENCE LENGTH
# =============================================================================
max_len = config.MAX_LEN

print(f"✓ Vocabulary Size : {vocab_size}")
print(f"✓ Max Sequence Len: {max_len}")

# =============================================================================
# 11. TRAIN MODEL
# =============================================================================
print("\n" + "=" * 60)
print("STEP 11: TRAINING")
print("=" * 60)

# =============================================================================
# MODEL INPUT CONFIG
# =============================================================================
v_size = int(
    min(
        len(tokenizer.word_index) + 1,
        config.MAX_WORDS
    )
)

m_len = config.MAX_LEN

# =============================================================================
# TRAIN MODEL
# =============================================================================
model, history = train_model(
    X_text_train,
    X_num_train,
    y_train,
    v_size,
    m_len,
)
# =============================================================================
# 12. EVALUATION
# =============================================================================
print("\n" + "=" * 60)
print("STEP 12: EVALUATION")
print("=" * 60)

results, pred, y_actual, qwk, r2, mae, rmse, mse = evaluate_model(
    model,
    test_data=(
        X_text_test,
        X_num_test,
        y_test
    )
)

# =============================================================================
# INVERSE NORMALIZATION
# =============================================================================
# training menggunakan y/100
# maka hasil prediksi dikembalikan ke skala 0-100

#pred = pred * 100.0
#y_actual = y_actual * 100.0

# safety clipping
#pred = np.clip(pred, 0, 100)
#y_actual = np.clip(y_actual, 0, 100)

print("\n✓ Evaluation completed")
#print(f"✓ Prediction shape : {pred.shape}")
#print(f"✓ Actual shape     : {y_actual.shape}")

# =============================================================================
# 13. SAVE ARTIFACTS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 13: SAVING MODEL & ARTIFACTS")
print("=" * 60)



# =============================================================================
# SAVE MODEL
# =============================================================================
model_path = (
    f"{config.OUTPUT_DIR}/"
    f"hybrid_model_v{totaldata}.keras"
)

model.save(model_path)

print(f"✓ Model saved: {model_path}")

# =============================================================================
# SAVE TOKENIZER
# =============================================================================
tokenizer_path = (
    f"{config.OUTPUT_DIR}/"
    f"tokenizer_v{totaldata}.pkl"
)

with open(tokenizer_path, "wb") as f:
    pickle.dump(tokenizer, f)

print(f"✓ Tokenizer saved: {tokenizer_path}")

# =============================================================================
# SAVE PROXY MODEL
# =============================================================================
if proxy_model is not None:

    proxy_path = (
        f"{config.OUTPUT_DIR}/"
        f"proxy_model_v{totaldata}.pkl"
    )

    with open(proxy_path, "wb") as f:
        pickle.dump(proxy_model, f)

    print(f"✓ Proxy model saved: {proxy_path}")

# =============================================================================
# SAVE EVALUATION REPORT
# =============================================================================
evaluation_path = (
    f"{config.OUTPUT_DIR}/"
    f"evaluation_v{totaldata}.json"
)

with open(evaluation_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"✓ Evaluation report saved: {evaluation_path}")

# =============================================================================
# SAVE TRAINING HISTORY
# =============================================================================
history_df = pd.DataFrame(history.history)

history_path = (
    f"{config.OUTPUT_DIR}/"
    f"training_history_v{totaldata}.csv"
)

history_df.to_csv(
    history_path,
    index=False
)

print(f"✓ Training history saved: {history_path}")

print("\n" + "=" * 60)
print("PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 60)

# =============================================================================
# 14. ADVANCED VISUALIZATION & STATISTICAL ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 14: ADVANCED VISUALIZATION")
print("=" * 60)



# =============================================================================
# PEARSON CORRELATION
# =============================================================================
pearson_corr, pearson_p = pearsonr(
    y_actual,
    pred
)

print(f"\n📊 Pearson Correlation")
print("-" * 40)
print(f"Correlation (r) : {pearson_corr:.4f}")
print(f"P-value         : {pearson_p:.6f}")

# =============================================================================
# 1. SCATTER PLOT
# =============================================================================
plt.figure(figsize=(8,6))

plt.scatter(
    y_actual,
    pred,
    alpha=0.5
)

# ideal line
plt.plot(
    [0,100],
    [0,100],
    linestyle='--'
)

plt.xlabel("Score Dosen")
plt.ylabel("Score Model")
plt.title(
    f"Score Dosen vs Score Model\n"
    f"QWK={qwk:.3f} | "
    f"R²={r2:.3f} | "
    f"r={pearson_corr:.3f}"
)

plt.xlim(0,100)
plt.ylim(0,100)

plt.grid(True)

scatter_path = (
    f"{config.OUTPUT_DIR}/"
    f"scatter_plot_v{totaldata}.png"
)

plt.savefig(
    scatter_path,
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print(f"✓ Scatter plot saved: {scatter_path}")

# =============================================================================
# 2. LINE COMPARISON PLOT
# =============================================================================
sample_size = min(50, len(y_actual))

plt.figure(figsize=(12,5))

plt.plot(
    y_actual[:sample_size],
    label='Score Dosen'
)

plt.plot(
    pred[:sample_size],
    label='Score Model'
)

plt.xlabel("Sample")
plt.ylabel("Score")

plt.title(
    "Perbandingan Score Dosen dan Score Model"
)

plt.legend()
plt.grid(True)

lineplot_path = (
    f"{config.OUTPUT_DIR}/"
    f"line_comparison_v{totaldata}.png"
)

plt.savefig(
    lineplot_path,
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print(f"✓ Line comparison saved: {lineplot_path}")

# =============================================================================
# 3. ERROR DISTRIBUTION
# =============================================================================
errors = np.abs(
    y_actual - pred
)

plt.figure(figsize=(8,5))

plt.hist(
    errors,
    bins=20
)

plt.xlabel("Absolute Error")
plt.ylabel("Frequency")

plt.title(
    "Distribusi Error Prediksi"
)

plt.grid(True)

error_path = (
    f"{config.OUTPUT_DIR}/"
    f"error_distribution_v{totaldata}.png"
)

plt.savefig(
    error_path,
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print(f"✓ Error distribution saved: {error_path}")

# =============================================================================
# 4. BLAND ALTMAN PLOT
# =============================================================================
mean_scores = (
    y_actual + pred
) / 2

diff_scores = (
    y_actual - pred
)

mean_diff = np.mean(diff_scores)
std_diff = np.std(diff_scores)

upper_limit = (
    mean_diff + 1.96 * std_diff
)

lower_limit = (
    mean_diff - 1.96 * std_diff
)

plt.figure(figsize=(8,6))

plt.scatter(
    mean_scores,
    diff_scores,
    alpha=0.5
)

plt.axhline(
    mean_diff,
    linestyle='--'
)

plt.axhline(
    upper_limit,
    linestyle='--'
)

plt.axhline(
    lower_limit,
    linestyle='--'
)

plt.xlabel("Mean Score")
plt.ylabel("Difference")

plt.title("Bland-Altman Plot")

plt.grid(True)

bland_path = (
    f"{config.OUTPUT_DIR}/"
    f"bland_altman_v{totaldata}.png"
)

plt.savefig(
    bland_path,
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print(f"✓ Bland-Altman plot saved: {bland_path}")

# =============================================================================
# 5. INTERVAL AGREEMENT
# =============================================================================
intervals = [
    0,10,20,30,40,
    50,60,70,80,90,100
]

actual_bins = np.digitize(
    y_actual,
    intervals
)

pred_bins = np.digitize(
    pred,
    intervals
)

interval_accuracy = np.mean(
    actual_bins == pred_bins
)

adjacent_accuracy = np.mean(
    np.abs(actual_bins - pred_bins) <= 1
)

print("\n📊 INTERVAL AGREEMENT")
print("-" * 40)

print(
    f"Exact Interval Accuracy    : "
    f"{interval_accuracy:.4f}"
)

print(
    f"Adjacent Interval Accuracy : "
    f"{adjacent_accuracy:.4f}"
)

# =============================================================================
# 6. CONFUSION INTERVAL MATRIX
# =============================================================================
cm = confusion_matrix(
    actual_bins,
    pred_bins
)

plt.figure(figsize=(8,6))

plt.imshow(cm)

plt.xlabel("Predicted Interval")
plt.ylabel("Actual Interval")

plt.title(
    "Interval Confusion Matrix"
)

plt.colorbar()

cm_path = (
    f"{config.OUTPUT_DIR}/"
    f"confusion_interval_v{totaldata}.png"
)

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print(f"✓ Confusion matrix saved: {cm_path}")

# =============================================================================
# 7. REGRESSION PLOT
# =============================================================================
z = np.polyfit(
    y_actual,
    pred,
    1
)

p = np.poly1d(z)

plt.figure(figsize=(8,6))

plt.scatter(
    y_actual,
    pred,
    alpha=0.5
)

# regression line
plt.plot(
    y_actual,
    p(y_actual)
)

# ideal line
plt.plot(
    [0,100],
    [0,100],
    linestyle='--'
)

plt.xlabel("Score Dosen")
plt.ylabel("Score Model")

plt.title(
    f"Regression Correlation\n"
    f"Pearson r={pearson_corr:.3f}"
)

plt.grid(True)

regression_path = (
    f"{config.OUTPUT_DIR}/"
    f"regression_plot_v{totaldata}.png"
)

plt.savefig(
    regression_path,
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print(f"✓ Regression plot saved: {regression_path}")

# =============================================================================
# 8. TRAINING LOSS CURVE
# =============================================================================
plt.figure(figsize=(8,5))

plt.plot(
    history.history['loss'],
    label='Train Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Training Curve"
)

plt.legend()
plt.grid(True)

loss_curve_path = (
    f"{config.OUTPUT_DIR}/"
    f"loss_curve_v{totaldata}.png"
)

plt.savefig(
    loss_curve_path,
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print(f"✓ Loss curve saved: {loss_curve_path}")

# =============================================================================
# 9. SAVE ADVANCED REPORT
# =============================================================================
advanced_report = {

    "pearson_correlation":
        float(pearson_corr),

    "pearson_p_value":
        float(pearson_p),

    "interval_accuracy":
        float(interval_accuracy),

    "adjacent_interval_accuracy":
        float(adjacent_accuracy),

    "mean_difference":
        float(mean_diff),

    "std_difference":
        float(std_diff),

    "upper_limit_agreement":
        float(upper_limit),

    "lower_limit_agreement":
        float(lower_limit)
}

advanced_report_path = (
    f"{config.OUTPUT_DIR}/"
    f"advanced_statistics_v{totaldata}.json"
)

with open(
    advanced_report_path,
    "w"
) as f:

    json.dump(
        advanced_report,
        f,
        indent=2
    )

print(
    f"✓ Advanced statistical report saved: "
    f"{advanced_report_path}"
)

print("\n" + "=" * 60)
print("ADVANCED VISUALIZATION COMPLETED")
print("=" * 60)