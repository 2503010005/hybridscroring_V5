import numpy as np
from sklearn.metrics import cohen_kappa_score, mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.losses import Huber
from metrics import quadratic_weighted_kappa


# =========================
# QWK METRIC
# =========================
def quadratic_weighted_kappa(y_true, y_pred):
    y_true = np.round(y_true * 100).astype(int)
    y_pred = np.round(y_pred * 100).astype(int)

    y_true = np.clip(y_true, 0, 100)
    y_pred = np.clip(y_pred, 0, 100)

    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


# =========================
# TRAIN FUNCTION
# =========================
def train_model(model, train_data, val_data, config):

    (
        X_essay_train,
        X_ref_train,
        X_sim_train,
        X_rub_train,
        X_len_train,
        y_train
    ) = train_data

    (
        X_essay_val,
        X_ref_val,
        X_sim_val,
        X_rub_val,
        X_len_val,
        y_val
    ) = val_data

    # =========================
    # OPTIMIZER + LR SCHEDULE
    # =========================
    lr_schedule = CosineDecay(
        initial_learning_rate=config.INITIAL_LR,
        decay_steps=config.EPOCHS * len(y_train) // config.BATCH_SIZE
    )

    optimizer = Adam(learning_rate=lr_schedule)

    # =========================
    # COMPILE MODEL
    # =========================
    model.compile(
        optimizer=optimizer,
        loss=Huber(delta=10.0),
        metrics=["mae", "mse"]
    )

    # =========================
    # CALLBACKS
    # =========================
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=f"{config.OUTPUT_DIR}/best_model.keras",
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    # =========================
    # TRAINING
    # =========================
    history = model.fit(
        [X_essay_train, X_ref_train, X_sim_train, X_rub_train, X_len_train],
        y_train,
        validation_data=(
            [X_essay_val, X_ref_val, X_sim_val, X_rub_val, X_len_val],
            y_val
        ),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    return history


# =========================
# EVALUATION FUNCTION
# =========================
def evaluate_model(model, test_data):

    (
        X_essay_test,
        X_ref_test,
        X_sim_test,
        X_rub_test,
        X_len_test,
        y_test
    ) = test_data

    # Predict
    pred = model.predict(
        [X_essay_test, X_ref_test, X_sim_test, X_rub_test, X_len_test]
    ).flatten()

    # Scale back
    pred = np.clip(pred * 100, 0, 100)
    y_true = y_test * 100

    # Metrics
    mse = mean_squared_error(y_true, pred)
    mae = mean_absolute_error(y_true, pred)
    r2 = r2_score(y_true, pred)
    qwk = quadratic_weighted_kappa(y_test, pred / 100)

    results = {
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": mae,
        "R2": r2,
        "QWK": qwk
    }

    print("\n=== EVALUATION RESULTS ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    # Interpretasi QWK
    if qwk >= 0.8:
        print("✓ Excellent agreement")
    elif qwk >= 0.6:
        print("✓ Good agreement")
    elif qwk >= 0.4:
        print("⚠ Moderate agreement")
    else:
        print("⚠ Low agreement")

    return results, pred, y_true, qwk, r2, mae, np.sqrt(mse), mse