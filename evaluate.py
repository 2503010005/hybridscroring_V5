# evaluate.py

import numpy as np

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    cohen_kappa_score
)


# =========================================================
# EVALUATE MODEL
# =========================================================
# evaluate.py

import numpy as np

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    cohen_kappa_score
)


def evaluate_model(
    model,
    test_data
):

    (
        X_text_test,
        X_num_test,
        y_test
    ) = test_data

    # =====================================================
    # PREDICT
    # =====================================================
    pred = model.predict(
        [X_text_test, X_num_test],
        verbose=0
    ).flatten()

    print("DEBUG MAX PRED:", np.max(pred))
    print("DEBUG MAX Y:", np.max(y_test))

    # =====================================================
    # SAFETY
    # =====================================================
    pred = np.nan_to_num(pred)
    y_test = np.nan_to_num(y_test)

    # =====================================================
    # AUTO SCALE DETECTION
    # =====================================================
    # Jika masih skala 0-1
    # maka convert ke 0-100
    # =====================================================

    if np.max(pred) <= 1.5:
        pred = pred * 100.0

    if np.max(y_test) <= 1.5:
        y_test = y_test * 100.0

    # =====================================================
    # CLIPPING
    # =====================================================
    pred = np.clip(pred, 0, 100)
    y_actual = np.clip(y_test, 0, 100)

    # =====================================================
    # METRICS
    # =====================================================
    mse = mean_squared_error(
        y_actual,
        pred
    )

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(
        y_actual,
        pred
    )

    r2 = r2_score(
        y_actual,
        pred
    )

    # =====================================================
    # QWK
    # =====================================================
    qwk = cohen_kappa_score(

        np.round(y_actual).astype(int),
        np.round(pred).astype(int),

        weights='quadratic'
    )

    # =====================================================
    # REPORT
    # =====================================================
    print("\n" + "=" * 30)
    print("   FINAL EVALUATION RESULTS")
    print("=" * 30)

    print(f"MSE   : {mse:.4f}")
    print(f"RMSE  : {rmse:.4f}")
    print(f"MAE   : {mae:.4f}")
    print(f"R2    : {r2:.4f}")
    print(f"QWK   : {qwk:.4f}")

    print("-" * 30)

    # =====================================================
    # INTERPRETATION
    # =====================================================
    if qwk >= 0.81:
        strength = "Almost Perfect"

    elif qwk >= 0.61:
        strength = "Substantial"

    elif qwk >= 0.41:
        strength = "Moderate"

    elif qwk >= 0.21:
        strength = "Fair"

    else:
        strength = "Poor/Fair"

    print(
        f"Strength of Agreement: {strength}"
    )

    print("=" * 30)

    # =====================================================
    # DEBUG
    # =====================================================
    print("Debug Pred:", pred[:5])
    print("Debug Actual:", y_actual[:5])

    # =====================================================
    # RESULTS
    # =====================================================
    results = {

        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "qwk": float(qwk)

    }

    return (
        results,
        pred,
        y_actual,
        qwk,
        r2,
        mae,
        rmse,
        mse
    )