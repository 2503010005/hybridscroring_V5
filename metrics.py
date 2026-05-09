import numpy as np
from sklearn.metrics import cohen_kappa_score

def quadratic_weighted_kappa(y_true, y_pred):
    y_true = np.round(y_true * 100).astype(int)
    y_pred = np.round(y_pred * 100).astype(int)

    y_true = np.clip(y_true, 0, 100)
    y_pred = np.clip(y_pred, 0, 100)

    return cohen_kappa_score(y_true, y_pred, weights="quadratic")