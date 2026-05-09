import numpy as np
from sklearn.metrics import cohen_kappa_score

def qwk(y_true, y_pred):
    print("QWK Evaluation....")
    y_true = np.clip(np.round(y_true),0,100)
    y_pred = np.clip(np.round(y_pred),0,100)
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")