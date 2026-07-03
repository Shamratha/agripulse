"""Crop type classification: Random Forest on multi-temporal features."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import train_test_split


def classify_crops(features, scene):
    """Train on ground-truth pixels, classify the full grid.

    Returns (crop_map (H,W), metrics dict).
    """
    H, W = scene["ndvi"].shape[1:]
    rows, cols, labels = scene["gt"]
    idx = rows * W + cols

    X, y = features[idx], labels
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=7)

    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                n_jobs=-1, random_state=7)
    rf.fit(X_tr, y_tr)

    pred_te = rf.predict(X_te)
    metrics = {
        "overall_accuracy": round(float(accuracy_score(y_te, pred_te)), 4),
        "kappa": round(float(cohen_kappa_score(y_te, pred_te)), 4),
        "confusion_matrix": confusion_matrix(y_te, pred_te).tolist(),
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
    }

    crop_map = rf.predict(features).reshape(H, W).astype(np.int32)
    return crop_map, metrics
