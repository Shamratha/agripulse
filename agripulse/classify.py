"""Crop type classification: Random Forest on multi-temporal features.

Validation uses a SPATIAL hold-out (train on the west block, test on the east
block) rather than a random pixel split. Random splits leak, because adjacent
pixels are highly correlated — a pixel's neighbour in the test set makes the
score look better than it is. A spatial split reports honest generalisation to
unseen ground.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix


def classify_crops(features, scene, split_frac=0.5):
    """Train on ground-truth pixels (spatial west/east hold-out), map full grid.

    Returns (crop_map (H,W), metrics dict).
    """
    H, W = scene["ndvi"].shape[1:]
    rows, cols, labels = scene["gt"]
    idx = rows * W + cols
    X, y = features[idx], labels

    split_col = W * split_frac
    train = cols < split_col
    test = ~train
    # fall back to a random split only if the spatial split starves a class
    if len(np.unique(y[train])) < len(np.unique(y)) or test.sum() < 20:
        rng = np.random.default_rng(7)
        perm = rng.permutation(len(y))
        cut = int(len(y) * 0.7)
        train = np.zeros(len(y), bool); train[perm[:cut]] = True
        test = ~train
        split = "random 70/30 (spatial split starved a class)"
    else:
        split = "spatial west/east hold-out"

    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                n_jobs=-1, random_state=7)
    rf.fit(X[train], y[train])

    pred = rf.predict(X[test])
    labels_present = sorted(np.unique(y).tolist())
    cm = confusion_matrix(y[test], pred, labels=labels_present)
    # no-information rate: accuracy of always guessing the majority test class
    nir = float(cm.sum(0).max() / cm.sum()) if cm.sum() else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.nan_to_num(cm.diagonal() / cm.sum(1))      # producer's accuracy
        precision = np.nan_to_num(cm.diagonal() / cm.sum(0))   # user's accuracy
        f1 = np.nan_to_num(2 * precision * recall / (precision + recall))
    metrics = {
        "overall_accuracy": round(float(accuracy_score(y[test], pred)), 4),
        "kappa": round(float(cohen_kappa_score(y[test], pred)), 4),
        "no_information_rate": round(nir, 4),
        "per_class_recall": dict(zip(labels_present, recall.round(3).tolist())),
        "per_class_precision": dict(zip(labels_present, precision.round(3).tolist())),
        "per_class_f1": dict(zip(labels_present, f1.round(3).tolist())),
        "macro_f1": round(float(f1.mean()), 4),
        "confusion_matrix": cm.tolist(),
        "labels": labels_present,
        "n_train": int(train.sum()), "n_test": int(test.sum()),
        "validation": split,
        "reference": "ESA WorldCereal 2021 (satellite product, not in-situ field data)"
                     if scene.get("wc_label") is not None else "synthetic field labels (sample mode)",
    }

    # retrain on all points for the delivered wall-to-wall map + per-pixel confidence
    rf.fit(X, y)
    proba = rf.predict_proba(features)
    crop_map = rf.classes_[proba.argmax(1)].reshape(H, W).astype(np.int32)
    confidence = proba.max(1).reshape(H, W).astype(np.float32)
    return crop_map, confidence, metrics
