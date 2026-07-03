"""Crop type classification: Random Forest on multi-temporal features.

Validation uses SPATIAL hold-outs (train on one spatial block, test on another)
rather than a random pixel split. Random splits leak, because adjacent pixels
are highly correlated — a pixel's neighbour in the test set makes the score look
better than it is. A spatial split reports honest generalisation to unseen ground.

To show the score is stable (not an artifact of one arbitrary boundary), the
classifier is evaluated over several spatial folds (west/east and north/south,
both directions) and the mean ± std of kappa/OA/macro-F1 is reported.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score)

RF_KW = dict(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=7)


def _per_class(cm):
    """Per-class recall (producer's), precision (user's), F1 from a confusion matrix."""
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.nan_to_num(cm.diagonal() / cm.sum(1))
        precision = np.nan_to_num(cm.diagonal() / cm.sum(0))
        f1 = np.nan_to_num(2 * precision * recall / (precision + recall))
    return recall, precision, f1


def _spatial_folds(rows, cols, H, W):
    """Yield (name, train_mask) for 4 spatial hold-outs (both directions)."""
    yield "west->east", cols < 0.5 * W
    yield "east->west", cols >= 0.5 * W
    yield "north->south", rows < 0.5 * H
    yield "south->north", rows >= 0.5 * H


def classify_crops(features, scene):
    """Train on ground-truth pixels, map the full grid.

    Returns (crop_map (H,W), confidence (H,W), metrics dict).
    """
    H, W = scene["ndvi"].shape[1:]
    rows, cols, labels = scene["gt"]
    idx = rows * W + cols
    X, y = features[idx], labels
    all_labels = sorted(np.unique(y).tolist())

    # --- spatial cross-validation: mean +/- std over 4 folds ---
    fold_scores, primary = [], None
    per_class_folds = {c: {"f1": [], "recall": [], "precision": []} for c in all_labels}
    for name, train in _spatial_folds(rows, cols, H, W):
        test = ~train
        if (len(np.unique(y[train])) < len(all_labels) or test.sum() < 20
                or len(np.unique(y[test])) < 2):
            continue
        rf = RandomForestClassifier(**RF_KW).fit(X[train], y[train])
        pred = rf.predict(X[test])
        rec, prec, f1c = _per_class(confusion_matrix(y[test], pred, labels=all_labels))
        for j, c in enumerate(all_labels):
            per_class_folds[c]["f1"].append(f1c[j])
            per_class_folds[c]["recall"].append(rec[j])
            per_class_folds[c]["precision"].append(prec[j])
        sc = {
            "fold": name,
            "kappa": round(float(cohen_kappa_score(y[test], pred)), 4),
            "overall_accuracy": round(float(accuracy_score(y[test], pred)), 4),
            "macro_f1": round(float(f1_score(y[test], pred, average="macro")), 4),
        }
        fold_scores.append(sc)
        if primary is None:  # first valid fold (west->east) carries the confusion matrix
            primary = (train, test, pred)

    if primary is None:  # spatial split starved a class -> single random fallback
        rng = np.random.default_rng(7)
        perm = rng.permutation(len(y)); cut = int(len(y) * 0.7)
        train = np.zeros(len(y), bool); train[perm[:cut]] = True
        test = ~train
        pred = RandomForestClassifier(**RF_KW).fit(X[train], y[train]).predict(X[test])
        primary = (train, test, pred)
        validation = "random 70/30 (spatial split starved a class)"
    else:
        validation = f"{len(fold_scores)}-fold spatial hold-out (west/east + north/south)"

    train, test, pred = primary
    cm = confusion_matrix(y[test], pred, labels=all_labels)
    nir = float(cm.sum(0).max() / cm.sum()) if cm.sum() else 0.0
    recall, precision, f1 = _per_class(cm)  # primary fold, for the confusion matrix

    def mean_std(key):
        v = np.array([s[key] for s in fold_scores]) if fold_scores else np.array([0.0])
        return round(float(v.mean()), 4), round(float(v.std()), 4)

    def per_class_mean_std(metric):
        mean, std = {}, {}
        for c in all_labels:
            v = np.array(per_class_folds[c][metric]) if per_class_folds[c][metric] else np.array([0.0])
            mean[c] = round(float(v.mean()), 3)
            std[c] = round(float(v.std()), 3)
        return mean, std

    k_m, k_s = mean_std("kappa")
    oa_m, oa_s = mean_std("overall_accuracy")
    f_m, f_s = mean_std("macro_f1")
    f1_mean, f1_std = per_class_mean_std("f1")
    rec_mean, _ = per_class_mean_std("recall")
    prec_mean, _ = per_class_mean_std("precision")

    metrics = {
        "kappa": k_m, "kappa_std": k_s,
        "overall_accuracy": oa_m, "overall_accuracy_std": oa_s,
        "macro_f1": f_m, "macro_f1_std": f_s,
        "n_folds": len(fold_scores),
        "fold_scores": fold_scores,
        "no_information_rate": round(nir, 4),
        # per-class shown as fold-averaged mean (+ std) — same rigor as the aggregates
        "per_class_f1": f1_mean, "per_class_f1_std": f1_std,
        "per_class_recall": rec_mean,
        "per_class_precision": prec_mean,
        # primary (west->east) fold, aligned with the confusion matrix below
        "per_class_f1_primary": dict(zip(all_labels, f1.round(3).tolist())),
        "per_class_recall_primary": dict(zip(all_labels, recall.round(3).tolist())),
        "per_class_precision_primary": dict(zip(all_labels, precision.round(3).tolist())),
        "confusion_matrix": cm.tolist(),
        "labels": all_labels,
        "n_train": int(train.sum()), "n_test": int(test.sum()),
        "validation": validation,
        "reference": "ESA WorldCereal 2021 (satellite product, not in-situ field data)"
                     if scene.get("wc_label") is not None else "synthetic field labels (sample mode)",
    }

    # final model on all points: wall-to-wall map + per-pixel confidence + importances
    rf = RandomForestClassifier(**RF_KW).fit(X, y)
    proba = rf.predict_proba(features)
    crop_map = rf.classes_[proba.argmax(1)].reshape(H, W).astype(np.int32)
    confidence = proba.max(1).reshape(H, W).astype(np.float32)
    metrics["feature_importances"] = rf.feature_importances_.round(5).tolist()
    return crop_map, confidence, metrics
