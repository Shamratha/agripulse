"""Per-pixel feature extraction and phenology metrics."""

import numpy as np


def build_features(scene):
    """Stack temporal spectral values + phenology summaries into (H*W, F)."""
    ndvi, ndwi, vv, vh = scene["ndvi"], scene["ndwi"], scene["vv"], scene["vh"]
    T, H, W = ndvi.shape

    layers = [ndvi, ndwi, vv, vh, vh - vv]  # ratio in dB space = difference
    if scene.get("evi") is not None:
        layers.append(scene["evi"])
    feats = [l.reshape(T, -1).T for l in layers]  # each (N, T)

    # phenology metrics per pixel
    n = ndvi.reshape(T, -1)
    s, e = sos(ndvi).ravel(), eos(ndvi).ravel()
    summaries = np.stack([
        n.max(0), n.mean(0), n.std(0), n.argmax(0),
        s, e, e - s,   # SOS, EOS, LGP (length of growing period)
    ], axis=1)

    return np.hstack(feats + [summaries]).astype(np.float32)


def feature_names(scene):
    """Column names matching build_features(), for feature-importance plots."""
    T = scene["ndvi"].shape[0]
    layers = ["NDVI", "NDWI", "VV", "VH", "VH-VV"]
    if scene.get("evi") is not None:
        layers.append("EVI")
    names = [f"{l} t{t}" for l in layers for t in range(T)]
    names += ["NDVI max", "NDVI mean", "NDVI std", "NDVI peak-t", "SOS", "EOS", "LGP"]
    return names


def importance_by_group(names, importances):
    """Aggregate per-column importances into readable groups (band / phenology)."""
    groups = {}
    for n, imp in zip(names, importances):
        key = n.split(" t")[0] if " t" in n else n  # collapse temporal columns per band
        groups[key] = groups.get(key, 0.0) + float(imp)
    return dict(sorted(groups.items(), key=lambda kv: kv[1], reverse=True))


def sos(ndvi, thresh=0.30):
    """Start of season: first composite where NDVI crosses thresh (T if never)."""
    T = ndvi.shape[0]
    above = ndvi > thresh
    first = np.argmax(above, axis=0).astype(np.float32)
    first[~above.any(axis=0)] = T
    return first


def eos(ndvi, thresh=0.30):
    """End of season: last composite where NDVI is above thresh (0 if never)."""
    T = ndvi.shape[0]
    above = ndvi > thresh
    last = (T - 1 - np.argmax(above[::-1], axis=0)).astype(np.float32)
    last[~above.any(axis=0)] = 0
    return last


def growth_stage(ndvi, t_now):
    """Per-pixel growth stage index at composite t_now.

    0 Sowing, 1 Vegetative, 2 Flowering (peak), 3 Maturity — from the pixel's
    own phenology (SOS, peak, EOS), so stress is interpreted stage-wise.
    """
    s, e = sos(ndvi), eos(ndvi)
    peak = np.argmax(ndvi, axis=0).astype(np.float32)
    stage = np.zeros(ndvi.shape[1:], dtype=np.int32)
    stage[t_now >= (s + (peak - s) * 0.35)] = 1
    stage[t_now >= (s + (peak - s) * 0.85)] = 2
    stage[t_now >= (peak + (e - peak) * 0.5)] = 3
    stage[t_now < s] = 0
    return stage
