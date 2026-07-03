"""Stage-aware moisture stress detection.

Primary signal is the **Vegetation Condition Index (VCI)** — an ABSOLUTE
condition measure the problem statement names explicitly:

    VCI = (NDVI - NDVI_lo) / (NDVI_hi - NDVI_lo)

where NDVI_lo/hi are the pixel's own multi-year 10th/90th-percentile envelope
for that 8-day window (fetched in data_gee._vci_baseline). VCI near 0 means the
crop is at the bottom of its historical range for this time of year (stressed);
near 1 means best-on-record vigour. Because it compares each pixel to its OWN
history rather than to its neighbours, it detects region-wide stress that a
spatial anomaly cannot.

VCI is combined with a canopy-water term (NDWI, relative to same-crop median)
so both vegetation vigour and moisture are represented. Growth stage is derived
per pixel so stress is interpreted stage-wise.

When no multi-year baseline is available (sample mode), it falls back to the
same-crop spatial anomaly.
"""

import numpy as np

from .features import growth_stage


def stress_assessment(scene, crop_map, t_now=None):
    ndvi, ndwi, vv = scene["ndvi"], scene["ndwi"], scene["vv"]
    T = ndvi.shape[0]
    if t_now is None:
        t_now = T - 1

    lo, hi = scene.get("ndvi_lo"), scene.get("ndvi_hi")
    if lo is not None and hi is not None:
        score, vci = _vci_score(ndvi, ndwi, crop_map, lo, hi, t_now)
        # VCI drought thresholds: >=0.5 none, 0.35-0.5 mild, 0.2-0.35 moderate, <0.2 severe
        stress = np.select(
            [vci >= 0.5, vci >= 0.35, vci >= 0.2],
            [0, 1, 2], default=3).astype(np.int32)
    else:
        score = _anomaly_score(ndvi, ndwi, vv, crop_map, t_now)
        vci = None
        stress = np.digitize(score, [0.75, 1.5, 2.5]).astype(np.int32)

    stress[crop_map == 0] = 0
    stage = growth_stage(ndvi, t_now)
    stage[crop_map == 0] = 0
    return stress, stage, score, vci


def _vci_score(ndvi, ndwi, crop_map, lo, hi, t_now):
    """VCI (absolute) blended with a canopy-water anomaly; returns (stress_score, vci)."""
    denom = np.maximum(hi[t_now] - lo[t_now], 0.05)
    vci = np.clip((ndvi[t_now] - lo[t_now]) / denom, 0.0, 1.0).astype(np.float32)

    # canopy-water deficit relative to same-crop median (0..1, higher = drier)
    ndwi_def = np.zeros_like(vci)
    for c in np.unique(crop_map):
        m = crop_map == c
        if c == 0 or m.sum() < 20:
            continue
        ref = np.median(ndwi[t_now][m])
        spread = np.median(np.abs(ndwi[t_now][m] - ref)) * 2 + 1e-6
        ndwi_def[m] = np.clip((ref - ndwi[t_now][m]) / spread, 0, 1)

    # stress score in 0..1: low VCI and/or high canopy-water deficit
    score = 0.7 * (1 - vci) + 0.3 * ndwi_def
    score[crop_map == 0] = 0
    return score.astype(np.float32), vci


def _anomaly_score(ndvi, ndwi, vv, crop_map, t_now):
    """Fallback: same-crop spatial anomaly (used when no VCI baseline, e.g. sample mode)."""
    score = np.zeros(ndvi.shape[1:], dtype=np.float32)
    for c in np.unique(crop_map):
        m = crop_map == c
        if c == 0 or m.sum() < 20:
            continue
        ref_ndvi, ref_ndwi, ref_vv = (np.median(a[t_now][m]) for a in (ndvi, ndwi, vv))
        mad = lambda x, r: np.median(np.abs(x[t_now][m] - r)) + 1e-6
        z_ndvi = (ref_ndvi - ndvi[t_now][m]) / mad(ndvi, ref_ndvi)
        z_ndwi = (ref_ndwi - ndwi[t_now][m]) / mad(ndwi, ref_ndwi)
        z_vv = (ref_vv - vv[t_now][m]) / mad(vv, ref_vv)
        score[m] = 0.25 * z_ndvi + 0.50 * z_ndwi + 0.25 * z_vv
    return score
