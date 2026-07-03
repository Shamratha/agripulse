"""Stage-aware moisture stress detection.

Stress score per pixel combines:
  - NDVI anomaly vs the median trajectory of its own crop class (VCI-style)
  - NDWI (canopy water) anomaly — most sensitive to moisture deficit
  - SAR VV backscatter anomaly (surface soil moisture, all-weather)
Anomalies are computed against same-crop medians at the same composite, which
makes the score phenology-aware: a maturing crop's natural dry-down is not
flagged as stress.
"""

import numpy as np

from .features import growth_stage


def stress_assessment(scene, crop_map, t_now=None):
    ndvi, ndwi, vv = scene["ndvi"], scene["ndwi"], scene["vv"]
    T = ndvi.shape[0]
    if t_now is None:
        t_now = T - 1

    score = np.zeros(ndvi.shape[1:], dtype=np.float32)
    for c in np.unique(crop_map):
        m = crop_map == c
        if c == 0 or m.sum() < 20:
            continue
        # robust same-crop reference at this composite
        ref_ndvi = np.median(ndvi[t_now][m])
        ref_ndwi = np.median(ndwi[t_now][m])
        ref_vv = np.median(vv[t_now][m])
        mad = lambda x, r: np.median(np.abs(x[m] - r)) + 1e-6

        z_ndvi = (ref_ndvi - ndvi[t_now][m]) / mad(ndvi[t_now], ref_ndvi)
        z_ndwi = (ref_ndwi - ndwi[t_now][m]) / mad(ndwi[t_now], ref_ndwi)
        z_vv = (ref_vv - vv[t_now][m]) / mad(vv[t_now], ref_vv)
        score[m] = 0.25 * z_ndvi + 0.50 * z_ndwi + 0.25 * z_vv

    # classes: 0 none, 1 mild, 2 moderate, 3 severe (MAD-z units)
    stress = np.digitize(score, [0.75, 1.5, 2.5]).astype(np.int32)
    stress[crop_map == 0] = 0

    stage = growth_stage(ndvi, t_now)
    stage[crop_map == 0] = 0
    return stress, stage, score
