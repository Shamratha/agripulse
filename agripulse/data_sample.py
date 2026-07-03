"""Synthetic pilot-area scene generator.

Produces multi-temporal NDVI / NDWI / Sentinel-1 VV / VH stacks with realistic
crop phenology, field structure (Voronoi parcels), moisture-stressed fields in
the canal tail-end, weather series, and ground-truth points — the same data
contract the GEE provider fills from real imagery (see data_gee.py).
"""

import numpy as np

from .config import GRID_SIZE, N_COMPOSITES

RNG = np.random.default_rng(42)

# Double-logistic phenology parameters per crop: (base, amplitude, sos_t, eos_t)
# Classes mirror the WorldCereal-grounded scheme in config.CROPS.
PHENOLOGY = {
    0: (0.12, 0.05, 4.0, 11.0),   # non-cropland: near-flat
    1: (0.15, 0.58, 2.2, 15.0),   # wheat / winter cereal
    2: (0.15, 0.50, 1.9, 13.0),   # other cropland (earlier, shorter — overlaps wheat)
}


def _double_logistic(t, base, amp, sos, eos, rate=1.6):
    rise = 1.0 / (1.0 + np.exp(-rate * (t - sos)))
    fall = 1.0 / (1.0 + np.exp(-rate * (t - eos)))
    return base + amp * (rise - fall)


def generate_scene(size=GRID_SIZE, n_fields=140, n_steps=N_COMPOSITES):
    # --- field parcels via nearest-seed (Voronoi) partition ---
    seeds = RNG.uniform(0, size, (n_fields, 2))
    yy, xx = np.mgrid[0:size, 0:size]
    d2 = (yy[..., None] - seeds[:, 0]) ** 2 + (xx[..., None] - seeds[:, 1]) ** 2
    field_id = np.argmin(d2, axis=-1)

    # crop assignment per field: wheat-dominant rabi mosaic (matches WorldCereal split)
    field_crop = RNG.choice([0, 1, 2], size=n_fields, p=[0.10, 0.70, 0.20])
    crop = field_crop[field_id]

    # --- moisture stress: tail-end of the canal command (east side) ---
    # probability of a field being water-stressed rises with distance from the canal head (west edge)
    tail_frac = seeds[:, 1] / size
    field_stressed = (RNG.uniform(0, 1, n_fields) < 0.55 * tail_frac ** 2) & (field_crop != 0)
    stressed = field_stressed[field_id]
    stress_onset = RNG.integers(6, 9, n_fields)[field_id]  # composite index when deficit begins

    # --- weather: rising ET0 through rabi, sparse winter rain (mm per 8 days) ---
    t = np.arange(n_steps)
    et0 = 18 + 1.6 * t + RNG.normal(0, 1.0, n_steps)
    rain = np.where(RNG.uniform(0, 1, n_steps) < 0.25, RNG.uniform(5, 28, n_steps), 0.0)
    rain[0] += 12  # sowing rain

    # --- spectral time series ---
    ndvi = np.zeros((n_steps, size, size), dtype=np.float32)
    ndwi = np.zeros_like(ndvi)
    vv = np.zeros_like(ndvi)
    vh = np.zeros_like(ndvi)

    # per-field random variation so parcels are not identical
    f_amp = RNG.normal(1.0, 0.13, n_fields)[field_id]
    f_shift = RNG.normal(0.0, 0.9, n_fields)[field_id]

    for ti in range(n_steps):
        base = np.zeros((size, size), dtype=np.float32)
        for c, (b, a, sos, eos) in PHENOLOGY.items():
            m = crop == c
            base[m] = _double_logistic(ti + f_shift[m], b, a * f_amp[m], sos, eos)

        # cumulative stress effect after onset: mild NDVI decline, strong NDWI decline
        sev = np.clip((ti - stress_onset) / 5.0, 0, 1) * stressed
        ndvi_t = base * (1 - 0.22 * sev) + RNG.normal(0, 0.045, (size, size))
        moisture = 0.55 * ndvi_t - 0.12 - 0.18 * sev + RNG.normal(0, 0.04, (size, size))
        vv_t = -15.5 + 6.0 * ndvi_t + 2.5 * (moisture + 0.2) + RNG.normal(0, 1.1, (size, size))
        vh_t = vv_t - 7.0 + 1.5 * ndvi_t + RNG.normal(0, 1.1, (size, size))

        ndvi[ti] = np.clip(ndvi_t, -0.1, 1.0)
        ndwi[ti] = np.clip(moisture, -0.6, 0.8)
        vv[ti] = vv_t
        vh[ti] = vh_t

    # --- ground truth: labelled points near field seeds (survey plots) ---
    gt_rows, gt_cols, gt_labels = [], [], []
    for fi in range(n_fields):
        r, c_ = int(seeds[fi, 0]), int(seeds[fi, 1])
        if 2 <= r < size - 2 and 2 <= c_ < size - 2 and field_id[r, c_] == fi:
            gt_rows.append(r)
            gt_cols.append(c_)
            gt_labels.append(field_crop[fi])

    return {
        "ndvi": ndvi, "ndwi": ndwi, "vv": vv, "vh": vh,
        "crop_truth": crop, "field_id": field_id, "stressed_truth": stressed,
        "et0": et0, "rain": rain,
        "gt": (np.array(gt_rows), np.array(gt_cols), np.array(gt_labels)),
    }
