"""End-to-end pipeline: data -> features -> crop map -> stress -> advisory.

Writes dashboard-ready outputs to outputs/:
  crop_map.png, stress_map.png, advisory_map.png, ndvi_latest.png (map overlays)
  summary.json (metrics, legends, time series, bounds)
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from .config import (ADVISORY_CLASSES, COMPOSITE_DAYS, CROPS, PILOT_BOUNDS,
                     SEASON_START, STAGES, STRESS_CLASSES)
from .classify import classify_crops
from .features import build_features
from .stress import stress_assessment
from .water import water_balance

OUT = Path(__file__).resolve().parent.parent / "outputs"


def _save_class_png(arr, classes, path, alpha=0.82):
    cmap = ListedColormap([classes[k]["color"] for k in sorted(classes)])
    rgba = cmap(arr / max(len(classes) - 1, 1))
    rgba[..., 3] = alpha
    plt.imsave(path, rgba)


def _save_scalar_png(arr, path, cmap="RdYlGn", alpha=0.82):
    norm = (arr - arr.min()) / (np.ptp(arr) + 1e-9)
    rgba = plt.get_cmap(cmap)(norm)
    rgba[..., 3] = alpha
    plt.imsave(path, rgba)


def run(mode="sample", t_now=None):
    OUT.mkdir(exist_ok=True)

    if mode == "gee":
        from .data_gee import generate_scene
    else:
        from .data_sample import generate_scene
    scene = generate_scene()
    T = scene["ndvi"].shape[0]
    if t_now is None or not (0 <= t_now < T):
        t_now = T - 1

    print("Extracting features...")
    features = build_features(scene)

    print("Classifying crops (Random Forest)...")
    crop_map, metrics = classify_crops(features, scene)
    print(f"  OA={metrics['overall_accuracy']:.2%}  kappa={metrics['kappa']:.3f} "
          f"({metrics['validation']} vs {metrics['reference']})")

    # wall-to-wall agreement with the full WorldCereal reference map (GEE mode)
    wc = scene.get("wc_label")
    if wc is not None:
        metrics["map_agreement_worldcereal"] = round(float((crop_map == wc).mean()), 4)
        print(f"  full-map agreement vs WorldCereal: {metrics['map_agreement_worldcereal']:.2%}")

    print(f"Assessing moisture stress (stage-aware) at composite {t_now}...")
    stress, stage, score, vci = stress_assessment(scene, crop_map, t_now)
    cropped_mask = crop_map != 0
    if vci is not None:
        print(f"  method: VCI (multi-year baseline); mean VCI over crops = "
              f"{float(vci[cropped_mask].mean()):.2f}")
    else:
        print("  method: same-crop spatial anomaly (no multi-year baseline)")

    print("Computing water balance & irrigation advisory...")
    deficit, advisory, weather = water_balance(scene, crop_map, stage, stress, score, t_now)

    # --- map overlays ---
    _save_class_png(crop_map, CROPS, OUT / "crop_map.png")
    _save_class_png(stress, STRESS_CLASSES, OUT / "stress_map.png")
    _save_class_png(advisory, ADVISORY_CLASSES, OUT / "advisory_map.png")
    _save_scalar_png(scene["ndvi"][t_now], OUT / "ndvi_latest.png")
    if vci is not None:
        vci_disp = np.where(crop_map != 0, vci, np.nan)
        _save_scalar_png(np.nan_to_num(vci_disp, nan=float(np.nanmax(vci_disp))),
                         OUT / "vci_map.png", cmap="RdYlGn")

    # --- time series: per-crop mean NDVI / NDWI ---
    dates = [(datetime.fromisoformat(SEASON_START) + timedelta(days=COMPOSITE_DAYS * i)
              ).strftime("%d %b") for i in range(scene["ndvi"].shape[0])]
    ts = {}
    for c, info in CROPS.items():
        m = crop_map == c
        if m.sum() == 0:
            continue
        ts[info["name"]] = {
            "color": info["color"],
            "ndvi": [round(float(scene["ndvi"][t][m].mean()), 3) for t in range(len(dates))],
            "ndwi": [round(float(scene["ndwi"][t][m].mean()), 3) for t in range(len(dates))],
        }

    cropped = crop_map != 0
    n_cropped = int(cropped.sum())
    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "analysis_date": dates[t_now],
        "bounds": PILOT_BOUNDS,
        "dates": dates,
        "metrics": metrics,
        "legends": {
            "crop": {info["name"]: info["color"] for info in CROPS.values()},
            "stress": {info["name"]: info["color"] for info in STRESS_CLASSES.values()},
            "advisory": {info["name"]: info["color"] for info in ADVISORY_CLASSES.values()},
        },
        "crop_area_pct": {info["name"]: round(float((crop_map == c).mean() * 100), 1)
                          for c, info in CROPS.items()},
        "stress_pct": {info["name"]: round(float(((stress == k) & cropped).sum() / n_cropped * 100), 1)
                       for k, info in STRESS_CLASSES.items()},
        "advisory_pct": {info["name"]: round(float(((advisory == k) & cropped).sum() / n_cropped * 100), 1)
                         for k, info in ADVISORY_CLASSES.items()},
        "stage_pct": {STAGES[s]: round(float(((stage == s) & cropped).sum() / n_cropped * 100), 1)
                      for s in range(len(STAGES))},
        "mean_deficit_mm": round(float(deficit[cropped].mean()), 1),
        "stress_method": "VCI (multi-year NDVI baseline) + canopy-water"
                         if vci is not None else "same-crop spatial anomaly",
        "mean_vci": round(float(vci[cropped].mean()), 3) if vci is not None else None,
        "weather": weather,
        "timeseries": ts,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Outputs written to {OUT}")
    return summary
