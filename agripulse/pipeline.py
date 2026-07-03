"""End-to-end pipeline: data -> features -> crop map -> stress -> advisory.

Writes dashboard-ready outputs to outputs/:
  {crop,stress,advisory,vci,smi,confidence,ndvi_latest}.png map overlays, each
  with a .wld + .prj sidecar so it loads as a georeferenced raster in QGIS.
  summary.json — metrics (kappa, per-class precision/recall/F1, no-info
  baseline), stage-wise stress per crop, hectares, recommended irrigation depth,
  stress-detector validation (sample mode), SMI cross-check, time series.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from .config import (ADVISORY_CLASSES, COMPOSITE_DAYS, CROPS, GRID_SIZE,
                     PILOT_BOUNDS, SEASON_START, STAGES, STRESS_CLASSES)
from .classify import classify_crops
from .features import build_features, feature_names, importance_by_group
from .stress import stress_assessment
from .water import water_balance

OUT = Path(__file__).resolve().parent.parent / "outputs"


_WGS84_WKT = ('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
              'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]]')


def _write_worldfile(path):
    """Write a .wld + .prj sidecar so the PNG is a georeferenced raster loadable
    in QGIS/ArcGIS — no GeoTIFF dependency needed."""
    w, s, e, n = PILOT_BOUNDS
    a = (e - w) / GRID_SIZE
    d = -(n - s) / GRID_SIZE
    lines = [a, 0.0, 0.0, d, w + a / 2, n + d / 2]
    path.with_suffix(".wld").write_text("\n".join(f"{v:.10f}" for v in lines) + "\n")
    path.with_suffix(".prj").write_text(_WGS84_WKT + "\n")


def _save_class_png(arr, classes, path, alpha=0.82):
    cmap = ListedColormap([classes[k]["color"] for k in sorted(classes)])
    rgba = cmap(arr / max(len(classes) - 1, 1))
    rgba[..., 3] = alpha
    plt.imsave(path, rgba)
    _write_worldfile(path)


def _save_scalar_png(arr, path, cmap="RdYlGn", alpha=0.82):
    norm = (arr - arr.min()) / (np.ptp(arr) + 1e-9)
    rgba = plt.get_cmap(cmap)(norm)
    rgba[..., 3] = alpha
    plt.imsave(path, rgba)
    _write_worldfile(path)


def _save_importance_png(groups, path):
    """Horizontal bar chart of grouped RF feature importances."""
    items = list(groups.items())[:14][::-1]
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(5.4, 4.2), dpi=120)
    ax.barh(labels, vals, color="#4caf7d")
    ax.set_xlabel("Importance (summed over temporal columns)")
    ax.set_title("Random Forest feature importance")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, transparent=True)
    plt.close(fig)


def _pixel_ha():
    """Approximate ground area of one grid pixel, in hectares."""
    import math
    w, s, e, n = PILOT_BOUNDS
    mid = math.radians((s + n) / 2)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(mid)
    px_w = (e - w) / GRID_SIZE * m_per_deg_lon
    px_h = (n - s) / GRID_SIZE * m_per_deg_lat
    return px_w * px_h / 10_000.0


def _validate_stress(stress, stressed_truth, cropped):
    """Binary (stressed vs not) recall/precision of the detector vs injected truth."""
    pred = (stress >= 2) & cropped          # moderate+severe = "stressed"
    truth = stressed_truth & cropped
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    return {
        "recall": round(tp / (tp + fn), 3) if tp + fn else 0.0,
        "precision": round(tp / (tp + fp), 3) if tp + fp else 0.0,
        "n_truth_stressed": int(truth.sum()),
        "definition": "stressed = moderate+severe class vs injected field stress (sample mode)",
    }


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
    crop_map, confidence, metrics = classify_crops(features, scene)
    print(f"  kappa={metrics['kappa']:.3f}±{metrics['kappa_std']:.3f}  "
          f"macro-F1={metrics['macro_f1']:.3f}±{metrics['macro_f1_std']:.3f}  "
          f"OA={metrics['overall_accuracy']:.2%} (no-info baseline {metrics['no_information_rate']:.2%})")
    print(f"  per-class F1: {metrics['per_class_f1']}")
    print(f"  ({metrics['validation']}, {metrics['n_folds']} folds; reference = {metrics['reference']})")

    # feature-importance plot (grouped by band + phenology)
    groups = importance_by_group(feature_names(scene), metrics.pop("feature_importances"))
    metrics["feature_importance_groups"] = {k: round(v, 4) for k, v in groups.items()}
    _save_importance_png(groups, OUT / "feature_importance.png")

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
    deficit, advisory, weather = water_balance(scene, crop_map, stage, stress, vci, t_now)

    # --- validate the stress detector against injected truth (sample mode only) ---
    stress_validation = None
    if scene.get("stressed_truth") is not None:
        stress_validation = _validate_stress(stress, scene["stressed_truth"], cropped_mask)
        print(f"  stress detector vs injected truth: "
              f"recall {stress_validation['recall']:.2f}, precision {stress_validation['precision']:.2f}")

    # --- map overlays ---
    _save_class_png(crop_map, CROPS, OUT / "crop_map.png")
    _save_class_png(stress, STRESS_CLASSES, OUT / "stress_map.png")
    _save_class_png(advisory, ADVISORY_CLASSES, OUT / "advisory_map.png")
    _save_scalar_png(scene["ndvi"][t_now], OUT / "ndvi_latest.png")
    _save_scalar_png(np.where(cropped_mask, confidence, 1.0), OUT / "confidence_map.png", cmap="viridis")
    if vci is not None:
        vci_disp = np.where(crop_map != 0, vci, np.nan)
        _save_scalar_png(np.nan_to_num(vci_disp, nan=float(np.nanmax(vci_disp))),
                         OUT / "vci_map.png", cmap="RdYlGn")

    # SMI: independent (reanalysis) soil-moisture layer + a physical sanity check.
    # ERA5 soil moisture is coarse (~9 km), so rather than claim pixel-scale
    # corroboration, we verify it behaves physically: region-mean soil moisture
    # should RISE the composite after rainfall. A positive lagged correlation
    # confirms the independent moisture signal is real and coherent.
    smi_crosscheck = None
    if scene.get("smi") is not None and vci is not None:
        smi_t = scene["smi"][t_now]
        _save_scalar_png(np.where(cropped_mask, smi_t, 1.0), OUT / "smi_map.png", cmap="YlGnBu")
        smi_season = np.array([scene["smi"][t][cropped_mask].mean() for t in range(T)])
        rain_season = np.asarray(scene["rain"], dtype=float)
        if smi_season[1:].std() > 1e-6 and rain_season[:-1].std() > 1e-6:
            r = float(np.corrcoef(smi_season[1:], rain_season[:-1])[0, 1])
            smi_crosscheck = {
                "smi_vs_rain_lag1_corr": round(r, 3),
                "note": "Pearson r between region-mean ERA5 soil moisture and the "
                        "PREVIOUS composite's rainfall — soil moisture rising after "
                        "rain confirms the independent moisture layer is physically real.",
            }
            print(f"  SMI cross-check: soil-moisture-vs-prior-rain correlation r = {r:.2f}")

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
    ha = _pixel_ha()

    # stage-wise stress breakdown per crop — the exact PS deliverable
    # ("% of flowering wheat under moderate stress"), computed not narrated.
    stress_by_stage_crop = {}
    for c, cinfo in CROPS.items():
        if c == 0:
            continue
        cm = crop_map == c
        if cm.sum() == 0:
            continue
        per_stage = {}
        for si, sname in enumerate(STAGES):
            sm = cm & (stage == si)
            if sm.sum() == 0:
                continue
            per_stage[sname] = {
                "area_ha": round(float(sm.sum() * ha), 1),
                "pct_moderate_severe": round(float(((stress >= 2) & sm).sum() / sm.sum() * 100), 1),
            }
        stress_by_stage_crop[cinfo["name"]] = per_stage

    # recommended irrigation depth (mm) = mean computed deficit among fields advised to irrigate
    advised = cropped & (advisory >= 1)
    rec_depth = round(float(deficit[advised].mean()), 1) if advised.any() else 0.0

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
        "crop_area_ha": {info["name"]: round(float((crop_map == c).sum() * ha), 1)
                         for c, info in CROPS.items()},
        "mean_deficit_mm": round(float(deficit[cropped].mean()), 1),
        "recommended_depth_mm": rec_depth,
        "stress_by_stage_crop": stress_by_stage_crop,
        "stress_validation": stress_validation,
        "smi_crosscheck": smi_crosscheck,
        "mean_cloud_fill_fraction": round(float(np.mean(scene["fill_fraction"])), 3)
                                    if scene.get("fill_fraction") is not None else None,
        "stress_method": "VCI (multi-year NDVI baseline) + canopy-water, stage-aware thresholds"
                         if vci is not None else "same-crop spatial anomaly (sample mode)",
        "mean_vci": round(float(vci[cropped].mean()), 3) if vci is not None else None,
        "pixel_ha": round(ha, 3),
        "weather": weather,
        "timeseries": ts,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Outputs written to {OUT}")
    return summary
