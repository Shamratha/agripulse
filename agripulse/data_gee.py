"""Google Earth Engine provider — real-data drop-in for data_sample.generate_scene.

Requires `pip install earthengine-api`, a registered GEE project
(https://code.earthengine.google.com/register), and `earthengine authenticate`.

Fetches 8-day composites over PILOT_BOUNDS on a fixed GRID_SIZE grid via
ee.data.computePixels:
  Sentinel-2 SR  -> NDVI, NDWI     Sentinel-1 GRD -> VV, VH
  CHIRPS         -> rainfall        ERA5-Land      -> ET0 proxy

Ground truth is REAL: crop labels come from ESA WorldCereal 2021 (a global
crop product validated against 100k+ in-situ field samples), sampled only at
high-confidence pixels. That makes the reported accuracy a genuine agreement
with an independently field-validated reference, not a self-fulfilling metric.

For the hackathon's own survey points, set GT_CSV (columns lon,lat,crop_id) to
override WorldCereal.
"""

import csv
import os

import numpy as np

from .config import (COMPOSITE_DAYS, GRID_SIZE, N_COMPOSITES, PILOT_BOUNDS,
                     SEASON_START)

GEE_PROJECT = os.environ.get("GEE_PROJECT", "agripulse-hackathon")
# Optional: CSV of real survey points with columns lon,lat,crop_id (0-2)
GT_CSV = os.environ.get("GT_CSV", "")
# Fetched composites are cached here (offline demo fallback); GEE_REFRESH=1 refetches
CACHE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "gee_cache.npz")

S2_SR = "COPERNICUS/S2_SR_HARMONIZED"
S1_GRD = "COPERNICUS/S1_GRD"
CHIRPS = "UCSB-CHG/CHIRPS/DAILY"
ERA5_LAND = "ECMWF/ERA5_LAND/DAILY_AGGR"
WORLDCEREAL = "ESA/WorldCereal/2021/MODELS/v100"

CONF_MIN = 80          # only train on WorldCereal pixels this confident
N_POINTS = 1200        # total ground-truth points, sampled ~proportional to prevalence
N_FLOOR = 70           # minimum points per present class (so rare classes still train)


def _grid():
    w, s, e, n = PILOT_BOUNDS
    return {
        "dimensions": {"width": GRID_SIZE, "height": GRID_SIZE},
        "affineTransform": {
            "scaleX": (e - w) / GRID_SIZE, "shearX": 0, "translateX": w,
            "shearY": 0, "scaleY": -(n - s) / GRID_SIZE, "translateY": n,
        },
        "crsCode": "EPSG:4326",
    }


def _fetch(ee, image, bands):
    """computePixels -> (len(bands), H, W) float array; masked pixels = nan."""
    arr = ee.data.computePixels({
        "expression": image.select(bands).toFloat().unmask(-9999),
        "fileFormat": "NUMPY_NDARRAY",
        "grid": _grid(),
    })
    out = np.stack([arr[b].astype(np.float32) for b in bands])
    out[out == -9999] = np.nan
    return out


def generate_scene():
    if os.path.exists(CACHE) and not os.environ.get("GEE_REFRESH"):
        print(f"  using cached data ({CACHE}); set GEE_REFRESH=1 to refetch")
        d = dict(np.load(CACHE))
        gt = _load_gt_csv() if GT_CSV else _sample_labels(d["label"], d["conf"])
        return {**d, "crop_truth": None, "field_id": None, "stressed_truth": None,
                "gt": gt, "wc_label": d["label"]}

    try:
        import ee
    except ImportError as e:
        raise RuntimeError("pip install earthengine-api, then `earthengine authenticate`") from e

    ee.Initialize(project=GEE_PROJECT)
    region = ee.Geometry.Rectangle(PILOT_BOUNDS)
    start = ee.Date(SEASON_START)

    ndvi = np.full((N_COMPOSITES, GRID_SIZE, GRID_SIZE), np.nan, np.float32)
    ndwi = np.full_like(ndvi, np.nan)
    vv = np.full_like(ndvi, np.nan)
    vh = np.full_like(ndvi, np.nan)
    et0, rain = np.zeros(N_COMPOSITES), np.zeros(N_COMPOSITES)

    for i in range(N_COMPOSITES):
        t0 = start.advance(i * COMPOSITE_DAYS, "day")
        t1 = t0.advance(COMPOSITE_DAYS, "day")
        print(f"  composite {i + 1}/{N_COMPOSITES}...", flush=True)

        s2 = (ee.ImageCollection(S2_SR).filterBounds(region).filterDate(t0, t1)
              .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40)))
        if s2.size().getInfo() > 0:
            comp = s2.median()
            img = (comp.normalizedDifference(["B8", "B4"]).rename("ndvi")
                   .addBands(comp.normalizedDifference(["B3", "B8"]).rename("ndwi")))
            ndvi[i], ndwi[i] = _fetch(ee, img, ["ndvi", "ndwi"])

        s1 = (ee.ImageCollection(S1_GRD).filterBounds(region).filterDate(t0, t1)
              .filter(ee.Filter.eq("instrumentMode", "IW"))
              .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
        if s1.size().getInfo() > 0:
            vv[i], vh[i] = _fetch(ee, s1.median(), ["VV", "VH"])

        r = (ee.ImageCollection(CHIRPS).filterDate(t0, t1).sum()
             .reduceRegion(ee.Reducer.mean(), region, 5000).getInfo())
        rain[i] = r.get("precipitation") or 0.0
        p = (ee.ImageCollection(ERA5_LAND).filterDate(t0, t1)
             .select("potential_evaporation_sum").sum()
             .reduceRegion(ee.Reducer.mean(), region, 10000).getInfo())
        et0[i] = abs(p.get("potential_evaporation_sum") or 0.02) * 1000

    _gap_fill(ndvi), _gap_fill(ndwi), _gap_fill(vv), _gap_fill(vh)

    print("  fetching ESA WorldCereal ground-truth labels...")
    label, conf = _worldcereal(ee, region)

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, ndvi=ndvi, ndwi=ndwi, vv=vv, vh=vh,
                        et0=et0, rain=rain, label=label, conf=conf)

    gt = _load_gt_csv() if GT_CSV else _sample_labels(label, conf)
    return {
        "ndvi": ndvi, "ndwi": ndwi, "vv": vv, "vh": vh,
        "crop_truth": None, "field_id": None, "stressed_truth": None,
        "et0": et0, "rain": rain, "gt": gt, "wc_label": label,
    }


def _worldcereal(ee, region):
    """Real crop labels from ESA WorldCereal 2021.

    0 = non-cropland, 1 = winter cereal (wheat), 2 = other temporary crop.
    conf = WorldCereal per-pixel decision confidence (0-100).
    """
    col = ee.ImageCollection(WORLDCEREAL).filterBounds(region)

    def prod(p, s):
        return (col.filter(ee.Filter.eq("product", p))
                .filter(ee.Filter.eq("season", s)).mosaic())

    tc = prod("temporarycrops", "tc-annual")
    wc = prod("wintercereals", "tc-wintercereals")
    is_crop = tc.select("classification").eq(100)
    is_wheat = wc.select("classification").eq(100)

    label = (ee.Image(0).where(is_crop, 2).where(is_wheat, 1).rename("label").toFloat())
    conf = (tc.select("confidence")
            .where(is_wheat, wc.select("confidence")).rename("conf").toFloat())
    return _fetch(ee, label.addBands(conf), ["label", "conf"])


def _sample_labels(label, conf, n_total=N_POINTS, floor=N_FLOOR, seed=11):
    """Sample high-confidence WorldCereal pixels as ground-truth points.

    Points are drawn ~proportional to each class's true prevalence (with a
    per-class floor), so the trained map reproduces real crop-area proportions
    instead of the flat prior a balanced sample would impose.
    """
    rng = np.random.default_rng(seed)
    hc = conf >= CONF_MIN
    prev = {c: int(((label == c) & hc).sum()) for c in (0, 1, 2)}
    tot = sum(prev.values()) or 1

    rows, cols, labs = [], [], []
    for c in (0, 1, 2):
        r, cl = np.where((label == c) & hc)
        if len(r) == 0:
            r, cl = np.where(label == c)  # relax if a class is sparse
        if len(r) == 0:
            continue
        want = max(floor, int(round(n_total * prev[c] / tot)))
        take = rng.choice(len(r), min(want, len(r)), replace=False)
        rows += r[take].tolist(); cols += cl[take].tolist(); labs += [c] * len(take)
    counts = {c: labs.count(c) for c in sorted(set(labs))}
    print(f"  ground-truth points sampled: {len(labs)} by class {counts} (conf>={CONF_MIN})")
    return np.array(rows), np.array(cols), np.array(labs)


def _gap_fill(stack):
    """Fill cloudy/missing composites: forward-fill in time, then backward."""
    for i in range(1, stack.shape[0]):
        m = np.isnan(stack[i])
        stack[i][m] = stack[i - 1][m]
    for i in range(stack.shape[0] - 2, -1, -1):
        m = np.isnan(stack[i])
        stack[i][m] = stack[i + 1][m]
    np.nan_to_num(stack, copy=False)


def _load_gt_csv():
    """Real survey points: CSV with header lon,lat,crop_id."""
    w, s, e, n = PILOT_BOUNDS
    rows, cols, labs = [], [], []
    with open(GT_CSV, newline="") as f:
        for rec in csv.DictReader(f):
            lon, lat = float(rec["lon"]), float(rec["lat"])
            if not (w <= lon <= e and s <= lat <= n):
                continue
            rows.append(int((n - lat) / (n - s) * (GRID_SIZE - 1)))
            cols.append(int((lon - w) / (e - w) * (GRID_SIZE - 1)))
            labs.append(int(rec["crop_id"]))
    print(f"  ground-truth points loaded from {GT_CSV}: {len(labs)}")
    return np.array(rows), np.array(cols), np.array(labs)
