"""Google Earth Engine provider — real-data drop-in for data_sample.generate_scene.

Requires `pip install earthengine-api`, a registered GEE project
(https://code.earthengine.google.com/register), and `earthengine authenticate`.

Fetches 8-day composites over PILOT_BOUNDS on a fixed GRID_SIZE grid via
ee.data.computePixels:
  Sentinel-2 SR  -> NDVI, NDWI     Sentinel-1 GRD -> VV, VH
  CHIRPS         -> rainfall        ERA5-Land      -> ET0 proxy

Crop labels come from ESA WorldCereal 2021 — a global crop product that is
itself satellite-derived (validated globally against ~100k in-situ samples, but
NOT field-verified for this specific tile). So the reported OA/kappa measure
AGREEMENT WITH A SATELLITE PRODUCT, not accuracy against ground truth here, and
share some spectral-confusion modes with our own classifier. Treat them as a
reasonable reference baseline, not field accuracy.

For real field accuracy, set GT_CSV (columns lon,lat,crop_id) to the hackathon's
own survey points; the pipeline then reports OA/kappa against THOSE.
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
VCI_YEARS = list(range(2019, 2025))   # multi-year NDVI baseline for VCI (per 8-day window)


def _fingerprint():
    """Identity of the fetch config, so a stale cache for a different region/season
    is auto-invalidated instead of silently served."""
    return f"{PILOT_BOUNDS}|{GRID_SIZE}|{SEASON_START}|{N_COMPOSITES}|{COMPOSITE_DAYS}|{VCI_YEARS}"


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


def _mask_s2_clouds(img):
    """Per-pixel cloud/shadow/cirrus mask from the Sentinel-2 SCL band.

    Scene-level CLOUDY_PIXEL_PERCENTAGE only drops whole tiles; this removes the
    individual contaminated pixels (SCL 3 shadow, 8/9 cloud, 10 cirrus) so the
    median composite is built from clear observations.
    """
    scl = img.select("SCL")
    clear = (scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)))
    return img.updateMask(clear)


def _despeckle(s1_db):
    """Speckle reduction for Sentinel-1: 3x3 focal median in linear power.

    SAR GRD is multiplicative-noise heavy; filtering in linear power (not dB)
    is the correct domain. This is a boxcar/median filter; Refined Lee is the
    edge-preserving production upgrade. Temporal median compositing already
    removes much speckle; this cleans residual pixel spikes.
    """
    import ee
    nat = ee.Image(10).pow(s1_db.divide(10))
    filt = nat.focal_median(1.5, "square", "pixels")
    return filt.log10().multiply(10)


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
        d = dict(np.load(CACHE))
        cached_fp = str(d.get("fingerprint", ""))
        if "ndvi_lo" not in d:
            print("  cache missing VCI baseline; refetching from GEE...")
        elif cached_fp != _fingerprint():
            print(f"  cache is for a different area/season config; refetching from GEE...")
        else:
            print(f"  using cached data ({CACHE}); set GEE_REFRESH=1 to refetch")
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
    evi = np.full_like(ndvi, np.nan)
    vv = np.full_like(ndvi, np.nan)
    vh = np.full_like(ndvi, np.nan)
    smi_raw = np.full_like(ndvi, np.nan)   # ERA5-Land volumetric soil water (m3/m3)
    et0, rain = np.zeros(N_COMPOSITES), np.zeros(N_COMPOSITES)
    fill_fraction = np.zeros(N_COMPOSITES)

    for i in range(N_COMPOSITES):
        t0 = start.advance(i * COMPOSITE_DAYS, "day")
        t1 = t0.advance(COMPOSITE_DAYS, "day")
        print(f"  composite {i + 1}/{N_COMPOSITES}...", flush=True)

        s2 = (ee.ImageCollection(S2_SR).filterBounds(region).filterDate(t0, t1)
              .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
              .map(_mask_s2_clouds))   # per-pixel SCL cloud/shadow masking
        if s2.size().getInfo() > 0:
            comp = s2.median()
            r_ = comp.select("B4").divide(10000)
            n_ = comp.select("B8").divide(10000)
            b_ = comp.select("B2").divide(10000)
            evi_band = n_.subtract(r_).multiply(2.5).divide(
                n_.add(r_.multiply(6)).subtract(b_.multiply(7.5)).add(1)).rename("evi")
            img = (comp.normalizedDifference(["B8", "B4"]).rename("ndvi")
                   .addBands(comp.normalizedDifference(["B3", "B8"]).rename("ndwi"))
                   .addBands(evi_band))
            ndvi[i], ndwi[i], evi[i] = _fetch(ee, img, ["ndvi", "ndwi", "evi"])

        s1 = (ee.ImageCollection(S1_GRD).filterBounds(region).filterDate(t0, t1)
              .filter(ee.Filter.eq("instrumentMode", "IW"))
              .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
        if s1.size().getInfo() > 0:
            vv[i], vh[i] = _fetch(ee, _despeckle(s1.median()), ["VV", "VH"])

        r = (ee.ImageCollection(CHIRPS).filterDate(t0, t1).sum()
             .reduceRegion(ee.Reducer.mean(), region, 5000).getInfo())
        rain[i] = r.get("precipitation") or 0.0
        era = ee.ImageCollection(ERA5_LAND).filterDate(t0, t1)
        p = (era.select("potential_evaporation_sum").sum()
             .reduceRegion(ee.Reducer.mean(), region, 10000).getInfo())
        et0[i] = abs(p.get("potential_evaporation_sum") or 0.02) * 1000
        sm = era.select("volumetric_soil_water_layer_1").mean()
        smi_raw[i] = _fetch(ee, sm.rename("sm"), ["sm"])[0]

        fill_fraction[i] = float(np.isnan(ndvi[i]).mean())

    _gap_fill(ndvi), _gap_fill(ndwi), _gap_fill(evi)
    _gap_fill(vv), _gap_fill(vh), _gap_fill(smi_raw)

    # SMI (Soil Moisture Index): per-pixel min-max normalisation of ERA5 soil
    # water over the season -> 0 (driest on record) .. 1 (wettest). An
    # INDEPENDENT (reanalysis) moisture signal to cross-check the optical VCI.
    sm_lo = np.nanmin(smi_raw, axis=0)
    sm_hi = np.nanmax(smi_raw, axis=0)
    smi = np.clip((smi_raw - sm_lo) / np.maximum(sm_hi - sm_lo, 1e-6), 0, 1).astype(np.float32)

    print("  fetching ESA WorldCereal ground-truth labels...")
    label, conf = _worldcereal(ee, region)

    print(f"  fetching multi-year NDVI baseline for VCI ({VCI_YEARS[0]}-{VCI_YEARS[-1]})...")
    ndvi_lo, ndvi_hi = _vci_baseline(ee, region, start)

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, ndvi=ndvi, ndwi=ndwi, evi=evi, vv=vv, vh=vh,
                        smi=smi, et0=et0, rain=rain, fill_fraction=fill_fraction,
                        label=label, conf=conf, ndvi_lo=ndvi_lo, ndvi_hi=ndvi_hi,
                        fingerprint=_fingerprint())

    gt = _load_gt_csv() if GT_CSV else _sample_labels(label, conf)
    return {
        "ndvi": ndvi, "ndwi": ndwi, "evi": evi, "vv": vv, "vh": vh, "smi": smi,
        "crop_truth": None, "field_id": None, "stressed_truth": None,
        "et0": et0, "rain": rain, "fill_fraction": fill_fraction,
        "gt": gt, "wc_label": label, "ndvi_lo": ndvi_lo, "ndvi_hi": ndvi_hi,
    }


def _vci_baseline(ee, region, start):
    """Per-composite multi-year NDVI 10th/90th percentile envelope (for VCI).

    For each 8-day window, reduces all Sentinel-2 NDVI across VCI_YEARS falling
    in the same day-of-year range. VCI = (NDVI - lo) / (hi - lo) then measures a
    pixel's ABSOLUTE condition against its own historical range at that time of
    year — so region-wide stress is detectable, unlike a spatial anomaly.
    """
    lo = np.full((N_COMPOSITES, GRID_SIZE, GRID_SIZE), np.nan, np.float32)
    hi = np.full_like(lo, np.nan)
    for i in range(N_COMPOSITES):
        d0 = start.advance(i * COMPOSITE_DAYS, "day")
        doy0 = d0.getRelative("day", "year").getInfo() + 1
        doy1 = doy0 + COMPOSITE_DAYS
        col = (ee.ImageCollection(S2_SR).filterBounds(region)
               .filter(ee.Filter.calendarRange(VCI_YEARS[0], VCI_YEARS[-1], "year"))
               .filter(ee.Filter.calendarRange(doy0, doy1, "day_of_year"))
               .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50))
               .map(lambda im: im.normalizedDifference(["B8", "B4"]).rename("ndvi")))
        if col.size().getInfo() == 0:
            continue
        pct = col.reduce(ee.Reducer.percentile([10, 90]))
        lo[i], hi[i] = _fetch(ee, pct, ["ndvi_p10", "ndvi_p90"])
        print(f"    baseline window {i + 1}/{N_COMPOSITES}", flush=True)
    _gap_fill(lo), _gap_fill(hi)
    return lo, hi


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
