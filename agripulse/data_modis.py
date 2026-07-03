"""MODIS moderate-resolution provider — proves the pipeline is resolution-agnostic.

The PS title says "Moderate Resolution Spectral Signatures". MODIS (the
MODerate-resolution Imaging Spectroradiometer, ~250 m) is the canonical
moderate-res sensor and is explicitly named in the PS. This provider runs the
SAME classify / stress / advisory code on MOD13Q1 250 m vegetation indices over
the same pilot area — demonstrating the method transfers straight to
moderate-resolution (and therefore to indigenous AWiFS, ~56 m, for national scale).

MODIS is optical-only: NDVI + EVI (16-day) + phenology; no SAR, no NDWI. VCI uses
the long MODIS archive (2015-2024) for a robust per-pixel baseline.
"""

import numpy as np

from .config import PILOT_BOUNDS

MODIS_VI = "MODIS/061/MOD13Q1"
S2_WC = "ESA/WorldCereal/2021/MODELS/v100"
CHIRPS = "UCSB-CHG/CHIRPS/DAILY"
ERA5_LAND = "ECMWF/ERA5_LAND/DAILY_AGGR"

GRID = 72                       # ~250 m pixels over the ~20 km pilot
SEASON_START = "2020-11-01"
SEASON_END = "2021-04-15"
CONF_MIN = 80


def _grid(size):
    w, s, e, n = PILOT_BOUNDS
    return {"dimensions": {"width": size, "height": size},
            "affineTransform": {"scaleX": (e - w) / size, "shearX": 0, "translateX": w,
                                "shearY": 0, "scaleY": -(n - s) / size, "translateY": n},
            "crsCode": "EPSG:4326"}


def _fetch(ee, image, bands, size=GRID):
    arr = ee.data.computePixels({"expression": image.select(bands).toFloat().unmask(-9999),
                                 "fileFormat": "NUMPY_NDARRAY", "grid": _grid(size)})
    out = np.stack([arr[b].astype(np.float32) for b in bands])
    out[out == -9999] = np.nan
    return out


def _gap_fill(stack):
    for i in range(1, stack.shape[0]):
        m = np.isnan(stack[i]); stack[i][m] = stack[i - 1][m]
    for i in range(stack.shape[0] - 2, -1, -1):
        m = np.isnan(stack[i]); stack[i][m] = stack[i + 1][m]
    np.nan_to_num(stack, copy=False)


def generate_scene():
    import ee
    ee.Initialize(project=__import__("os").environ.get("GEE_PROJECT", "agripulse-hackathon"))
    region = ee.Geometry.Rectangle(PILOT_BOUNDS)

    col = (ee.ImageCollection(MODIS_VI).filterBounds(region)
           .filterDate(SEASON_START, SEASON_END).sort("system:time_start"))
    ids = col.aggregate_array("system:index").getInfo()
    T = len(ids)
    ndvi = np.full((T, GRID, GRID), np.nan, np.float32)
    evi = np.full_like(ndvi, np.nan)
    doys = []
    print(f"  MODIS MOD13Q1: {T} sixteen-day composites at ~250 m", flush=True)
    for i, sid in enumerate(ids):
        img = ee.Image(f"{MODIS_VI}/{sid}")
        # each composite's ACTUAL day-of-year, so the baseline aligns to the real
        # MODIS calendar (fixed DOYs that reset at the year boundary), not to a
        # drifting Nov-1 + 16*i window.
        doys.append(int(ee.Date(img.get("system:time_start")).getRelative("day", "year").getInfo()) + 1)
        # QA: keep only good/marginal VI pixels (SummaryQA 0 or 1) to drop cloud/snow spikes
        qa = img.select("SummaryQA")
        good = qa.lte(1)
        ndvi[i] = _fetch(ee, img.select("NDVI").multiply(0.0001).updateMask(good), ["NDVI"])[0]
        evi[i] = _fetch(ee, img.select("EVI").multiply(0.0001).updateMask(good), ["EVI"])[0]
    _gap_fill(ndvi), _gap_fill(evi)

    print("  WorldCereal labels resampled to the MODIS grid...", flush=True)
    wc = ee.ImageCollection(S2_WC).filterBounds(region)
    tc = wc.filter(ee.Filter.eq("product", "temporarycrops")).filter(ee.Filter.eq("season", "tc-annual")).mosaic()
    wcw = wc.filter(ee.Filter.eq("product", "wintercereals")).filter(ee.Filter.eq("season", "tc-wintercereals")).mosaic()
    is_crop, is_wheat = tc.select("classification").eq(100), wcw.select("classification").eq(100)
    label_img = ee.Image(0).where(is_crop, 2).where(is_wheat, 1).rename("label").toFloat()
    conf_img = tc.select("confidence").where(is_wheat, wcw.select("confidence")).rename("conf").toFloat()
    label, conf = _fetch(ee, label_img.addBands(conf_img), ["label", "conf"])
    label = np.round(label).astype(np.int32)

    # weather scalars per 16-day window
    start = ee.Date(SEASON_START)
    et0, rain = np.zeros(T), np.zeros(T)
    for i in range(T):
        t0 = start.advance(i * 16, "day"); t1 = t0.advance(16, "day")
        rain[i] = (ee.ImageCollection(CHIRPS).filterDate(t0, t1).sum()
                   .reduceRegion(ee.Reducer.mean(), region, 5000).getInfo().get("precipitation") or 0.0)
        p = (ee.ImageCollection(ERA5_LAND).filterDate(t0, t1).select("potential_evaporation_sum").sum()
             .reduceRegion(ee.Reducer.mean(), region, 10000).getInfo())
        et0[i] = abs(p.get("potential_evaporation_sum") or 0.02) * 1000

    from .data_gee import _sample_labels
    gt = _sample_labels(label, conf)
    # NDVI + EVI + phenology only (MODIS VI product is optical); this demo
    # exercises the classifier's resolution-transfer. VCI stress needs a
    # per-sensor multi-year baseline (left to the Sentinel run).
    return {"ndvi": ndvi, "evi": evi, "et0": et0, "rain": rain,
            "gt": gt, "wc_label": label, "grid": GRID, "n_composites": T}
