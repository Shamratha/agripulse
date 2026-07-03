"""Google Earth Engine provider — real-data drop-in for data_sample.generate_scene.

Requires: `pip install earthengine-api` and a registered GEE project
(sign up at https://code.earthengine.google.com/register — free for
non-commercial use). Then: `earthengine authenticate`.

Fills the same dict contract as generate_scene():
  ndvi/ndwi/vv/vh: (T, H, W) arrays of 8-day composites over PILOT_BOUNDS
  et0/rain: per-composite weather series
  gt: ground-truth points (load from your survey CSV / Bhuvan crop points)
"""

import numpy as np

from .config import PILOT_BOUNDS, GRID_SIZE, N_COMPOSITES, SEASON_START, COMPOSITE_DAYS

# GEE collection IDs used by this provider
S2_SR = "COPERNICUS/S2_SR_HARMONIZED"      # optical: NDVI (B8,B4), NDWI (B3,B8)
S1_GRD = "COPERNICUS/S1_GRD"               # SAR: VV, VH backscatter
CHIRPS = "UCSB-CHG/CHIRPS/DAILY"           # rainfall (mm/day)
ERA5_LAND = "ECMWF/ERA5_LAND/DAILY_AGGR"   # reference ET proxy / temperature


def generate_scene():
    try:
        import ee
    except ImportError as e:
        raise RuntimeError(
            "earthengine-api not installed. Run in sample mode (--mode sample) "
            "or `pip install earthengine-api` and authenticate."
        ) from e

    ee.Initialize()
    region = ee.Geometry.Rectangle(PILOT_BOUNDS)
    start = ee.Date(SEASON_START)

    ndvi_stack, ndwi_stack, vv_stack, vh_stack = [], [], [], []
    et0, rain = [], []

    for i in range(N_COMPOSITES):
        t0 = start.advance(i * COMPOSITE_DAYS, "day")
        t1 = t0.advance(COMPOSITE_DAYS, "day")

        s2 = (ee.ImageCollection(S2_SR).filterBounds(region).filterDate(t0, t1)
              .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40)).median())
        ndvi = s2.normalizedDifference(["B8", "B4"])
        ndwi = s2.normalizedDifference(["B3", "B8"])

        s1 = (ee.ImageCollection(S1_GRD).filterBounds(region).filterDate(t0, t1)
              .filter(ee.Filter.eq("instrumentMode", "IW")).median())
        vv, vh = s1.select("VV"), s1.select("VH")

        for img, stack in ((ndvi, ndvi_stack), (ndwi, ndwi_stack), (vv, vv_stack), (vh, vh_stack)):
            arr = np.array(
                img.sampleRectangle(region, defaultValue=0).getInfo()["properties"]
                [list(img.bandNames().getInfo())[0]], dtype=np.float32)
            stack.append(arr)

        rain_img = ee.ImageCollection(CHIRPS).filterDate(t0, t1).sum()
        rain.append(rain_img.reduceRegion(ee.Reducer.mean(), region, 5000)
                    .getInfo().get("precipitation", 0.0))
        # Simple ET0 proxy from ERA5 potential evaporation (m -> mm, sign flip)
        pev = (ee.ImageCollection(ERA5_LAND).filterDate(t0, t1)
               .select("potential_evaporation_sum").sum()
               .reduceRegion(ee.Reducer.mean(), region, 10000).getInfo())
        et0.append(abs(pev.get("potential_evaporation_sum", 0.003)) * 1000)

    return {
        "ndvi": np.stack(ndvi_stack), "ndwi": np.stack(ndwi_stack),
        "vv": np.stack(vv_stack), "vh": np.stack(vh_stack),
        "crop_truth": None, "field_id": None, "stressed_truth": None,
        "et0": np.array(et0), "rain": np.array(rain),
        # TODO: replace with real survey points (CSV of lat, lon, crop label)
        "gt": (np.array([]), np.array([]), np.array([])),
    }
