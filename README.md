# AgriPulse 🌾🛰️

AI-driven crop type mapping, stage-aware moisture stress detection, and 8-day
irrigation advisories from multi-source satellite data (optical + SAR).
Built for the Bharat Antariksh Hackathon problem statement.

**Pilot:** ~20 × 18 km in **Ludhiana district, Punjab** (Payal / Malerkotla,
Sirhind canal belt), **rabi 2020–21**. Real Sentinel-1/2 + CHIRPS + ERA5 via
Google Earth Engine; crop ground truth from **ESA WorldCereal 2021** (a global
product validated against 100k+ in-situ field samples).

## Validation (honest numbers)

Classifier is validated against WorldCereal using a **spatial west/east hold-out**
(not a random pixel split — random splits leak through neighbouring correlated
pixels and inflate scores). **Kappa / macro-F1 are the headline, not OA** — in an
~86%-wheat monoculture an "always-wheat" classifier already scores ~90%.

| metric | value | meaning |
|---|---|---|
| **Kappa** | **~0.63** | skill above chance — the honest headline |
| Macro-F1 | ~0.74 | mean F1 across the three classes |
| Overall accuracy | ~91.3% | only just beats the ~89.6% "always-wheat" baseline |
| Full-map agreement | ~88.9% | wall-to-wall vs WorldCereal |
| Wheat F1 | 0.96 | winter-cereal detection is strong |
| Non-crop F1 | 0.77 | rare class (~4% of area) |
| Other-crop F1 | 0.50 | heterogeneous catch-all — the genuine hard part |

> **On the ground truth:** WorldCereal is itself a *satellite-derived* product
> (globally validated against ~100k field samples, but **not** field-verified for
> this tile). So OA/kappa here measure **agreement with a satellite reference**,
> not accuracy against ground truth, and share some spectral-confusion modes with
> our classifier. Set `GT_CSV` (lon,lat,crop_id) to the hackathon's own survey
> points to report accuracy against *real* field data.

**What is validated, not just asserted:**
- **Crop map:** spatial hold-out with full confusion matrix, per-class
  precision/recall/F1, and the no-information baseline printed alongside OA.
- **Stress detector:** in sample mode, scored against injected field stress
  (recall ~0.86 / precision ~0.87) — it demonstrably recovers real stress.
- **Soil moisture (SMI):** independent ERA5 soil moisture rises the composite
  *after* rainfall (r ≈ 0.32), confirming the moisture layer is physically real.

## Pipeline

```
Optical (Sentinel-2 NDVI/NDWI) ─┐
                                ├─ 8-day composites ─ features ─ RF crop map
SAR (Sentinel-1 VV/VH) ─────────┘                        │
                                                         ▼
Weather (CHIRPS rain, ERA5 ET₀) ──► Kc water balance ◄─ stage-aware stress
                                          │                (VCI/NDWI/SAR anomaly)
                                          ▼
                            irrigation advisory maps + dashboard
```

- **Preprocessing** — Sentinel-2 L2A surface reflectance with per-pixel SCL
  cloud/shadow masking; Sentinel-1 GRD with a focal-median speckle filter
  (linear power); 8-day temporal-median compositing; cloudy gaps forward/back
  filled and the **fill fraction reported**.
- **Features** — multi-temporal NDVI, **EVI**, NDWI, VV, VH, VH−VV ratio +
  phenology metrics (SOS, EOS, **LGP**, peak timing, temporal stats).
- **Crop classification** — Random Forest; reported with kappa, macro-F1,
  per-class precision/recall/F1, no-information baseline, and a per-pixel
  **confidence** raster (`predict_proba`).
- **Moisture stress** — primary signal is **VCI (Vegetation Condition Index)**,
  an *absolute* index: each pixel's NDVI vs its own multi-year (2019–24) 10th/90th
  percentile envelope for that 8-day window (VCI≈0 = worst-on-record, ≈1 = best),
  blended with a canopy-water (NDWI) term. **Stage-dependent thresholds** —
  flowering is flagged at a higher VCI than maturity. Cross-checked against an
  independent **SMI** (ERA5 soil moisture). Sample mode falls back to a
  same-crop spatial anomaly.
- **Irrigation advisory** — FAO-56 Kc × ET₀ demand minus effective rainfall
  (with soil-storage carry-over) → *no action / schedule within 8 days /
  irrigate now*; "now" requires **confirmed VCI stress**, keeping the advisory
  coherent with the stress layer. Recommended depth = the computed deficit (mm).
- **Outputs** — every map PNG ships with a `.wld` + `.prj` sidecar so it loads
  as a **georeferenced raster in QGIS**; `summary.json` carries stage-wise stress
  per crop, areas in hectares, and all validation metrics.

Tunable coefficients (stress weights, VCI bands, Kc, deficit thresholds) live in
`config.py`, not as magic numbers. Run tests with `.venv\Scripts\python -m pytest`.

## Run it

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python run_pipeline.py --mode sample   # synthetic pilot area, no accounts needed
.venv\Scripts\python -m uvicorn dashboard.server:app --port 8010
# open http://localhost:8010
```

## Data modes

- `--mode sample` (default) — synthetic Sirhind-canal pilot area with realistic
  rabi phenology, field parcels, and tail-end stress. Demo never blocks on data access.
- `--mode gee` — real data via Google Earth Engine (`pip install earthengine-api`,
  register at https://code.earthengine.google.com/register, then
  `earthengine authenticate`; project id via `GEE_PROJECT`, default
  `agripulse-hackathon`). Collections wired in `agripulse/data_gee.py`:
  Sentinel-2 SR, Sentinel-1 GRD, CHIRPS rainfall, ERA5-Land ET. Swap in
  LISS-III/AWiFS from Bhoonidhi for the indigenous-data story.

GEE notes:
- Fetched composites **and** WorldCereal labels are cached in
  `outputs/gee_cache.npz` (offline demo fallback); set `GEE_REFRESH=1` to refetch.
- Ground-truth labels come from **ESA WorldCereal 2021**, sampled at
  high-confidence (≥80) pixels ~proportional to class prevalence so the map
  reproduces real crop-area proportions. Override with your own survey points
  via `GT_CSV=path\to\points.csv` (columns `lon,lat,crop_id`; 0=non-crop,
  1=wheat, 2=other).
- `--at T` picks the composite to analyse for stress/advisory (0-based,
  default last). Mid/late season (e.g. `--at 14`, ~21 Feb) is the interesting
  irrigation window for rabi wheat.

## Layout

```
agripulse/
  config.py       pilot bounds, crops, Kc table, class legends
  data_sample.py  synthetic scene generator (same contract as GEE provider)
  data_gee.py     Google Earth Engine provider (real collection IDs)
  features.py     temporal features + phenology (SOS/EOS/stage)
  classify.py     Random Forest + OA/kappa validation
  stress.py       stage-aware stress scoring
  water.py        Kc water balance → advisory classes
  pipeline.py     orchestration; writes outputs/ (PNG overlays + summary.json)
dashboard/        FastAPI + Leaflet/Chart.js dashboard
```
