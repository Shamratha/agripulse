# AgriPulse 🌾🛰️

AI-driven crop type mapping, stage-aware moisture stress detection, and 8-day
irrigation advisories from multi-source satellite data (optical + SAR).
Built for the Bharat Antariksh Hackathon problem statement.

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

- **Crop classification** — Random Forest on multi-temporal NDVI/NDWI/VV/VH +
  phenology metrics (SOS, EOS, peak timing); validated with OA and kappa.
- **Moisture stress** — per-pixel anomaly vs same-crop, same-composite medians
  (phenology-aware: natural senescence is not flagged); classes none/mild/moderate/severe.
- **Irrigation advisory** — FAO-56 Kc × ET₀ demand minus effective rainfall,
  escalated by observed stress → *no action / irrigate within 8 days / irrigate now*.

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
- Fetched composites are cached in `outputs/gee_cache.npz` (offline demo
  fallback); set `GEE_REFRESH=1` to refetch.
- Without real survey points, training labels are **pseudo-labels** from NDVI
  trajectory rules — fine for testing, replace before judging: set
  `GT_CSV=path\to\points.csv` (columns `lon,lat,crop_id`).
- `--at T` picks the composite to analyse for stress/advisory (0-based,
  default last). Mid-season (e.g. `--at 12`, early Feb) is the interesting
  irrigation window for rabi.

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
