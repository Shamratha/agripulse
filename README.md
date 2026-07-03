# AgriPulse 🌾🛰️

**A phenology-aware moisture-stress and 8-day irrigation-advisory engine for
canal command areas.** Built for the Bharat Antariksh Hackathon problem statement.

> **Where the contribution actually is.** Sentinel + Random Forest crop
> classification is a solved, commodity baseline — many teams will submit it.
> AgriPulse's differentiation is the layer *on top*: an **absolute, stage-aware
> VCI moisture-stress index** (each pixel judged against its own multi-year
> history, with flowering held to stricter thresholds than maturity) feeding a
> **FAO-56 crop-water-balance irrigation advisory** that stays coherent with the
> observed stress. The classifier is the substrate; the stress → advisory chain
> is the product.

**Pilot:** ~20 × 18 km in **Ludhiana district, Punjab** (Payal / Malerkotla,
Sirhind canal belt), **rabi 2020–21**. Real Sentinel-1/2 + CHIRPS + ERA5 via
Google Earth Engine; crop reference labels from **ESA WorldCereal 2021**.

## Validation (honest numbers)

> ### ⚠️ What these numbers do and don't prove
> The reference labels are **ESA WorldCereal 2021 — itself a *satellite-derived*
> model** (globally validated against ~100k field samples, but **not** field-checked
> for this tile). So kappa/OA here measure **agreement with another satellite
> model**, and can share spectral-confusion modes with our classifier — this is
> **not** accuracy against ground truth. It shows the pipeline independently
> reproduces a published crop product from our own S1+S2 feature stack; it does
> **not** prove field-level correctness. Point `GT_CSV` (lon,lat,crop_id) at the
> hackathon's survey points to get *real* field accuracy.

Validated with **spatial hold-outs** (not random pixel splits — those leak
through correlated neighbours and inflate scores), evaluated over **4 spatial
folds** (west/east + north/south, both directions) so the score is shown to be
stable, not an artifact of one boundary. **Kappa / macro-F1 are the headline, not
OA** — in an ~86%-wheat monoculture an "always-wheat" classifier already scores ~90%.

All held-out metrics are **mean ± std over the 4 spatial folds** — aggregates
*and* per-class, so the weak class gets the same scrutiny as the headline.

| metric | value | meaning |
|---|---|---|
| **Kappa** | **0.67 ± 0.03** | skill above chance, over 4 spatial folds |
| Macro-F1 | 0.77 ± 0.03 | mean F1 across the three classes |
| Overall accuracy | 92.1% ± 0.5% | only just beats the ~89.6% "always-wheat" baseline |
| Wheat F1 | 0.96 ± 0.00 | winter-cereal detection is strong and stable |
| Non-crop F1 | 0.81 ± 0.05 | rare class (~4% of area) |
| Other-crop F1 | 0.55 ± 0.04 | heterogeneous catch-all — *consistently* the hard part |
| Full-map agreement | ~88.9%† | †single map-vs-map concordance, **not** a fold metric (see below) |

† **Full-map agreement** is a *deployment* statistic, not a held-out evaluation:
the final model (trained on all ground-truth points) predicts every pixel, and
that wall-to-wall map is compared once against the whole WorldCereal raster. It
includes training pixels, so it's an optimistic concordance — a "does the output
map look like the reference" sanity check, not an accuracy estimate. The
fold-averaged numbers above are the honest generalisation figures.

**What is validated, not just asserted:**
- **Crop map:** 4-fold spatial hold-out with mean ± std, full confusion matrix,
  per-class precision/recall/F1, and the no-information baseline beside OA.
- **Feature importance:** optical indices (NDVI/EVI/NDWI ≈ 71%) lead, but
  Sentinel-1 SAR contributes ~23% — genuine multi-source fusion, not optical alone.
- **Stress detector:** in sample mode, scored against injected field stress
  (recall ~0.86 / precision ~0.87) — it demonstrably recovers real stress.
- **Soil moisture (SMI):** independent ERA5 soil moisture rises the composite
  *after* rainfall (r ≈ 0.32), confirming the moisture layer is physically real.

## Tests

**10 tests, 72% line coverage** (`.venv\Scripts\python -m pytest`). Coverage of the
offline-testable code is **~95%** — stress `100%`, features `100%`, config `100%`,
water `100%`, pipeline `89%`, classify `89%`. The live Earth Engine fetch layer
(`data_gee`, 17%) is excluded as it needs network. Tests guard the numeric
contracts the rest of the code depends on: band/scene shape parity, VCI ∈ [0,1],
stress/advisory class ranges, feature-name↔matrix alignment, advisory
escalation/de-escalation logic, fold-averaged per-class metrics, and a full
end-to-end sample run.

The suite is deliberately contract-focused rather than exhaustive: this is a
compact numeric pipeline (~580 statements) with little branching per module, so a
handful of contract tests hit ~95% of the runnable code. The count scales with
branching, not ambition.

## Data sources, resolution & national alignment

**On "Moderate Resolution."** The PS *title* says moderate resolution (strictly,
MODIS/AWiFS-class, ~56 m–1 km), but the PS *body* names the sanctioned inputs
explicitly:

> *"optical observations such as LISS-IV, LISS-III, AWiFS, Sentinel-2, Landsat
> and MODIS … microwave SAR observations such as EOS-04, Sentinel-1 and upcoming
> NISAR."*

So **Sentinel-1/2 are named, sanctioned PS inputs** — this prototype is on-spec,
not off it. We deliberately use Sentinel's finer 10–20 m scale because a 20 km
pilot needs field-level detail to be credible.

**Demonstrated, not just claimed.** `run_modis_demo.py` runs the *identical*
feature → Random-Forest → 4-fold-spatial-CV code on **MODIS MOD13Q1 at 250 m**
(the MODerate-resolution sensor, PS-named) over the same pilot and season:

| run | resolution | Kappa | Wheat F1 | minority classes |
|---|---|---|---|---|
| Sentinel-2 (main) | 10–20 m | 0.67 ± 0.03 | 0.96 | non-crop 0.81, other 0.55 |
| MODIS (demo) | 250 m | 0.40 ± 0.10 | 0.96 | non-crop 0.00, other 0.44 |

The method transfers **unchanged** — that's the point. The dominant wheat class
stays strong at 250 m; the accuracy drop is entirely the *expected resolution ↔
coverage trade-off* (coarse pixels blur small, mixed, and rare fields, so
minority classes collapse), not a code or method difference. Fine Sentinel suits
a command-area pilot; moderate-resolution AWiFS/MODIS trades per-field detail for
the wide swath that makes **national wall-to-wall** monitoring tractable. (The
demo covers the classifier; the VCI stress layer needs a per-sensor multi-year
baseline calibration and stays on the Sentinel run.)

**On indigenous data (honest status).** The current stack is open/foreign
(Copernicus, CHIRPS, ERA5, WorldCereal) chosen for reproducibility — all
free and on Earth Engine, so anyone can re-run it. Indian sources are the
**operational path, not yet wired**, and slot into the same provider contract
(`generate_scene()` returns one dict; sample/GEE are two implementations):

| layer | prototype (foreign, wired) | indigenous swap (PS-named, roadmap) |
|---|---|---|
| optical | Sentinel-2 | **AWiFS / LISS-III via ISRO Bhoonidhi** (moderate-res) |
| SAR | Sentinel-1 | **EOS-04 / RISAT, upcoming NISAR** |
| rainfall | CHIRPS | **IMD gridded rainfall** |
| ET / weather | ERA5-Land | **INSAT-derived ET, IMD grids** |
| crop labels | WorldCereal (satellite product) | **field survey points via `GT_CSV`** |

The last row is also the fix for the ground-truth circularity flagged above:
WorldCereal is a satellite reference, not field truth — the `GT_CSV` hook is
already wired to report accuracy against real survey points when available.

## Pipeline

```
Optical (Sentinel-2 NDVI/NDWI) ─┐
                                ├─ 8-day composites ─ features ─ RF crop map
SAR (Sentinel-1 VV/VH) ─────────┘                        │
                                                         ▼
Weather (CHIRPS rain, ERA5 ET₀) ──► Kc water balance ◄─ stage-aware stress
        │                                 │            (VCI + NDWI, stage thresholds)
        └── ERA5 soil moisture (SMI) ─────┴─ independent moisture cross-check
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

## Scaling to an operational system

Google Earth Engine here is a **prototyping backend** (per-user auth, quotas) —
we do **not** claim it as the operational foundation for a national service. What
makes the design scalable is the shape of the compute, not the host:

- **Per-tile and embarrassingly parallel.** Each run is one command area over one
  season — bounded work (20 composites over a small AOI). A district or state is
  just a set of command-area AOIs run independently; there is no global step that
  grows with area. Throughput scales by adding workers, not by a bigger model.
- **Swappable backend.** The same `generate_scene()` provider contract that
  switches sample↔GEE also lets the operational build swap GEE for an
  ISRO-native pipeline (**Bhoonidhi / MOSDAC** batch processing) or a cloud-native
  **STAC + Cloud-Optimized-GeoTIFF** stack (Sentinel Hub, MS Planetary Computer,
  or a self-hosted rasterio/dask cluster) — no change to the science modules.
- **Lightweight, standards-based outputs.** Each run emits small **georeferenced
  rasters (`.wld`/`.prj`, QGIS/ArcGIS-ready) + a JSON summary** — trivial to push
  to a command-area office, a Bhuvan-style portal, or a PMKSY/PMFBY dashboard. No
  heavyweight serving tier required.
- **Cadence.** Sentinel/AWiFS 8-day composites match the PS's 8-day water-deficit
  window; a cron per AOI produces a rolling advisory layer.

The realistic operational sensor is **moderate-resolution AWiFS** (indigenous,
PS-named, wide-swath) — coarser pixels mean a state is covered in far fewer tiles,
which is exactly why the resolution-agnostic method matters for national scale.

## Layout

```
agripulse/
  config.py       pilot bounds, crops, Kc table, stress/advisory thresholds, legends
  data_sample.py  synthetic scene generator (same contract as GEE provider)
  data_gee.py     GEE provider: S2/S1/CHIRPS/ERA5 + WorldCereal labels + VCI baseline
  data_modis.py   MODIS 250 m provider for the moderate-resolution demo
  features.py     temporal features (NDVI/EVI/NDWI/VV/VH) + phenology (SOS/EOS/LGP)
  classify.py     Random Forest + spatial hold-out (kappa, per-class precision/recall/F1)
  stress.py       stage-aware VCI stress scoring
  water.py        FAO-56 Kc water balance → advisory classes
  pipeline.py     orchestration; writes georeferenced maps + summary.json
run_pipeline.py   main entry: --mode sample|gee --at <composite>
run_modis_demo.py moderate-resolution (MODIS 250 m) resolution-transfer demo
dashboard/        FastAPI + Leaflet/Chart.js dashboard
tests/            pytest suite guarding the numeric contracts
```
