"""Pilot-area and crop configuration for the AgriPulse prototype.

Pilot area: ~20 x 18 km in Ludhiana district, Punjab (Payal / Malerkotla belt,
Sirhind Canal command), a textbook rabi wheat zone.

Season is fixed to rabi 2020-21 because that is where independent, field-
validated ground truth exists: ESA WorldCereal 2021 (winter-cereal, temporary-
crop and irrigation products, validated against 100k+ in-situ samples). The
methodology is season-agnostic; point SEASON_START and GT at the hackathon's
provided current-season labels to run it operationally.
"""

# Geographic bounds of the pilot command area [west, south, east, north]
PILOT_BOUNDS = [75.90, 30.50, 76.10, 30.66]
GRID_SIZE = 160  # pixels per side (~115 m/pixel over this extent)

# Rabi 2020-21: 20 eight-day composites, Nov 1 -> ~Apr 10 (wheat green-up to senescence)
N_COMPOSITES = 20
SEASON_START = "2020-11-01"
COMPOSITE_DAYS = 8

# Crop classes grounded in ESA WorldCereal products (not invented rules)
CROPS = {
    0: {"name": "Non-cropland", "color": "#9e9e7a"},
    1: {"name": "Wheat (winter cereal)", "color": "#2e7d32"},
    2: {"name": "Other cropland", "color": "#f9a825"},
}

# Growth stages indexed by fraction of the crop's season elapsed
STAGES = ["Sowing", "Vegetative", "Flowering", "Maturity"]

# FAO-56 style crop coefficients per stage (Kc)
KC = {
    1: [0.35, 0.75, 1.15, 0.40],  # wheat / winter cereal
    2: [0.35, 0.70, 1.05, 0.45],  # other temporary crops (mustard/potato/veg mix)
}

STRESS_CLASSES = {
    0: {"name": "No stress", "color": "#1a9850"},
    1: {"name": "Mild", "color": "#fee08b"},
    2: {"name": "Moderate", "color": "#f46d43"},
    3: {"name": "Severe", "color": "#d73027"},
}

ADVISORY_CLASSES = {
    0: {"name": "No irrigation needed", "color": "#4575b4"},
    1: {"name": "Schedule irrigation (within 8 days)", "color": "#fdae61"},
    2: {"name": "Irrigate now (deficit critical)", "color": "#d73027"},
}

# --- tunable coefficients (kept here, not buried as magic numbers) ---

# Stress score blend (VCI is absolute vegetation condition; NDWI is canopy water)
STRESS_WEIGHTS = {"vci": 0.7, "ndwi": 0.3}

# VCI stress bands are STAGE-DEPENDENT: flowering is the most drought-sensitive
# wheat stage, maturity's natural dry-down is tolerated. Per stage index
# (0 Sowing,1 Vegetative,2 Flowering,3 Maturity): [none>=, mild>=, moderate>=];
# below the last value is severe. Higher thresholds => flagged sooner.
STRESS_VCI_BANDS = {
    0: [0.40, 0.28, 0.15],
    1: [0.45, 0.32, 0.18],
    2: [0.55, 0.40, 0.25],
    3: [0.40, 0.25, 0.12],
}

# Water balance / advisory
EFF_RAIN_NOW = 0.8        # fraction of this composite's rain that is effective
EFF_RAIN_PREV = 0.3       # soil-storage carry-over from the previous composite
DEFICIT_TRIGGER_MM = 8    # 8-day deficit above which irrigation is advised
DEFICIT_CRITICAL_MM = 35  # deficit above which "irrigate now" regardless of stress
VCI_WET_DEESCALATE = 0.75 # VCI above which a pixel is treated as recently watered

