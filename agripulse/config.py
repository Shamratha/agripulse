"""Pilot-area and crop configuration for the AgriPulse prototype.

Pilot area: a patch of the Sirhind Canal command area, Punjab (rabi season).
The sample-data generator synthesizes this area; the GEE provider clips the
real collections to the same bounds.
"""

# Geographic bounds of the pilot command area [west, south, east, north]
PILOT_BOUNDS = [75.90, 30.50, 76.10, 30.66]
GRID_SIZE = 160  # pixels per side (~110 m/pixel over this extent)

# Rabi season: 15 eight-day composites, Nov 1 -> Mar 31 (approx)
N_COMPOSITES = 15
SEASON_START = "2025-11-01"
COMPOSITE_DAYS = 8

CROPS = {
    0: {"name": "Fallow / Other", "color": "#9e9e7a"},
    1: {"name": "Wheat", "color": "#2e7d32"},
    2: {"name": "Mustard", "color": "#f9a825"},
    3: {"name": "Sugarcane", "color": "#00695c"},
}

# Growth stages indexed by fraction of the crop's season elapsed
STAGES = ["Sowing", "Vegetative", "Flowering", "Maturity"]

# FAO-56 style crop coefficients per stage (Kc)
KC = {
    1: [0.35, 0.75, 1.15, 0.40],  # wheat
    2: [0.35, 0.70, 1.10, 0.35],  # mustard
    3: [0.50, 0.90, 1.25, 0.75],  # sugarcane
}

STRESS_CLASSES = {
    0: {"name": "No stress", "color": "#1a9850"},
    1: {"name": "Mild", "color": "#fee08b"},
    2: {"name": "Moderate", "color": "#f46d43"},
    3: {"name": "Severe", "color": "#d73027"},
}

ADVISORY_CLASSES = {
    0: {"name": "No irrigation needed", "color": "#4575b4"},
    1: {"name": "Irrigate within 8 days (~25 mm)", "color": "#fdae61"},
    2: {"name": "Irrigate now (~50 mm)", "color": "#d73027"},
}
