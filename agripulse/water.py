"""8-day crop water balance and irrigation advisory.

ETc = Kc(crop, stage) * ET0            (FAO-56 crop coefficient method)
Deficit = ETc - effective rainfall - soil moisture drawdown proxy
The moisture-stress score modulates the advisory: a pixel already showing
canopy/soil water stress escalates to "irrigate now".
"""

import numpy as np

from .config import KC


def water_balance(scene, crop_map, stage, stress, t_now=None):
    et0, rain = scene["et0"], scene["rain"]
    T = scene["ndvi"].shape[0]
    if t_now is None:
        t_now = T - 1

    eff_rain = 0.8 * rain[t_now]

    kc = np.zeros_like(crop_map, dtype=np.float32)
    for c, kcs in KC.items():
        m = crop_map == c
        kc[m] = np.take(kcs, stage[m])

    etc = kc * et0[t_now]                      # mm per 8 days
    deficit = np.maximum(etc - eff_rain, 0.0)  # unmet demand

    # advisory: 0 none, 1 within 8 days (~25mm), 2 now (~50mm)
    advisory = np.zeros_like(crop_map, dtype=np.int32)
    advisory[(deficit > 12)] = 1
    advisory[(deficit > 12) & (stress >= 2)] = 2
    advisory[crop_map == 0] = 0

    series = {
        "et0": [round(float(v), 1) for v in et0],
        "rain": [round(float(v), 1) for v in rain],
    }
    return deficit, advisory, series
