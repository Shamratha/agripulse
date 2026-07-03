"""8-day crop water balance and irrigation advisory.

ETc = Kc(crop, stage) * ET0            (FAO-56 crop coefficient method)
Deficit = ETc - effective rainfall (this composite + soil-storage carry-over)
The observed VCI stress escalates the advisory (a confirmed-stressed pixel with
a deficit -> "irrigate now"); a pixel whose vegetation condition is well above
typical (VCI high, i.e. recently irrigated) is de-escalated.

Recommended irrigation depth is the actual computed deficit (mm), not a fixed
label.
"""

import numpy as np

from .config import (DEFICIT_CRITICAL_MM, DEFICIT_TRIGGER_MM, EFF_RAIN_NOW,
                     EFF_RAIN_PREV, KC, VCI_WET_DEESCALATE)


def water_balance(scene, crop_map, stage, stress, vci, t_now=None):
    et0, rain = scene["et0"], scene["rain"]
    T = scene["ndvi"].shape[0]
    if t_now is None:
        t_now = T - 1

    # effective rain: this composite + soil-storage carry-over from the last one
    eff_rain = EFF_RAIN_NOW * rain[t_now] + (EFF_RAIN_PREV * rain[t_now - 1] if t_now > 0 else 0.0)

    kc = np.zeros_like(crop_map, dtype=np.float32)
    for c, kcs in KC.items():
        m = crop_map == c
        kc[m] = np.take(kcs, stage[m])

    etc = kc * et0[t_now]                      # mm per 8 days
    deficit = np.maximum(etc - eff_rain, 0.0)  # unmet demand (= recommended depth)

    # advisory: 0 none, 1 schedule (within 8 days), 2 irrigate now
    # "now" means the crop has an unmet demand AND is showing stress (VCI) — this
    # keeps the advisory coherent with the stress layer rather than escalating on
    # raw ET demand alone. A very large deficit escalates only if also stressed.
    advisory = np.zeros_like(crop_map, dtype=np.int32)
    advisory[deficit > DEFICIT_TRIGGER_MM] = 1
    escalate = (deficit > DEFICIT_TRIGGER_MM) & (stress >= 2)
    advisory[escalate] = 2
    # de-escalate: vegetation condition well above typical => effectively watered.
    # VCI is 0..1 (GEE mode); if unavailable (sample mode) skip de-escalation.
    if vci is not None:
        advisory[(vci >= VCI_WET_DEESCALATE) & (deficit < DEFICIT_CRITICAL_MM) & (stress < 2)] = 0
    advisory[crop_map == 0] = 0

    series = {
        "et0": [round(float(v), 1) for v in et0],
        "rain": [round(float(v), 1) for v in rain],
    }
    return deficit, advisory, series
