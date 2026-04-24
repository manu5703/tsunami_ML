"""
Elevation/slope zone lookup for the Forest Covertype dataset.

Usage in SQL:
    SELECT COUNT(*) FROM data WHERE Zone = 'Krummholz'
    SELECT COUNT(*) FROM data WHERE Zone = 'HighAlpine' AND Horiz_Dist_Roadways > 1500
    SELECT AVG(Slope) FROM data WHERE Zone = 'Lowland'

Type  zones  at the tsunami> prompt to see all zones.
"""

import pandas as pd

# Each entry: col -> (lo, hi)  — applied as range constraints on the covtype columns
_ZONES = {
    'Krummholz':  {'Elevation': (3400, 3858), 'Slope': (20, 90)},
    'HighAlpine': {'Elevation': (3200, 3858), 'Slope': (15, 90)},
    'Subalpine':  {'Elevation': (2800, 3200), 'Slope': (8,  22)},
    'Montane':    {'Elevation': (2400, 2800), 'Slope': (5,  18)},
    'Lowland':    {'Elevation': (1863, 2400), 'Slope': (0,  12)},
    'Riparian':   {'Elevation': (1863, 2200), 'Slope': (0,   8)},
}

_DESCRIPTIONS = {
    'Krummholz':  'Alpine treeline — stunted trees, rocky, wind-exposed  [TRAINING zone]',
    'HighAlpine': 'Expanded alpine/subalpine boundary                    [NEARBY / Group A]',
    'Subalpine':  'Dominant Spruce-Fir + Lodgepole Pine belt',
    'Montane':    'Mid-elevation Ponderosa Pine zone',
    'Lowland':    'Lower valley terrain                                  [FAR / Group B]',
    'Riparian':   'Valley streams, Cottonwood/Willow                     [FAR subset]',
}

_PCT = {
    'Krummholz':  '~4%',
    'HighAlpine': '~12%',
    'Subalpine':  '~65%',
    'Montane':    '~15%',
    'Lowland':    '~10%',
    'Riparian':   '~4%',
}


def lookup(name):
    key = next((k for k in _ZONES if k.lower() == name.lower()), None)
    if key is None:
        raise KeyError(
            f"Unknown zone '{name}'. Known zones: {', '.join(_ZONES)}"
        )
    return _ZONES[key]


def list_zones():
    rows = []
    for name, cols in _ZONES.items():
        e = cols['Elevation']
        s = cols['Slope']
        rows.append({
            'Zone':        name,
            'Elevation':   f"{e[0]}-{e[1]} m",
            'Slope':       f"{s[0]}-{s[1]} deg",
            '% of data':   _PCT[name],
            'Description': _DESCRIPTIONS[name],
        })
    return pd.DataFrame(rows)
