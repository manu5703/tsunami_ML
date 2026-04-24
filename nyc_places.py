"""
NYC Neighbourhoods Lookup Table
================================
Bounding boxes for NYC neighbourhoods and boroughs.
Used when querying Airbnb / Taxi datasets with  Neighbourhood = 'name'.
"""

import pandas as pd

NYC_PLACES = [
    # ── Boroughs ──────────────────────────────────────────────────────────────
    {"Place": "Manhattan",          "Lat_min": 40.700, "Lat_max": 40.882, "Lon_min": -74.020, "Lon_max": -73.907},
    {"Place": "Brooklyn",           "Lat_min": 40.570, "Lat_max": 40.740, "Lon_min": -74.042, "Lon_max": -73.833},
    {"Place": "Queens",             "Lat_min": 40.541, "Lat_max": 40.800, "Lon_min": -73.962, "Lon_max": -73.700},
    {"Place": "Bronx",              "Lat_min": 40.785, "Lat_max": 40.917, "Lon_min": -73.933, "Lon_max": -73.748},
    {"Place": "Staten Island",      "Lat_min": 40.477, "Lat_max": 40.651, "Lon_min": -74.259, "Lon_max": -74.034},

    # ── Manhattan neighbourhoods ───────────────────────────────────────────────
    {"Place": "Midtown",            "Lat_min": 40.748, "Lat_max": 40.768, "Lon_min": -74.000, "Lon_max": -73.970},
    {"Place": "Lower Manhattan",    "Lat_min": 40.700, "Lat_max": 40.720, "Lon_min": -74.020, "Lon_max": -73.990},
    {"Place": "Upper East Side",    "Lat_min": 40.762, "Lat_max": 40.785, "Lon_min": -73.965, "Lon_max": -73.942},
    {"Place": "Upper West Side",    "Lat_min": 40.762, "Lat_max": 40.800, "Lon_min": -73.995, "Lon_max": -73.970},
    {"Place": "Harlem",             "Lat_min": 40.800, "Lat_max": 40.830, "Lon_min": -73.960, "Lon_max": -73.920},
    {"Place": "East Village",       "Lat_min": 40.720, "Lat_max": 40.733, "Lon_min": -73.995, "Lon_max": -73.975},
    {"Place": "West Village",       "Lat_min": 40.730, "Lat_max": 40.742, "Lon_min": -74.010, "Lon_max": -73.998},
    {"Place": "SoHo",               "Lat_min": 40.720, "Lat_max": 40.730, "Lon_min": -74.007, "Lon_max": -73.993},
    {"Place": "Tribeca",            "Lat_min": 40.714, "Lat_max": 40.724, "Lon_min": -74.015, "Lon_max": -74.000},
    {"Place": "Chelsea",            "Lat_min": 40.740, "Lat_max": 40.752, "Lon_min": -74.005, "Lon_max": -73.990},
    {"Place": "Hell's Kitchen",     "Lat_min": 40.752, "Lat_max": 40.768, "Lon_min": -74.005, "Lon_max": -73.988},
    {"Place": "Financial District", "Lat_min": 40.700, "Lat_max": 40.712, "Lon_min": -74.020, "Lon_max": -74.000},
    {"Place": "Chinatown",          "Lat_min": 40.713, "Lat_max": 40.720, "Lon_min": -74.002, "Lon_max": -73.993},
    {"Place": "Gramercy",           "Lat_min": 40.733, "Lat_max": 40.745, "Lon_min": -73.990, "Lon_max": -73.978},
    {"Place": "Murray Hill",        "Lat_min": 40.745, "Lat_max": 40.753, "Lon_min": -73.985, "Lon_max": -73.970},
    {"Place": "Flatiron",           "Lat_min": 40.738, "Lat_max": 40.745, "Lon_min": -73.993, "Lon_max": -73.985},
    {"Place": "Morningside Heights","Lat_min": 40.800, "Lat_max": 40.815, "Lon_min": -73.968, "Lon_max": -73.950},
    {"Place": "Washington Heights", "Lat_min": 40.833, "Lat_max": 40.868, "Lon_min": -73.950, "Lon_max": -73.920},
    {"Place": "Inwood",             "Lat_min": 40.865, "Lat_max": 40.882, "Lon_min": -73.940, "Lon_max": -73.910},

    # ── Brooklyn neighbourhoods ───────────────────────────────────────────────
    {"Place": "Williamsburg",       "Lat_min": 40.700, "Lat_max": 40.720, "Lon_min": -73.975, "Lon_max": -73.940},
    {"Place": "Brooklyn Heights",   "Lat_min": 40.692, "Lat_max": 40.702, "Lon_min": -74.002, "Lon_max": -73.990},
    {"Place": "Park Slope",         "Lat_min": 40.660, "Lat_max": 40.678, "Lon_min": -73.990, "Lon_max": -73.975},
    {"Place": "Bushwick",           "Lat_min": 40.693, "Lat_max": 40.713, "Lon_min": -73.935, "Lon_max": -73.905},
    {"Place": "Bed-Stuy",           "Lat_min": 40.676, "Lat_max": 40.695, "Lon_min": -73.960, "Lon_max": -73.920},
    {"Place": "Crown Heights",      "Lat_min": 40.659, "Lat_max": 40.678, "Lon_min": -73.955, "Lon_max": -73.930},
    {"Place": "Sunset Park",        "Lat_min": 40.634, "Lat_max": 40.655, "Lon_min": -74.010, "Lon_max": -73.985},
    {"Place": "Bay Ridge",          "Lat_min": 40.615, "Lat_max": 40.640, "Lon_min": -74.042, "Lon_max": -74.010},
    {"Place": "Coney Island",       "Lat_min": 40.570, "Lat_max": 40.585, "Lon_min": -74.010, "Lon_max": -73.980},
    {"Place": "DUMBO",              "Lat_min": 40.700, "Lat_max": 40.706, "Lon_min": -73.995, "Lon_max": -73.985},
    {"Place": "Flatbush",           "Lat_min": 40.627, "Lat_max": 40.648, "Lon_min": -73.965, "Lon_max": -73.940},
    {"Place": "Greenpoint",         "Lat_min": 40.720, "Lat_max": 40.735, "Lon_min": -73.960, "Lon_max": -73.940},

    # ── Queens neighbourhoods ─────────────────────────────────────────────────
    {"Place": "Astoria",            "Lat_min": 40.765, "Lat_max": 40.785, "Lon_min": -73.940, "Lon_max": -73.910},
    {"Place": "Long Island City",   "Lat_min": 40.742, "Lat_max": 40.760, "Lon_min": -73.955, "Lon_max": -73.930},
    {"Place": "Flushing",           "Lat_min": 40.755, "Lat_max": 40.775, "Lon_min": -73.840, "Lon_max": -73.810},
    {"Place": "Jackson Heights",    "Lat_min": 40.745, "Lat_max": 40.760, "Lon_min": -73.895, "Lon_max": -73.870},
    {"Place": "Jamaica",            "Lat_min": 40.685, "Lat_max": 40.710, "Lon_min": -73.820, "Lon_max": -73.780},
    {"Place": "JFK Airport",        "Lat_min": 40.615, "Lat_max": 40.650, "Lon_min": -73.820, "Lon_max": -73.760},

    # ── Bronx neighbourhoods ──────────────────────────────────────────────────
    {"Place": "South Bronx",        "Lat_min": 40.790, "Lat_max": 40.820, "Lon_min": -73.933, "Lon_max": -73.895},
    {"Place": "Fordham",            "Lat_min": 40.855, "Lat_max": 40.870, "Lon_min": -73.910, "Lon_max": -73.885},
    {"Place": "Riverdale",          "Lat_min": 40.883, "Lat_max": 40.905, "Lon_min": -73.925, "Lon_max": -73.897},
]

nyc_df = pd.DataFrame(NYC_PLACES).set_index("Place")


def lookup(name: str):
    name = name.strip()
    for key in nyc_df.index:
        if key.lower() == name.lower():
            row = nyc_df.loc[key]
            return row.Lat_min, row.Lat_max, row.Lon_min, row.Lon_max
    matches = [k for k in nyc_df.index if name.lower() in k.lower()]
    if len(matches) == 1:
        row = nyc_df.loc[matches[0]]
        return row.Lat_min, row.Lat_max, row.Lon_min, row.Lon_max
    if len(matches) > 1:
        raise KeyError(f"Ambiguous '{name}'. Did you mean: {', '.join(matches)}?")
    raise KeyError(f"Unknown neighbourhood '{name}'. Type  neighbourhoods  to see all.")


def list_places():
    return nyc_df.reset_index()[["Place", "Lat_min", "Lat_max", "Lon_min", "Lon_max"]]
