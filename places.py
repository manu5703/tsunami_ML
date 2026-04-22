"""
California Places Table
=======================
Bounding boxes for California cities and regions.
Used by query_cli.py to resolve  Place = 'name'  into lat/lon ranges.
"""

import pandas as pd

# ── Places table ──────────────────────────────────────────────────────────────
PLACES = [
    # ── Major cities ──────────────────────────────────────────────────────────
    {"Place": "San Francisco",   "Lat_min": 37.70, "Lat_max": 37.92, "Lon_min": -122.55, "Lon_max": -122.33},
    {"Place": "Los Angeles",     "Lat_min": 33.50, "Lat_max": 34.50, "Lon_min": -119.00, "Lon_max": -117.50},
    {"Place": "San Diego",       "Lat_min": 32.50, "Lat_max": 33.20, "Lon_min": -117.50, "Lon_max": -116.50},
    {"Place": "Sacramento",      "Lat_min": 38.30, "Lat_max": 38.80, "Lon_min": -121.70, "Lon_max": -121.10},
    {"Place": "San Jose",        "Lat_min": 37.20, "Lat_max": 37.50, "Lon_min": -122.10, "Lon_max": -121.70},
    {"Place": "Oakland",         "Lat_min": 37.70, "Lat_max": 37.90, "Lon_min": -122.35, "Lon_max": -122.10},
    {"Place": "Berkeley",        "Lat_min": 37.83, "Lat_max": 37.92, "Lon_min": -122.35, "Lon_max": -122.22},
    {"Place": "Fresno",          "Lat_min": 36.60, "Lat_max": 37.00, "Lon_min": -120.00, "Lon_max": -119.50},
    {"Place": "Long Beach",      "Lat_min": 33.70, "Lat_max": 33.90, "Lon_min": -118.30, "Lon_max": -118.00},
    {"Place": "Santa Barbara",   "Lat_min": 34.30, "Lat_max": 34.60, "Lon_min": -120.20, "Lon_max": -119.50},
    {"Place": "Monterey",        "Lat_min": 36.50, "Lat_max": 36.70, "Lon_min": -122.00, "Lon_max": -121.70},
    {"Place": "Bakersfield",     "Lat_min": 35.20, "Lat_max": 35.60, "Lon_min": -119.20, "Lon_max": -118.80},
    {"Place": "Stockton",        "Lat_min": 37.80, "Lat_max": 38.10, "Lon_min": -121.50, "Lon_max": -121.10},
    {"Place": "Riverside",       "Lat_min": 33.80, "Lat_max": 34.10, "Lon_min": -117.60, "Lon_max": -117.20},
    {"Place": "Anaheim",         "Lat_min": 33.75, "Lat_max": 33.90, "Lon_min": -118.00, "Lon_max": -117.70},
    {"Place": "Santa Rosa",      "Lat_min": 38.35, "Lat_max": 38.60, "Lon_min": -122.85, "Lon_max": -122.50},
    {"Place": "Oxnard",          "Lat_min": 34.10, "Lat_max": 34.30, "Lon_min": -119.30, "Lon_max": -119.00},
    {"Place": "Irvine",          "Lat_min": 33.60, "Lat_max": 33.75, "Lon_min": -117.90, "Lon_max": -117.70},
    {"Place": "Chula Vista",     "Lat_min": 32.55, "Lat_max": 32.70, "Lon_min": -117.15, "Lon_max": -116.95},
    {"Place": "Modesto",         "Lat_min": 37.55, "Lat_max": 37.75, "Lon_min": -121.10, "Lon_max": -120.80},

    # ── Tech / wealthy enclaves ───────────────────────────────────────────────
    {"Place": "Silicon Valley",  "Lat_min": 37.20, "Lat_max": 37.60, "Lon_min": -122.20, "Lon_max": -121.70},
    {"Place": "Palo Alto",       "Lat_min": 37.38, "Lat_max": 37.48, "Lon_min": -122.18, "Lon_max": -122.07},
    {"Place": "Sunnyvale",       "Lat_min": 37.33, "Lat_max": 37.42, "Lon_min": -122.08, "Lon_max": -121.97},
    {"Place": "Mountain View",   "Lat_min": 37.35, "Lat_max": 37.43, "Lon_min": -122.13, "Lon_max": -122.00},
    {"Place": "Cupertino",       "Lat_min": 37.28, "Lat_max": 37.35, "Lon_min": -122.10, "Lon_max": -121.97},
    {"Place": "Santa Monica",    "Lat_min": 34.00, "Lat_max": 34.05, "Lon_min": -118.52, "Lon_max": -118.44},
    {"Place": "Beverly Hills",   "Lat_min": 34.05, "Lat_max": 34.10, "Lon_min": -118.43, "Lon_max": -118.37},
    {"Place": "Malibu",          "Lat_min": 34.00, "Lat_max": 34.10, "Lon_min": -118.95, "Lon_max": -118.60},

    # ── Broad regions ─────────────────────────────────────────────────────────
    {"Place": "Bay Area",        "Lat_min": 37.00, "Lat_max": 38.50, "Lon_min": -123.00, "Lon_max": -121.00},
    {"Place": "LA Metro",        "Lat_min": 33.50, "Lat_max": 34.50, "Lon_min": -119.00, "Lon_max": -117.50},
    {"Place": "SF Metro",        "Lat_min": 37.40, "Lat_max": 37.90, "Lon_min": -122.60, "Lon_max": -121.80},
    {"Place": "Central Valley",  "Lat_min": 35.00, "Lat_max": 39.00, "Lon_min": -121.50, "Lon_max": -118.50},
    {"Place": "Coastal SoCal",   "Lat_min": 32.50, "Lat_max": 34.50, "Lon_min": -119.00, "Lon_max": -117.00},
    {"Place": "Coastal NorCal",  "Lat_min": 37.50, "Lat_max": 41.95, "Lon_min": -124.35, "Lon_max": -122.00},
    {"Place": "Northern CA",     "Lat_min": 38.50, "Lat_max": 41.95, "Lon_min": -124.35, "Lon_max": -114.31},
    {"Place": "Southern CA",     "Lat_min": 32.54, "Lat_max": 36.00, "Lon_min": -121.00, "Lon_max": -114.31},
    {"Place": "Inland Empire",   "Lat_min": 33.60, "Lat_max": 34.30, "Lon_min": -117.70, "Lon_max": -116.50},
    {"Place": "Orange County",   "Lat_min": 33.40, "Lat_max": 33.95, "Lon_min": -118.10, "Lon_max": -117.40},
    {"Place": "Wine Country",    "Lat_min": 38.00, "Lat_max": 39.00, "Lon_min": -123.50, "Lon_max": -122.00},
    {"Place": "Gold Country",    "Lat_min": 37.50, "Lat_max": 39.50, "Lon_min": -121.00, "Lon_max": -119.50},
]

places_df = pd.DataFrame(PLACES).set_index("Place")


def lookup(name: str):
    """
    Return (lat_min, lat_max, lon_min, lon_max) for a place name.
    Case-insensitive, partial match allowed.
    Raises KeyError if not found.
    """
    name = name.strip()
    # exact match first (case-insensitive)
    for key in places_df.index:
        if key.lower() == name.lower():
            row = places_df.loc[key]
            return row.Lat_min, row.Lat_max, row.Lon_min, row.Lon_max
    # partial match
    matches = [k for k in places_df.index if name.lower() in k.lower()]
    if len(matches) == 1:
        row = places_df.loc[matches[0]]
        return row.Lat_min, row.Lat_max, row.Lon_min, row.Lon_max
    if len(matches) > 1:
        raise KeyError(f"Ambiguous place '{name}'. Did you mean: {', '.join(matches)}?")
    raise KeyError(f"Unknown place '{name}'. Type  places  to see all options.")


def list_places():
    return places_df.reset_index()[["Place", "Lat_min", "Lat_max", "Lon_min", "Lon_max"]]
