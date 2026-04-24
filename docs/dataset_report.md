# Tsunami Index — Dataset Report

---

## Dataset 1 — California Housing (Real)

### Data Overview

| Property | Value |
|---|---|
| Source | 1990 US Census, via `sklearn.datasets.fetch_california_housing` |
| Raw rows | 20,640 block groups |
| Rows used | 2,000,000 (oversampled with 1% Gaussian jitter) |
| Dimensions | 9 continuous columns |
| Lookup table | `Place` — maps city name to (Lat_min, Lat_max, Lon_min, Lon_max) |

### Column Table

| Column | Unit | Range | Description |
|---|---|---|---|
| MedInc | x$10K | 0.5 – 15.0 | Median household income |
| HouseAge | years | 1 – 52 | Median house age in block group |
| AveRooms | count | 0.8 – 20.0 | Average rooms per household |
| AveBedrms | count | 0.3 – 5.0 | Average bedrooms per household |
| Population | count | 3 – 10,000 | Block group population |
| AveOccup | count | 0.7 – 20.0 | Average occupants per household |
| Latitude | degrees | 32.54 – 41.95 | Block group latitude |
| Longitude | degrees | -124.35 – -114.31 | Block group longitude |
| MedHouseVal | x$100K | 0.15 – 5.0 | Median house value |

### Training, Nearby, and Far Query Zones

| Group | Zone | Geographic Region | Key Filters | Approx. Selectivity |
|---|---|---|---|---|
| Training | Bay Area + housing attributes | Lat 37.0–38.5, Lon -122.5–-121.0 | MedInc, AveRooms, AveBedrms, MedHouseVal ranges | ~5–8% of data |
| Nearby (Group A) | Silicon Valley cities | Sunnyvale, Mountain View, Palo Alto, Cupertino | Same housing attributes within city bounding boxes | ~3–6% per city |
| Far (Group B) | Southern California | Los Angeles, Irvine, Chula Vista | Same housing attributes, 600+ km south of Bay Area | ~3–6% per city |

Training queries directly specify latitude/longitude ranges for the Bay Area combined with housing attribute ranges. Nearby queries use the Place lookup table to resolve Silicon Valley city names into tight lat/lon bounding boxes — geographic sub-clusters within the training region. Far queries apply the same lookup for SoCal cities, which lie entirely outside the trained region in 9-dimensional feature space.

### Place Lookup Table (subset)

| Place | Lat min | Lat max | Lon min | Lon max |
|---|---|---|---|---|
| Bay Area | 37.00 | 38.50 | -123.00 | -121.00 |
| Sunnyvale | 37.33 | 37.42 | -122.08 | -121.97 |
| Mountain View | 37.35 | 37.43 | -122.13 | -122.00 |
| Palo Alto | 37.38 | 37.48 | -122.18 | -122.07 |
| Cupertino | 37.28 | 37.35 | -122.10 | -121.97 |
| Los Angeles | 33.50 | 34.50 | -119.00 | -117.50 |
| Irvine | 33.60 | 33.75 | -117.90 | -117.70 |
| Chula Vista | 32.55 | 32.70 | -117.15 | -116.95 |

---

## Dataset 2 — NYC Yellow Taxi (Real)

### Data Overview

| Property | Value |
|---|---|
| Source | NYC TLC Yellow Taxi Trip Records, June 2016 |
| Raw rows | ~10–11 million trips |
| Rows used | 2,000,000 (subsample or oversample with 1% jitter) |
| Dimensions | 10 continuous columns |
| Lookup table | `Neighbourhood` — maps NYC neighborhood name to (Lat_min, Lat_max, Lon_min, Lon_max) |

Note: 2016 is the last year the TLC dataset included raw pickup/dropoff latitude and longitude.
From 2017 onward TLC replaced coordinates with zone IDs, which cannot be used as numeric
range features for the Tsunami index.

### Column Table

| Column | Unit | Description |
|---|---|---|
| Latitude | degrees | Pickup latitude (~40.60 – 40.85) |
| Longitude | degrees | Pickup longitude (~-74.05 – -73.75) |
| dropoff_lat | degrees | Dropoff latitude |
| dropoff_lon | degrees | Dropoff longitude |
| trip_distance | miles | Distance traveled |
| fare_amount | USD | Base metered fare |
| tip_amount | USD | Tip paid |
| passenger_count | count | Number of passengers (1–6) |
| duration_min | minutes | Trip duration |
| total_amount | USD | Total charged (fare + tip + surcharges) |

### Training, Nearby, and Far Query Zones

| Group | Zone | Key Filters | Approx. Selectivity |
|---|---|---|---|
| Training | Expanded premium / medium-long trips | fare_amount > 30, trip_distance > 8 | ~10–15% of data |
| Nearby (Group A) | Tight premium / long-haul trips | fare_amount > 40, trip_distance > 10 | ~5% of data |
| Far (Group B) | Budget / short trips | fare_amount < 12, trip_distance < 2 | ~35–45% of data |

Training queries cover the broader premium zone (fare > $30, distance > 8 miles) — mid-to-high fare rides including airport runs and outer-borough trips. Tsunami builds fine-grained partitions across this region. Nearby queries tighten the thresholds to the core long-haul cluster (fare > $40, distance > 10 miles), which is a strict subset of the training zone — Tsunami's partitions are maximally precise here, yielding low scan% and high speedup. Far queries target the opposite end of the feature space: short, cheap rides (fare < $12, distance < 2 miles) that make up the most common ride type in NYC. These land in coarse unoptimized partitions, resulting in high scan% and lower speedup.

### Neighbourhood Lookup Table (subset)

| Neighbourhood | Lat min | Lat max | Lon min | Lon max |
|---|---|---|---|---|
| Midtown | 40.748 | 40.768 | -74.000 | -73.970 |
| Chelsea | 40.740 | 40.752 | -74.005 | -73.990 |
| Gramercy | 40.733 | 40.745 | -73.990 | -73.978 |
| Flatiron | 40.738 | 40.745 | -73.993 | -73.985 |
| Murray Hill | 40.745 | 40.753 | -73.985 | -73.970 |
| West Village | 40.730 | 40.742 | -74.010 | -73.998 |
| SoHo | 40.720 | 40.730 | -74.007 | -73.993 |
| Flushing | 40.755 | 40.775 | -73.840 | -73.810 |
| Jamaica | 40.685 | 40.710 | -73.820 | -73.780 |
| JFK Airport | 40.615 | 40.650 | -73.820 | -73.760 |
| Coney Island | 40.570 | 40.585 | -74.010 | -73.980 |
| Fordham | 40.855 | 40.870 | -73.910 | -73.885 |
| Riverdale | 40.883 | 40.905 | -73.925 | -73.897 |
| Flatbush | 40.627 | 40.648 | -73.965 | -73.940 |

---

## Dataset 3 — Forest Covertype (Real)

### Data Overview

| Property | Value |
|---|---|
| Source | Blackard & Dean (1999), Roosevelt National Forest, Colorado; via `sklearn.datasets.fetch_covtype` |
| Raw rows | 581,012 forest land observations |
| Rows used | 2,000,000 (oversampled with 1% Gaussian jitter) |
| Dimensions | 10 continuous columns (binary wilderness/soil columns excluded) |
| Lookup table | `Zone` — maps ecological zone name to (Elevation range, Slope range) |

### Column Table

| Column | Unit | Range | Description |
|---|---|---|---|
| Elevation | meters | 1,863 – 3,858 | Elevation above sea level |
| Aspect | degrees | 0 – 360 | Compass direction the slope faces |
| Slope | degrees | 0 – 66 | Steepness of terrain |
| Horiz_Dist_Hydrology | meters | 0 – 1,397 | Horizontal distance to nearest water source |
| Vert_Dist_Hydrology | meters | -173 – 601 | Vertical distance to nearest water source |
| Horiz_Dist_Roadways | meters | 0 – 7,117 | Horizontal distance to nearest road |
| Hillshade_9am | 0–255 | 0 – 254 | Hillshade index at 9 AM (summer solstice) |
| Hillshade_Noon | 0–255 | 0 – 254 | Hillshade index at noon (summer solstice) |
| Hillshade_3pm | 0–255 | 0 – 254 | Hillshade index at 3 PM (summer solstice) |
| Horiz_Dist_Fire_Points | meters | 0 – 7,173 | Distance to nearest wildfire ignition point |

### Zone Lookup Table

| Zone | Elevation | Slope | % of Dataset | Ecological Description |
|---|---|---|---|---|
| Krummholz | 3,400–3,858 m | 20–90 deg | ~4% | Alpine treeline — stunted trees, rocky, wind-exposed (Nearby / Group A) |
| HighAlpine | 3,200–3,858 m | 15–90 deg | ~12% | Expanded alpine/subalpine boundary (Training zone) |
| Subalpine | 2,800–3,200 m | 8–22 deg | ~65% | Dominant Spruce-Fir + Lodgepole Pine belt |
| Montane | 2,400–2,800 m | 5–18 deg | ~15% | Mid-elevation Ponderosa Pine zone |
| Lowland | 1,863–2,400 m | 0–12 deg | ~10% | Lower valley terrain (Far / Group B) |
| Riparian | 1,863–2,200 m | 0–8 deg | ~4% | Valley streams, Cottonwood/Willow |

### Training, Nearby, and Far Query Zones

| Group | Zone | Elevation | Slope | Approx. Selectivity |
|---|---|---|---|---|
| Training | HighAlpine | > 3,200 m | > 15 deg | ~8–12% of dataset |
| Nearby (Group A) | Krummholz | > 3,400 m | > 20 deg | ~3–5% of dataset |
| Far (Group B) | Lowland | < 2,400 m | < 12 deg | ~6–10% of dataset |

Krummholz is a geographic and ecological subset of HighAlpine. Tsunami is trained on the
broader alpine zone and builds highly precise partitions covering the entire alpine band,
including the tighter Krummholz sub-cluster. Lowland terrain sits at the opposite extreme
of the elevation gradient in coarse unoptimized partitions.

---

## How a Query Flows Through Tsunami's Components

Example query (Forest Covertype):
```sql
SELECT COUNT(*) FROM data WHERE Elevation > 3400 AND Slope > 20 AND Hillshade_9am < 200
```

### Step 1 — SQL Parsing

The WHERE clause is split on AND into individual conditions. Each condition is converted
into a per-column numeric range constraint. For lookup keywords (Place, Neighbourhood, Zone),
the name is resolved to column ranges before the Query object is assembled. Columns with no
constraint receive [col_min, col_max] (unconstrained).

```
Elevation:            [3400, 3858]   <- constrained by > 3400
Slope:                [20,     66]   <- constrained by > 20
Hillshade_9am:        [0,     200]   <- constrained by < 200
All other columns:    [min,   max]   <- unconstrained
```

### Step 2 — Grid Tree (GT) Traversal

The Tsunami index organises data in a recursive binary tree of axis-aligned partitions
(gt_max_depth=4, up to 16 leaf nodes). At each internal node the query bounding box is tested:

- No overlap: prune the entire subtree — zero rows scanned.
- Full containment: include all rows without further descent.
- Partial overlap: recurse into both children.

### Step 3 — AGD-Optimized Partition Boundaries

During index construction, Active Gradient Descent (AGD) shifts the GT split thresholds to
minimize total rows scanned across all training queries. For the HighAlpine training workload,
AGD concentrates several leaf partitions in the alpine band, making each leaf small and precise.
The dominant subalpine zone (~65% of data) is covered by only a few large coarse leaves.

### Step 4 — Leaf Scan and Aggregation

For each surviving leaf partition, rows are checked against exact query ranges and the
aggregation (COUNT / AVG / SUM) is accumulated. The result includes value, n_scanned,
and n_matched.

### Step 5 — Brute Force Baseline

idx.brute_force(q) scans all 2,000,000 rows sequentially. Used as baseline:

```
Speedup = Brute Force time / Tsunami time
Scan %  = n_scanned / 2,000,000 x 100
```

### Why Nearby Queries Outperform Far Queries

| Scenario | GT Behavior | Scan % | Speedup |
|---|---|---|---|
| Nearby (Krummholz — subset of training zone) | Query hits fine alpine partitions; few small leaves selected | Low (~5–10%) | High |
| Far (Lowland — opposite elevation extreme) | Query hits coarse boundary leaves covering wide elevation band | Higher (~15–30%) | Lower |

Both groups return a similar number of matching rows (~3–10% of data), so the speedup
difference is purely due to partition resolution — the core claim of workload-driven
index construction in the Tsunami paper.
