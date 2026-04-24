# Tsunami Index — Testing Instructions

This document provides step-by-step instructions to install, run, and test all features
of the Tsunami Index implementation across three real datasets.

---

## 1. Prerequisites

### Python version
Python 3.9 or higher is required.

### Install dependencies
```
pip install numpy pandas scikit-learn matplotlib pyarrow
```

All code files must be in the same directory. The working directory must contain:
- `query_cli.py`
- `batch_test.py`
- `tsunami_index.py`  (the Tsunami index implementation)
- `places.py`         (California city bounding boxes)
- `nyc_places.py`     (NYC neighbourhood bounding boxes)
- `covtype_zones.py`  (Forest Covertype ecological zones)
- All `*_queries.txt` files

---

## 2. Datasets

Three real datasets are supported. Data is downloaded or fetched automatically on first run.

| Flag | Dataset | Source | Rows |
|---|---|---|---|
| `--dataset california_real` | California Housing | sklearn (1990 US Census) | 20,640 real → 2M oversampled |
| `--dataset nyc_taxi_real` | NYC Yellow Taxi | TLC 2016 (auto-downloaded) | ~10M real → 2M subsample |
| `--dataset covtype` | Forest Covertype | sklearn (581K real rows) | 581,012 real → 2M oversampled |

---

## 3. Batch Benchmark (recommended first step)

`batch_test.py` builds the Tsunami index from a training workload, runs two test groups
(nearby / in-distribution and far / out-of-distribution), prints a side-by-side comparison,
saves a CSV of all results, and generates speedup plots.

### 3a. California Housing

```
python batch_test.py --dataset california_real -w training_queries.txt --group-a nearby_queries.txt --group-b far_queries.txt
```

- Training zone: Bay Area (Lat 37–38.5, Lon -122.5–-121)
- Nearby (Group A): Silicon Valley cities — Sunnyvale, Mountain View, Palo Alto, Cupertino
- Far (Group B): Southern California — Los Angeles, Irvine, Chula Vista

### 3b. NYC Yellow Taxi

```
python batch_test.py --dataset nyc_taxi_real -w nyc_training_queries.txt --group-a nyc_nearby_queries.txt --group-b nyc_far_queries.txt
```

- Training zone: Central Manhattan — Midtown, Chelsea, Gramercy
- Nearby (Group A): Adjacent Manhattan — Flatiron, Murray Hill, Hell's Kitchen, West Village, SoHo
- Far (Group B): Outer Boroughs — Flushing, Jamaica, JFK Airport, Coney Island, Fordham, Riverdale, Flatbush

Note: On first run the TLC 2016 parquet file (~500 MB) is downloaded automatically.
If the download fails, place any 2015–2016 TLC yellow taxi parquet file in the
working directory; it will be detected automatically.

### 3c. Forest Covertype

```
python batch_test.py --dataset covtype -w covtype_training_queries.txt --group-a covtype_nearby_queries.txt --group-b covtype_far_queries.txt
```

- Training zone: HighAlpine — Elevation > 3200 m, Slope > 15 deg (~12% of data)
- Nearby (Group A): Krummholz — Elevation > 3400 m, Slope > 20 deg (~4% of data, tight subset of training zone)
- Far (Group B): Lowland — Elevation < 2400 m, Slope < 12 deg (~10% of data)

### Batch output

Each run prints:
1. Per-query table: Tsunami time, NumPy time, Brute Force time, speedup, scan%, answer correctness
2. Group summary (average and median across all queries)
3. Overall comparison table: Group A vs Group B side by side
4. CSV saved as `results_<dataset>.csv`
5. Speedup plots saved as PNG files

### Save results to a custom CSV

```
python batch_test.py --dataset california_real -w training_queries.txt --group-a nearby_queries.txt --group-b far_queries.txt --csv-out my_results.csv
```

### Reduce row count for faster testing (optional)

```
python batch_test.py --dataset covtype -w covtype_training_queries.txt --group-a covtype_nearby_queries.txt --group-b covtype_far_queries.txt -n 500000
```

---

## 4. Interactive Query CLI

`query_cli.py` launches an interactive SQL prompt against any dataset. The Tsunami index
can be built with or without a training workload.

### 4a. Launch with California Housing (no training workload)

```
python query_cli.py --dataset california_real
```

### 4b. Launch with training workload (index optimized for Bay Area queries)

```
python query_cli.py --dataset california_real -w training_queries.txt
```

### 4c. Launch with NYC Taxi

```
python query_cli.py --dataset nyc_taxi_real -w nyc_training_queries.txt
```

### 4d. Launch with Forest Covertype

```
python query_cli.py --dataset covtype -w covtype_training_queries.txt
```

### 4e. Launch with a custom CSV or Parquet file

```
python query_cli.py mydata.csv
python query_cli.py mydata.parquet -w myworkload.txt
```

---

## 5. Interactive CLI — Commands and Features

Once the `tsunami>` prompt appears, the following commands and query types are available.

### 5a. Help and navigation

```
help          -- show available commands and SQL syntax
columns       -- list all columns with min/max values
tables        -- show dataset info and all available lookup tables
quit          -- exit
```

### 5b. Show lookup tables

```
places            -- show all California city bounding boxes (california_real dataset)
neighbourhoods    -- show all NYC neighbourhood bounding boxes (nyc_taxi_real dataset)
zones             -- show all Forest Covertype ecological zones (covtype dataset)
```

### 5c. COUNT queries

**Single condition:**
```sql
SELECT COUNT(*) FROM data WHERE MedInc >= 7.0
SELECT COUNT(*) FROM data WHERE Elevation > 3400
SELECT COUNT(*) FROM data WHERE fare_amount BETWEEN 10.0 AND 30.0
```

**Multiple conditions (AND):**
```sql
SELECT COUNT(*) FROM data WHERE MedInc >= 7.0 AND MedHouseVal >= 3.0
SELECT COUNT(*) FROM data WHERE Elevation > 3400 AND Slope > 20
SELECT COUNT(*) FROM data WHERE Latitude BETWEEN 37.3 AND 37.5 AND Longitude BETWEEN -122.1 AND -121.9
```

**Using Place lookup (California Housing):**
```sql
SELECT COUNT(*) FROM data WHERE place = 'Sunnyvale'
SELECT COUNT(*) FROM data WHERE place = 'Palo Alto' AND MedInc >= 9.0
SELECT COUNT(*) FROM data WHERE place = 'Los Angeles' AND MedHouseVal >= 4.0
SELECT COUNT(*) FROM data WHERE place = 'Bay Area' AND MedInc >= 5.0 AND HouseAge <= 20
```

**Using Neighbourhood lookup (NYC Taxi):**
```sql
SELECT COUNT(*) FROM data WHERE neighbourhood = 'Midtown'
SELECT COUNT(*) FROM data WHERE neighbourhood = 'Chelsea' AND fare_amount >= 15.0
SELECT COUNT(*) FROM data WHERE neighbourhood = 'JFK Airport' AND trip_distance >= 5.0
SELECT COUNT(*) FROM data WHERE neighbourhood = 'Flushing' AND fare_amount >= 10.0 AND tip_amount >= 2.0
```

**Using Zone lookup (Forest Covertype):**
```sql
SELECT COUNT(*) FROM data WHERE zone = 'Krummholz'
SELECT COUNT(*) FROM data WHERE zone = 'HighAlpine' AND Horiz_Dist_Roadways > 1500
SELECT COUNT(*) FROM data WHERE zone = 'Lowland' AND Horiz_Dist_Hydrology < 200
SELECT COUNT(*) FROM data WHERE zone = 'Subalpine'
```

### 5d. AVG queries

```sql
SELECT AVG(MedHouseVal) FROM data WHERE place = 'Sunnyvale'
SELECT AVG(fare_amount) FROM data WHERE neighbourhood = 'Midtown' AND trip_distance >= 2.0
SELECT AVG(Elevation) FROM data WHERE zone = 'Krummholz'
SELECT AVG(Slope) FROM data WHERE Elevation > 3400 AND Horiz_Dist_Roadways > 2000
SELECT AVG(MedInc) FROM data WHERE Latitude BETWEEN 37.3 AND 37.5
```

### 5e. SUM queries

```sql
SELECT SUM(Population) FROM data WHERE place = 'Sunnyvale'
SELECT SUM(total_amount) FROM data WHERE neighbourhood = 'Midtown' AND fare_amount >= 10.0
SELECT SUM(Horiz_Dist_Roadways) FROM data WHERE zone = 'HighAlpine'
SELECT SUM(Horiz_Dist_Hydrology) FROM data WHERE Elevation > 3400 AND Slope > 20
```

### 5f. MIN and MAX queries

```sql
SELECT MIN(MedHouseVal) FROM data WHERE place = 'Los Angeles'
SELECT MAX(fare_amount) FROM data WHERE neighbourhood = 'JFK Airport'
SELECT MIN(Elevation) FROM data WHERE zone = 'Krummholz'
SELECT MAX(Slope) FROM data WHERE Elevation > 3200
```

### 5g. Row-level SELECT (returns actual rows)

```sql
SELECT * FROM data WHERE place = 'Palo Alto' LIMIT 10
SELECT * FROM data WHERE neighbourhood = 'Midtown' AND fare_amount >= 20.0 LIMIT 20
SELECT * FROM data WHERE zone = 'Krummholz' LIMIT 5
SELECT MedInc, MedHouseVal, Latitude, Longitude FROM data WHERE place = 'Sunnyvale' LIMIT 15
SELECT fare_amount, tip_amount, trip_distance FROM data WHERE neighbourhood = 'Chelsea' ORDER BY fare_amount DESC LIMIT 10
SELECT Elevation, Slope, Hillshade_9am FROM data WHERE zone = 'HighAlpine' ORDER BY Elevation DESC LIMIT 10
```

### 5h. Rebuild index with a new workload (without restarting)

```
workload covtype_training_queries.txt
workload nyc_training_queries.txt
workload training_queries.txt
```

### 5i. Load a new CSV or Parquet file at runtime

```
load myfile.csv
load myfile.parquet
```

---

## 6. Feature Summary Table

| Feature | CLI command / query type | Datasets |
|---|---|---|
| COUNT aggregation | `SELECT COUNT(*) FROM data WHERE ...` | All |
| AVG aggregation | `SELECT AVG(col) FROM data WHERE ...` | All |
| SUM aggregation | `SELECT SUM(col) FROM data WHERE ...` | All |
| MIN / MAX aggregation | `SELECT MIN/MAX(col) FROM data WHERE ...` | All |
| Row-level SELECT | `SELECT * FROM data WHERE ... LIMIT N` | All |
| ORDER BY | `SELECT ... ORDER BY col DESC LIMIT N` | All |
| BETWEEN condition | `col BETWEEN lo AND hi` | All |
| Greater/less than | `col >= val`, `col < val` | All |
| Place lookup | `place = 'CityName'` | california_real |
| Neighbourhood lookup | `neighbourhood = 'Name'` | nyc_taxi_real |
| Zone lookup | `zone = 'ZoneName'` | covtype |
| Show lookup table | `places` / `neighbourhoods` / `zones` | Per dataset |
| Show columns | `columns` | All |
| Rebuild index | `workload <file.txt>` | All |
| Load new file | `load <file>` | All |
| Batch benchmark | `python batch_test.py ...` | All |
| Results CSV export | `--csv-out results.csv` | All |
| Speedup plots (PNG) | auto-generated by batch_test.py | All |

---

## 7. Expected Output Structure (Batch Test)

```
Loading Forest Covertype (2,000,000 rows)...
  Fetching Forest Covertype dataset from sklearn...
  Loaded real Forest Covertype (581,012 rows).
  Oversampled to 2,000,000 rows, 152 MB.
Loading training queries from 'covtype_training_queries.txt'...
  Loaded 22 training queries.
Building Tsunami index...
  Tsunami index built in XXXX ms
  KDTree skipped (2,000,000 rows > 500,000 limit).
  Ready.

===[ GROUP A -- NEARBY [IN-DISTRIBUTION] ]===
  #   Query                       Tsunami    Ans-TS    NumPy    Ans-NP   BruteF    Ans-BF   Speedup  Scan%  Match
  1   SELECT COUNT(*) WHERE ...   12.34ms    123456   45.67ms   123456  89.01ms   123456     7.2x   4.2%   PASS
  ...

===[ GROUP B -- FAR [OUT-OF-DISTRIBUTION] ]===
  ...

==[ OVERALL COMPARISON ]==
  Metric                           Group A (Nearby)  Group B (Far)
  Avg Speedup  BF / Tsunami               8.5x           2.1x
  Avg Scan %                              5.2%          18.7%
```

---

## 8. Lookup Table Reference

### California Places (full list via `places` command)
Bay Area, SF Metro, Silicon Valley, Palo Alto, Sunnyvale, Mountain View, Cupertino,
San Francisco, Oakland, Berkeley, Sacramento, San Jose, Stockton, Fresno,
Los Angeles, LA Metro, Santa Monica, Beverly Hills, Long Beach, Anaheim, Irvine,
Chula Vista, San Diego, Riverside, Inland Empire, Orange County, Santa Barbara,
Monterey, Bakersfield, Oxnard, Modesto, Santa Rosa, Wine Country, Gold Country,
Coastal SoCal, Coastal NorCal, Central Valley, Northern CA, Southern CA, Malibu

### NYC Neighbourhoods (full list via `neighbourhoods` command)
**Boroughs:** Manhattan, Brooklyn, Queens, Bronx, Staten Island

**Manhattan:** Midtown, Chelsea, Gramercy, Flatiron, Murray Hill, Hell's Kitchen,
West Village, SoHo, Tribeca, East Village, Upper East Side, Upper West Side,
Harlem, Lower Manhattan, Financial District, Chinatown, Morningside Heights,
Washington Heights, Inwood

**Brooklyn:** Williamsburg, Brooklyn Heights, Park Slope, Bushwick, Bed-Stuy,
Crown Heights, Sunset Park, Bay Ridge, Coney Island, DUMBO, Flatbush, Greenpoint

**Queens:** Astoria, Long Island City, Flushing, Jackson Heights, Jamaica, JFK Airport

**Bronx:** South Bronx, Fordham, Riverdale

### Covertype Zones (full list via `zones` command)
Krummholz, HighAlpine, Subalpine, Montane, Lowland, Riparian
