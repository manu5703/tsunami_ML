# Tsunami Index — Testing Instructions

Step-by-step instructions to install, run, and test all features of the Tsunami Index
implementation across three real datasets.

---

## 1. Prerequisites

### Python version
Python 3.9 or higher is required.

### Install dependencies
```
pip install numpy pandas scikit-learn matplotlib pyarrow
```

### Folder structure
```
tsunami/
├── query_cli.py              # Interactive SQL CLI
├── batch_test.py             # Batch benchmark runner
├── tsunami_index.py          # Core Tsunami index
├── augmented_grid.py
├── grid_tree.py
├── cost_model_agd.py
├── places.py                 # California city lookup table
├── nyc_places.py             # NYC neighbourhood lookup table
├── covtype_zones.py          # Covertype ecological zone lookup table
├── queries/
│   ├── california/
│   │   ├── training_queries.txt
│   │   ├── nearby_queries.txt
│   │   └── far_queries.txt
│   ├── nyc_taxi/
│   │   ├── nyc_training_queries.txt
│   │   ├── nyc_nearby_queries.txt
│   │   └── nyc_far_queries.txt
│   └── covertype/
│       ├── covtype_training_queries.txt
│       ├── covtype_nearby_queries.txt
│       └── covtype_far_queries.txt
├── results/                  # CSVs and plots saved here
└── docs/                     # This file and dataset_report.md
```

All commands below must be run from the `tsunami/` root directory.

---

## 2. Datasets

| Flag | Dataset | Source | Rows used |
|---|---|---|---|
| `--dataset california_real` | California Housing | sklearn (1990 US Census) | 2,000,000 |
| `--dataset nyc_taxi_real` | NYC Yellow Taxi 2016 | TLC (auto-downloaded) | 2,000,000 |
| `--dataset covtype` | Forest Covertype | sklearn (581K rows) | 2,000,000 |

---

## 3. Batch Benchmark

Builds the Tsunami index from a training workload, runs two test groups (nearby / far),
prints a comparison table, saves a CSV to `results/`, and generates speedup PNG plots.

### California Housing
```
python batch_test.py --dataset california_real -w queries/california/training_queries.txt --group-a queries/california/nearby_queries.txt --group-b queries/california/far_queries.txt --csv-out results/results_california_real.csv
```
- Training: Bay Area (Lat 37–38.5, Lon -122.5–-121)
- Nearby (A): Silicon Valley — Sunnyvale, Mountain View, Palo Alto, Cupertino
- Far (B): Southern California — Los Angeles, Irvine, Chula Vista

### NYC Yellow Taxi
```
python batch_test.py --dataset nyc_taxi_real -w queries/nyc_taxi/nyc_training_queries.txt --group-a queries/nyc_taxi/nyc_nearby_queries.txt --group-b queries/nyc_taxi/nyc_far_queries.txt --csv-out results/results_nyc_taxi.csv
```
- Training: Expanded premium trips — fare > $30, distance > 8 miles (~10-15% of data)
- Nearby (A): Tight premium / long-haul trips — fare > $40, distance > 10 miles (~5%, subset of training)
- Far (B): Budget / short trips — fare < $12, distance < 2 miles (~35-45%)

Note: On first run the TLC 2016 parquet downloads automatically. The 2016 parquet schema
no longer includes lat/lon columns (TLC switched to zone IDs), so queries use trip
attributes (fare_amount, trip_distance, total_amount, tip_amount) directly.

### Forest Covertype
```
python batch_test.py --dataset covtype -w queries/covertype/covtype_training_queries.txt --group-a queries/covertype/covtype_nearby_queries.txt --group-b queries/covertype/covtype_far_queries.txt --csv-out results/results_covtype.csv
```
- Training: HighAlpine zone — Elevation > 3200 m, Slope > 15 deg (~12% of data)
- Nearby (A): Krummholz zone — Elevation > 3400 m, Slope > 20 deg (~4%, tight subset of training)
- Far (B): Lowland zone — Elevation < 2400 m, Slope < 12 deg (~10% of data)

### Optional: reduce rows for faster testing
```
python batch_test.py --dataset covtype -w queries/covertype/covtype_training_queries.txt --group-a queries/covertype/covtype_nearby_queries.txt --group-b queries/covertype/covtype_far_queries.txt -n 500000
```

---

## 4. Interactive CLI

Launches a live SQL prompt. The Tsunami index is built on startup (with or without a workload).

### California Housing
```
python query_cli.py --dataset california_real -w queries/california/training_queries.txt
```

### NYC Yellow Taxi
```
python query_cli.py --dataset nyc_taxi_real -w queries/nyc_taxi/nyc_training_queries.txt
```

### Forest Covertype
```
python query_cli.py --dataset covtype -w queries/covertype/covtype_training_queries.txt
```

### Load a custom CSV or Parquet file
```
python query_cli.py mydata.csv
python query_cli.py mydata.parquet -w myworkload.txt
```

---

## 5. CLI Commands and Query Types

Once the `tsunami>` prompt appears:

### Navigation
```
help            show syntax reference
columns         list all columns with min/max
tables          show dataset info and available lookup tables
quit            exit
```

### Show lookup tables
```
places          California city bounding boxes      (california_real)
neighbourhoods  NYC neighbourhood bounding boxes    (nyc_taxi_real)
zones           Covertype ecological zones          (covtype)
```

### COUNT
```sql
SELECT COUNT(*) FROM data WHERE MedInc >= 7.0
SELECT COUNT(*) FROM data WHERE Elevation > 3400 AND Slope > 20
SELECT COUNT(*) FROM data WHERE fare_amount BETWEEN 10.0 AND 30.0
SELECT COUNT(*) FROM data WHERE place = 'Sunnyvale'
SELECT COUNT(*) FROM data WHERE place = 'Palo Alto' AND MedInc >= 9.0
SELECT COUNT(*) FROM data WHERE neighbourhood = 'Midtown'
SELECT COUNT(*) FROM data WHERE neighbourhood = 'Chelsea' AND fare_amount >= 15.0
SELECT COUNT(*) FROM data WHERE zone = 'Krummholz'
SELECT COUNT(*) FROM data WHERE zone = 'HighAlpine' AND Horiz_Dist_Roadways > 1500
```

### AVG
```sql
SELECT AVG(MedHouseVal) FROM data WHERE place = 'Sunnyvale'
SELECT AVG(fare_amount) FROM data WHERE neighbourhood = 'Midtown' AND trip_distance >= 2.0
SELECT AVG(Elevation) FROM data WHERE zone = 'Krummholz'
SELECT AVG(Slope) FROM data WHERE Elevation > 3400 AND Horiz_Dist_Roadways > 2000
```

### SUM
```sql
SELECT SUM(Population) FROM data WHERE place = 'Sunnyvale'
SELECT SUM(total_amount) FROM data WHERE neighbourhood = 'Midtown' AND fare_amount >= 10.0
SELECT SUM(Horiz_Dist_Roadways) FROM data WHERE zone = 'HighAlpine'
```

### MIN / MAX
```sql
SELECT MIN(MedHouseVal) FROM data WHERE place = 'Los Angeles'
SELECT MAX(fare_amount) FROM data WHERE neighbourhood = 'JFK Airport'
SELECT MAX(Slope) FROM data WHERE Elevation > 3200
```

### Row SELECT with ORDER BY and LIMIT
```sql
SELECT * FROM data WHERE place = 'Palo Alto' LIMIT 10
SELECT * FROM data WHERE neighbourhood = 'Midtown' AND fare_amount >= 20.0 LIMIT 20
SELECT * FROM data WHERE zone = 'Krummholz' LIMIT 5
SELECT fare_amount, tip_amount, trip_distance FROM data WHERE neighbourhood = 'Chelsea' ORDER BY fare_amount DESC LIMIT 10
SELECT Elevation, Slope, Hillshade_9am FROM data WHERE zone = 'HighAlpine' ORDER BY Elevation DESC LIMIT 10
```

### Rebuild index with a different workload (without restarting)
```
workload queries/california/training_queries.txt
workload queries/nyc_taxi/nyc_training_queries.txt
workload queries/covertype/covtype_training_queries.txt
```

### Load a new file at runtime
```
load myfile.csv
load myfile.parquet
```

---

## 6. Feature Summary

| Feature | How to use |
|---|---|
| COUNT / AVG / SUM / MIN / MAX | `SELECT AGG(col) FROM data WHERE ...` |
| Row-level results | `SELECT * FROM data WHERE ... LIMIT N` |
| Sort results | `ORDER BY col DESC LIMIT N` |
| Range filter | `col BETWEEN lo AND hi` |
| Comparison filter | `col >= val`, `col < val` |
| Place lookup (California) | `place = 'CityName'` |
| Neighbourhood lookup (NYC) | `neighbourhood = 'Name'` |
| Zone lookup (Covertype) | `zone = 'ZoneName'` |
| View lookup table | `places` / `neighbourhoods` / `zones` |
| View columns | `columns` |
| Rebuild index live | `workload <path/to/file.txt>` |
| Load new dataset live | `load <file>` |
| Batch benchmark | `python batch_test.py ...` |
| Save results CSV | `--csv-out results/myfile.csv` |
| Speedup plots | Auto-generated as PNG alongside CSV |

---

## 7. Lookup Table Reference

### California Places  (type `places` to see all)
Bay Area, SF Metro, Silicon Valley, Palo Alto, Sunnyvale, Mountain View, Cupertino,
San Francisco, Oakland, Berkeley, Sacramento, San Jose, Los Angeles, LA Metro,
Santa Monica, Beverly Hills, Irvine, Chula Vista, San Diego, Orange County,
Santa Barbara, Monterey, Bakersfield, Fresno, Stockton, Modesto, Riverside,
Inland Empire, Central Valley, Northern CA, Southern CA, Coastal SoCal, Wine Country

### NYC Neighbourhoods  (type `neighbourhoods` to see all)
Manhattan, Brooklyn, Queens, Bronx, Staten Island — and within Manhattan:
Midtown, Chelsea, Gramercy, Flatiron, Murray Hill, Hell's Kitchen, West Village,
SoHo, Tribeca, East Village, Upper East Side, Upper West Side, Harlem,
Lower Manhattan, Financial District, Chinatown
— and outer areas: Williamsburg, Park Slope, Flushing, Jamaica, JFK Airport,
Coney Island, Flatbush, Fordham, Riverdale, Astoria, and more.

### Covertype Zones  (type `zones` to see all)
| Zone | Elevation | Slope | % of data | Role |
|---|---|---|---|---|
| Krummholz | 3400–3858 m | 20–90 deg | ~4% | Nearby / Group A |
| HighAlpine | 3200–3858 m | 15–90 deg | ~12% | Training zone |
| Subalpine | 2800–3200 m | 8–22 deg | ~65% | Background |
| Montane | 2400–2800 m | 5–18 deg | ~15% | Background |
| Lowland | 1863–2400 m | 0–12 deg | ~10% | Far / Group B |
| Riparian | 1863–2200 m | 0–8 deg | ~4% | Far subset |
