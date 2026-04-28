# Tsunami Index — Step-by-Step Implementation

---

## 1. Overview

Tsunami is a learned multi-dimensional index designed for correlated data and skewed workloads. It partitions the data space using a Grid Tree, augments each region with a learned model, and answers range-aggregate queries by scanning only the relevant cells.

---

## 2. Core Components

**grid_tree.py** — Builds the Grid Tree that partitions d-dimensional space into non-overlapping regions. Uses Earth Mover's Distance (EMD) to measure and reduce query skew across regions.

**augmented_grid.py** — Fits a learned model (functional mapping) on top of each grid region. Stores per-cell aggregates (count, sum) to answer queries without scanning raw data where possible.

**cost_model_agd.py** — Automatic Gradient Descent optimizer that tunes the index configuration (grid resolution, split thresholds) by minimizing estimated query cost.

**tsunami_index.py** — Top-level index. Orchestrates data loading, Grid Tree construction, Augmented Grid fitting, and query execution. Exposes `TsunamiIndex` and `Query`.

**query_cli.py** — Command-line interface. Loads datasets (CSV, Parquet, Excel), parses SQL-like queries, and runs them through the index.

---

## 3. Step-by-Step Implementation

**Step 1 — Load and normalize data**
Raw data is loaded via `query_cli.py`. Datetime columns are decomposed into hour, day-of-week, and month features. All columns are cast to float64 and stored as a NumPy array.

**Step 2 — Build the Grid Tree**
`grid_tree.py` receives the data array and a training query workload. It recursively splits each region along the dimension that most reduces EMD-based query skew, producing a tree of non-overlapping axis-aligned regions.

**Step 3 — Fit the Augmented Grid**
`augmented_grid.py` takes each leaf region from the Grid Tree and fits a functional mapping over the sorted data within that region. Pre-aggregated statistics (count, sum, avg) are stored per cell so that qualifying cells can be resolved without re-scanning rows.

**Step 4 — Optimize with AGD**
`cost_model_agd.py` runs a gradient-descent loop over the configuration space (number of grid lines, resolution per dimension). It estimates query cost using a weighted model and updates parameters to minimize average scan fraction.

**Step 5 — Answer a query**
At query time, `TsunamiIndex.query()` maps the incoming range predicate to the Grid Tree regions it intersects, looks up pre-aggregated cell values, and scans only the boundary cells that partially overlap the query range. The result is returned in milliseconds.

**Step 6 — Benchmark**
`batch_test.py` runs two query groups — nearby (selective) and far (broad) — against Tsunami, NumPy, Z-Order, and Brute Force, recording per-query latency and scan percentage. Results are written to a CSV file and plotted as speedup charts.

---

## 4. Running with Docker

**Build the image**

```bash
docker build -t tsunami .
```

**Run the interactive query CLI**

```bash
docker run -it --rm tsunami python query_cli.py --dataset california_real
```

**Run the batch benchmark and export results**

```bash
docker run -it --rm -v "%cd%/results:/app/results" tsunami \
  python batch_test.py --dataset california_real \
  -w queries/california/training_queries.txt \
  --group-a queries/california/nearby_queries.txt \
  --group-b queries/california/far_queries.txt \
  --csv-out results/results_california_real.csv
```

Replace `%cd%` with `$(pwd)` on Linux/Mac. The `results/` folder is mounted so output files are accessible on the host after the container exits.

---

## 5. Dependencies

| Package | Purpose |
|---|---|
| numpy | Data storage and vectorized filtering |
| pandas | CSV / Parquet / Excel loading |
| scikit-learn | KDTree baseline and DBSCAN clustering |
| matplotlib | Benchmark plots |
| scipy | Statistical utilities |
| pyarrow | Parquet file support |

All dependencies are installed automatically inside the Docker image via `pip`.

---

*Please convert this document to a .docx file with plain formatting — no colored boxes or highlights. Use standard headings and a simple table style.*
