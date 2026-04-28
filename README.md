## 1. Implementation steps

**Clone the repo**
```bash
git clone https://github.com/manu5703/tsunami_ML.git
```

**Build the image**

```bash
docker build -t tsunami .
```

**Run a single interactive query**

California Housing:

```bash
docker run -it --rm tsunami python query_cli.py --dataset california_real
```

Once the index is built, type a query at the prompt:

```
SELECT COUNT(*) FROM data WHERE MedInc >= 7.0 AND HouseAge <= 20
```

NYC Taxi:

```bash
docker run -it --rm tsunami python query_cli.py --dataset nyc_taxi_real
```

Example query:

```
SELECT COUNT(*) FROM data WHERE fare_amount > 40 AND trip_distance > 10
```

Covertype:

```bash
docker run -it --rm tsunami python query_cli.py --dataset covtype
```

Example query:

```
SELECT COUNT(*) FROM data WHERE Elevation > 3400 AND Slope > 20
```


**CLI Commands (inside the prompt)**

Once the index is built and the `tsunami>` prompt appears, the following commands are available:

| Command | Description |
|---|---|
| `help` | Show available commands and query syntax |
| `columns` | List all columns with their min and max values |
| `tables` | List all loaded tables and their dimensions |
| `load <file.csv>` | Load a new CSV/Parquet file and rebuild the index |
| `workload <file.txt>` | Reload the index using a new training query file |
| `place` | Show the California place lookup table |
| `neighbourhood` | Show the NYC neighbourhood lookup table |
| `zone` | Show the Covertype elevation/slope zone table |
| `quit` / `exit` | Exit the CLI |

**Supported query syntax:**

```
SELECT COUNT(*)   FROM data WHERE <filters>
SELECT AVG(col)   FROM data WHERE <filters>
SELECT SUM(col)   FROM data WHERE <filters>
SELECT MIN(col)   FROM data WHERE <filters>
SELECT MAX(col)   FROM data WHERE <filters>
SELECT *          FROM data [WHERE <filters>] [ORDER BY col [ASC|DESC]] [LIMIT N]
```

**Filter operators:**

```
col BETWEEN lo AND hi
col >= val    col <= val    col > val    col < val
```

---

## 2. Overview

Tsunami is a learned multi-dimensional index for range-aggregate queries over correlated, skewed data. It partitions the data space using a Grid Tree, fits a learned model over each region, and at query time scans only the cells that intersect the query range.

---

## 3. Core Components

**grid_tree.py** — Partitions d-dimensional data space into non-overlapping leaf regions. Uses Earth Mover's Distance (EMD) as the skew metric to decide where to split.

**augmented_grid.py** — Fits a functional mapping over each Grid Tree leaf region. Pre-computes per-cell aggregates (count, sum, avg) to avoid rescanning raw rows for interior cells.

**cost_model_agd.py** — Gradient-descent optimizer that tunes grid resolution per region by minimizing estimated scan cost over the training workload.

**tsunami_index.py** — Orchestrates the full pipeline: data ingestion, Grid Tree build, Augmented Grid fitting, AGD optimization, and single-query execution via `TsunamiIndex.query()`.

**query_cli.py** — Loads a dataset (CSV, Parquet, or Excel), parses SQL-like query strings, and runs them through the index interactively.

---

## 4. Step-by-Step: Single Query Execution

**Step 1 — Load and preprocess data**
The dataset is loaded using `query_cli.py`. Datetime columns are decomposed into hour, day-of-week, and month features. All columns are cast to float64, constant or near-constant columns are dropped, and the result is stored as a NumPy array of shape (N, d).

**Step 2 — Build the Grid Tree**
`TsunamiIndex.build()` passes the data and a training query workload to `grid_tree.py`. The Grid Tree recursively splits each region along the dimension whose split most reduces EMD-based query skew, producing a set of non-overlapping leaf regions, each holding the row indices of the data points it contains.

**Step 3 — Fit Augmented Grids per region**
For each leaf region, `augmented_grid.py` fits a learned functional mapping over the data sorted along the chosen sort dimension. A lookup table of pre-aggregated cell statistics is built so that cells fully inside a query range contribute their aggregate directly without scanning individual rows.

**Step 4 — Optimize grid resolution (AGD)**
`cost_model_agd.py` runs up to 20 gradient-descent iterations over the grid resolution parameters. It samples 30% of the training queries, estimates query cost using a weighted scan model, and adjusts the number of partitions per dimension to minimize average scan fraction.

**Step 5 — Parse and execute one query**
The user submits a SQL-like string such as:

```
SELECT COUNT(*) FROM data WHERE MedInc >= 5.0 AND MedHouseVal <= 4.0
```

`query_cli.py` parses this into a `Query` object — a list of (lo, hi) range pairs, one per dimension, with (-inf, +inf) for unconstrained dimensions.

`TsunamiIndex.query()` then:
1. Iterates over all leaf regions and skips those whose bounding box does not intersect the query range.
2. For each intersecting region, identifies which Augmented Grid cells fall fully inside, partially inside, or outside the query range.
3. Fully inside cells contribute their pre-computed aggregate directly.
4. Boundary cells (partially overlapping) are scanned row-by-row against the query predicate.
5. Partial aggregates from all regions are merged into a single result.

The final answer and query latency in milliseconds are returned as a `QueryResult`.

---
