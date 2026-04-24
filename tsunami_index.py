"""
Tsunami — Complete Integrated Index
======================================
Paper: "Tsunami: A Learned Multi-dimensional Index for Correlated Data
        and Skewed Workloads", Ding et al., PVLDB 2021

This module wires together every component built across the series:

  grid_tree.py          — Grid Tree (§4):  EMD-based skew reduction,
                          DBSCAN query-type clustering, skew-tree DP
  augmented_grid.py     — Augmented Grid (§5): FM, CCDF, independent CDF
  cost_model_agd.py     — Cost model (§5.3.1) + AGD optimizer (§5.3.2)
  physical_storage.py   — Column store + lookup table (§2.2, §6.1)
  sort_dim_optimizer.py — Sort dimension selection (§2.2 footnote)
  workload_shift.py     — Workload shift detection (§8)
  delta_index.py        — Delta index for inserts/updates/deletes (§8)

Public API
----------
  TsunamiConfig         — All tuning knobs in one dataclass.
  TsunamiIndex          — Main class.

    .build(data, queries)          — Full offline optimisation + build
    .query(q)  → QueryResult       — Execute one analytical query
    .batch(qs) → BatchResult       — Execute many queries + stats
    .insert(row)   → RowId         — Buffer an insert
    .update(rid, row) → RowId      — Buffer an update
    .delete(rid)                   — Buffer a delete
    .observe(q)                    — Feed query to workload monitor
    .rebuild_if_shifted()          — Re-optimise if monitor fired
    .stats()   → dict              — Runtime statistics
    .brute_force(data, q)          — Correctness reference

Dependencies:
    All modules above must be in the same directory.
    pip install numpy scipy scikit-learn
"""

from __future__ import annotations

import sys
import math
import time
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable

# ── Import every component ────────────────────────────────────────────────────
# Each import is wrapped so the file degrades gracefully if a module is absent.

try:
    from grid_tree import (
        GridTree, GridTreeNode, Query as _GTQuery,
        cluster_query_types, compute_skew,
    )
    _HAS_GT = True
except ImportError:
    _HAS_GT = False
    print("[Tsunami] WARNING: grid_tree.py not found — using fallback partitioner")

try:
    from augmented_grid import (
        AugmentedGrid, StrategyKind, DimStrategy, Skeleton,
        initialise_skeleton, fit_functional_mapping,
    )
    _HAS_AG = True
except ImportError:
    _HAS_AG = False
    print("[Tsunami] WARNING: augmented_grid.py not found — skipping Augmented Grid")

try:
    from cost_model_agd import (
        AGDOptimizer, AGDConfig, CostModelWeights,
        optimise as agd_optimise,
    )
    _HAS_AGD = True
except ImportError:
    _HAS_AGD = False
    print("[Tsunami] WARNING: cost_model_agd.py not found — skipping AGD")

try:
    from physical_storage import (
        PhysicalStorage, GridSpec,
        assign_cell_keys_bulk, ColumnStore, LookupTable,
    )
    _HAS_PS = True
except ImportError:
    _HAS_PS = False
    print("[Tsunami] WARNING: physical_storage.py not found")

try:
    from sort_dim_optimizer import SortDimOptimizer, apply_sort_dim
    _HAS_SD = True
except ImportError:
    _HAS_SD = False

try:
    from workload_shift import (
        WorkloadMonitor, RegionBounds, ShiftEvent,
        Query as _WSQuery,
    )
    _HAS_WS = True
except ImportError:
    _HAS_WS = False
    print("[Tsunami] WARNING: workload_shift.py not found — no shift detection")

try:
    from delta_index import (
        DeltaIndexNode, MergePolicy, MergeResult,
        Query as _DIQuery,
    )
    _HAS_DI = True
except ImportError:
    _HAS_DI = False
    print("[Tsunami] WARNING: delta_index.py not found — read-only mode")


# ─────────────────────────────────────────────────────────────────────────────
# Unified query type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Query:
    """
    A d-dimensional range query.
    ranges[i] = (lo, hi);  use (-inf, +inf) for unfiltered dimensions.
    """
    ranges:  list[tuple[float, float]]
    agg_fn:  str           = "count"   # count|sum|min|max|avg
    agg_col: Optional[int] = None      # column index for sum/min/max/avg
    label:   str           = ""

    @property
    def ndim(self) -> int:
        return len(self.ranges)

    def matches(self, row: np.ndarray) -> bool:
        for dim, (lo, hi) in enumerate(self.ranges):
            if math.isinf(lo) and math.isinf(hi):
                continue
            if row[dim] < lo or row[dim] > hi:
                return False
        return True

    def _as_gt_query(self):
        """Convert to grid_tree.Query (same structure, different class)."""
        if _HAS_GT:
            return _GTQuery(ranges=self.ranges)
        return self

    def _as_ws_query(self):
        """Convert to workload_shift.Query."""
        if _HAS_WS:
            return _WSQuery(ranges=self.ranges)
        return self

    def _as_di_query(self):
        """Convert to delta_index.Query."""
        if _HAS_DI:
            return _DIQuery(ranges=self.ranges)
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    query:         Query
    value:         float          # final aggregated value
    n_matched:     int            # rows satisfying all predicates
    n_scanned:     int            # total rows read
    n_regions:     int            # Grid Tree leaf regions visited
    n_cells:       int            # grid cells identified
    n_ranges:      int            # non-contiguous storage ranges
    delta_rows:    int            # rows from delta buffer
    t_total_ms:    float          # wall-clock time (ms)
    from_cache:    bool = False

    @property
    def scan_efficiency(self) -> float:
        return self.n_matched / max(1, self.n_scanned)

    def __repr__(self) -> str:
        return (f"QueryResult(value={self.value:.4g}, matched={self.n_matched}, "
                f"scanned={self.n_scanned}, t={self.t_total_ms:.2f}ms)")


@dataclass
class BatchResult:
    results:        list[QueryResult]
    n_queries:      int
    throughput_qps: float
    lat_avg_ms:     float
    lat_p50_ms:     float
    lat_p95_ms:     float
    lat_p99_ms:     float
    avg_scanned:    float

    def __repr__(self) -> str:
        return (f"BatchResult(n={self.n_queries}, "
                f"tput={self.throughput_qps:.0f}q/s, "
                f"avg={self.lat_avg_ms:.2f}ms, "
                f"p99={self.lat_p99_ms:.2f}ms)")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TsunamiConfig:
    """All tuning knobs for the integrated Tsunami index."""

    # ── Grid Tree (§4) ────────────────────────────────────────────────────
    gt_min_skew_reduction: float = 0.05
    gt_min_points_frac:    float = 0.01
    gt_min_queries_frac:   float = 0.01
    gt_n_bins:             int   = 128
    gt_max_depth:          int   = 6

    # ── Augmented Grid / AGD (§5) ─────────────────────────────────────────
    ag_default_parts:  int   = 8
    ag_fm_thresh:      float = 0.10    # functional mapping error bound
    ag_ccdf_thresh:    float = 0.25    # fraction empty cells for CCDF
    agd_max_iter:      int   = 20
    agd_part_step:     int   = 2
    agd_sample_frac:   float = 0.30
    agd_enabled:       bool  = True    # set False to skip AGD (use heuristic)

    # ── Sort dimension (§2.2) ─────────────────────────────────────────────
    sort_dim_enabled:  bool  = True

    # ── Delta index (§8) ─────────────────────────────────────────────────
    delta_enabled:         bool  = True
    delta_size_threshold:  int   = 5_000
    delta_time_threshold_s:float = 300.0
    delta_ratio_threshold: float = 0.10

    # ── Workload shift detection (§8) ─────────────────────────────────────
    shift_detection_enabled: bool  = True
    shift_kl_threshold:      float = 0.20
    shift_drift_ratio:       float = 2.0
    shift_cooldown_s:        float = 60.0
    shift_type_window:       int   = 200
    shift_freq_window:       int   = 100
    shift_skew_window:       int   = 100

    # ── General ───────────────────────────────────────────────────────────
    col_names:         Optional[list[str]] = None
    verbose:           bool  = False


# ─────────────────────────────────────────────────────────────────────────────
# Per-region index bundle
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _RegionBundle:
    """Everything owned by one Grid Tree leaf region."""
    region_id:   int
    lo:          np.ndarray
    hi:          np.ndarray
    row_indices: np.ndarray         # global row indices of points in region

    # Augmented Grid (set after AGD)
    ag:          Optional[object]  = None   # AugmentedGrid instance
    skeleton:    Optional[list]    = None
    n_parts:     Optional[list]    = None

    # Physical storage
    storage:     Optional[object]  = None   # PhysicalStorage instance
    sort_dim:    Optional[int]     = None

    # Delta index node
    delta_node:  Optional[object]  = None   # DeltaIndexNode instance

    # Column arrays (sorted order) for direct scan fallback
    cols:        Optional[list]    = None   # list of (N,) np.ndarray

    def intersects(self, ranges: list[tuple[float, float]]) -> bool:
        for dim in range(len(self.lo)):
            qlo, qhi = ranges[dim]
            if qhi <= self.lo[dim] or qlo >= self.hi[dim]:
                return False
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation helper
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate(vals: np.ndarray, fn: str) -> float:
    if fn == "sum":  return float(vals.sum())
    if fn == "min":  return float(vals.min())
    if fn == "max":  return float(vals.max())
    if fn == "avg":  return float(vals.mean())
    return float(len(vals))


def _merge_aggs(parts: list[tuple[int, float]], fn: str) -> float:
    """Merge (n, partial_agg) pairs from multiple regions."""
    total_n = sum(n for n, _ in parts)
    if total_n == 0:
        return 0.0
    if fn == "count": return float(total_n)
    if fn == "sum":   return sum(v for _, v in parts)
    if fn == "min":   return min((v for n, v in parts if n > 0), default=0.0)
    if fn == "max":   return max((v for n, v in parts if n > 0), default=0.0)
    if fn == "avg":
        s = sum(n * v for n, v in parts)
        return s / total_n
    return float(total_n)

# ─────────────────────────────────────────────────────────────────────────────
# TsunamiIndex
# ─────────────────────────────────────────────────────────────────────────────

class TsunamiIndex:
    """
    Complete integrated Tsunami index.

    Build
    -----
    idx = TsunamiIndex(TsunamiConfig(verbose=True))
    idx.build(data, queries)

    Query
    -----
    result = idx.query(Query([(40.0, 60.0), (30.0, 50.0)], agg_fn="count"))
    batch  = idx.batch([q1, q2, ...])

    Mutations (requires delta_enabled=True)
    ----------------------------------------
    rid = idx.insert(np.array([55.0, 42.0]))
    rid = idx.update(rid, np.array([56.0, 43.0]))
    idx.delete(rid)

    Workload monitoring
    -------------------
    idx.observe(incoming_query)        # feed to shift detector
    idx.rebuild_if_shifted()           # re-optimise if shift detected
    """

    def __init__(self, config: Optional[TsunamiConfig] = None):
        self.cfg     = config or TsunamiConfig()
        self._data:  Optional[np.ndarray]  = None
        self._d:     int                   = 0
        self._N:     int                   = 0
        self._col_names: list[str]         = []

        self._regions: list[_RegionBundle] = []
        self._gt:      Optional[object]    = None   # GridTree

        # Workload monitor
        self._monitor: Optional[object]   = None
        self._shift_pending: bool         = False
        self._last_queries: list[Query]   = []      # kept for re-optimisation

        # Build timing
        self._build_stats: dict           = {}
        self._lock = threading.Lock()

    # ─────────────────────────────────────────────────────────────────────────
    # Build
    # ─────────────────────────────────────────────────────────────────────────

    def build(self, data: np.ndarray, queries: list[Query]) -> None:
        """
        Full offline optimisation pipeline:

          1. Grid Tree construction  (§4)
          2. Per-region AGD          (§5.3)
          3. Sort dimension selection (§2.2)
          4. Physical storage build  (§2.2, §6.1)
          5. Delta index init        (§8)
          6. Workload monitor init   (§8)
        """
        t_total = time.perf_counter()
        cfg     = self.cfg

        N, d        = data.shape
        self._data  = data.copy()
        self._d     = d
        self._N     = N
        self._col_names = (cfg.col_names or [f"d{i}" for i in range(d)])
        self._last_queries = list(queries)

        # Convert queries to grid_tree.Query objects
        gt_queries = [q._as_gt_query() for q in queries]

        if cfg.verbose:
            print(f"[Tsunami] Building index: N={N:,}  d={d}  queries={len(queries)}")

        # ── Step 1: Grid Tree ─────────────────────────────────────────────
        t0 = time.perf_counter()
        regions = self._build_grid_tree(data, gt_queries)
        self._build_stats["gt_ms"] = (time.perf_counter() - t0) * 1000

        if cfg.verbose:
            print(f"  [GT]  {len(regions)} leaf regions  "
                  f"({self._build_stats['gt_ms']:.0f}ms)")

        # ── Steps 2–5: Per-region optimisation ───────────────────────────
        t0 = time.perf_counter()
        self._regions = []
        for rb in regions:
            region_data    = data[rb.row_indices]
            region_queries = [q for q in queries
                              if rb.intersects(q.ranges)]
            if len(region_data) == 0:
                continue
            self._build_region(rb, region_data, region_queries)
            self._regions.append(rb)

        self._build_stats["region_ms"] = (time.perf_counter() - t0) * 1000

        if cfg.verbose:
            total_cells = sum(
                len(rb.ag.lookup) if rb.ag is not None and hasattr(rb.ag, "lookup") else 0
                for rb in self._regions)
            print(f"  [AG]  total grid cells: {total_cells:,}  "
                  f"({self._build_stats['region_ms']:.0f}ms)")

        # ── Step 6: Workload monitor ──────────────────────────────────────
        if cfg.shift_detection_enabled and _HAS_WS:
            t0 = time.perf_counter()
            self._build_monitor(data, queries)
            self._build_stats["monitor_ms"] = (time.perf_counter() - t0) * 1000
            if cfg.verbose:
                print(f"  [WS]  monitor ready  "
                      f"({self._build_stats['monitor_ms']:.0f}ms)")

        self._build_stats["total_ms"] = (time.perf_counter() - t_total) * 1000

        if cfg.verbose:
            print(f"[Tsunami] Build complete  "
                  f"({self._build_stats['total_ms']:.0f}ms total)")

    # ─────────────────────────────────────────────────────────────────────────
    # Query
    # ─────────────────────────────────────────────────────────────────────────

    def query(self, q: Query) -> QueryResult:
        """
        Execute one analytical query through the full pipeline:

          1. Grid Tree traversal → intersecting leaf regions
          2. Per-region Augmented Grid cell identification
          3. Physical storage scan + aggregation
          4. Delta buffer scan + merge
        """
        t0 = time.perf_counter()

        parts:     list[tuple[int, float]] = []
        n_scanned  = 0
        n_regions  = 0
        n_cells    = 0
        n_ranges   = 0
        delta_rows = 0

        for rb in self._regions:
            if not rb.intersects(q.ranges):
                continue
            n_regions += 1

            # ── Cell identification via Augmented Grid ────────────────────
            if rb.ag is not None and _HAS_AG:
                ag_q = _GTQuery(ranges=q.ranges) if _HAS_GT else q
                candidate_idx = rb.ag.query(ag_q)
                cells_hit     = 1
                n_cells  += cells_hit if cells_hit > 0 else 1
                n_ranges += 1 if len(candidate_idx) > 0 else 0
                n_scanned += len(candidate_idx)

                # Aggregate over candidates
                if len(candidate_idx) > 0:
                    global_idx  = rb.row_indices[candidate_idx]
                    rows        = self._data[global_idx]
                    mask        = np.ones(len(rows), dtype=bool)
                    for dim, (lo, hi) in enumerate(q.ranges):
                        if math.isinf(lo) and math.isinf(hi):
                            continue
                        mask &= (rows[:, dim] >= lo) & (rows[:, dim] <= hi)
                    n  = int(mask.sum())
                    if n > 0:
                        v = (_aggregate(rows[mask, q.agg_col], q.agg_fn)
                             if q.agg_fn != "count" and q.agg_col is not None
                             else float(n))
                        parts.append((n, v))

            elif rb.cols is not None:
                # Fallback: scan region columns directly
                n_s, n_m, v = self._scan_cols(rb.cols, q)
                n_scanned += n_s
                n_ranges  += 1
                if n_m > 0:
                    parts.append((n_m, v))

            # ── Delta buffer scan ─────────────────────────────────────────
            if rb.delta_node is not None and _HAS_DI:
                di_q = q._as_di_query()
                dr   = rb.delta_node.query(di_q, agg_fn=q.agg_fn,
                                           agg_col=q.agg_col)
                dn = dr.get("n_matched", 0)
                if dn > 0:
                    parts.append((dn, dr.get("agg_value", float(dn))))
                    delta_rows += dn

        value     = _merge_aggs(parts, q.agg_fn)
        n_matched = sum(n for n, _ in parts)
        t_ms      = (time.perf_counter() - t0) * 1000

        return QueryResult(
            query=q,
            value=value,
            n_matched=n_matched,
            n_scanned=n_scanned,
            n_regions=n_regions,
            n_cells=n_cells,
            n_ranges=n_ranges,
            delta_rows=delta_rows,
            t_total_ms=t_ms,
        )

    def batch(self, queries: list[Query]) -> BatchResult:
        """Execute a list of queries and return throughput statistics."""
        results: list[QueryResult] = []
        t0 = time.perf_counter()
        for q in queries:
            results.append(self.query(q))
        total_s = time.perf_counter() - t0

        lats = np.array([r.t_total_ms for r in results])
        return BatchResult(
            results=results,
            n_queries=len(queries),
            throughput_qps=len(queries) / max(total_s, 1e-9),
            lat_avg_ms=float(lats.mean()),
            lat_p50_ms=float(np.percentile(lats, 50)),
            lat_p95_ms=float(np.percentile(lats, 95)),
            lat_p99_ms=float(np.percentile(lats, 99)),
            avg_scanned=float(np.mean([r.n_scanned for r in results])),
        )

    def brute_force(self, q: Query) -> tuple[int, float]:
        """Brute-force full scan — correctness reference."""
        assert self._data is not None
        mask = np.ones(self._N, dtype=bool)
        for dim, (lo, hi) in enumerate(q.ranges):
            if math.isinf(lo) and math.isinf(hi):
                continue
            mask &= (self._data[:, dim] >= lo) & (self._data[:, dim] <= hi)
        n = int(mask.sum())
        if n == 0 or q.agg_fn == "count":
            return n, float(n)
        ac   = q.agg_col if q.agg_col is not None else 0
        vals = self._data[mask, ac]
        return n, _aggregate(vals, q.agg_fn)

    # ─────────────────────────────────────────────────────────────────────────
    # Mutations
    # ─────────────────────────────────────────────────────────────────────────

    def insert(self, row: np.ndarray) -> int:
        """Buffer an insert into the appropriate leaf region's delta index."""
        assert _HAS_DI, "delta_index.py required for mutations"
        rb = self._region_for_row(row)
        if rb is None or rb.delta_node is None:
            raise RuntimeError("No region found for row — build() must be called first")
        return rb.delta_node.insert(row)

    def update(self, row_id: int, new_row: np.ndarray) -> int:
        """Buffer an update (tombstone old, insert new)."""
        assert _HAS_DI
        rb = self._region_for_rowid(row_id)
        if rb is None or rb.delta_node is None:
            raise RuntimeError(f"RowId {row_id} not found in any region")
        return rb.delta_node.update(row_id, new_row)

    def delete(self, row_id: int) -> None:
        """Buffer a delete (tombstone the row)."""
        assert _HAS_DI
        rb = self._region_for_rowid(row_id)
        if rb is not None and rb.delta_node is not None:
            rb.delta_node.delete(row_id)

    def merge_deltas(self) -> list[MergeResult]:
        """Force merge of all delta buffers. Returns one MergeResult per region."""
        results = []
        for rb in self._regions:
            if rb.delta_node is not None:
                mr = rb.delta_node.merge_now()
                results.append(mr)
        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Workload monitoring
    # ─────────────────────────────────────────────────────────────────────────

    def observe(self, q: Query) -> list:
        """
        Feed an incoming query to the workload shift monitor.
        Returns a list of ShiftEvents if any trigger fired.
        """
        if self._monitor is None or not _HAS_WS:
            return []
        ws_q   = q._as_ws_query()
        events = self._monitor.observe(ws_q)
        if events:
            self._shift_pending = True
        return events

    def rebuild_if_shifted(self) -> bool:
        """
        If the workload monitor detected a shift, re-run the full
        build pipeline with the queries seen since the last build.
        Returns True if a rebuild was triggered.
        """
        if not self._shift_pending or self._data is None:
            return False
        if self.cfg.verbose:
            print("[Tsunami] Workload shift detected — rebuilding index")
        self.build(self._data, self._last_queries)
        self._shift_pending = False
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return a summary of the current index state."""
        n_cells_total = 0
        n_delta_live  = 0
        n_delta_tombs = 0
        region_sizes  = []

        for rb in self._regions:
            region_sizes.append(len(rb.row_indices))
            if rb.ag is not None and hasattr(rb.ag, "lookup"):
                n_cells_total += len(rb.ag.lookup)
            if rb.delta_node is not None:
                ds = rb.delta_node.stats()
                n_delta_live  += ds.get("delta_n_live", 0)
                n_delta_tombs += ds.get("delta_n_tombstones", 0)

        return {
            "n_rows":             self._N,
            "n_dims":             self._d,
            "n_regions":          len(self._regions),
            "n_cells_total":      n_cells_total,
            "region_sizes":       region_sizes,
            "delta_live_rows":    n_delta_live,
            "delta_tombstones":   n_delta_tombs,
            "shift_pending":      self._shift_pending,
            "build_ms":           self._build_stats.get("total_ms", 0.0),
            "shift_events":       (len(self._monitor.shift_history)
                                   if self._monitor else 0),
        }

    def print_summary(self) -> None:
        """Print a human-readable index summary."""
        s = self.stats()
        print(f"\n{'─'*50}")
        print(f"  Tsunami Index Summary")
        print(f"{'─'*50}")
        print(f"  Dataset     : {s['n_rows']:,} rows × {s['n_dims']} dims")
        print(f"  GT regions  : {s['n_regions']}")
        print(f"  Grid cells  : {s['n_cells_total']:,}")
        if s['region_sizes']:
            print(f"  Region sizes: min={min(s['region_sizes'])}"
                  f"  max={max(s['region_sizes'])}"
                  f"  avg={int(np.mean(s['region_sizes']))}")
        print(f"  Delta live  : {s['delta_live_rows']}")
        print(f"  Shift det.  : {'yes' if self._monitor else 'no'}"
              f"  (pending={s['shift_pending']})")
        print(f"  Build time  : {s['build_ms']:.0f} ms")
        print(f"{'─'*50}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Private — build helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_grid_tree(self,
                         data:    np.ndarray,
                         queries: list) -> list[_RegionBundle]:
        """Step 1: partition data space via the Grid Tree."""
        cfg = self.cfg
        N, d = data.shape

        if _HAS_GT:
            gt = GridTree(
                min_skew_reduction=cfg.gt_min_skew_reduction,
                min_points_frac=cfg.gt_min_points_frac,
                min_queries_frac=cfg.gt_min_queries_frac,
                n_bins=cfg.gt_n_bins,
                max_depth=cfg.gt_max_depth,
            )
            gt.build(data, queries)
            self._gt = gt

            regions = []
            for i, leaf in enumerate(gt.leaves()):
                if leaf.point_indices is None or len(leaf.point_indices) == 0:
                    continue
                regions.append(_RegionBundle(
                    region_id=i,
                    lo=leaf.region_lo.copy(),
                    hi=leaf.region_hi.copy(),
                    row_indices=leaf.point_indices.copy(),
                ))
            return regions

        else:
            # Fallback: median-split partitioner
            return self._fallback_partition(data, N, d)

    def _fallback_partition(self, data: np.ndarray,
                            N: int, d: int) -> list[_RegionBundle]:
        """Simple variance-based binary partition when GridTree unavailable."""
        lo = data.min(axis=0).astype(float)
        hi = data.max(axis=0).astype(float) + 1e-9
        regions: list[_RegionBundle] = []
        rid = [0]

        n_target = max(500, N // 8)

        def _split(idx: np.ndarray, lo: np.ndarray,
                   hi: np.ndarray, depth: int):
            if len(idx) <= n_target or depth >= 4:
                regions.append(_RegionBundle(
                    region_id=rid[0], lo=lo.copy(), hi=hi.copy(),
                    row_indices=idx))
                rid[0] += 1
                return
            sub = data[idx]
            dim = int(np.argmax(sub.std(axis=0)))
            mid = float(np.median(sub[:, dim]))
            lm  = sub[:, dim] < mid
            rm  = sub[:, dim] >= mid
            if lm.sum() == 0 or rm.sum() == 0:
                regions.append(_RegionBundle(
                    region_id=rid[0], lo=lo.copy(), hi=hi.copy(),
                    row_indices=idx))
                rid[0] += 1
                return
            hl = hi.copy(); hl[dim] = mid
            lr = lo.copy(); lr[dim] = mid
            _split(idx[lm], lo, hl, depth + 1)
            _split(idx[rm], lr, hi, depth + 1)

        _split(np.arange(N), lo, hi, 0)
        return regions

    def _build_region(self,
                      rb:            _RegionBundle,
                      region_data:   np.ndarray,
                      region_queries: list[Query]) -> None:
        """Steps 2–5 for one region: AGD → sort dim → storage → delta."""
        cfg = self.cfg
        N, d = region_data.shape

        # ── Step 2: Augmented Grid via AGD ────────────────────────────────
        if _HAS_AG:
            if _HAS_AGD and cfg.agd_enabled and region_queries:
                try:
                    from cost_model_agd import Query as _AGDQuery
                    agd_queries = [_AGDQuery(ranges=q.ranges) for q in region_queries]
                    agd_cfg  = AGDConfig(
                        max_iter=cfg.agd_max_iter,
                        part_step=cfg.agd_part_step,
                        sample_frac=cfg.agd_sample_frac,
                        max_parts=16,
                        verbose=False,
                    )
                    agd_result = agd_optimise(region_data, agd_queries, config=agd_cfg)
                    rb.skeleton = agd_result.skeleton
                    rb.n_parts  = agd_result.n_parts
                except Exception as e:
                    if cfg.verbose:
                        print(f"  [AGD] region {rb.region_id}: AGD failed ({e}), "
                              f"using heuristic")
                    rb.skeleton, rb.n_parts = initialise_skeleton(
                        region_data,
                        fm_error_thresh=cfg.ag_fm_thresh,
                        ccdf_empty_thresh=cfg.ag_ccdf_thresh,
                        default_parts=4,
                    )
            else:
                # No training queries for this region — use heuristic INDEPENDENT CDF
                # with fewer partitions to keep cell count small
                rb.skeleton, rb.n_parts = initialise_skeleton(
                    region_data,
                    fm_error_thresh=cfg.ag_fm_thresh,
                    ccdf_empty_thresh=cfg.ag_ccdf_thresh,
                    default_parts=4,
                )

            if rb.skeleton is not None:
                try:
                    rb.ag = AugmentedGrid(region_data, rb.skeleton, rb.n_parts)
                except Exception:
                    rb.ag = None

        # ── Step 3: Sort dimension ────────────────────────────────────────
        if cfg.sort_dim_enabled and _HAS_SD and region_queries:
            try:
                from sort_dim_optimizer import SortDimOptimizer
                from sort_dim_optimizer import Query as _SDQuery
                sd_queries = [_SDQuery(ranges=q.ranges) for q in region_queries]
                n_parts_dict  = {i: (rb.n_parts[i] if rb.n_parts else cfg.ag_default_parts)
                                 for i in range(d)}
                boundaries_sd = {}
                if rb.ag is not None:
                    boundaries_sd = {
                        dim: bds for dim, bds in enumerate(rb.ag.boundaries)
                        if bds is not None
                    }
                opt = SortDimOptimizer()
                rb.sort_dim = opt.select_per_region(
                    region_data, sd_queries,
                    n_parts=n_parts_dict,
                    boundaries=boundaries_sd,
                    candidates=[i for i in range(d)
                                 if n_parts_dict.get(i, 0) > 0],
                )
            except Exception:
                rb.sort_dim = 0
        else:
            rb.sort_dim = 0

        # ── Step 4: Physical storage ──────────────────────────────────────
        # Build column arrays sorted by cell key (+ optional sort dim)
        if rb.ag is not None:
            sorted_data = region_data[rb.ag.sorted_indices]
        else:
            order = np.argsort(region_data[:, rb.sort_dim or 0], kind="stable")
            sorted_data = region_data[order]

        rb.cols = [sorted_data[:, dim].copy() for dim in range(d)]

        # ── Step 5: Delta index ───────────────────────────────────────────
        if cfg.delta_enabled and _HAS_DI:
            policy = MergePolicy(
                size_threshold=cfg.delta_size_threshold,
                time_threshold_s=cfg.delta_time_threshold_s,
                ratio_threshold=cfg.delta_ratio_threshold,
            )
            rb.delta_node = DeltaIndexNode(
                d=d,
                col_names=self._col_names,
                policy=policy,
                sort_dim=rb.sort_dim,
                on_merge=lambda mr, _rb=rb: self._on_region_merge(mr, _rb),
            )
            rb.delta_node.load(sorted_data)

    def _build_monitor(self, data: np.ndarray,
                       queries: list[Query]) -> None:
        """Step 6: initialise the workload shift monitor."""
        cfg = self.cfg
        ws_queries = [q._as_ws_query() for q in queries]
        bounds     = (data.min(axis=0), data.max(axis=0))
        ws_regions = [
            RegionBounds(lo=rb.lo.copy(), hi=rb.hi.copy(),
                         region_id=rb.region_id)
            for rb in self._regions
        ]
        self._monitor = WorkloadMonitor.from_build(
            build_queries=ws_queries,
            regions=ws_regions,
            data_bounds=bounds,
            on_shift=self._on_shift_event,
            kl_threshold=cfg.shift_kl_threshold,
            drift_ratio_threshold=cfg.shift_drift_ratio,
            cooldown_s=cfg.shift_cooldown_s,
            type_window=cfg.shift_type_window,
            freq_window=cfg.shift_freq_window,
            skew_window=cfg.shift_skew_window,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Private — callbacks and utilities
    # ─────────────────────────────────────────────────────────────────────────

    def _on_shift_event(self, event) -> None:
        self._shift_pending = True
        if self.cfg.verbose:
            print(f"[Tsunami] Shift event: trigger={event.trigger}  "
                  f"queries_seen={event.queries_seen}")

    def _on_region_merge(self, mr, rb: _RegionBundle) -> None:
        """
        Called after a delta merge for one region.
        Rebuilds the region's Augmented Grid and column arrays from
        the merged data.
        """
        if self.cfg.verbose:
            print(f"[Tsunami] Region {rb.region_id} delta merge: "
                  f"{mr.n_main_rows} + {mr.n_delta_rows} → {mr.n_total} rows")

        if mr.n_total == 0:
            return

        merged = mr.merged_data
        rb.row_indices = np.arange(mr.n_total)   # local indices after merge

        # Rebuild AG
        if _HAS_AG and rb.skeleton is not None:
            try:
                rb.ag = AugmentedGrid(merged, rb.skeleton, rb.n_parts)
            except Exception:
                rb.ag = None

        # Rebuild column arrays
        if rb.ag is not None:
            sorted_data = merged[rb.ag.sorted_indices]
        else:
            order = np.argsort(merged[:, rb.sort_dim or 0], kind="stable")
            sorted_data = merged[order]
        rb.cols = [sorted_data[:, dim].copy() for dim in range(self._d)]

    def _region_for_row(self, row: np.ndarray) -> Optional[_RegionBundle]:
        """Find the leaf region a row belongs to (for inserts)."""
        for rb in self._regions:
            ranges = [(float(row[dim]), float(row[dim])) for dim in range(self._d)]
            if rb.intersects(ranges):
                return rb
        return self._regions[0] if self._regions else None

    def _region_for_rowid(self, row_id: int) -> Optional[_RegionBundle]:
        """Find the leaf region owning a given RowId."""
        for rb in self._regions:
            if rb.delta_node is not None:
                ids = rb.delta_node._main_row_ids
                if row_id in ids:
                    return rb
                ids_buf = rb.delta_node._buffer._row_ids[:rb.delta_node._buffer._size]
                if row_id in ids_buf:
                    return rb
        return None

    @staticmethod
    def _scan_cols(cols:  list[np.ndarray],
                   q:     Query
                   ) -> tuple[int, int, float]:
        """Linear scan fallback when no Augmented Grid is available."""
        N    = len(cols[0])
        mask = np.ones(N, dtype=bool)
        for dim, (lo, hi) in enumerate(q.ranges):
            if dim >= len(cols):
                break
            if math.isinf(lo) and math.isinf(hi):
                continue
            mask &= (cols[dim] >= lo) & (cols[dim] <= hi)
        n = int(mask.sum())
        if n == 0 or q.agg_fn == "count" or q.agg_col is None:
            return N, n, float(n)
        vals = cols[q.agg_col][mask]
        return N, n, _aggregate(vals, q.agg_fn)


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(0)
    rng = np.random.default_rng(0)

    print("=" * 60)
    print("  Tsunami — Complete Integrated Index Demo")
    print("=" * 60)

    # ── Dataset: 15 000 rows, 3 dims ─────────────────────────────────────
    N, D = 15_000, 3
    d0 = rng.uniform(0, 100, N)
    d1 = 0.85 * d0 + rng.normal(0, 4, N)   # tight FM correlation
    d2 = rng.uniform(0, 100, N)             # independent
    d1 = np.clip(d1, 0, 100)
    data      = np.column_stack([d0, d1, d2])
    col_names = ["price", "cost", "weight"]

    # ── Workload: two skewed query types ──────────────────────────────────
    queries = (
        [Query([(float(lo), float(lo+10)),
                (-math.inf, math.inf),
                (float(lo), float(lo+10))])
         for lo in rng.uniform(60, 90, 80)]
      + [Query([(float(lo), float(lo+30)),
                (-math.inf, math.inf),
                (-math.inf, math.inf)])
         for lo in rng.uniform(0, 70, 40)]
    )

    # ── Build ─────────────────────────────────────────────────────────────
    print("\n── Building index ──")
    cfg = TsunamiConfig(
        col_names=col_names,
        gt_max_depth=3,
        agd_max_iter=10,
        agd_enabled=_HAS_AGD,
        delta_enabled=_HAS_DI,
        shift_detection_enabled=_HAS_WS,
        verbose=True,
    )
    idx = TsunamiIndex(cfg)
    idx.build(data, queries)
    idx.print_summary()

    # ── Queries ───────────────────────────────────────────────────────────
    print("── Query correctness ──")
    test_qs = [
        Query([(65.0, 80.0), (-math.inf, math.inf), (60.0, 75.0)],
              agg_fn="count", label="narrow"),
        Query([(20.0, 70.0), (-math.inf, math.inf), (-math.inf, math.inf)],
              agg_fn="count", label="wide"),
        Query([(40.0, 80.0), (-math.inf, math.inf), (-math.inf, math.inf)],
              agg_fn="sum", agg_col=2, label="SUM(weight)"),
        Query([(0.0, 100.0), (-math.inf, math.inf), (-math.inf, math.inf)],
              agg_fn="avg", agg_col=0, label="AVG(price)"),
    ]

    for q in test_qs:
        r        = idx.query(q)
        bf_n, bf_v = idx.brute_force(q)
        ok       = (r.n_matched == bf_n and
                    abs(r.value - bf_v) < max(1.0, abs(bf_v) * 0.01))
        print(f"  [{q.label:12s}]  matched={r.n_matched:4d}  "
              f"value={r.value:10.2f}  bf={bf_v:10.2f}  "
              f"scan={r.n_scanned:5d}  t={r.t_total_ms:.2f}ms  "
              f"{'✓' if ok else '✗'}")

    # ── Batch ─────────────────────────────────────────────────────────────
    print("\n── Batch (100 queries) ──")
    batch_qs = [
        Query([(float(lo), float(lo+15)), (-math.inf, math.inf),
               (float(lo), float(lo+15))])
        for lo in rng.uniform(10, 85, 100)
    ]
    br = idx.batch(batch_qs)
    print(f"  {br}")

    # ── Mutations ─────────────────────────────────────────────────────────
    if _HAS_DI:
        print("\n── Mutations (insert / update / delete) ──")
        rid1 = idx.insert(np.array([72.0, 61.0, 68.0]))
        print(f"  Inserted row  → RowId={rid1}")

        rid2 = idx.update(rid1, np.array([73.0, 62.0, 69.0]))
        print(f"  Updated row   → new RowId={rid2}")

        idx.delete(rid2)
        print(f"  Deleted RowId={rid2}")

        q_mut = Query([(70.0, 76.0), (-math.inf, math.inf), (65.0, 72.0)])
        r_mut = idx.query(q_mut)
        print(f"  Post-mutation query: matched={r_mut.n_matched}  "
              f"delta_rows={r_mut.delta_rows}")

    # ── Workload shift simulation ─────────────────────────────────────────
    if _HAS_WS:
        print("\n── Workload shift simulation ──")
        # Feed 150 new queries of a completely different type
        for lo in rng.uniform(0, 40, 150):
            shift_q = Query([(float(lo), float(lo+2)),
                             (-math.inf, math.inf),
                             (-math.inf, math.inf)])
            events = idx.observe(shift_q)
            if events:
                print(f"  Shift detected: {events[0].trigger}")
                break
        rebuilt = idx.rebuild_if_shifted()
        print(f"  Rebuild triggered: {rebuilt}")

    # ── Final stats ───────────────────────────────────────────────────────
    print("\n── Final stats ──")
    s = idx.stats()
    for k, v in s.items():
        if k != "region_sizes":
            print(f"  {k:28s}: {v}")
