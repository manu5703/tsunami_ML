"""
Augmented Grid Implementation — from Tsunami paper
"Tsunami: A Learned Multi-dimensional Index for Correlated Data and Skewed Workloads"
Ding et al., PVLDB 2021  (§5)

The Augmented Grid handles data correlation inside each Grid Tree region.
Each dimension uses one of three partitioning strategies:
  1. Independent CDF  — CDF(X)        (Flood-style, for uncorrelated dims)
  2. Functional Mapping — F: X → Y    (for tight monotonic correlations)
  3. Conditional CDF   — CDF(X | Y)   (for loose / generic correlations)

The best skeleton (combination of strategies) and partition counts are found
via Adaptive Gradient Descent (AGD) on a linear cost model.

Dependencies:
    pip install numpy scikit-learn scipy
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum, auto
from scipy.stats import linregress


# ============================================================
# Re-use Query dataclass from grid_tree.py
# ============================================================

@dataclass
class Query:
    """A d-dimensional range query: each entry is (lo, hi) per dimension."""
    ranges: list[tuple[float, float]]

    @property
    def ndim(self) -> int:
        return len(self.ranges)


# ============================================================
# §5.2  Skeleton — partitioning strategy per dimension
# ============================================================

class StrategyKind(Enum):
    INDEPENDENT = auto()   # partition X using CDF(X)
    FUNCTIONAL  = auto()   # remove X, map X-filter → Y via linear regression
    CONDITIONAL = auto()   # partition X using CDF(X | Y)


@dataclass
class DimStrategy:
    kind: StrategyKind
    # FUNCTIONAL:  other_dim = target dimension Y;  lr_slope/intercept/el/eu
    # CONDITIONAL: other_dim = base dimension Y
    other_dim:    Optional[int]   = None
    lr_slope:     float           = 0.0
    lr_intercept: float           = 0.0
    el:           float           = 0.0   # lower error bound
    eu:           float           = 0.0   # upper error bound

    def __repr__(self):
        if self.kind == StrategyKind.INDEPENDENT:
            return "CDF(X)"
        if self.kind == StrategyKind.FUNCTIONAL:
            return f"F:X→dim{self.other_dim}"
        return f"CDF(X|dim{self.other_dim})"


Skeleton = list[DimStrategy]   # one entry per dimension


# ============================================================
# §5.2.1  Functional Mapping helpers
# ============================================================

def fit_functional_mapping(data: np.ndarray, x_dim: int, y_dim: int,
                           error_thresh: float = 0.10) -> Optional[DimStrategy]:
    """
    Fit a linear regression X = f(Y) and return a DimStrategy if the
    max absolute error (normalised by Y's domain) is below error_thresh.
    Returns None if the correlation is not tight enough.
    """
    X = data[:, x_dim].astype(float)
    Y = data[:, y_dim].astype(float)
    if len(X) < 4:
        return None

    res = linregress(Y, X)
    X_pred = res.slope * Y + res.intercept
    residuals = X - X_pred
    el = float(-residuals.min())   # lower error bound (positive)
    eu = float( residuals.max())   # upper error bound (positive)

    y_range = Y.max() - Y.min() + 1e-12
    if (el + eu) / y_range <= error_thresh:
        return DimStrategy(
            kind=StrategyKind.FUNCTIONAL,
            other_dim=y_dim,
            lr_slope=res.slope,
            lr_intercept=res.intercept,
            el=el,
            eu=eu,
        )
    return None


def apply_functional_mapping(strat: DimStrategy,
                             x_lo: float, x_hi: float
                             ) -> tuple[float, float]:
    """
    Given a filter [x_lo, x_hi] on the FUNCTIONAL dim X, derive a tighter
    filter on the target dim Y.

    Model: X = slope * Y + intercept  (with residual bounds el, eu)
    So:    Y = (X - intercept) / slope  ±  error/|slope|

    el = max(X_pred - X)  →  model overestimates by at most el
    eu = max(X - X_pred)  →  model underestimates by at most eu

    Y bounds that cover all (X, Y) pairs with X ∈ [x_lo, x_hi]:
      lower: (x_lo - eu - intercept) / slope
      upper: (x_hi + el - intercept) / slope
    """
    slope = strat.lr_slope
    if abs(slope) < 1e-12:
        return -1e18, 1e18   # degenerate slope → no Y constraint

    y1 = (x_lo - strat.eu - strat.lr_intercept) / slope
    y2 = (x_hi + strat.el - strat.lr_intercept) / slope
    return (min(y1, y2), max(y1, y2))


# ============================================================
# §5.2.2  Conditional CDF helpers
# ============================================================

def check_conditional_needed(data: np.ndarray, x_dim: int, y_dim: int,
                              p_x: int, p_y: int,
                              empty_thresh: float = 0.25) -> bool:
    """
    Return True if more than `empty_thresh` fraction of cells in the X-Y
    grid hyperplane would be empty when partitioning independently.
    """
    X = data[:, x_dim]
    Y = data[:, y_dim]
    # build a p_x × p_y grid using independent quantile boundaries
    x_bounds = np.quantile(X, np.linspace(0, 1, p_x + 1))
    y_bounds = np.quantile(Y, np.linspace(0, 1, p_y + 1))
    occupied = 0
    for i in range(p_x):
        for j in range(p_y):
            mask = ((X >= x_bounds[i]) & (X < x_bounds[i+1]) &
                    (Y >= y_bounds[j]) & (Y < y_bounds[j+1]))
            if mask.sum() > 0:
                occupied += 1
    empty_frac = 1.0 - occupied / (p_x * p_y)
    return empty_frac > empty_thresh


# ============================================================
# §5.2  Augmented Grid
# ============================================================

class AugmentedGrid:
    """
    An Augmented Grid for one region of the Grid Tree.

    Parameters
    ----------
    n_partitions : list[int]
        Number of partitions per non-mapped dimension (length = d).
        Mapped dimensions are ignored (they contribute 0 partitions).
    skeleton : Skeleton
        Partitioning strategy per dimension (length = d).
    """

    def __init__(self, data: np.ndarray, skeleton: Skeleton,
                 n_partitions: list[int]):
        self.d          = data.shape[1]
        self.skeleton   = skeleton
        self.n_parts    = n_partitions   # one entry per dim (0 if mapped)
        self._build(data)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self, data: np.ndarray):
        """Pre-compute CDF boundaries and conditional histograms."""
        N, d = data.shape
        self.boundaries: list[Optional[np.ndarray]] = [None] * d
        # For CONDITIONAL dims: store p_base histograms over the dep dim.
        # cond_bounds[dep_dim] = array of shape (p_base, p_dep+1)
        self.cond_bounds: dict[int, np.ndarray] = {}

        for dim, strat in enumerate(self.skeleton):
            p = self.n_parts[dim]
            if p < 1:
                continue

            if strat.kind == StrategyKind.INDEPENDENT:
                # CDF(X): quantile-based boundaries
                quantiles = np.linspace(0, 1, p + 1)
                self.boundaries[dim] = np.quantile(data[:, dim], quantiles)

            elif strat.kind == StrategyKind.FUNCTIONAL:
                # Mapped dim: no own boundaries; target dim handles it.
                pass

            elif strat.kind == StrategyKind.CONDITIONAL:
                base_dim = strat.other_dim
                p_base   = self.n_parts[base_dim]
                if p_base < 2:
                    p_base = 8   # base dim may be FUNCTIONAL (p=0); use a default
                base_bds = self.boundaries[base_dim]
                if base_bds is None:
                    # Fallback: build base boundaries now
                    base_bds = np.quantile(data[:, base_dim],
                                           np.linspace(0, 1, p_base + 1))
                    self.boundaries[base_dim] = base_bds

                # For each partition of base_dim, compute quantile boundaries
                # of dim conditioned on being in that base partition.
                cond = np.zeros((p_base, p + 1), dtype=float)
                for k in range(p_base):
                    lo_b = base_bds[k]
                    hi_b = base_bds[k + 1]
                    mask = ((data[:, base_dim] >= lo_b) &
                            (data[:, base_dim] <  hi_b))
                    sub = data[mask, dim]
                    if len(sub) == 0:
                        cond[k] = np.linspace(data[:, dim].min(),
                                              data[:, dim].max(), p + 1)
                    else:
                        cond[k] = np.quantile(sub, np.linspace(0, 1, p + 1))
                self.cond_bounds[dim] = cond
                self.boundaries[dim]  = None   # not used for CONDITIONAL

        # Precompute the topo-sorted active dim list once.
        # All cell keys (lookup table and queries) use this order so they match.
        raw_active = [
            dim for dim, strat in enumerate(self.skeleton)
            if self.n_parts[dim] >= 1 and strat.kind != StrategyKind.FUNCTIONAL
        ]
        self._active_dims: list[int] = self._topo_sort_dims(raw_active)

        # Build lookup table: maps cell tuple → (start, end) in sorted storage
        # Sort data by the grid cell order and store sorted indices.
        self._sort_data(data)

    def _cell_index(self, point: np.ndarray) -> tuple:
        """Map a single point to its grid cell tuple (in _active_dims order)."""
        cell = []
        for dim in self._active_dims:
            strat = self.skeleton[dim]
            p     = self.n_parts[dim]
            if strat.kind == StrategyKind.INDEPENDENT:
                bds = self.boundaries[dim]
                idx = int(np.searchsorted(bds[1:-1], point[dim]))
                cell.append(min(idx, p - 1))
            elif strat.kind == StrategyKind.CONDITIONAL:
                base_dim = strat.other_dim
                base_bds = self.boundaries[base_dim]
                if base_bds is None:
                    cell.append(0)
                    continue
                base_idx = int(np.searchsorted(base_bds[1:-1], point[base_dim]))
                base_idx = min(base_idx, self.n_parts[base_dim] - 1)
                cond_bds = self.cond_bounds[dim][base_idx]
                idx = int(np.searchsorted(cond_bds[1:-1], point[dim]))
                cell.append(min(idx, p - 1))
        return tuple(cell)

    def _sort_data(self, data: np.ndarray):
        """
        Fully vectorised: assign each row its cell tuple using numpy searchsorted,
        lexsort to group rows, then build the lookup table and vectorised query arrays.
        No Python loop over rows — scales to tens of millions of rows.
        """
        N         = len(data)
        n_active  = len(self._active_dims)

        if n_active == 0 or N == 0:
            self.sorted_indices = np.arange(N, dtype=np.int64)
            self.lookup         = {(): (0, N)} if N > 0 else {}
            if N > 0:
                self._vkeys   = np.zeros((1, 0), dtype=np.int32)
                self._vstarts = np.array([0],    dtype=np.int64)
                self._vends   = np.array([N],    dtype=np.int64)
            else:
                self._vkeys = self._vstarts = self._vends = None
            return

        # key_matrix[row, col] = partition index for that row in the col-th active dim
        key_matrix = np.zeros((N, n_active), dtype=np.int32)

        for col_idx, dim in enumerate(self._active_dims):
            strat = self.skeleton[dim]
            p     = self.n_parts[dim]

            if strat.kind == StrategyKind.INDEPENDENT:
                bds   = self.boundaries[dim]
                idx   = np.searchsorted(bds[1:-1], data[:, dim])
                key_matrix[:, col_idx] = np.clip(idx, 0, p - 1)

            elif strat.kind == StrategyKind.CONDITIONAL:
                base_dim = strat.other_dim
                try:
                    base_col  = self._active_dims.index(base_dim)
                    base_asgn = key_matrix[:, base_col]
                except ValueError:
                    base_asgn = np.zeros(N, dtype=np.int32)

                p_base = self.n_parts[base_dim] if base_dim < len(self.n_parts) else 8
                if p_base < 2:
                    p_base = 8

                indices = np.zeros(N, dtype=np.int32)
                for k in range(p_base):        # loop is over partitions (≤16), not rows
                    mask = (base_asgn == k)
                    if not mask.any():
                        continue
                    if dim in self.cond_bounds and k < len(self.cond_bounds[dim]):
                        cond_bds = self.cond_bounds[dim][k]
                        idx = np.searchsorted(cond_bds[1:-1], data[mask, dim])
                        indices[mask] = np.clip(idx, 0, p - 1)
                key_matrix[:, col_idx] = indices

        # Lexicographic sort: first active dim is primary key.
        # np.lexsort uses last key as primary, so pass columns reversed.
        sort_keys = [key_matrix[:, i] for i in range(n_active - 1, -1, -1)]
        order     = np.lexsort(sort_keys)
        self.sorted_indices = order.astype(np.int64)

        sorted_km = key_matrix[order]   # reordered cell-key matrix

        # Find group boundaries (positions where consecutive rows differ)
        if N > 1:
            diffs = np.any(sorted_km[1:] != sorted_km[:-1], axis=1)
            bdy   = np.concatenate(([0], np.where(diffs)[0] + 1, [N]))
        else:
            bdy = np.array([0, N], dtype=np.int64)

        starts = bdy[:-1]
        ends   = bdy[1:]

        # Build lookup dict and vectorised query arrays in one pass
        self.lookup: dict[tuple, tuple[int, int]] = {}
        for i in range(len(starts)):
            key = tuple(int(x) for x in sorted_km[starts[i]])
            self.lookup[key] = (int(starts[i]), int(ends[i]))

        self._vkeys   = sorted_km[starts].astype(np.int32)
        self._vstarts = starts.astype(np.int64)
        self._vends   = ends.astype(np.int64)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, q: Query) -> np.ndarray:
        """
        Return the sorted_indices of all candidate points for query q.
        The caller should verify the actual predicate match.
        """
        # Apply functional mappings to tighten query filters (§5.2.1)
        adjusted = list(q.ranges)
        for dim, strat in enumerate(self.skeleton):
            if strat.kind == StrategyKind.FUNCTIONAL:
                y_dim = strat.other_dim
                y_lo, y_hi = q.ranges[dim]
                x_lo, x_hi = apply_functional_mapping(strat, y_lo, y_hi)
                cur_lo, cur_hi = adjusted[y_dim]
                adjusted[y_dim] = (max(cur_lo, x_lo), min(cur_hi, x_hi))

        if not self._active_dims:
            return self.sorted_indices.copy()

        if self._vkeys is None or len(self._vkeys) == 0:
            return np.array([], dtype=np.int64)

        # ── Vectorised cell matching ──────────────────────────────────────
        # mask[j] = True  iff  cell j intersects the query on every active dim.
        mask = np.ones(len(self._vkeys), dtype=bool)

        # Precompute base-dim column index in _active_dims for CONDITIONAL dims.
        cond_base_col: dict[int, int] = {}
        for i, dim in enumerate(self._active_dims):
            strat = self.skeleton[dim]
            if strat.kind == StrategyKind.CONDITIONAL and strat.other_dim is not None:
                try:
                    cond_base_col[i] = self._active_dims.index(strat.other_dim)
                except ValueError:
                    cond_base_col[i] = -1

        for i, dim in enumerate(self._active_dims):
            if not mask.any():
                return np.array([], dtype=np.int64)   # all cells pruned

            strat = self.skeleton[dim]
            p     = self.n_parts[dim]
            qlo, qhi = adjusted[dim]

            if strat.kind == StrategyKind.INDEPENDENT:
                bds   = self.boundaries[dim]
                first = max(0, int(np.searchsorted(bds[1:], qlo)))
                last  = min(p - 1, int(np.searchsorted(bds[:-1], qhi, side='right')) - 1)
                if first > last:
                    return np.array([], dtype=np.int64)
                col = self._vkeys[:, i]
                mask &= (col >= first) & (col <= last)

            elif strat.kind == StrategyKind.CONDITIONAL:
                base_col = cond_base_col.get(i, -1)
                col_mask = np.zeros(len(self._vkeys), dtype=bool)
                # Only evaluate still-alive cells to minimise work.
                for j in np.where(mask)[0]:
                    base_k   = int(self._vkeys[j, base_col]) if base_col >= 0 else 0
                    k        = int(self._vkeys[j, i])
                    cond_bds = self.cond_bounds[dim][base_k]
                    first    = max(0, int(np.searchsorted(cond_bds[1:], qlo)))
                    last     = min(p - 1,
                                   int(np.searchsorted(cond_bds[:-1], qhi, side='right')) - 1)
                    col_mask[j] = (k >= first) and (k <= last)
                mask &= col_mask

        # ── Slice-based result assembly (no Python int list) ──────────────
        # Each matching cell is a contiguous run in sorted_indices; concatenate
        # the numpy views directly instead of building a flat position list.
        hits = np.where(mask)[0]
        if len(hits) == 0:
            return np.array([], dtype=np.int64)
        vstarts = self._vstarts[hits]
        vends   = self._vends[hits]
        sizes   = vends - vstarts
        total   = int(sizes.sum())
        if total == 0:
            return np.array([], dtype=np.int64)
        if len(hits) == 1:
            return self.sorted_indices[vstarts[0]:vends[0]].copy()
        # Fully vectorised gather — avoids Python loop + slow np.concatenate on many tiny arrays
        repeated_starts = np.repeat(vstarts, sizes)
        local_offsets   = np.arange(total, dtype=np.int64) - np.repeat(
                              np.concatenate(([0], np.cumsum(sizes[:-1]))), sizes)
        return self.sorted_indices[repeated_starts + local_offsets]

    def _topo_sort_dims(self, dims: list[int]) -> list[int]:
        """Return dims reordered so CONDITIONAL base dims precede their dependents."""
        dim_set = set(dims)
        ordered: list[int] = []
        visited: set[int]  = set()

        def visit(d: int) -> None:
            if d in visited:
                return
            visited.add(d)
            strat = self.skeleton[d]
            if (strat.kind == StrategyKind.CONDITIONAL
                    and strat.other_dim is not None
                    and strat.other_dim in dim_set):
                visit(strat.other_dim)
            ordered.append(d)

        for d in dims:
            visit(d)
        return ordered


# ============================================================
# §5.3.1  Cost Model
# ============================================================

def estimate_cost(data: np.ndarray, queries: list[Query],
                  skeleton: Skeleton, n_parts: list[int],
                  w0: float = 1.0, w1: float = 0.01) -> float:
    """
    Linear cost model:
        Time = w0 * (#cell_ranges) + w1 * (#scanned_points) * (#filtered_dims)

    Uses a sample of queries for speed.
    """
    sample = queries[:min(50, len(queries))]
    if not sample:
        return float("inf")

    ag = AugmentedGrid(data, skeleton, n_parts)

    total_cost = 0.0
    for q in sample:
        candidates = ag.query(q)
        n_scanned  = len(candidates)

        # Estimate cell ranges as unique consecutive groups in sorted order
        if n_scanned == 0:
            continue
        sorted_pos = np.sort(candidates)
        n_ranges   = 1 + int(np.sum(np.diff(sorted_pos) > 1))

        n_filtered = sum(1 for dim, strat in enumerate(skeleton)
                         if strat.kind != StrategyKind.FUNCTIONAL
                         and n_parts[dim] > 0)

        total_cost += w0 * n_ranges + w1 * n_scanned * n_filtered

    return total_cost / len(sample)


# ============================================================
# §5.3  Skeleton heuristic initialiser
# ============================================================

def initialise_skeleton(data: np.ndarray,
                        fm_error_thresh: float = 0.10,
                        ccdf_empty_thresh: float = 0.25,
                        default_parts: int = 8) -> tuple[Skeleton, list[int]]:
    """
    Heuristic skeleton initialisation (§5.3.2, step 1):
      1. Use functional mapping F:X→Y if error bound ≤ 10% of Y's domain.
      2. Else use CDF(X|Y) if >25% of X-Y cells would be empty independently.
      3. Else use independent CDF(X).
    """
    d = data.shape[1]
    skeleton: Skeleton = [DimStrategy(StrategyKind.INDEPENDENT)] * d
    n_parts  = [default_parts] * d

    # Track which dims are already used as targets (cannot be mapped dims too)
    mapped_targets:    set[int] = set()
    conditional_bases: set[int] = set()

    for x_dim in range(d):
        best_fm  = None
        best_err = float("inf")
        for y_dim in range(d):
            if y_dim == x_dim:
                continue
            if y_dim in mapped_targets:
                continue
            strat = fit_functional_mapping(data, x_dim, y_dim, fm_error_thresh)
            if strat is not None:
                err = strat.el + strat.eu
                if err < best_err:
                    best_err = err
                    best_fm  = strat

        if best_fm is not None:
            skeleton[x_dim]  = best_fm
            n_parts[x_dim]   = 0        # mapped dim has no own partitions
            mapped_targets.add(best_fm.other_dim)
            continue

        # Try conditional CDF
        for y_dim in range(d):
            if y_dim == x_dim:
                continue
            if skeleton[y_dim].kind in (StrategyKind.FUNCTIONAL,
                                        StrategyKind.CONDITIONAL):
                continue   # base cannot itself be mapped or conditional
            if y_dim in conditional_bases:
                continue
            p_x = default_parts
            p_y = default_parts
            if check_conditional_needed(data, x_dim, y_dim, p_x, p_y,
                                        ccdf_empty_thresh):
                skeleton[x_dim] = DimStrategy(
                    kind=StrategyKind.CONDITIONAL,
                    other_dim=y_dim,
                )
                conditional_bases.add(y_dim)
                break

    return skeleton, n_parts


# ============================================================
# §5.3.2  Adaptive Gradient Descent (AGD)
# ============================================================

def _neighbours(skeleton: Skeleton, n_parts: list[int],
                dim: int, d: int) -> list[tuple[Skeleton, list[int]]]:
    """
    Generate all skeletons one 'hop' away from current by changing
    the strategy of a single dimension (§5.3.2, step 3).
    """
    neighbours = []
    current = skeleton[dim]

    strategies_to_try = []

    # Always try INDEPENDENT
    if current.kind != StrategyKind.INDEPENDENT:
        strategies_to_try.append(DimStrategy(StrategyKind.INDEPENDENT))

    # Try FUNCTIONAL to each other dim
    for y in range(d):
        if y == dim:
            continue
        if skeleton[y].kind == StrategyKind.FUNCTIONAL:
            continue   # target cannot itself be a mapped dim
        new_strat = DimStrategy(StrategyKind.FUNCTIONAL, other_dim=y)
        if current.kind != StrategyKind.FUNCTIONAL or current.other_dim != y:
            strategies_to_try.append(new_strat)

    # Try CONDITIONAL on each other dim
    for y in range(d):
        if y == dim:
            continue
        if skeleton[y].kind in (StrategyKind.FUNCTIONAL,
                                StrategyKind.CONDITIONAL):
            continue   # base cannot be mapped/conditional
        new_strat = DimStrategy(StrategyKind.CONDITIONAL, other_dim=y)
        if current.kind != StrategyKind.CONDITIONAL or current.other_dim != y:
            strategies_to_try.append(new_strat)

    for new_strat in strategies_to_try:
        new_skel   = list(skeleton)
        new_skel[dim] = new_strat
        new_parts  = list(n_parts)
        if new_strat.kind == StrategyKind.FUNCTIONAL:
            new_parts[dim] = 0
        else:
            if new_parts[dim] == 0:
                new_parts[dim] = 4   # restore a default
        neighbours.append((new_skel, new_parts))

    return neighbours


def optimise_augmented_grid(
        data: np.ndarray,
        queries: list[Query],
        max_iter: int = 20,
        lr: float = 0.5,
        min_parts: int = 2,
        max_parts: int = 32,
        w0: float = 1.0,
        w1: float = 0.01,
) -> tuple[Skeleton, list[int]]:
    """
    Adaptive Gradient Descent (§5.3.2) to find the (skeleton, n_parts) pair
    that minimises the cost model.

    Returns the best (skeleton, n_parts) found.
    """
    d = data.shape[1]

    # Step 1: heuristic initialisation
    skeleton, n_parts = initialise_skeleton(data)
    # Re-fit functional mappings on actual data
    for dim, strat in enumerate(skeleton):
        if strat.kind == StrategyKind.FUNCTIONAL:
            fitted = fit_functional_mapping(data, dim, strat.other_dim)
            if fitted is not None:
                skeleton[dim] = fitted

    best_skeleton = list(skeleton)
    best_parts    = list(n_parts)
    best_cost     = estimate_cost(data, queries, skeleton, n_parts, w0, w1)

    for iteration in range(max_iter):
        improved = False

        # Step 2: gradient descent over n_parts
        for dim in range(d):
            if n_parts[dim] == 0:
                continue

            cost_minus = estimate_cost(
                data, queries, skeleton,
                [p if i != dim else max(min_parts, p - 1) for i, p in enumerate(n_parts)],
                w0, w1)
            cost_plus  = estimate_cost(
                data, queries, skeleton,
                [p if i != dim else min(max_parts, p + 1) for i, p in enumerate(n_parts)],
                w0, w1)

            grad = cost_plus - cost_minus
            if grad > 0:   # decreasing parts reduces cost
                n_parts[dim] = max(min_parts, n_parts[dim] - int(lr) - 1)
            elif grad < 0: # increasing parts reduces cost
                n_parts[dim] = min(max_parts, n_parts[dim] + int(lr) + 1)

        current_cost = estimate_cost(data, queries, skeleton, n_parts, w0, w1)
        if current_cost < best_cost:
            best_cost     = current_cost
            best_skeleton = list(skeleton)
            best_parts    = list(n_parts)
            improved      = True

        # Step 3: local search over skeletons (one hop)
        for dim in range(d):
            for new_skel, new_parts in _neighbours(skeleton, n_parts, dim, d):
                # Re-fit FM parameters
                for d2, s2 in enumerate(new_skel):
                    if s2.kind == StrategyKind.FUNCTIONAL:
                        fitted = fit_functional_mapping(data, d2, s2.other_dim)
                        if fitted is not None:
                            new_skel[d2] = fitted
                        else:
                            new_skel[d2] = DimStrategy(StrategyKind.INDEPENDENT)
                            new_parts[d2] = 4

                c = estimate_cost(data, queries, new_skel, new_parts, w0, w1)
                if c < best_cost:
                    best_cost     = c
                    best_skeleton = list(new_skel)
                    best_parts    = list(new_parts)
                    skeleton      = list(new_skel)
                    n_parts       = list(new_parts)
                    improved      = True
                    break

        if not improved:
            break

    return best_skeleton, best_parts


# ============================================================
# Full Tsunami-style index: Grid Tree + Augmented Grid per leaf
# ============================================================

# (Minimal Grid Tree re-implementation to keep this file self-contained)

@dataclass
class _GTNode:
    lo: np.ndarray
    hi: np.ndarray
    point_idx: Optional[np.ndarray] = None
    split_dim: int = -1
    split_vals: list[float] = field(default_factory=list)
    children: list["_GTNode"] = field(default_factory=list)
    is_leaf: bool = True
    aug_grid: Optional[AugmentedGrid] = None


def _gt_split(node: _GTNode, data: np.ndarray, queries: list[Query],
              depth: int, max_depth: int = 4,
              min_pts: int = 200):
    if depth >= max_depth or len(node.point_idx) < min_pts:
        return
    d = data.shape[1]
    best_dim, best_val = -1, None
    best_score = -1.0

    for dim in range(d):
        vals = data[node.point_idx, dim]
        mid  = np.median(vals)
        # simple skew proxy: std of query coverage on each side
        left_q  = [q for q in queries if q.ranges[dim][0] < mid]
        right_q = [q for q in queries if q.ranges[dim][1] > mid]
        score = abs(len(left_q) - len(right_q))
        if best_dim == -1 or score < best_score:
            best_score = score
            best_dim   = dim
            best_val   = mid

    if best_dim == -1:
        return

    node.split_dim  = best_dim
    node.split_vals = [best_val]
    node.is_leaf    = False

    lo_arr = data[node.point_idx, best_dim]
    left_mask  = lo_arr <  best_val
    right_mask = lo_arr >= best_val

    for mask, side in [(left_mask, "L"), (right_mask, "R")]:
        child_lo = node.lo.copy()
        child_hi = node.hi.copy()
        if side == "L":
            child_hi[best_dim] = best_val
        else:
            child_lo[best_dim] = best_val
        child_pts = node.point_idx[mask]
        child_q   = [q for q in queries
                     if not (q.ranges[best_dim][1] <= child_lo[best_dim] or
                             q.ranges[best_dim][0] >= child_hi[best_dim])]
        child = _GTNode(lo=child_lo, hi=child_hi, point_idx=child_pts)
        node.children.append(child)
        _gt_split(child, data, child_q, depth + 1, max_depth, min_pts)


class TsunamiIndex:
    """
    Minimal Tsunami index: Grid Tree + one Augmented Grid per leaf region.

    Usage
    -----
    idx = TsunamiIndex()
    idx.build(data, queries)
    results = idx.query(q)   # returns array of matching row indices
    """

    def __init__(self, max_gt_depth: int = 3, min_pts_per_leaf: int = 200,
                 agd_iter: int = 10):
        self.max_gt_depth     = max_gt_depth
        self.min_pts_per_leaf = min_pts_per_leaf
        self.agd_iter         = agd_iter
        self.root: Optional[_GTNode] = None
        self._data: Optional[np.ndarray] = None

    def build(self, data: np.ndarray, queries: list[Query]):
        self._data = data
        N, d = data.shape
        lo = data.min(axis=0).astype(float)
        hi = data.max(axis=0).astype(float) + 1e-9
        self.root = _GTNode(lo=lo, hi=hi, point_idx=np.arange(N))
        _gt_split(self.root, data, queries, 0,
                  self.max_gt_depth, self.min_pts_per_leaf)
        self._attach_grids(self.root, data, queries)

    def _attach_grids(self, node: _GTNode, data: np.ndarray,
                      queries: list[Query]):
        if node.is_leaf:
            pts = data[node.point_idx]
            if len(pts) < 4:
                return
            # Only keep queries that intersect this leaf region
            leaf_q = [q for q in queries if self._intersects(node, q)]
            if not leaf_q:
                leaf_q = queries[:10]   # fallback sample
            skel, parts = optimise_augmented_grid(
                pts, leaf_q, max_iter=self.agd_iter)
            # Re-index: AugmentedGrid is built on local pts; we remap later
            node.aug_grid = AugmentedGrid(pts, skel, parts)
            return
        for child in node.children:
            self._attach_grids(child, data, queries)

    @staticmethod
    def _intersects(node: _GTNode, q: Query) -> bool:
        for dim in range(len(node.lo)):
            if q.ranges[dim][1] <= node.lo[dim] or \
               q.ranges[dim][0] >= node.hi[dim]:
                return False
        return True

    def query(self, q: Query) -> np.ndarray:
        if self.root is None or self._data is None:
            return np.array([], dtype=np.int64)
        results = []
        self._traverse(self.root, q, results)
        if not results:
            return np.array([], dtype=np.int64)
        candidates = np.concatenate(results)
        # Final predicate check
        data = self._data
        mask = np.ones(len(candidates), dtype=bool)
        for dim in range(data.shape[1]):
            lo_q, hi_q = q.ranges[dim]
            mask &= (data[candidates, dim] >= lo_q)
            mask &= (data[candidates, dim] <= hi_q)
        return candidates[mask]

    def _traverse(self, node: _GTNode, q: Query, results: list):
        if not self._intersects(node, q):
            return
        if node.is_leaf:
            if node.aug_grid is not None:
                # AugmentedGrid was built on local pts → remap to global
                local_q = Query([
                    (max(q.ranges[dim][0], node.lo[dim]),
                     min(q.ranges[dim][1], node.hi[dim]))
                    for dim in range(len(node.lo))
                ])
                local_idx = node.aug_grid.query(local_q)
                if len(local_idx) > 0:
                    results.append(node.point_idx[local_idx])
            else:
                results.append(node.point_idx)
            return
        for child in node.children:
            self._traverse(child, q, results)


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    np.random.seed(0)

    # -------------------------------------------------------
    # Dataset: 5000 points, 2-D, with strong linear correlation
    # dim0 = uniform, dim1 = 0.8*dim0 + noise
    # -------------------------------------------------------
    N = 5_000
    dim0 = np.random.uniform(0, 100, N)
    dim1 = 0.8 * dim0 + np.random.normal(0, 5, N)
    dim1 = np.clip(dim1, 0, 100)
    data = np.column_stack([dim0, dim1])

    # -------------------------------------------------------
    # Queries: skewed — many narrow queries in upper region
    # -------------------------------------------------------
    queries = (
        [Query([(np.random.uniform(60, 90), np.random.uniform(65, 95)),
                (np.random.uniform(60, 90), np.random.uniform(65, 95))])
         for _ in range(80)]
        +
        [Query([(np.random.uniform(0, 100), np.random.uniform(5, 100)),
                (np.random.uniform(0, 100), np.random.uniform(5, 100))])
         for _ in range(20)]
    )

    print("=" * 55)
    print("1. Standalone AugmentedGrid (one region, full data)")
    print("=" * 55)
    skel, parts = initialise_skeleton(data)
    print("Initial skeleton :", [str(s) for s in skel])
    print("Initial n_parts  :", parts)

    skel_opt, parts_opt = optimise_augmented_grid(
        data, queries, max_iter=8)
    print("Optimised skeleton:", [str(s) for s in skel_opt])
    print("Optimised n_parts :", parts_opt)

    ag = AugmentedGrid(data, skel_opt, parts_opt)

    test_q = Query([(65.0, 85.0), (50.0, 75.0)])
    candidates = ag.query(test_q)

    # Brute-force ground truth
    mask_gt = ((data[:, 0] >= 65) & (data[:, 0] <= 85) &
               (data[:, 1] >= 50) & (data[:, 1] <= 75))
    gt = np.where(mask_gt)[0]

    # Filter candidates to actual matches
    if len(candidates) > 0:
        match_mask = ((data[candidates, 0] >= 65) & (data[candidates, 0] <= 85) &
                      (data[candidates, 1] >= 50) & (data[candidates, 1] <= 75))
        true_hits = candidates[match_mask]
    else:
        true_hits = np.array([], dtype=np.int64)

    recall    = len(np.intersect1d(true_hits, gt)) / (len(gt) + 1e-9)
    precision = len(true_hits) / (len(candidates) + 1e-9)
    print(f"\nQuery [(65,85), (50,75)]:")
    print(f"  Candidates scanned : {len(candidates)}")
    print(f"  True matches       : {len(gt)}")
    print(f"  Recall             : {recall:.3f}  (target=1.0)")
    print(f"  Precision          : {precision:.3f}")

    print("\n" + "=" * 55)
    print("2. Full TsunamiIndex (Grid Tree + Augmented Grid)")
    print("=" * 55)
    idx = TsunamiIndex(max_gt_depth=2, min_pts_per_leaf=300, agd_iter=5)
    idx.build(data, queries)
    print("Index built successfully.")

    results = idx.query(test_q)
    recall2 = len(np.intersect1d(results, gt)) / (len(gt) + 1e-9)
    print(f"\nQuery [(65,85), (50,75)] via TsunamiIndex:")
    print(f"  Returned results : {len(results)}")
    print(f"  True matches     : {len(gt)}")
    print(f"  Recall           : {recall2:.3f}  (target=1.0)")
