"""
Tsunami — Cost Model & Adaptive Gradient Descent Optimizer
===========================================================
Paper: "Tsunami: A Learned Multi-dimensional Index for Correlated Data
        and Skewed Workloads", Ding et al., PVLDB 2021  (§5.3)

This module provides:

  CostModel          — §5.3.1  Calibrated linear cost model
                         Time = w0*(#cell_ranges) + w1*(#scanned_pts)*(#filtered_dims)
                         Weights w0, w1 are fitted from micro-benchmarks on the
                         actual machine, then used to evaluate (skeleton, n_parts)
                         configurations without running every query.

  SkeletonNeighbours — §5.3.2  Enumerate all skeletons one "hop" away from
                         the current skeleton (change one dim's strategy).

  AGDOptimizer       — §5.3.2  Adaptive Gradient Descent that jointly optimises
                         the skeleton S (discrete) and partition counts P (continuous)
                         using the cost model as the objective.

  optimise           — Convenience entry point.  Given data + queries, returns
                         the best (skeleton, n_parts) found.

Dependencies:
    pip install numpy scipy scikit-learn
"""

from __future__ import annotations

import time
import math
import numpy as np
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional
# Import shared types from augmented_grid so both modules use the same enum
# objects — this prevents comparison failures inside AugmentedGrid._build.
from augmented_grid import (
    StrategyKind, DimStrategy, Skeleton,
    fit_functional_mapping, initialise_skeleton,
)


# ─────────────────────────────────────────────────────────────────────────────
# Query type (local — only used within this module)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Query:
    """A d-dimensional range query: each filter is (lo, hi) per dimension."""
    ranges: list[tuple[float, float]]

    @property
    def ndim(self) -> int:
        return len(self.ranges)


# ─────────────────────────────────────────────────────────────────────────────
# §5.3.1  Cost Model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CostModelWeights:
    """
    Calibrated weights for the linear cost model.

    w0 : time (seconds) to process one cell-range boundary lookup +
         cache-miss penalty for jumping to a new storage range.
    w1 : time (seconds) to scan one point across one filtered dimension.
    """
    w0: float = 1.0e-6   # ~1 µs per cell-range boundary (cache miss)
    w1: float = 2.0e-9   # ~2 ns per point per dimension (L1/L2 scan)


class CostModel:
    """
    §5.3.1  Linear cost model used to predict average query time for a
    given (skeleton, n_parts) configuration without running real queries.

        predicted_time(q) = w0 * R(q) + w1 * S(q) * F(q)

    where
        R(q) = number of cell ranges (contiguous runs in physical storage)
        S(q) = number of scanned points  (estimated via a data sample)
        F(q) = number of filtered (non-mapped) dimensions

    Parameters
    ----------
    data       : full dataset, shape (N, d)
    queries    : sample query workload
    weights    : CostModelWeights (calibrated or default)
    sample_frac: fraction of queries used for cost evaluation (for speed)
    """

    def __init__(self,
                 data:        np.ndarray,
                 queries:     list[Query],
                 weights:     Optional[CostModelWeights] = None,
                 sample_frac: float = 0.30):
        self.data        = data
        self.queries     = queries
        self.weights     = weights or CostModelWeights()
        self.sample_frac = sample_frac
        self._sample_idx = self._draw_sample()
        self._call_count = 0          # tracks how many evaluations we've done

    # ── public ───────────────────────────────────────────────────────────────

    def predict(self, skeleton: Skeleton, n_parts: list[int]) -> float:
        """
        Return the average predicted query time (seconds) for (skeleton, n_parts).
        Lower is better.
        """
        self._call_count += 1
        sample_queries = [self.queries[i] for i in self._sample_idx]
        total = 0.0
        for q in sample_queries:
            r, s, f = self._features(q, skeleton, n_parts)
            total += self.weights.w0 * r + self.weights.w1 * s * f
        return total / max(1, len(sample_queries))

    def calibrate(self,
                  skeleton:  Skeleton,
                  n_parts:   list[int],
                  n_repeats: int = 5) -> CostModelWeights:
        """
        Fit w0 and w1 from actual query timings on the current machine.

        Runs two micro-benchmarks:
          bench_w0 : many tiny queries that each touch exactly 1 point
                     → isolates per-cell-range overhead
          bench_w1 : a few wide queries that scan many points
                     → isolates per-point scan cost

        Returns the fitted CostModelWeights and updates self.weights.
        """
        # We need the AugmentedGrid to run real timings.
        # Import lazily to avoid circular dependency in production.
        try:
            from augmented_grid import AugmentedGrid
        except ImportError:
            print("[CostModel.calibrate] augmented_grid not found — "
                  "using default weights.")
            return self.weights

        ag = AugmentedGrid(self.data, skeleton, n_parts)

        # ── bench_w0: point queries (1 cell each) ─────────────────────────
        point_queries = self._make_point_queries(20)
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            for pq in point_queries:
                ag.query(pq)
        t_point = (time.perf_counter() - t0) / (n_repeats * len(point_queries))

        # Expected features for a point query: R≈1, S≈small, F=active dims
        f_active = sum(1 for dim, s in enumerate(skeleton)
                       if s.is_active() and n_parts[dim] > 0)
        # Estimate S for point queries
        s_point = max(1, len(self.data) / math.prod(
            max(1, n_parts[dim]) for dim, s in enumerate(skeleton)
            if s.is_active() and n_parts[dim] > 0) or 1)

        # ── bench_w1: wide queries (many points, few ranges) ──────────────
        wide_queries = self._make_wide_queries(10)
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            for wq in wide_queries:
                ag.query(wq)
        t_wide = (time.perf_counter() - t0) / (n_repeats * len(wide_queries))

        # Wide queries: R≈1 (nearly one big range), S ≈ many points
        s_wide = len(self.data) * 0.5   # ~50% selectivity for a "wide" query

        # Solve 2×2 system:
        #   t_point = w0 * 1       + w1 * s_point * f_active
        #   t_wide  = w0 * 1       + w1 * s_wide  * f_active
        # → w1 = (t_wide - t_point) / ((s_wide - s_point) * f_active)
        # → w0 = t_point - w1 * s_point * f_active
        denom = (s_wide - s_point) * max(1, f_active)
        if abs(denom) < 1e-15:
            return self.weights   # degenerate; keep defaults

        w1 = max(1e-12, (t_wide - t_point) / denom)
        w0 = max(1e-12, t_point - w1 * s_point * f_active)

        self.weights = CostModelWeights(w0=w0, w1=w1)
        return self.weights

    def prediction_error(self,
                         skeleton: Skeleton,
                         n_parts:  list[int]) -> dict:
        """
        Measure the relative error between the model's prediction and the
        actual average query time on the full query sample.

        Returns a dict with keys: predicted, actual, relative_error.
        """
        try:
            from augmented_grid import AugmentedGrid
        except ImportError:
            return {"error": "augmented_grid not available"}

        ag      = AugmentedGrid(self.data, skeleton, n_parts)
        sample  = [self.queries[i] for i in self._sample_idx]

        predicted_times = []
        actual_times    = []
        for q in sample:
            r, s, f = self._features(q, skeleton, n_parts)
            predicted_times.append(self.weights.w0 * r + self.weights.w1 * s * f)

            t0 = time.perf_counter()
            ag.query(q)
            actual_times.append(time.perf_counter() - t0)

        pred_avg   = float(np.mean(predicted_times))
        actual_avg = float(np.mean(actual_times))
        rel_err    = abs(pred_avg - actual_avg) / max(actual_avg, 1e-12)

        return {
            "predicted_avg_s": pred_avg,
            "actual_avg_s":    actual_avg,
            "relative_error":  rel_err,
        }

    # ── private ──────────────────────────────────────────────────────────────

    def _features(self,
                  q:        Query,
                  skeleton: Skeleton,
                  n_parts:  list[int]) -> tuple[float, float, int]:
        """
        Analytically compute (R, S, F) for a query given (skeleton, n_parts).

        R  = estimated number of cell-range boundaries
        S  = estimated number of scanned points
        F  = number of filtered (non-mapped) dimensions
        """
        d     = self.data.shape[1]
        N     = len(self.data)

        # F: non-mapped active dimensions
        F = sum(1 for dim in range(d)
                if skeleton[dim].is_active() and n_parts[dim] > 0)

        # Estimate fraction of cells that intersect the query
        frac = 1.0
        for dim in range(d):
            strat = skeleton[dim]
            p     = n_parts[dim]
            if p < 1 or not strat.is_active():
                continue

            qlo, qhi   = q.ranges[dim]
            dlo        = float(self.data[:, dim].min())
            dhi        = float(self.data[:, dim].max()) + 1e-9
            domain_len = dhi - dlo

            if strat.kind == StrategyKind.FUNCTIONAL:
                # mapped dim: after applying FM the effective range shrinks
                # approximate: use the FM error bounds to tighten the filter
                eff_lo, eff_hi = self._apply_fm_approx(strat, qlo, qhi)
                q_len = max(0.0, eff_hi - eff_lo)
            else:
                q_len = max(0.0, min(qhi, dhi) - max(qlo, dlo))

            part_hit_frac = min(1.0, q_len / domain_len + 1.0 / p)
            frac *= part_hit_frac

        # S: expected scanned points
        S = max(1.0, frac * N)

        # R: expected number of cell ranges
        # Each intersected cell can be a separate range; in practice cells that
        # are adjacent in the last active dimension merge into one range.
        # Approximate: #ranges ≈ product of intersecting partitions in all dims
        # except the innermost sort dimension.
        R = max(1.0, S / max(1, N / max(1, math.prod(
            max(1, n_parts[dim])
            for dim in range(d)
            if skeleton[dim].is_active() and n_parts[dim] > 0
        ) or 1)))
        # simple lower-bound: at least 1 range per distinct row in higher dims
        R = max(1.0, R)

        return R, S, F

    @staticmethod
    def _apply_fm_approx(strat: DimStrategy,
                         y_lo: float, y_hi: float) -> tuple[float, float]:
        """Apply a functional mapping to produce a tighter X range."""
        x_lo = strat.lr_slope * y_lo + strat.lr_intercept
        x_hi = strat.lr_slope * y_hi + strat.lr_intercept
        if x_lo > x_hi:
            x_lo, x_hi = x_hi, x_lo
        return x_lo - strat.el, x_hi + strat.eu

    def _draw_sample(self) -> list[int]:
        n = max(1, int(len(self.queries) * self.sample_frac))
        rng = np.random.default_rng(42)
        return rng.choice(len(self.queries), size=n, replace=False).tolist()

    def _make_point_queries(self, n: int) -> list[Query]:
        """Generate n near-point queries by sampling data points."""
        rng  = np.random.default_rng(0)
        idxs = rng.choice(len(self.data), n, replace=False)
        eps  = (self.data.max(axis=0) - self.data.min(axis=0)) * 0.001 + 1e-9
        return [
            Query([(self.data[i, d] - eps[d], self.data[i, d] + eps[d])
                   for d in range(self.data.shape[1])])
            for i in idxs
        ]

    def _make_wide_queries(self, n: int) -> list[Query]:
        """Generate n queries that each cover ~50% of each dimension's range."""
        rng   = np.random.default_rng(1)
        lo    = self.data.min(axis=0)
        hi    = self.data.max(axis=0)
        span  = hi - lo
        queries = []
        for _ in range(n):
            starts = lo + rng.random(self.data.shape[1]) * span * 0.5
            ends   = starts + span * 0.5
            queries.append(Query([(float(s), float(e))
                                  for s, e in zip(starts, ends)]))
        return queries


# ─────────────────────────────────────────────────────────────────────────────
# §5.3.2  Skeleton neighbourhood
# ─────────────────────────────────────────────────────────────────────────────

class SkeletonNeighbours:
    """
    Generate all skeletons that differ from the current one by exactly one
    dimension's strategy ("one hop"), as described in §5.3.2.

    Restrictions (from §5.2):
      • A target dimension (of a FUNCTIONAL mapping) cannot itself be mapped.
      • A base dimension (of a CONDITIONAL CDF) cannot be mapped or conditional.
      • A dimension cannot be its own target/base.
    """

    def __init__(self, data: np.ndarray):
        self.data = data
        self.d    = data.shape[1]

    def neighbours(self,
                   skeleton: Skeleton,
                   n_parts:  list[int],
                   dim:      int,
                   default_parts: int = 8
                   ) -> list[tuple[Skeleton, list[int]]]:
        """Return all (skeleton, n_parts) pairs one hop from current on `dim`."""
        result    = []
        current   = skeleton[dim]

        # Which dims are currently used as FM targets / CCDF bases?
        fm_targets   = {s.other_dim for s in skeleton
                        if s.kind == StrategyKind.FUNCTIONAL and s.other_dim is not None}
        ccdf_bases   = {s.other_dim for s in skeleton
                        if s.kind == StrategyKind.CONDITIONAL and s.other_dim is not None}

        candidates: list[DimStrategy] = []

        # ── 1. INDEPENDENT ────────────────────────────────────────────────
        if current.kind != StrategyKind.INDEPENDENT:
            candidates.append(DimStrategy(StrategyKind.INDEPENDENT))

        # ── 2. FUNCTIONAL to each eligible target ─────────────────────────
        for y in range(self.d):
            if y == dim:
                continue
            # target cannot be a mapped dim itself
            if skeleton[y].kind == StrategyKind.FUNCTIONAL:
                continue
            # don't re-propose current strategy
            if current.kind == StrategyKind.FUNCTIONAL and current.other_dim == y:
                continue
            strat = fit_functional_mapping(self.data, dim, y)
            if strat is not None:
                candidates.append(strat)

        # ── 3. CONDITIONAL on each eligible base ──────────────────────────
        for y in range(self.d):
            if y == dim:
                continue
            # base cannot itself be mapped or conditional
            if skeleton[y].kind in (StrategyKind.FUNCTIONAL,
                                    StrategyKind.CONDITIONAL):
                continue
            # don't re-propose current strategy
            if current.kind == StrategyKind.CONDITIONAL and current.other_dim == y:
                continue
            candidates.append(DimStrategy(StrategyKind.CONDITIONAL, other_dim=y))

        # ── Build (skeleton, n_parts) pairs ──────────────────────────────
        for cand in candidates:
            new_skel  = list(skeleton)
            new_skel[dim] = cand
            new_parts = list(n_parts)
            if cand.kind == StrategyKind.FUNCTIONAL:
                new_parts[dim] = 0          # mapped dims have no own partitions
            elif new_parts[dim] == 0:
                new_parts[dim] = default_parts   # restore
            result.append((new_skel, new_parts))

        return result

    def all_neighbours(self,
                       skeleton: Skeleton,
                       n_parts:  list[int]
                       ) -> list[tuple[Skeleton, list[int], int]]:
        """
        Return all one-hop neighbours across all dimensions.
        Each entry is (skeleton, n_parts, changed_dim).
        """
        out = []
        for dim in range(self.d):
            for skel, parts in self.neighbours(skeleton, n_parts, dim):
                out.append((skel, parts, dim))
        return out


# ─────────────────────────────────────────────────────────────────────────────
# §5.3.2  Adaptive Gradient Descent
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AGDConfig:
    """Hyperparameters for the AGD optimiser."""
    max_iter:        int   = 30      # maximum outer iterations
    min_parts:       int   = 2       # minimum partitions per active dim
    max_parts:       int   = 64      # maximum partitions per active dim
    part_step:       int   = 2       # step size for partition gradient
    convergence_tol: float = 1e-4    # stop if relative improvement < tol
    sample_frac:     float = 0.30    # fraction of queries used per eval
    verbose:         bool  = False   # print iteration log


@dataclass
class AGDResult:
    """Returned by AGDOptimizer.run()."""
    skeleton:      Skeleton
    n_parts:       list[int]
    best_cost:     float
    initial_cost:  float
    n_iterations:  int
    n_cost_evals:  int
    history:       list[float] = field(default_factory=list)

    @property
    def improvement(self) -> float:
        """Relative cost improvement over the initial configuration."""
        if self.initial_cost < 1e-15:
            return 0.0
        return (self.initial_cost - self.best_cost) / self.initial_cost


class AGDOptimizer:
    """
    §5.3.2  Adaptive Gradient Descent

    Jointly optimises the discrete skeleton S and continuous partition
    counts P to minimise average predicted query time.

    Algorithm outline (per iteration):
      Step 2  — Gradient descent over P (holding S fixed).
                For each active dimension, try P[dim] ± step and accept
                the direction that lowers predicted cost.
      Step 3  — Local skeleton search (holding P fixed).
                Try all one-hop neighbour skeletons; accept the best one
                if it improves cost.
      Repeat until convergence or max_iter reached.

    Parameters
    ----------
    data    : dataset, shape (N, d)
    queries : sample query workload
    config  : AGDConfig hyperparameters
    weights : CostModelWeights; if None, defaults are used
    """

    def __init__(self,
                 data:    np.ndarray,
                 queries: list[Query],
                 config:  Optional[AGDConfig]          = None,
                 weights: Optional[CostModelWeights]   = None):
        self.data     = data
        self.queries  = queries
        self.config   = config  or AGDConfig()
        self.nbrs     = SkeletonNeighbours(data)
        self.model    = CostModel(data, queries,
                                  weights=weights,
                                  sample_frac=self.config.sample_frac)

    # ── public ───────────────────────────────────────────────────────────────

    def run(self,
            init_skeleton: Optional[Skeleton]   = None,
            init_n_parts:  Optional[list[int]]  = None
            ) -> AGDResult:
        """
        Run AGD from an initial (skeleton, n_parts).

        If init_skeleton / init_n_parts are None, heuristic initialisation
        is used (§5.3.2 step 1).
        """
        cfg = self.config

        # ── Step 1: initialise ────────────────────────────────────────────
        if init_skeleton is None or init_n_parts is None:
            skeleton, n_parts = initialise_skeleton(self.data)
            # re-fit FM parameters on actual data
            for dim, strat in enumerate(skeleton):
                if strat.kind == StrategyKind.FUNCTIONAL:
                    fitted = fit_functional_mapping(
                        self.data, dim, strat.other_dim)
                    skeleton[dim] = fitted if fitted else DimStrategy(
                        StrategyKind.INDEPENDENT)
                    if skeleton[dim].kind == StrategyKind.INDEPENDENT:
                        n_parts[dim] = cfg.min_parts * 2
        else:
            skeleton = deepcopy(init_skeleton)
            n_parts  = list(init_n_parts)

        best_cost    = self.model.predict(skeleton, n_parts)
        initial_cost = best_cost
        history      = [best_cost]

        if cfg.verbose:
            print(f"[AGD] iter=0  cost={best_cost:.6e}  "
                  f"skel={[str(s) for s in skeleton]}  parts={n_parts}")

        prev_cost = float("inf")

        for iteration in range(1, cfg.max_iter + 1):

            # ── Step 2: gradient descent over P ──────────────────────────
            skeleton, n_parts, best_cost = self._gradient_step_P(
                skeleton, n_parts, best_cost)

            # ── Step 3: local skeleton search ────────────────────────────
            skeleton, n_parts, best_cost = self._local_search_S(
                skeleton, n_parts, best_cost)

            history.append(best_cost)

            if cfg.verbose:
                print(f"[AGD] iter={iteration:2d}  cost={best_cost:.6e}  "
                      f"skel={[str(s) for s in skeleton]}  parts={n_parts}")

            # ── convergence check ─────────────────────────────────────────
            rel_improvement = (prev_cost - best_cost) / max(prev_cost, 1e-15)
            if rel_improvement < cfg.convergence_tol and iteration > 2:
                if cfg.verbose:
                    print(f"[AGD] converged at iter={iteration} "
                          f"(rel_improvement={rel_improvement:.2e})")
                break
            prev_cost = best_cost

        return AGDResult(
            skeleton=skeleton,
            n_parts=n_parts,
            best_cost=best_cost,
            initial_cost=initial_cost,
            n_iterations=iteration,
            n_cost_evals=self.model._call_count,
            history=history,
        )

    def run_with_naive_init(self) -> AGDResult:
        """
        AGD-NI variant: start from all-independent skeleton (§6.6).
        Useful as a robustness check.
        """
        d        = self.data.shape[1]
        skel_ni  = [DimStrategy(StrategyKind.INDEPENDENT) for _ in range(d)]
        parts_ni = [self.config.min_parts * 2] * d
        return self.run(init_skeleton=skel_ni, init_n_parts=parts_ni)

    # ── private ──────────────────────────────────────────────────────────────

    def _gradient_step_P(self,
                         skeleton:   Skeleton,
                         n_parts:    list[int],
                         best_cost:  float
                         ) -> tuple[Skeleton, list[int], float]:
        """
        Numerical gradient descent over partition counts P (Step 2).

        For each active dimension:
          - evaluate cost at P[dim] - step  (if ≥ min_parts)
          - evaluate cost at P[dim] + step  (if ≤ max_parts)
          - move in whichever direction is cheaper; stay if neither helps.
        """
        cfg      = self.config
        n_parts  = list(n_parts)   # work on a copy

        for dim in range(self.data.shape[1]):
            if not skeleton[dim].is_active() or n_parts[dim] == 0:
                continue

            p_cur = n_parts[dim]

            # Cost at p - step
            if p_cur - cfg.part_step >= cfg.min_parts:
                p_lo          = list(n_parts)
                p_lo[dim]     = p_cur - cfg.part_step
                cost_lo       = self.model.predict(skeleton, p_lo)
            else:
                cost_lo       = float("inf")

            # Cost at p + step
            if p_cur + cfg.part_step <= cfg.max_parts:
                p_hi          = list(n_parts)
                p_hi[dim]     = p_cur + cfg.part_step
                cost_hi       = self.model.predict(skeleton, p_hi)
            else:
                cost_hi       = float("inf")

            best_local = min(best_cost, cost_lo, cost_hi)
            if best_local < best_cost:
                best_cost = best_local
                if cost_lo <= cost_hi:
                    n_parts[dim] = p_cur - cfg.part_step
                else:
                    n_parts[dim] = p_cur + cfg.part_step

        return skeleton, n_parts, best_cost

    def _local_search_S(self,
                        skeleton:   Skeleton,
                        n_parts:    list[int],
                        best_cost:  float
                        ) -> tuple[Skeleton, list[int], float]:
        """
        One-hop local search over skeletons (Step 3).

        Iterates through all one-hop neighbours.  Accepts the first
        neighbour that strictly improves cost (first-improvement).
        Re-fits FM parameters for any FUNCTIONAL strategies in the
        candidate skeleton.
        """
        for cand_skel, cand_parts, changed_dim in \
                self.nbrs.all_neighbours(skeleton, n_parts):

            # Re-fit FM parameters on actual data for any functional mappings
            for dim, strat in enumerate(cand_skel):
                if strat.kind == StrategyKind.FUNCTIONAL:
                    fitted = fit_functional_mapping(
                        self.data, dim, strat.other_dim)
                    if fitted is not None:
                        cand_skel[dim] = fitted
                    else:
                        # FM not viable after re-fit → fall back to INDEPENDENT
                        cand_skel[dim]   = DimStrategy(StrategyKind.INDEPENDENT)
                        cand_parts[dim]  = self.config.min_parts * 2

            c = self.model.predict(cand_skel, cand_parts)
            if c < best_cost:
                return cand_skel, cand_parts, c

        return skeleton, n_parts, best_cost


# ─────────────────────────────────────────────────────────────────────────────
# Convenience entry point
# ─────────────────────────────────────────────────────────────────────────────

def optimise(data:    np.ndarray,
             queries: list[Query],
             config:  Optional[AGDConfig]        = None,
             weights: Optional[CostModelWeights] = None,
             verbose: bool = False
             ) -> AGDResult:
    """
    Full Tsunami optimisation pipeline for one region:

      1. Calibrate cost-model weights on the current machine (optional).
      2. Run AGD from heuristic initialisation.
      3. Return AGDResult with best (skeleton, n_parts).

    Parameters
    ----------
    data    : region data, shape (N, d)
    queries : queries that intersect this region
    config  : AGDConfig  (None → sensible defaults)
    weights : pre-calibrated CostModelWeights  (None → use defaults)
    verbose : print iteration log
    """
    if config is None:
        config = AGDConfig(verbose=verbose)
    else:
        config.verbose = verbose

    optimizer = AGDOptimizer(data, queries, config=config, weights=weights)
    return optimizer.run()


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(0)
    print("=" * 60)
    print("Tsunami Cost Model + AGD Optimizer — demo")
    print("=" * 60)

    # ── Dataset: 3-D, 8000 points ──────────────────────────────────────────
    # dim0: uniform
    # dim1: strongly correlated with dim0  (tight linear → FM should fire)
    # dim2: loosely correlated with dim0   (CCDF expected)
    N = 8_000
    dim0 = np.random.uniform(0, 100, N)
    dim1 = 0.85 * dim0 + np.random.normal(0, 2,  N)   # tight
    dim2 = 0.50 * dim0 + np.random.normal(0, 15, N)   # loose
    dim1 = np.clip(dim1, 0, 100)
    dim2 = np.clip(dim2, 0, 100)
    data = np.column_stack([dim0, dim1, dim2])

    # ── Workload: skewed — many narrow queries in high-value region ─────────
    rng = np.random.default_rng(42)
    queries = (
        # narrow queries in upper region (skewed)
        [Query([(float(lo), float(lo + 10)),
                (float(lo), float(lo + 10)),
                (float(lo), float(lo + 10))])
         for lo in rng.uniform(60, 90, 80)]
        +
        # wide queries across the full space
        [Query([(float(lo), float(lo + 40)),
                (float(lo), float(lo + 40)),
                (float(lo), float(lo + 40))])
         for lo in rng.uniform(0, 60, 40)]
    )

    # ── 1. Cost Model stand-alone ───────────────────────────────────────────
    print("\n── 1. Cost Model feature estimation ──")
    skel0, parts0 = initialise_skeleton(data)
    print(f"  Initial skeleton : {[str(s) for s in skel0]}")
    print(f"  Initial n_parts  : {parts0}")

    model = CostModel(data, queries, sample_frac=0.4)
    cost0 = model.predict(skel0, parts0)
    print(f"  Predicted cost   : {cost0:.4e} s/query")
    print(f"  Cost model calls : {model._call_count}")

    # ── 2. Run full AGD ─────────────────────────────────────────────────────
    print("\n── 2. Adaptive Gradient Descent ──")
    cfg    = AGDConfig(max_iter=25, part_step=2, verbose=True)
    result = optimise(data, queries, config=cfg)

    print(f"\n  Best skeleton    : {[str(s) for s in result.skeleton]}")
    print(f"  Best n_parts     : {result.n_parts}")
    print(f"  Initial cost     : {result.initial_cost:.4e} s/query")
    print(f"  Best cost        : {result.best_cost:.4e} s/query")
    print(f"  Improvement      : {result.improvement * 100:.1f}%")
    print(f"  Iterations       : {result.n_iterations}")
    print(f"  Cost model calls : {result.n_cost_evals}")

    # ── 3. AGD-NI (naive init) vs AGD ──────────────────────────────────────
    print("\n── 3. AGD-NI (naive skeleton init) ──")
    optimizer = AGDOptimizer(data, queries,
                             config=AGDConfig(max_iter=25, verbose=False))
    result_ni = optimizer.run_with_naive_init()
    print(f"  AGD-NI best cost : {result_ni.best_cost:.4e} s/query")
    print(f"  AGD-NI skeleton  : {[str(s) for s in result_ni.skeleton]}")

    winner = "AGD" if result.best_cost <= result_ni.best_cost else "AGD-NI"
    print(f"  Winner           : {winner}")

    # ── 4. Neighbour enumeration sanity check ───────────────────────────────
    print("\n── 4. Skeleton neighbourhood (dim 1) ──")
    nbrs = SkeletonNeighbours(data)
    hops = nbrs.neighbours(result.skeleton, result.n_parts, dim=1)
    print(f"  Current strategy for dim1 : {result.skeleton[1]}")
    print(f"  One-hop neighbours        : {len(hops)}")
    for skel_n, parts_n in hops[:3]:
        c = model.predict(skel_n, parts_n)
        print(f"    {[str(s) for s in skel_n]}  parts={parts_n}  "
              f"cost={c:.4e}")
    if len(hops) > 3:
        print(f"    ... ({len(hops) - 3} more)")

    # ── 5. Cost history ─────────────────────────────────────────────────────
    print("\n── 5. Cost history ──")
    for i, c in enumerate(result.history):
        bar = "█" * int(40 * c / max(result.history))
        print(f"  iter {i:2d}  {c:.4e}  {bar}")
