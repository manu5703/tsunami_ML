"""
Grid Tree Implementation - from Tsunami paper
"Tsunami: A Learned Multi-dimensional Index for Correlated Data and Skewed Workloads"
Ding et al., PVLDB 2021

The Grid Tree partitions d-dimensional data space into non-overlapping regions
to reduce query skew, using Earth Mover's Distance (EMD) as the skew metric.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from sklearn.cluster import DBSCAN


@dataclass
class Query:
    """A d-dimensional range query: each entry is (lo, hi) per dimension."""
    ranges: list[tuple[float, float]]

    @property
    def ndim(self):
        return len(self.ranges)


@dataclass
class GridTreeNode:
    """One node in the Grid Tree."""
    region_lo: np.ndarray
    region_hi: np.ndarray
    split_dim: Optional[int] = None
    split_values: list[float] = field(default_factory=list)
    children: list["GridTreeNode"] = field(default_factory=list)
    is_leaf: bool = True
    point_indices: Optional[np.ndarray] = None


def emd_1d(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """
    Compute EMD between two 1-D histograms of the same length.
    For equal-length 1-D histograms this equals the L1 distance
    of their CDFs (Wasserstein-1).
    Both histograms should sum to the same total mass.
    """
    h1 = hist1 / (hist1.sum() + 1e-12)
    h2 = hist2 / (hist2.sum() + 1e-12)
    return float(np.abs(np.cumsum(h1) - np.cumsum(h2)).sum())


def build_histogram(queries: list[Query], dim: int,
                    lo: float, hi: float, n_bins: int = 128) -> np.ndarray:
    """
    Build a histogram of query mass over [lo, hi) in dimension `dim`.
    Each query contributes 1/m mass to each of the m bins it intersects.
    """
    bins = np.zeros(n_bins, dtype=float)
    width = (hi - lo) / n_bins

    for q in queries:
        qlo, qhi = q.ranges[dim]
        qlo = max(qlo, lo)
        qhi = min(qhi, hi)
        if qlo >= qhi:
            continue
        b_start = int((qlo - lo) / width)
        b_end   = int((qhi - lo) / width)
        b_start = max(0, min(b_start, n_bins - 1))
        b_end   = max(0, min(b_end,   n_bins - 1))
        m = b_end - b_start + 1
        bins[b_start:b_end + 1] += 1.0 / m

    return bins


def compute_skew(queries: list[Query], dim: int,
                 lo: float, hi: float, n_bins: int = 128) -> float:
    """Skew = EMD(uniform, empirical query PDF) over [lo, hi) in dimension dim."""
    if not queries:
        return 0.0
    hist = build_histogram(queries, dim, lo, hi, n_bins)
    uniform = np.full(n_bins, hist.sum() / n_bins)
    return emd_1d(uniform, hist)


def cluster_query_types(queries: list[Query]) -> list[list[Query]]:
    """
    Cluster queries into types using DBSCAN on per-dimension selectivity
    embeddings, as described in §4.3.1 of the paper.
    """
    if not queries:
        return []

    ndim = queries[0].ndim
    embeddings = []
    for q in queries:
        sel = [(q.ranges[d][1] - q.ranges[d][0]) for d in range(ndim)]
        embeddings.append(sel)

    X = np.array(embeddings, dtype=float)

    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=0.0)
    
    xmin = np.min(X, axis=0)
    xmax = np.max(X, axis=0)
    rng = xmax - xmin + 1e-12
    
    X_norm = (X - xmin) / rng

    labels = DBSCAN(eps=0.2, min_samples=1).fit_predict(X_norm)

    clusters: dict[int, list[Query]] = {}
    for q, lbl in zip(queries, labels):
        clusters.setdefault(lbl, []).append(q)
    return list(clusters.values())


def find_best_split_values(queries: list[Query], dim: int,
                           lo: float, hi: float,
                           n_bins: int = 128,
                           merge_tol: float = 0.10) -> tuple[float, list[float]]:
    """
    Build the skew tree and use dynamic programming to find split values
    that minimise total skew in dimension `dim` over [lo, hi).

    Returns (skew_reduction, split_values).
    """
    hist = build_histogram(queries, dim, lo, hi, n_bins)
    total_skew = compute_skew(queries, dim, lo, hi, n_bins)

    n_leaves = n_bins // 2

    def range_skew(start: int, end: int) -> float:
        if end <= start + 1:
            return 0.0
        sub = hist[start:end]
        if sub.sum() == 0:
            return 0.0
        uniform = np.full(len(sub), sub.sum() / len(sub))
        return emd_1d(uniform, sub)

    inf = float("inf")
    min_skew = [inf] * (n_leaves + 1)
    best_split = [None] * (n_leaves + 1)
    min_skew[n_leaves] = 0.0

    for i in range(n_leaves - 1, -1, -1):
        for j in range(i + 2, n_leaves + 1):
            cost = range_skew(i * 2, j * 2) + min_skew[j]
            if cost < min_skew[i]:
                min_skew[i] = cost
                best_split[i] = j

    boundaries = []
    cur = 0
    while cur < n_leaves:
        nxt = best_split[cur]
        if nxt is None or nxt >= n_leaves:
            break
        boundaries.append(nxt)
        cur = nxt

    bin_width = (hi - lo) / n_bins
    split_vals = [lo + b * 2 * bin_width for b in boundaries]

    split_vals = _merge_splits(split_vals, queries, dim, lo, hi,
                               merge_tol, hist, n_bins)

    skew_reduction = total_skew - min_skew[0]
    return max(0.0, skew_reduction), split_vals


def _merge_splits(splits: list[float], queries: list[Query], dim: int,
                  lo: float, hi: float, tol: float,
                  hist: np.ndarray, n_bins: int) -> list[float]:
    """Merge adjacent segments if combined skew is within tol of their sum."""
    if len(splits) <= 1:
        return splits

    boundaries = [lo] + splits + [hi]
    bin_width = (hi - lo) / n_bins

    def seg_skew(a: float, b: float) -> float:
        i = int((a - lo) / bin_width)
        j = int((b - lo) / bin_width)
        sub = hist[i:j]
        if sub.sum() == 0:
            return 0.0
        u = np.full(len(sub), sub.sum() / len(sub))
        return emd_1d(u, sub)

    merged = True
    while merged and len(boundaries) > 2:
        merged = False
        new_bounds = [boundaries[0]]
        i = 0
        while i < len(boundaries) - 2:
            s1 = seg_skew(boundaries[i],     boundaries[i + 1])
            s2 = seg_skew(boundaries[i + 1], boundaries[i + 2])
            s12 = seg_skew(boundaries[i],    boundaries[i + 2])
            if s12 <= (s1 + s2) * (1 + tol):
                new_bounds.append(boundaries[i + 2])
                i += 2
                merged = True
            else:
                new_bounds.append(boundaries[i + 1])
                i += 1
        if i == len(boundaries) - 2:
            new_bounds.append(boundaries[-1])
        boundaries = new_bounds

    return boundaries[1:-1]


class GridTree:
    """
    Grid Tree as described in §4 of the Tsunami paper.

    Parameters
    ----------
    min_skew_reduction : float
        Stop splitting if skew reduction falls below this fraction of |Q|.
    min_points_frac : float
        Stop splitting if region has fewer than this fraction of total points.
    min_queries_frac : float
        Stop splitting if region has fewer than this fraction of total queries.
    n_bins : int
        Number of histogram bins used to approximate the query PDF.
    max_depth : int
        Hard cap on tree depth.
    """

    def __init__(self,
                 min_skew_reduction: float = 0.05,
                 min_points_frac: float = 0.01,
                 min_queries_frac: float = 0.01,
                 n_bins: int = 128,
                 max_depth: int = 10):
        self.min_skew_reduction = min_skew_reduction
        self.min_points_frac    = min_points_frac
        self.min_queries_frac   = min_queries_frac
        self.n_bins             = n_bins
        self.max_depth          = max_depth
        self.root: Optional[GridTreeNode] = None
        self._total_points = 0
        self._total_queries = 0


    def build(self, data: np.ndarray, queries: list[Query]):
        """
        Build the Grid Tree from `data` (shape [N, d]) and a list of queries.
        """
        N, d = data.shape
        self._total_points  = N
        self._total_queries = len(queries)

        lo = data.min(axis=0).astype(float)
        hi = data.max(axis=0).astype(float) + 1e-9

        self.root = GridTreeNode(
            region_lo=lo,
            region_hi=hi,
            point_indices=np.arange(N),
        )
        self._split(self.root, data, queries, depth=0)

    def query(self, q: Query) -> list[np.ndarray]:
        """
        Traverse the Grid Tree and return point-index arrays from all leaf
        regions that intersect the query.
        """
        results = []
        self._traverse(self.root, q, results)
        return results

    def leaves(self) -> list[GridTreeNode]:
        """Return all leaf nodes."""
        out = []
        self._collect_leaves(self.root, out)
        return out


    def _split(self, node: GridTreeNode, data: np.ndarray,
               queries: list[Query], depth: int):
        """Recursively split a node."""
        if depth >= self.max_depth:
            return

        n_pts = len(node.point_indices) if node.point_indices is not None else 0
        n_qry = len(queries)

        if (n_pts  < self.min_points_frac  * self._total_points or
                n_qry  < self.min_queries_frac * self._total_queries):
            return

        query_types = cluster_query_types(queries)

        d = len(node.region_lo)
        best_dim, best_reduction, best_splits = -1, 0.0, []

        for dim in range(d):
            lo = node.region_lo[dim]
            hi = node.region_hi[dim]

            total_reduction = 0.0
            all_splits: list[float] = []
            for qt in query_types:
                red, splits = find_best_split_values(
                    qt, dim, lo, hi, self.n_bins)
                total_reduction += red
                all_splits = splits

            if total_reduction > best_reduction:
                best_reduction = total_reduction
                best_dim       = dim
                best_splits    = all_splits

        threshold = self.min_skew_reduction * n_qry
        if best_dim == -1 or best_reduction < threshold or not best_splits:
            return

        node.split_dim    = best_dim
        node.split_values = best_splits
        node.is_leaf      = False

        boundaries = ([node.region_lo[best_dim]]
                      + best_splits
                      + [node.region_hi[best_dim]])

        pts = data[node.point_indices]

        for k in range(len(boundaries) - 1):
            child_lo = node.region_lo.copy()
            child_hi = node.region_hi.copy()
            child_lo[best_dim] = boundaries[k]
            child_hi[best_dim] = boundaries[k + 1]

            mask = ((pts[:, best_dim] >= boundaries[k]) &
                    (pts[:, best_dim] <  boundaries[k + 1]))
            child_pts = node.point_indices[mask]

            child_queries = [
                q for q in queries
                if (q.ranges[best_dim][0] < boundaries[k + 1] and
                    q.ranges[best_dim][1] > boundaries[k])
            ]

            child = GridTreeNode(
                region_lo=child_lo,
                region_hi=child_hi,
                point_indices=child_pts,
            )
            node.children.append(child)
            self._split(child, data, child_queries, depth + 1)

    def _traverse(self, node: GridTreeNode, q: Query,
                  results: list[np.ndarray]):
        if node is None:
            return
        for dim in range(len(node.region_lo)):
            if (q.ranges[dim][1] <= node.region_lo[dim] or
                    q.ranges[dim][0] >= node.region_hi[dim]):
                return

        if node.is_leaf:
            if node.point_indices is not None and len(node.point_indices) > 0:
                results.append(node.point_indices)
            return

        for child in node.children:
            self._traverse(child, q, results)

    def _collect_leaves(self, node: Optional[GridTreeNode],
                        out: list[GridTreeNode]):
        if node is None:
            return
        if node.is_leaf:
            out.append(node)
        for child in node.children:
            self._collect_leaves(child, out)


    def print_tree(self, node: Optional[GridTreeNode] = None,
                   indent: int = 0):
        if node is None:
            node = self.root
        prefix = "  " * indent
        n_pts = (len(node.point_indices)
                 if node.point_indices is not None else 0)
        if node.is_leaf:
            print(f"{prefix}[LEAF] pts={n_pts} "
                  f"lo={np.round(node.region_lo,2)} "
                  f"hi={np.round(node.region_hi,2)}")
        else:
            print(f"{prefix}[NODE] split_dim={node.split_dim} "
                  f"splits={np.round(node.split_values,2)} pts={n_pts}")
            for child in node.children:
                self.print_tree(child, indent + 1)


if __name__ == "__main__":
    np.random.seed(42)

    N, D = 10_000, 2
    data = np.random.uniform(0, 100, size=(N, D))

    qr = [Query([(np.random.uniform(0, 90), np.random.uniform(10, 100)),
                 (np.random.uniform(0, 90), np.random.uniform(10, 100))])
          for _ in range(100)]

    qg = [Query([(np.random.uniform(70, 95), np.random.uniform(75, 100)),
                 (np.random.uniform(70, 95), np.random.uniform(75, 100))])
          for _ in range(100)]

    all_queries = qr + qg

    gt = GridTree(
        min_skew_reduction=0.03,
        min_points_frac=0.01,
        min_queries_frac=0.01,
        n_bins=64,
        max_depth=4,
    )
    gt.build(data, all_queries)

    print("=== Grid Tree Structure ===")
    gt.print_tree()

    leaves = gt.leaves()
    print(f"\nTotal leaf regions: {len(leaves)}")
    for i, lf in enumerate(leaves):
        n = len(lf.point_indices) if lf.point_indices is not None else 0
        print(f"  Region {i}: {n} points  "
              f"lo={np.round(lf.region_lo,1)}  hi={np.round(lf.region_hi,1)}")

    test_query = Query([(75.0, 95.0), (75.0, 95.0)])
    candidate_sets = gt.query(test_query)
    all_candidates = (np.concatenate(candidate_sets)
                      if candidate_sets else np.array([], dtype=int))

    mask = ((data[:, 0] >= 75) & (data[:, 0] <= 95) &
            (data[:, 1] >= 75) & (data[:, 1] <= 95))
    ground_truth = np.where(mask)[0]

    print(f"\n=== Query [(75,95), (75,95)] ===")
    print(f"  Grid Tree candidates : {len(all_candidates)}")
    print(f"  True matches         : {len(ground_truth)}")
    recall = (len(np.intersect1d(all_candidates, ground_truth))
              / (len(ground_truth) + 1e-9))
    print(f"  Recall               : {recall:.3f}  (should be 1.0)")
