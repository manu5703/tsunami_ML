"""
Batch Query Benchmark
=====================
Builds the Tsunami index using a TRAINING workload file, then runs
two separate TEST query files against all four methods:

  Tsunami | NumPy | KDTree | Brute Force

Usage:
    python batch_test.py -w bay_area_queries.txt --group-a nearby_queries.txt --group-b far_queries.txt
    python batch_test.py housing.csv -w bay_area_queries.txt --group-a nearby_queries.txt --group-b far_queries.txt
    python batch_test.py housing.csv -w bay_area_queries.txt --group-a nearby_queries.txt --group-b far_queries.txt -n 100000

Arguments:
    -w / --workload   Training queries used to BUILD the Tsunami index
    --group-a         Test queries for Group A (in-distribution)
    --group-b         Test queries for Group B (out-of-distribution)
    csv               (optional) CSV/Parquet file — omit for synthetic data
    -n                (optional) load only first N rows
"""

import sys, os, time, csv, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
sys.path.insert(0, '.')

from query_cli import (
    generate_california_real, generate_nyc_taxi_real, generate_covtype,
    load_csv, build_index,
    load_workload, parse_query,
    numpy_query, kdtree_query, timed, format_value,
)
from sklearn.neighbors import KDTree


def load_test_queries(path):
    """Read SQL lines from a file, skip blank lines and # comments."""
    queries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            queries.append(line)
    return queries


def shorten(sql, width=50):
    return (sql[:width - 1] + '…') if len(sql) > width else sql


def _close(a, b):
    if isinstance(a, float) and np.isnan(a): return False
    if isinstance(b, float) and np.isnan(b): return False
    return abs(a - b) / max(abs(b), 1e-9) < 1e-4


def run_group(label, queries, data, col_names, idx, tree):
    N = len(data)
    rows = []

    W = 140
    print(f"\n{'═' * W}")
    print(f"  {label}  ({len(queries)} queries)")
    print(f"{'═' * W}")
    print(f"  {'#':>3}  {'Query':<52}  "
          f"{'Tsunami':>9} {'Ans-TS':>12}  "
          f"{'NumPy':>9} {'Ans-NP':>12}  "
          f"{'BruteF':>9} {'Ans-BF':>12}  "
          f"{'Speedup':>8}  {'Scan%':>6}  {'Match':>5}")
    print(f"  {'-' * (W-2)}")

    for i, sql in enumerate(queries, 1):
        try:
            q, agg_fn, _ = parse_query(sql, data, col_names)
        except Exception as e:
            print(f"  {i:>3}  PARSE ERROR: {e}")
            continue

        try:
            (r_ts, t_ts) = timed(lambda: idx.query(q))
            (r_np, t_np) = timed(lambda: numpy_query(data, q))
            if tree is not None:
                (r_kd, t_kd) = timed(lambda: kdtree_query(tree, data, q))
            else:
                r_kd, t_kd = None, float('nan')
            (r_bf, t_bf) = timed(lambda: idx.brute_force(q))
        except Exception as e:
            print(f"  {i:>3}  RUNTIME ERROR: {e}")
            continue

        speedup  = t_bf / t_ts if t_ts > 0 else 0
        scan_pct = r_ts.n_scanned / N * 100

        if scan_pct == 0.0:
            continue

        v_ts = format_value(r_ts.value,  agg_fn)
        v_np = format_value(r_np[0],     agg_fn)
        v_bf = format_value(r_bf[1],     agg_fn)

        np_ok  = _close(r_np[0],  r_ts.value)
        bf_ok  = _close(r_bf[1],  r_ts.value)
        match  = "PASS" if (np_ok and bf_ok) else "FAIL"

        print(f"  {i:>3}  {shorten(sql):<52}  "
              f"{t_ts:>8.2f}ms {v_ts:>12}  "
              f"{t_np:>8.2f}ms {v_np:>12}  "
              f"{t_bf:>8.2f}ms {v_bf:>12}  "
              f"{speedup:>7.1f}x  {scan_pct:>5.1f}%  {match:>5}")

        rows.append(dict(
            query=sql,
            t_ts=t_ts,    ans_ts=r_ts.value,
            t_np=t_np,    ans_np=r_np[0],
            t_kd=t_kd,
            t_bf=t_bf,    ans_bf=r_bf[1],
            speedup=speedup, scan_pct=scan_pct, match=match,
        ))

    if not rows:
        return rows

    def avg(key): return sum(r[key] for r in rows) / len(rows)
    def med(key):
        vals = sorted(r[key] for r in rows if not (isinstance(r[key], float) and r[key] != r[key]))
        return vals[len(vals) // 2] if vals else float('nan')

    print(f"\n  {'— SUMMARY —':<55}  {'Tsunami':>9}{'':>13}  {'NumPy':>9}{'':>13}  {'BruteF':>9}{'':>13}  {'Speedup':>8}  {'Scan%':>6}")
    print(f"  {'-' * (W-2)}")
    print(f"  {'Average':<55}  "
          f"{avg('t_ts'):>8.2f}ms {'':>12}  "
          f"{avg('t_np'):>8.2f}ms {'':>12}  "
          f"{avg('t_bf'):>8.2f}ms {'':>12}  "
          f"{avg('speedup'):>7.1f}x  {avg('scan_pct'):>5.1f}%")
    print(f"  {'Median':<55}  "
          f"{med('t_ts'):>8.2f}ms {'':>12}  "
          f"{med('t_np'):>8.2f}ms {'':>12}  "
          f"{med('t_bf'):>8.2f}ms {'':>12}  "
          f"{med('speedup'):>7.1f}x  {med('scan_pct'):>5.1f}%")

    return rows


def extract_location(sql):
    """Pull the place/neighbourhood/zone name out of a SQL string."""
    m = re.search(r"(?:place|neighbourhood|zone|tier|pricetier)\s*=\s*'([^']+)'", sql, re.IGNORECASE)
    return m.group(1) if m else "Unknown"


def plot_speedup(rows_a, rows_b, dataset_label, out_prefix):
    """
    For each group (nearby / far), draw one line per location showing
    speedup across that location's queries.  Also draw a combined
    avg-speedup comparison across all locations.
    """

    def group_by_location(rows):
        loc_map = {}
        for r in rows:
            loc = extract_location(r['query'])
            loc_map.setdefault(loc, []).append(r['speedup'])
        return loc_map

    loc_a = group_by_location(rows_a)
    loc_b = group_by_location(rows_b)

    def _plot_group(loc_map, group_name, color_start):
        if not loc_map:
            return
        locations = list(loc_map.keys())
        colors    = cm.tab10(np.linspace(color_start, color_start + 0.5, len(locations)))

        fig, ax = plt.subplots(figsize=(12, 5))
        for loc, color in zip(locations, colors):
            speedups = loc_map[loc]
            ax.plot(range(1, len(speedups) + 1), speedups,
                    marker='o', markersize=3, linewidth=1.5,
                    label=loc, color=color)

        ax.set_title(f"{dataset_label}  —  {group_name}  |  Speedup per Query by Location",
                     fontsize=13, fontweight='bold')
        ax.set_xlabel("Query index within location", fontsize=11)
        ax.set_ylabel("Speedup  (Brute Force / Tsunami)", fontsize=11)
        ax.axhline(1.0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = f"{out_prefix}_{group_name.replace(' ', '_').lower()}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"  Plot saved → {fname}")

    _plot_group(loc_a, "Group_A_Nearby",   0.0)
    _plot_group(loc_b, "Group_B_Far_Away", 0.5)

    # ── Combined bar chart: avg speedup per location, both groups ────────────
    all_locs    = sorted(set(list(loc_a.keys()) + list(loc_b.keys())))
    avg_a = [np.mean(loc_a.get(l, [0])) for l in all_locs]
    avg_b = [np.mean(loc_b.get(l, [0])) for l in all_locs]

    x      = np.arange(len(all_locs))
    width  = 0.35
    fig, ax = plt.subplots(figsize=(max(10, len(all_locs) * 1.1), 5))
    bars_a = ax.bar(x - width/2, avg_a, width, label='Nearby (Group A)',   color='steelblue')
    bars_b = ax.bar(x + width/2, avg_b, width, label='Far Away (Group B)', color='tomato')

    ax.set_title(f"{dataset_label}  —  Avg Speedup per Location  (Brute Force / Tsunami)",
                 fontsize=13, fontweight='bold')
    ax.set_xlabel("Location", fontsize=11)
    ax.set_ylabel("Avg Speedup", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(all_locs, rotation=30, ha='right', fontsize=9)
    ax.axhline(1.0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)

    for bar in bars_a:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.05,
                    f'{h:.1f}x', ha='center', va='bottom', fontsize=7)
    for bar in bars_b:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.05,
                    f'{h:.1f}x', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    fname = f"{out_prefix}_combined_avg_speedup.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Plot saved → {fname}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("csv",        nargs="?",
                    help="CSV/Parquet file (omit for synthetic California Housing)")
    ap.add_argument("-w", "--workload", required=True,
                    help="Training query file used to BUILD the Tsunami index  e.g. bay_area_queries.txt")
    ap.add_argument("--group-a", required=True,
                    help="Test query file for Group A (in-distribution)   e.g. nearby_queries.txt")
    ap.add_argument("--group-b", required=True,
                    help="Test query file for Group B (out-of-distribution) e.g. far_queries.txt")
    ap.add_argument("-n", "--nrows", type=int, default=0,
                    help="Load only first N rows")
    ap.add_argument("--dataset",
                    choices=["california_real", "nyc_taxi_real", "covtype"],
                    default="california_real",
                    help="Dataset: california_real, nyc_taxi_real, covtype")
    ap.add_argument("--csv-out", default=None,
                    help="Save all results to this CSV file  e.g. results.csv")
    args = ap.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.csv:
        print(f"Loading '{args.csv}'…")
        data, col_names = load_csv(args.csv, max_rows=args.nrows)
    elif args.dataset == "nyc_taxi_real":
        n = args.nrows if args.nrows > 0 else 2_000_000
        print(f"Loading real NYC Taxi ({n:,} rows)…")
        data, col_names = generate_nyc_taxi_real(n=n)
    elif args.dataset == "covtype":
        n = args.nrows if args.nrows > 0 else 2_000_000
        print(f"Loading Forest Covertype ({n:,} rows)…")
        data, col_names = generate_covtype(n=n)
    else:
        n = args.nrows if args.nrows > 0 else 2_000_000
        print(f"Loading real California Housing ({n:,} rows)…")
        data, col_names = generate_california_real(n=n)

    # ── Build index from TRAINING workload ────────────────────────────────────
    workload = []
    if args.workload and os.path.exists(args.workload):
        print(f"Loading training queries from '{args.workload}'…")
        workload = load_workload(args.workload, data, col_names)
    else:
        print("  (no training file found — building index without workload)")

    print("Building Tsunami index…")
    idx = build_index(data, col_names, workload)

    _KDTREE_LIMIT = 500_000
    if len(data) <= _KDTREE_LIMIT:
        print("Building KDTree…")
        tree = KDTree(data)
    else:
        print(f"  KDTree skipped ({len(data):,} rows > {_KDTREE_LIMIT:,} limit).")
        tree = None

    print("  Ready.\n")

    # ── Load TEST query files ─────────────────────────────────────────────────
    if not os.path.exists(args.group_a):
        print(f"ERROR: Group A file not found: '{args.group_a}'")
        sys.exit(1)
    if not os.path.exists(args.group_b):
        print(f"ERROR: Group B file not found: '{args.group_b}'")
        sys.exit(1)

    group_a_queries = load_test_queries(args.group_a)
    group_b_queries = load_test_queries(args.group_b)

    print(f"Test queries loaded:")
    print(f"  Group A ({args.group_a}): {len(group_a_queries)} queries")
    print(f"  Group B ({args.group_b}): {len(group_b_queries)} queries")

    # ── Run benchmarks ────────────────────────────────────────────────────────
    rows_a = run_group(
        f"GROUP A — NEARBY  [IN-DISTRIBUTION]  ({args.group_a})",
        group_a_queries, data, col_names, idx, tree,
    )
    rows_b = run_group(
        f"GROUP B — FAR AWAY  [OUT-OF-DISTRIBUTION]  ({args.group_b})",
        group_b_queries, data, col_names, idx, tree,
    )

    # ── Final side-by-side comparison ────────────────────────────────────────
    if rows_a and rows_b:
        def avg(rows, key):
            vals = [r[key] for r in rows if not (isinstance(r[key], float) and r[key] != r[key])]
            return sum(vals) / len(vals) if vals else float('nan')

        print(f"\n{'═' * 62}")
        print("  OVERALL COMPARISON  (avg across all queries in each group)")
        print(f"{'═' * 62}")
        print(f"  {'Metric':<32} {'Group A (Nearby)':>14} {'Group B (Far)':>14}")
        print(f"  {'-' * 60}")
        for key, label, fmt in [
            ('t_ts',     'Avg Tsunami time (ms)',      '{:.2f}'),
            ('t_np',     'Avg NumPy time (ms)',         '{:.2f}'),
            ('t_bf',     'Avg Brute Force time (ms)',   '{:.2f}'),
            ('speedup',  'Avg Speedup  BF / Tsunami',  '{:.1f}x'),
            ('scan_pct', 'Avg Scan %',                 '{:.1f}%'),
        ]:
            nv = avg(rows_a, key)
            fv = avg(rows_b, key)
            print(f"  {label:<32} {fmt.format(nv):>14} {fmt.format(fv):>14}")
        print(f"{'═' * 62}\n")

    # ── Write CSV ─────────────────────────────────────────────────────────────
    csv_path = args.csv_out or f"results_{args.dataset}.csv"
    fieldnames = [
        'dataset', 'group', 'query_no', 'query',
        'tsunami_ms', 'tsunami_answer',
        'numpy_ms',   'numpy_answer',
        'bruteforce_ms', 'bruteforce_answer',
        'speedup_x', 'scan_pct', 'match',
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        dataset_label = args.dataset
        for group_label, rows in [('A_nearby', rows_a), ('B_far', rows_b)]:
            for i, r in enumerate(rows, 1):
                writer.writerow({
                    'dataset':          dataset_label,
                    'group':            group_label,
                    'query_no':         i,
                    'query':            r['query'],
                    'tsunami_ms':       round(r['t_ts'], 4),
                    'tsunami_answer':   round(r['ans_ts'], 4) if isinstance(r['ans_ts'], float) else r['ans_ts'],
                    'numpy_ms':         round(r['t_np'], 4),
                    'numpy_answer':     round(r['ans_np'], 4) if isinstance(r['ans_np'], float) else r['ans_np'],
                    'bruteforce_ms':    round(r['t_bf'], 4),
                    'bruteforce_answer':round(r['ans_bf'], 4) if isinstance(r['ans_bf'], float) else r['ans_bf'],
                    'speedup_x':        round(r['speedup'], 2),
                    'scan_pct':         round(r['scan_pct'], 2),
                    'match':            r['match'],
                })
    print(f"  Results saved to '{csv_path}'")

    # ── Generate plots ────────────────────────────────────────────────────────
    if rows_a or rows_b:
        print("\nGenerating plots…")
        out_prefix = csv_path.replace('.csv', '') if csv_path.endswith('.csv') else csv_path
        plot_speedup(rows_a, rows_b, args.dataset, out_prefix)
