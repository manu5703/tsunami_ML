import sys, re, time, csv, os
import numpy as np
import pandas as pd
sys.path.insert(0, '.')

from tsunami_index import TsunamiIndex, TsunamiConfig, Query
from sklearn.neighbors import KDTree

def _ansi(c): return f"\033[{c}m"
BOLD=_ansi("1"); RESET=_ansi("0"); GREEN=_ansi("32"); CYAN=_ansi("36")
YELLOW=_ansi("33"); RED=_ansi("31"); DIM=_ansi("2")
def bold(s):   return f"{BOLD}{s}{RESET}"
def green(s):  return f"{GREEN}{s}{RESET}"
def cyan(s):   return f"{CYAN}{s}{RESET}"
def yellow(s): return f"{YELLOW}{s}{RESET}"
def dim(s):    return f"{DIM}{s}{RESET}"
def red(s):    return f"{RED}{s}{RESET}"


def _extract_datetime_features(df):
    for col in list(df.columns):
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col + '_hour']      = df[col].dt.hour.astype(np.float64)
            df[col + '_dayofweek'] = df[col].dt.dayofweek.astype(np.float64)
            df[col + '_month']     = df[col].dt.month.astype(np.float64)
            df = df.drop(columns=[col])
        elif df[col].dtype == object:
            try:
                parsed = pd.to_datetime(df[col], errors='raise')
                df[col + '_hour']      = parsed.dt.hour.astype(np.float64)
                df[col + '_dayofweek'] = parsed.dt.dayofweek.astype(np.float64)
                df[col + '_month']     = parsed.dt.month.astype(np.float64)
                df = df.drop(columns=[col])
            except Exception:
                df = df.drop(columns=[col])
    return df


def load_csv(path, max_rows=0):
    ext = os.path.splitext(path)[1].lower()
    name = os.path.basename(path)

    if ext == '.parquet':
        print(f"  Reading parquet '{name}'…")
        df = pd.read_parquet(path)
    elif ext in ('.xlsx', '.xls'):
        print(f"  Reading Excel '{name}'…")
        df = pd.read_excel(path)
    else:
        file_mb = os.path.getsize(path) / 1_048_576
        if file_mb > 500:
            print(f"  Reading CSV '{name}' ({file_mb:,.0f} MB) in chunks…")
            chunks = []
            rows   = 0
            for chunk in pd.read_csv(path, chunksize=500_000, low_memory=False):
                chunks.append(chunk)
                rows += len(chunk)
                print(f"    {rows:,} rows loaded…", end='\r')
                if max_rows and rows >= max_rows:
                    break
            print()
            df = pd.concat(chunks, ignore_index=True)
            if max_rows:
                df = df.iloc[:max_rows]
        else:
            df = pd.read_csv(path, low_memory=False)

    df = _extract_datetime_features(df)
    df = df.select_dtypes(include=[np.number])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='all')
    df = df.dropna()

    if df.empty:
        raise ValueError(f"No usable numeric columns in '{name}'")

    # Sanitize column names: remove dots/special chars so the SQL parser can handle them
    # e.g. 'PM2.5' → 'PM25'
    df.columns = [
        re.sub(r'[^A-Za-z0-9_]', '', col.replace(' ', '_').replace('-', '_')) or f'col_{i}'
        for i, col in enumerate(df.columns)
    ]

    # Drop columns that cause Tsunami's linregress to fail inside leaf regions:
    #   (a) near-constant: ≥75% of rows share one value (e.g. Is, Ir)
    #   (b) low-cardinality: <50 distinct values (e.g. year=5, month=12, hour=24)
    #       — a leaf covering one season can make such a column all-identical.
    dropped = []
    for col in list(df.columns):
        vc = df[col].value_counts(normalize=True)
        if len(vc) == 0 or vc.iloc[0] >= 0.75 or len(vc) < 50:
            df = df.drop(columns=[col])
            dropped.append(col)
    if dropped:
        print(f"  Dropped column(s): {', '.join(dropped)}")

    if df.empty:
        raise ValueError(f"No usable columns remain after filtering in '{name}'")

    print(f"  Loaded {len(df):,} rows x {len(df.columns)} columns from '{name}'")
    return df.values.astype(np.float64), list(df.columns)


def generate_california(n=200_000):
    rng = np.random.default_rng(42)
    med_inc    = np.clip(rng.lognormal(1.05, 0.65, n), 0.5, 15.0)
    house_age  = rng.uniform(1, 52, n)
    ave_rooms  = np.clip(rng.lognormal(1.7, 0.4, n), 0.8, 20.0)
    ave_bedrms = np.clip(ave_rooms / rng.uniform(3.5, 5.5, n), 0.3, 5.0)
    population = np.clip(rng.lognormal(6.5, 1.0, n), 3, 10_000).astype(float)
    ave_occup  = np.clip(rng.lognormal(1.1, 0.5, n), 0.7, 20.0)
    latitude   = rng.uniform(32.54, 41.95, n)
    longitude  = rng.uniform(-124.35, -114.31, n)
    coast      = np.exp(-0.3 * np.abs(longitude + 119))
    target     = np.clip(0.45*med_inc + 0.3*coast*med_inc
                         + 0.05*(52-house_age)/52 + rng.normal(0, 0.4, n), 0.15, 5.0)
    data = np.column_stack([med_inc, house_age, ave_rooms, ave_bedrms,
                            population, ave_occup, latitude, longitude, target])
    col_names = ['MedInc','HouseAge','AveRooms','AveBedrms',
                 'Population','AveOccup','Latitude','Longitude','MedHouseVal']
    mb = data.nbytes / 1_048_576
    print(f"  Loaded synthetic California Housing ({n:,} rows, {mb:,.0f} MB).")
    return data, col_names


def generate_nyc_taxi(n=200_000):
    rng = np.random.default_rng(7)

    # Pickup location — concentrated in Manhattan / Brooklyn / Queens
    pickup_lat = np.clip(rng.normal(40.730, 0.060, n), 40.60, 40.85)
    pickup_lon = np.clip(rng.normal(-73.980, 0.055, n), -74.05, -73.75)

    # Dropoff — correlated with pickup (short trips stay nearby)
    dropoff_lat = np.clip(pickup_lat + rng.normal(0.0, 0.025, n), 40.60, 40.85)
    dropoff_lon = np.clip(pickup_lon + rng.normal(0.0, 0.025, n), -74.05, -73.75)

    # Trip distance — log-normal, correlated with lat/lon displacement
    displacement = np.sqrt((dropoff_lat - pickup_lat)**2 + (dropoff_lon - pickup_lon)**2)
    trip_dist    = np.clip(displacement * 120 + rng.lognormal(0.5, 0.6, n), 0.1, 30.0)

    # Fare — strongly correlated with distance
    fare   = np.clip(2.5 + 2.5 * trip_dist + rng.normal(0, 1.5, n), 2.5, 150.0)

    # Tip — correlated with fare (NYC ~20% tip culture)
    tip    = np.clip(0.18 * fare + rng.normal(0, 0.8, n), 0.0, 40.0)

    # Passenger count — 1-6
    passengers = rng.integers(1, 7, n).astype(float)

    # Trip duration in minutes — correlated with distance
    duration = np.clip(trip_dist * 3.5 + rng.normal(0, 3.0, n), 1.0, 120.0)

    # Total amount
    total = np.clip(fare + tip + rng.uniform(0.3, 1.0, n), 2.5, 200.0)

    data = np.column_stack([
        pickup_lat, pickup_lon,
        dropoff_lat, dropoff_lon,
        trip_dist, fare, tip,
        passengers, duration, total,
    ])
    col_names = [
        'Latitude', 'Longitude',
        'dropoff_lat', 'dropoff_lon',
        'trip_distance', 'fare_amount', 'tip_amount',
        'passenger_count', 'duration_min', 'total_amount',
    ]
    mb = data.nbytes / 1_048_576
    print(f"  Loaded synthetic NYC Taxi ({n:,} rows, {mb:,.0f} MB).")
    return data, col_names



def _oversample(real_data, n, rng, jitter_scale=0.01):
    m = len(real_data)
    if n <= m:
        return real_data[rng.choice(m, n, replace=False)]
    repeats = (n // m) + 1
    tiled = np.tile(real_data, (repeats, 1))[:n]
    stds = real_data.std(axis=0)
    stds[stds == 0] = 1.0
    tiled += rng.normal(0, jitter_scale, tiled.shape) * stds
    return tiled


def generate_california_real(n=2_000_000):
    from sklearn.datasets import fetch_california_housing
    rng = np.random.default_rng(42)
    print("  Fetching real California Housing dataset from sklearn…")
    ds = fetch_california_housing()
    col_names = list(ds.feature_names) + ['MedHouseVal']
    real_data = np.column_stack([ds.data, ds.target])
    print(f"  Loaded real California Housing ({len(real_data):,} rows).")
    real_data = _oversample(real_data, n, rng)
    mb = real_data.nbytes / 1_048_576
    print(f"  Oversampled to {n:,} rows, {mb:,.0f} MB.")
    return real_data, col_names


def generate_nyc_taxi_real(n=2_000_000):
    rng = np.random.default_rng(7)
    # Look for a locally downloaded parquet file first
    for fname in sorted(os.listdir('.')):
        if ('tripdata' in fname.lower() or 'taxi' in fname.lower()) \
                and fname.endswith(('.parquet', '.csv')):
            print(f"  Loading local taxi file: {fname}")
            data, col_names = load_csv(fname)
            data = _oversample(data, n, rng)
            return data, col_names
    # Try downloading 2016 yellow taxi (last year with lat/lon columns)
    url = ("https://d37ci6vzurychx.cloudfront.net/trip-data/"
           "yellow_tripdata_2016-06.parquet")
    try:
        print("  Downloading NYC Yellow Taxi 2016-06 from TLC…")
        df = pd.read_parquet(url)
        rename = {
            'pickup_latitude':  'Latitude',
            'pickup_longitude': 'Longitude',
            'dropoff_latitude':  'dropoff_lat',
            'dropoff_longitude': 'dropoff_lon',
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        df = df.select_dtypes(include=[np.number])
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        # Drop near-constant / low-cardinality columns
        for col in list(df.columns):
            vc = df[col].value_counts(normalize=True)
            if len(vc) == 0 or vc.iloc[0] >= 0.75 or len(vc) < 50:
                df = df.drop(columns=[col])
        print(f"  Downloaded {len(df):,} rows, cols: {list(df.columns)}")
        real_data = df.values.astype(np.float64)
        col_names = list(df.columns)
        real_data = _oversample(real_data, n, rng)
        mb = real_data.nbytes / 1_048_576
        print(f"  Oversampled to {n:,} rows, {mb:,.0f} MB.")
        return real_data, col_names
    except Exception as e:
        raise RuntimeError(
            f"Could not download NYC Taxi data: {e}\n"
            "  Download a monthly parquet from:\n"
            "    https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page\n"
            "  (use a 2015–2016 file for lat/lon columns), then run:\n"
            "    python batch_test.py yellow_tripdata_2016-06.parquet "
            "-w nyc_training_queries.txt ..."
        )


def generate_covtype(n=2_000_000):
    from sklearn.datasets import fetch_covtype
    rng = np.random.default_rng(99)
    print("  Fetching Forest Covertype dataset from sklearn…")
    ds = fetch_covtype()
    col_names = [
        'Elevation', 'Aspect', 'Slope',
        'Horiz_Dist_Hydrology', 'Vert_Dist_Hydrology',
        'Horiz_Dist_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
        'Horiz_Dist_Fire_Points',
    ]
    real_data = ds.data[:, :10].astype(np.float64)   # first 10 = continuous features
    print(f"  Loaded real Forest Covertype ({len(real_data):,} rows).")
    real_data = _oversample(real_data, n, rng)
    mb = real_data.nbytes / 1_048_576
    print(f"  Oversampled to {n:,} rows, {mb:,.0f} MB.")
    return real_data, col_names


def build_index(data, col_names, workload=[]):
    use_agd = len(workload) > 0
    # Deeper tree + more AGD iterations when a training workload drives the build.
    # gt_max_depth=4 → up to 16 leaf regions, enough to isolate a 1-2% tight cluster.
    # gt_min_points_frac=0.005 → allows leaves as small as 0.5% of data.
    gt_depth  = 4 if use_agd else 2
    agd_iters = 10 if use_agd else 3
    cfg = TsunamiConfig(
        col_names=col_names,
        gt_max_depth=gt_depth,
        gt_min_points_frac=0.005,
        agd_max_iter=agd_iters,
        agd_enabled=use_agd, delta_enabled=False,
        shift_detection_enabled=False, verbose=False,
    )
    idx = TsunamiIndex(cfg)
    t0  = time.perf_counter()
    idx.build(data, workload)
    ms  = (time.perf_counter() - t0) * 1000
    print(f"  Tsunami index built in {ms:.0f} ms")
    return idx


_LOOKUP_TABLES = {}


def _register_lookup(keyword, handler, describe_fn):
    _LOOKUP_TABLES[keyword.lower()] = {"handler": handler, "describe": describe_fn}


try:
    from places import lookup as _place_lookup, list_places as _list_places

    def _place_handler(value, col_names, ranges):
        lat_col = next((c for c in col_names if c.lower() == 'latitude'),  None)
        lon_col = next((c for c in col_names if c.lower() == 'longitude'), None)
        if lat_col is None or lon_col is None:
            raise ParseError("Place filter requires 'Latitude' and 'Longitude' columns in this dataset.")
        try:
            lat_min, lat_max, lon_min, lon_max = _place_lookup(value)
        except KeyError as e:
            raise ParseError(str(e))
        ranges[lat_col][0] = max(ranges[lat_col][0], lat_min)
        ranges[lat_col][1] = min(ranges[lat_col][1], lat_max)
        ranges[lon_col][0] = max(ranges[lon_col][0], lon_min)
        ranges[lon_col][1] = min(ranges[lon_col][1], lon_max)

    def _place_describe():
        df = _list_places()
        print()
        print(f"  {'Place':<20} {'Lat min':>8} {'Lat max':>8} {'Lon min':>9} {'Lon max':>9}")
        print("  " + "-" * 60)
        for _, row in df.iterrows():
            print(f"  {row.Place:<20} {row.Lat_min:>8.2f} {row.Lat_max:>8.2f}"
                  f" {row.Lon_min:>9.2f} {row.Lon_max:>9.2f}")
        print()

    _register_lookup("place", _place_handler, _place_describe)

except ImportError:
    pass


try:
    from nyc_places import lookup as _nyc_lookup, list_places as _nyc_list

    def _nyc_handler(value, col_names, ranges):
        lat_col = next((c for c in col_names if c.lower() == 'latitude'),  None)
        lon_col = next((c for c in col_names if c.lower() == 'longitude'), None)
        if lat_col is None or lon_col is None:
            raise ParseError("Neighbourhood filter requires 'latitude' and 'longitude' columns.")
        try:
            lat_min, lat_max, lon_min, lon_max = _nyc_lookup(value)
        except KeyError as e:
            raise ParseError(str(e))
        ranges[lat_col][0] = max(ranges[lat_col][0], lat_min)
        ranges[lat_col][1] = min(ranges[lat_col][1], lat_max)
        ranges[lon_col][0] = max(ranges[lon_col][0], lon_min)
        ranges[lon_col][1] = min(ranges[lon_col][1], lon_max)

    def _nyc_describe():
        df = _nyc_list()
        print()
        print(f"  {'Neighbourhood':<22} {'Lat min':>8} {'Lat max':>8} {'Lon min':>9} {'Lon max':>9}")
        print("  " + "-" * 62)
        for _, row in df.iterrows():
            print(f"  {row.Place:<22} {row.Lat_min:>8.3f} {row.Lat_max:>8.3f}"
                  f" {row.Lon_min:>9.3f} {row.Lon_max:>9.3f}")
        print()

    _register_lookup("neighbourhood", _nyc_handler, _nyc_describe)

except ImportError:
    pass


try:
    from covtype_zones import lookup as _zone_lookup, list_zones as _zone_list

    def _zone_handler(value, col_names, ranges):
        try:
            zone_ranges = _zone_lookup(value)
        except KeyError as e:
            raise ParseError(str(e))
        for col, (lo, hi) in zone_ranges.items():
            if col in ranges:
                ranges[col][0] = max(ranges[col][0], lo)
                ranges[col][1] = min(ranges[col][1], hi)

    def _zone_describe():
        df = _zone_list()
        print()
        print(f"  {'Zone':<14} {'Elevation':>14} {'Slope':>10} {'% data':>7}  Description")
        print("  " + "-" * 90)
        for _, row in df.iterrows():
            print(f"  {row['Zone']:<14} {row['Elevation']:>14} {row['Slope']:>10} "
                  f"{row['% of data']:>7}  {row['Description']}")
        print()

    _register_lookup("zone", _zone_handler, _zone_describe)

except ImportError:
    pass


class ParseError(Exception): pass


def parse_query(sql, data, col_names):
    sql   = sql.strip().rstrip(';')
    upper = sql.upper()

    m = re.match(r'SELECT\s+(.+?)\s+FROM\s+\S+', sql, re.IGNORECASE)
    if not m:
        raise ParseError("Expected:  SELECT <AGG>(<col>|*) FROM <table> WHERE ...")
    sel = m.group(1).strip()

    agg_m = re.match(r'(COUNT|AVG|SUM|MIN|MAX)\s*\(\s*(\*|\w+)\s*\)', sel, re.IGNORECASE)
    if not agg_m:
        raise ParseError(f"Unknown aggregation '{sel}'. Use COUNT(*), AVG(col), SUM(col), MIN(col), MAX(col).")

    agg_fn       = agg_m.group(1).lower()
    agg_col_name = agg_m.group(2)

    if agg_fn == 'count':
        agg_col, agg_col_name = None, None
    else:
        if agg_col_name == '*':
            raise ParseError(f"{agg_fn.upper()}(*) is not valid — specify a column name.")
        match = next((c for c in col_names if c.lower() == agg_col_name.lower()), None)
        if match is None:
            raise ParseError(f"Unknown column '{agg_col_name}'. Available: {', '.join(col_names)}")
        agg_col_name = match
        agg_col      = col_names.index(agg_col_name)

    col_lo = {c: float(data[:, i].min()) for i, c in enumerate(col_names)}
    col_hi = {c: float(data[:, i].max()) for i, c in enumerate(col_names)}
    ranges = {c: [col_lo[c], col_hi[c]] for c in col_names}

    where_m = re.search(r'WHERE\s+(.+)$', sql, re.IGNORECASE)
    if where_m:
        for cond in _split_conditions(where_m.group(1).strip()):
            _apply_condition(cond.strip(), ranges, col_names)

    range_list = [(ranges[c][0], ranges[c][1]) for c in col_names]
    return Query(range_list, agg_fn=agg_fn, agg_col=agg_col), agg_fn, agg_col_name


def parse_row_query(sql, data, col_names):
    sql = sql.strip().rstrip(';')

    limit = 50
    m = re.search(r'\bLIMIT\s+(\d+)\b', sql, re.IGNORECASE)
    if m:
        limit = int(m.group(1))
        sql   = sql[:m.start()].strip()

    order_col  = None
    order_dir  = "ASC"
    order_cidx = None
    m = re.search(r'\bORDER\s+BY\s+(\w+)(?:\s+(ASC|DESC))?\b', sql, re.IGNORECASE)
    if m:
        order_col_raw = m.group(1)
        order_dir     = (m.group(2) or "ASC").upper()
        order_col     = _resolve_col(order_col_raw, col_names)
        order_cidx    = col_names.index(order_col)
        sql           = sql[:m.start()].strip()

    sel_m = re.match(r'SELECT\s+(.+?)\s+FROM\s+\S+', sql, re.IGNORECASE)
    if not sel_m:
        raise ParseError("Expected: SELECT * FROM data [WHERE ...] [ORDER BY col] [LIMIT N]")
    sel_raw = sel_m.group(1).strip()

    if sel_raw == '*':
        display_cols = col_names
        display_cidx = list(range(len(col_names)))
    else:
        raw_cols     = [c.strip() for c in sel_raw.split(',')]
        display_cols = [_resolve_col(c, col_names) for c in raw_cols]
        display_cidx = [col_names.index(c) for c in display_cols]

    col_lo = {c: float(data[:, i].min()) for i, c in enumerate(col_names)}
    col_hi = {c: float(data[:, i].max()) for i, c in enumerate(col_names)}
    ranges = {c: [col_lo[c], col_hi[c]] for c in col_names}

    where_m = re.search(r'\bWHERE\s+(.+)$', sql, re.IGNORECASE)
    if where_m:
        for cond in _split_conditions(where_m.group(1).strip()):
            _apply_condition(cond.strip(), ranges, col_names)

    mask = np.ones(len(data), dtype=bool)
    for i, c in enumerate(col_names):
        mask &= (data[:, i] >= ranges[c][0]) & (data[:, i] <= ranges[c][1])

    filtered = data[mask]

    if order_cidx is not None and len(filtered):
        sort_idx = np.argsort(filtered[:, order_cidx])
        if order_dir == "DESC":
            sort_idx = sort_idx[::-1]
        filtered = filtered[sort_idx]

    total_matched = len(filtered)
    filtered      = filtered[:limit]

    return (filtered[:, display_cidx], display_cols,
            order_col, order_dir, limit, total_matched, ranges)


def _split_conditions(where_str):
    placeholders = {}
    idx = [0]

    def save(m):
        key = f"__PH{idx[0]}__"
        placeholders[key] = m.group(0)
        idx[0] += 1
        return key

    s = where_str
    s = re.sub(r"\w+\s*=\s*['\"].+?['\"]", save, s, flags=re.IGNORECASE)
    s = re.sub(r'\w+\s+BETWEEN\s+[^\s]+\s+AND\s+[^\s]+', save, s, flags=re.IGNORECASE)

    parts = re.split(r'\bAND\b', s, flags=re.IGNORECASE)
    result = []
    for p in parts:
        for k, v in placeholders.items():
            p = p.replace(k, v)
        p = p.strip()
        if p:
            result.append(p)
    return result


def _apply_condition(cond, ranges, col_names):
    m = re.match(r"(\w+)\s*=\s*['\"](.+?)['\"]", cond, re.IGNORECASE)
    if m:
        keyword, value = m.group(1).lower(), m.group(2)
        if keyword in _LOOKUP_TABLES:
            _LOOKUP_TABLES[keyword]["handler"](value, col_names, ranges)
            return
        raise ParseError(f"Unknown lookup table keyword '{m.group(1)}'. "
                         f"Known: {', '.join(k.title() for k in _LOOKUP_TABLES)}")

    m = re.match(r'(\w+)\s+BETWEEN\s+([^\s]+)\s+AND\s+([^\s]+)', cond, re.IGNORECASE)
    if m:
        col = _resolve_col(m.group(1), col_names)
        lo, hi = float(m.group(2)), float(m.group(3))
        ranges[col][0] = max(ranges[col][0], lo)
        ranges[col][1] = min(ranges[col][1], hi)
        return

    m = re.match(r'(\w+)\s*(>=|<=|!=|>|<|=)\s*([^\s]+)', cond, re.IGNORECASE)
    if m:
        col, op, val = _resolve_col(m.group(1), col_names), m.group(2), float(m.group(3))
        if op in ('>=', '>'):  ranges[col][0] = max(ranges[col][0], val)
        if op in ('<=', '<'):  ranges[col][1] = min(ranges[col][1], val)
        if op == '=':          ranges[col][0] = ranges[col][1] = val
        return

    raise ParseError(f"Cannot parse condition: '{cond}'")


def _resolve_col(name, col_names):
    match = next((c for c in col_names if c.lower() == name.lower()), None)
    if match is None:
        raise ParseError(f"Unknown column '{name}'. Available: {', '.join(col_names)}")
    return match


def numpy_query(data, q):
    mask = np.ones(len(data), dtype=bool)
    for dim, (lo, hi) in enumerate(q.ranges):
        mask &= (data[:, dim] >= lo) & (data[:, dim] <= hi)
    filtered = data[mask]
    if len(filtered) == 0:
        return (0 if q.agg_fn == 'count' else float('nan')), 0
    fns = {'count': len, 'avg': lambda f: f[:, q.agg_col].mean(),
           'sum': lambda f: f[:, q.agg_col].sum(),
           'min': lambda f: f[:, q.agg_col].min(),
           'max': lambda f: f[:, q.agg_col].max()}
    return fns[q.agg_fn](filtered), len(filtered)


def kdtree_query(tree, data, q):
    lo = np.array([r[0] for r in q.ranges])
    hi = np.array([r[1] for r in q.ranges])
    radius = np.linalg.norm((hi - lo) / 2)
    cands  = tree.query_radius(((lo+hi)/2).reshape(1, -1), r=radius)[0]
    rows   = [data[i] for i in cands
              if all(lo[d] <= data[i,d] <= hi[d] for d in range(len(lo)))]
    filtered = np.array(rows) if rows else np.empty((0, data.shape[1]))
    if len(filtered) == 0:
        return (0 if q.agg_fn == 'count' else float('nan')), 0
    fns = {'count': len, 'avg': lambda f: f[:, q.agg_col].mean(),
           'sum': lambda f: f[:, q.agg_col].sum(),
           'min': lambda f: f[:, q.agg_col].min(),
           'max': lambda f: f[:, q.agg_col].max()}
    return fns[q.agg_fn](filtered), len(filtered)


def timed(fn, reps=3):
    t0 = time.perf_counter()
    for _ in range(reps): result = fn()
    return result, (time.perf_counter() - t0) / reps * 1000


def format_value(val, agg_fn):
    if agg_fn == 'count':        return f"{int(val):,}"
    if abs(val) >= 1_000_000:    return f"{val:,.2f}"
    if abs(val) >= 1_000:        return f"{val:,.2f}"
    return f"{val:.4f}"


def print_sql_result(agg_fn, agg_col_name, r_ts, r_np, r_kd, r_bf,
                     t_ts, t_np, t_kd, t_bf, n_matched, n_scanned, n_total):
    col_label = f"{agg_fn.upper()}({agg_col_name or '*'})"
    val_str   = format_value(r_ts.value, agg_fn)
    width     = max(len(col_label), len(val_str), 20)
    bar       = "+" + "-" * (width + 2) + "+"

    print()
    print(bar)
    print(f"| {bold(col_label):<{width + len(BOLD+RESET)}} |")
    print(bar)
    print(f"| {green(val_str):<{width + len(GREEN+RESET)}} |")
    print(bar)
    print(f"  {dim(f'{n_matched:,} row(s) matched  ·  scanned {n_scanned:,}/{n_total:,} ({n_scanned/n_total*100:.0f}%)')}")

    ref = r_ts.value
    methods = [("Tsunami",    t_ts, r_ts.value),
               ("NumPy",      t_np, r_np[0])]
    if r_kd is not None:
        methods.append(("KDTree", t_kd, r_kd[0]))
    methods.append(("Brute Force", t_bf, r_bf[1]))
    fastest = min(t for _, t, _ in methods)

    def close(a, b):
        if isinstance(a, float) and np.isnan(a): return False
        return abs(a - b) / max(abs(b), 1e-9) < 1e-4

    print()
    print(f"  {'Method':<14} {'Time':>9}  {'Result':>18}  Note")
    print("  " + "-" * 58)
    for name, t, v in methods:
        flag    = yellow(" <-- fastest") if t == fastest else ""
        v_str   = format_value(v, agg_fn) if not (isinstance(v, float) and np.isnan(v)) else "n/a"
        correct = green("PASS") if close(v, ref) else red("FAIL")
        print(f"  {name:<14} {t:>7.3f}ms  {v_str:>18}  {correct}{flag}")
    if r_kd is None:
        print(f"  {'KDTree':<14} {'n/a':>9}  {'(skipped—too large)':>18}")

    bf_su = t_bf / t_ts if t_ts > 0 else 0
    print()
    print(f"  {dim(f'Tsunami {bf_su:.1f}x faster than Brute Force')}")


def print_row_table(rows, col_names, order_col, order_dir, limit, total_matched):
    if len(rows) == 0:
        print(f"  {yellow('No rows matched.')}")
        return

    col_w = [max(len(c), 9) for c in col_names]
    for row in rows:
        for i, v in enumerate(row):
            col_w[i] = max(col_w[i], len(f"{v:.4g}"))

    sep  = "  +" + "+".join("-" * (w + 2) for w in col_w) + "+"
    head = "  |" + "|".join(f" {c:^{w}} " for c, w in zip(col_names, col_w)) + "|"
    print()
    print(sep)
    print(head)
    print(sep)
    for row in rows:
        cells = [f"{v:>{w}.4g}" for v, w in zip(row, col_w)]
        print("  |" + "|".join(f" {c} " for c in cells) + "|")
    print(sep)

    shown = len(rows)
    order_note = f"  ORDER BY {order_col} {order_dir}" if order_col else ""
    print(f"  {shown} of {total_matched:,} matched rows shown "
          f"(LIMIT {limit}){order_note}")
    print()


HELP_MSG = """
  SELECT COUNT(*)        FROM data WHERE <filters>
  SELECT AVG(col)        FROM data WHERE <filters>
  SELECT SUM(col)        FROM data WHERE <filters>
  SELECT MIN(col)        FROM data WHERE <filters>
  SELECT MAX(col)        FROM data WHERE <filters>

  SELECT * FROM data [WHERE <filters>] [ORDER BY col [ASC|DESC]] [LIMIT N]

  Filters:
    col BETWEEN lo AND hi
    col >= val   col <= val   col > val   col < val

  Commands:
    load <file.csv>        load a new CSV and rebuild the index
    workload <file.txt>    rebuild index with training queries
    columns                show all columns + min/max
    tables                 list all loaded tables
    places                 show California place lookup table
    neighbourhoods         show NYC neighbourhood lookup table
    zones                  show Covertype elevation/slope zone table
    help                   show this message
    quit                   exit
"""


def print_columns(data, col_names):
    print()
    print(f"  {'#':<4} {'Column':<18} {'Min':>12} {'Max':>12}")
    print("  " + "-" * 50)
    for i, c in enumerate(col_names):
        print(f"  {i:<4} {c:<18} {data[:,i].min():>12.4f} {data[:,i].max():>12.4f}")
    print()


def run_repl(init_data, init_col_names, init_idx, init_tree, init_source):
    state = {
        "data":      init_data,
        "col_names": init_col_names,
        "idx":       init_idx,
        "tree":      init_tree,
        "source":    init_source,
    }

    def header():
        N, D = state["data"].shape
        print(cyan(bold(f"\n  Tsunami Query CLI  —  {state['source']}")))
        print(dim(f"  {N:,} rows x {D} columns  ·  columns / tables / help / quit"))

    header()
    print()

    while True:
        try:
            sql = input(bold("tsunami> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye!")
            break

        if not sql:
            continue
        low = sql.lower()

        if low in ('quit', 'exit', 'q'):
            print("  Bye!")
            break

        if low == 'help':
            print(HELP_MSG)
            continue

        if low == 'columns':
            print_columns(state["data"], state["col_names"])
            continue

        if low == 'tables':
            print()
            print(f"  Table 1 (main dataset) — {state['source']}")
            N, D = state["data"].shape
            print(f"    {N:,} rows x {D} columns: {', '.join(state['col_names'])}")
            if _LOOKUP_TABLES:
                for i, (kw, tbl) in enumerate(_LOOKUP_TABLES.items(), 2):
                    print(f"  Table {i} (lookup) — {kw.title()} table")
                    print(f"    Type  '{kw}'  at the prompt to see contents.")
            print()
            continue

        if low in _LOOKUP_TABLES:
            _LOOKUP_TABLES[low]["describe"]()
            continue

        if low.startswith('workload '):
            wpath = sql[9:].strip().strip('"\'')
            if not os.path.exists(wpath):
                print(f"  {red('File not found:')} {wpath}\n")
                continue
            try:
                wl = load_workload(wpath, state["data"], state["col_names"])
                if not wl:
                    print(f"  {yellow('No valid queries found in file.')}\n")
                    continue
                print(f"  Rebuilding Tsunami index with {len(wl)} training queries…")
                state["idx"] = build_index(state["data"], state["col_names"], wl)
                print()
            except Exception as e:
                print(f"  {red('Error:')} {e}\n")
            continue

        if low.startswith('load '):
            path = sql[5:].strip().strip('"').strip("'")
            if not os.path.exists(path):
                print(f"  {red('File not found:')} {path}\n")
                continue
            try:
                mb = os.path.getsize(path) / 1_048_576
                print(f"  Loading '{path}' ({mb:,.0f} MB)…")
                new_data, new_cols = load_csv(path)
                print("  Building Tsunami index…")
                new_idx  = build_index(new_data, new_cols)
                if len(new_data) <= _KDTREE_ROW_LIMIT:
                    print("  Building KDTree…")
                    new_tree = KDTree(new_data)
                else:
                    print(f"  KDTree skipped ({len(new_data):,} rows).")
                    new_tree = None
                state.update(data=new_data, col_names=new_cols,
                             idx=new_idx, tree=new_tree,
                             source=os.path.basename(path))
                header()
                print()
            except Exception as e:
                print(f"  {red('Load failed:')} {e}\n")
            continue

        if not low.startswith('select'):
            print(f"  {yellow('?')} Type a SELECT statement or  help  for syntax.")
            continue

        sel_m   = re.match(r'SELECT\s+(.+?)\s+FROM\b', sql, re.IGNORECASE)
        sel_tok = sel_m.group(1).strip() if sel_m else ""
        is_row_query = (
            sel_tok == '*' or
            re.search(r'\bORDER\s+BY\b', sql, re.IGNORECASE) or
            re.search(r'\bLIMIT\b',      sql, re.IGNORECASE) or
            (sel_tok and not re.match(r'(COUNT|AVG|SUM|MIN|MAX)\s*\(', sel_tok, re.IGNORECASE))
        )

        if is_row_query:
            try:
                rows, dcols, ord_col, ord_dir, lim, total, _ = \
                    parse_row_query(sql, state["data"], state["col_names"])
            except ParseError as e:
                print(f"  {red('Parse error:')} {e}\n")
                continue
            print_row_table(rows, dcols, ord_col, ord_dir, lim, total)
            continue

        try:
            q, agg_fn, agg_col_name = parse_query(sql, state["data"], state["col_names"])
        except ParseError as e:
            print(f"  {red('Parse error:')} {e}\n")
            continue

        print(dim("  Running…"))
        try:
            (r_ts, t_ts) = timed(lambda: state["idx"].query(q))
            (r_np, t_np) = timed(lambda: numpy_query(state["data"], q))
            if state["tree"] is not None:
                (r_kd, t_kd) = timed(lambda: kdtree_query(state["tree"], state["data"], q))
            else:
                r_kd, t_kd = None, 0.0
            (r_bf, t_bf) = timed(lambda: state["idx"].brute_force(q))
        except Exception as e:
            print(f"  {red('Runtime error:')} {e}\n")
            continue

        print_sql_result(agg_fn, agg_col_name, r_ts, r_np, r_kd, r_bf,
                         t_ts, t_np, t_kd, t_bf,
                         r_ts.n_matched, r_ts.n_scanned, len(state["data"]))
        print()


def load_workload(path, data, col_names):
    queries = []
    errors  = []
    with open(path) as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                q, _, _ = parse_query(line, data, col_names)
                queries.append(q)
            except Exception as e:
                errors.append(f"  line {lineno}: {e}")
    if errors:
        print(f"  {yellow('Skipped')} {len(errors)} unparseable line(s):")
        for err in errors[:5]:
            print(err)
    print(f"  Loaded {len(queries)} training queries from '{os.path.basename(path)}'.")
    return queries


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Tsunami Index CLI")
    ap.add_argument("csv", nargs="?", help="CSV or Parquet file to load")
    ap.add_argument("-w", "--workload", default=None, metavar="FILE",
                    help="Training-query file")
    ap.add_argument("-n", "--nrows", type=int, default=0, metavar="N",
                    help="Load only first N rows")
    ap.add_argument("--dataset",
                    choices=["california_real", "nyc_taxi_real", "covtype"],
                    default="california_real",
                    help="Dataset: california_real, nyc_taxi_real, covtype")
    args = ap.parse_args()

    if args.csv:
        print(f"Loading '{args.csv}'…")
        data, col_names = load_csv(args.csv, max_rows=args.nrows)
        source = os.path.basename(args.csv)
    elif args.dataset == "nyc_taxi_real":
        n = args.nrows if args.nrows > 0 else 2_000_000
        print(f"Loading real NYC Taxi ({n:,} rows)…")
        data, col_names = generate_nyc_taxi_real(n=n)
        source = "NYC Taxi (Real)"
    elif args.dataset == "covtype":
        n = args.nrows if args.nrows > 0 else 2_000_000
        print(f"Loading Forest Covertype ({n:,} rows)…")
        data, col_names = generate_covtype(n=n)
        source = "Forest Covertype"
    else:
        n = args.nrows if args.nrows > 0 else 2_000_000
        print(f"Loading real California Housing ({n:,} rows)…")
        data, col_names = generate_california_real(n=n)
        source = "California Housing (Real)"

    workload = []
    if args.workload:
        print(f"Loading training queries from '{args.workload}'…")
        workload = load_workload(args.workload, data, col_names)

    print("Building Tsunami index…")
    idx = build_index(data, col_names, workload)

    _KDTREE_ROW_LIMIT = 2_000_000
    if len(data) <= _KDTREE_ROW_LIMIT:
        print("Building KDTree…")
        tree = KDTree(data)
    else:
        print(f"  KDTree skipped ({len(data):,} rows > {_KDTREE_ROW_LIMIT:,} limit).")
        tree = None
    print("  Ready.")

    run_repl(data, col_names, idx, tree, source)
