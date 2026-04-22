"""
Tsunami Index — GUI (terminal-style, all CLI features)
Run:  python gui_app.py
Requires only tkinter (built into Python) + project dependencies.

All CLI commands work here:
  SELECT COUNT(*) FROM data WHERE col >= 5
  SELECT AVG(col) FROM data WHERE Place = 'San Francisco'
  SELECT SUM(col) FROM data WHERE Neighbourhood = 'Williamsburg'
  load mydata.csv
  columns / places / neighbourhoods / tables / help / quit
"""

import sys, os, time, threading as _threading, re
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
sys.path.insert(0, os.path.dirname(__file__))

from tsunami_index import TsunamiIndex, TsunamiConfig, Query
from sklearn.neighbors import KDTree

# ── Import CLI parser & lookup tables ────────────────────────────────────────
from query_cli import (
    parse_query, parse_row_query, ParseError,
    numpy_query, kdtree_query,
    generate_california, load_csv, build_index,
    _LOOKUP_TABLES,
)
try:
    from places import list_places
except ImportError:
    list_places = None
try:
    from nyc_places import list_places as nyc_list
except ImportError:
    nyc_list = None

# ── Palette ───────────────────────────────────────────────────────────────────
BG       = "#1e1e2e"
PANEL    = "#181825"
ACCENT   = "#cba6f7"
FG       = "#cdd6f4"
FG_DIM   = "#6c7086"
GREEN    = "#a6e3a1"
RED      = "#f38ba8"
YELLOW   = "#f9e2af"
CYAN     = "#89dceb"
BLUE     = "#89b4fa"
FONT     = ("Consolas", 10)
FONT_B   = ("Consolas", 10, "bold")
FONT_LG  = ("Consolas", 12, "bold")
FONT_T   = ("Consolas", 11)   # terminal font


# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt(v, fn):
    if fn == "count":        return f"{int(v):,}"
    if abs(v) >= 1_000_000:  return f"{v:,.2f}"
    if abs(v) >= 1_000:      return f"{v:,.2f}"
    return f"{v:.4f}"

def close_enough(a, b):
    if isinstance(a, float) and np.isnan(a): return False
    return abs(a - b) / max(abs(b), 1e-9) < 1e-4

def timed(fn, reps=3):
    t0 = time.perf_counter()
    for _ in range(reps): r = fn()
    return r, (time.perf_counter() - t0) / reps * 1000


# ═══════════════════════════════════════════════════════════════════════════════
class VisualizationWindow(tk.Toplevel):
# ═══════════════════════════════════════════════════════════════════════════════
    """
    Separate window showing three live views after each query:
      Tab 1 — Scatter Plot  : data points coloured by role (pruned / scanned / matched)
                              + query bounding box + region boundaries
      Tab 2 — Exec Steps    : animated step-by-step of what Tsunami does internally
      Tab 3 — Speed Chart   : horizontal bar chart comparing all 4 methods
    """

    MPL_BG   = "#1e1e2e"
    MPL_AX   = "#181825"
    MPL_TEXT = "#cdd6f4"

    def __init__(self, parent):
        super().__init__(parent)
        self.parent   = parent
        self.title("Tsunami — Live Visualisation")
        self.geometry("960x680")
        self.configure(bg=BG)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._last_q     = None
        self._last_data  = None
        self._last_cols  = None
        self._last_stats = None   # dict filled by update_query()

        self._build_ui()
        self._show_idle()

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        # axis selectors (top bar)
        bar = tk.Frame(self, bg=PANEL)
        bar.pack(fill=tk.X, padx=6, pady=(6,0))

        tk.Label(bar, text="X axis:", font=FONT, bg=PANEL, fg=FG_DIM).pack(side=tk.LEFT, padx=(8,2))
        self._x_var = tk.StringVar()
        self._x_cb  = ttk.Combobox(bar, textvariable=self._x_var, state="readonly", width=16)
        self._x_cb.pack(side=tk.LEFT, padx=(0,10))
        self._x_cb.bind("<<ComboboxSelected>>", lambda e: self._refresh_scatter())

        tk.Label(bar, text="Y axis:", font=FONT, bg=PANEL, fg=FG_DIM).pack(side=tk.LEFT, padx=(0,2))
        self._y_var = tk.StringVar()
        self._y_cb  = ttk.Combobox(bar, textvariable=self._y_var, state="readonly", width=16)
        self._y_cb.pack(side=tk.LEFT)
        self._y_cb.bind("<<ComboboxSelected>>", lambda e: self._refresh_scatter())

        tk.Label(bar, text="  sample:", font=FONT, bg=PANEL, fg=FG_DIM).pack(side=tk.LEFT, padx=(10,2))
        self._sample_var = tk.IntVar(value=3000)
        ttk.Combobox(bar, textvariable=self._sample_var,
                     values=[500, 1000, 2000, 3000, 5000], state="readonly", width=6
                     ).pack(side=tk.LEFT)

        tk.Button(bar, text="Refresh", command=self._refresh_scatter,
                  bg="#313244", fg=FG, font=FONT, relief="flat",
                  activebackground="#45475a", cursor="hand2"
                  ).pack(side=tk.LEFT, padx=8)

        # notebook tabs
        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self._tab_scatter = tk.Frame(nb, bg=self.MPL_BG)
        self._tab_steps   = tk.Frame(nb, bg=BG)

        nb.add(self._tab_scatter, text="  📍 Scatter Plot  ")
        nb.add(self._tab_steps,   text="  🔍 Execution Steps  ")

        self._build_scatter_tab()
        self._build_steps_tab()

    # ── Tab 1: Scatter ────────────────────────────────────────────────────────

    def _build_scatter_tab(self):
        self._fig_s, self._ax_s = plt.subplots(figsize=(8, 5),
                                                facecolor=self.MPL_BG)
        self._ax_s.set_facecolor(self.MPL_AX)
        self._canvas_s = FigureCanvasTkAgg(self._fig_s, master=self._tab_scatter)
        self._canvas_s.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _refresh_scatter(self):
        if self._last_data is None:
            return
        data  = self._last_data
        cols  = self._last_cols
        q     = self._last_q
        stats = self._last_stats

        xi = cols.index(self._x_var.get()) if self._x_var.get() in cols else 0
        yi = cols.index(self._y_var.get()) if self._y_var.get() in cols else min(1, len(cols)-1)

        N      = len(data)
        n_samp = min(self._sample_var.get(), N)
        idx_s  = np.random.choice(N, n_samp, replace=False)

        ax = self._ax_s
        ax.clear()
        ax.set_facecolor(self.MPL_AX)

        # Compute matched mask on full data
        if q is not None:
            mask = np.ones(N, dtype=bool)
            for d, (lo, hi) in enumerate(q.ranges):
                mask &= (data[:, d] >= lo) & (data[:, d] <= hi)
            matched_idx = np.where(mask)[0]
        else:
            mask        = np.zeros(N, dtype=bool)
            matched_idx = np.array([], dtype=int)

        # Sample indices split into matched / not-matched
        samp_matched = idx_s[mask[idx_s]]
        samp_other   = idx_s[~mask[idx_s]]

        # Plot: background (gray), then matched (green)
        ax.scatter(data[samp_other,  xi], data[samp_other,  yi],
                   s=4, c="#45475a", alpha=0.4, linewidths=0, label="Not matched")
        if len(samp_matched):
            ax.scatter(data[samp_matched, xi], data[samp_matched, yi],
                       s=18, c="#a6e3a1", alpha=0.9, linewidths=0, label=f"Matched ({len(matched_idx):,})")

        # Query bounding box on these axes
        if q is not None:
            lo_x, hi_x = q.ranges[xi]
            lo_y, hi_y = q.ranges[yi]
            dx = data[:, xi]
            dy = data[:, yi]
            # clamp to data range
            lo_x = max(lo_x, dx.min()); hi_x = min(hi_x, dx.max())
            lo_y = max(lo_y, dy.min()); hi_y = min(hi_y, dy.max())
            rect = mpatches.FancyBboxPatch(
                (lo_x, lo_y), hi_x - lo_x, hi_y - lo_y,
                boxstyle="square,pad=0",
                linewidth=2, edgecolor="#f9e2af", facecolor="#f9e2af22",
                label="Query box", zorder=5)
            ax.add_patch(rect)

        # Region bounding boxes (if accessible from index)
        if stats and stats.get("regions"):
            for rb in stats["regions"]:
                lo_x_r = rb[xi][0]; hi_x_r = rb[xi][1]
                lo_y_r = rb[yi][0]; hi_y_r = rb[yi][1]
                dx = data[:, xi]; dy = data[:, yi]
                lo_x_r = max(lo_x_r, dx.min()); hi_x_r = min(hi_x_r, dx.max())
                lo_y_r = max(lo_y_r, dy.min()); hi_y_r = min(hi_y_r, dy.max())
                ax.add_patch(mpatches.FancyBboxPatch(
                    (lo_x_r, lo_y_r), hi_x_r - lo_x_r, hi_y_r - lo_y_r,
                    boxstyle="square,pad=0",
                    linewidth=1.2, edgecolor="#89b4fa44", facecolor="none",
                    linestyle="--", zorder=3))

        ax.set_xlabel(cols[xi], color=self.MPL_TEXT, fontsize=9)
        ax.set_ylabel(cols[yi], color=self.MPL_TEXT, fontsize=9)
        ax.tick_params(colors=self.MPL_TEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#313244")
        ax.set_title(
            f"{cols[xi]}  vs  {cols[yi]}  —  "
            f"{len(matched_idx):,} matched / {n_samp:,} sampled",
            color=self.MPL_TEXT, fontsize=10)
        leg = ax.legend(fontsize=8, facecolor="#313244",
                        labelcolor=self.MPL_TEXT, framealpha=0.9)
        self._fig_s.tight_layout()
        self._canvas_s.draw()

    # ── Tab 2: Execution steps ────────────────────────────────────────────────

    def _build_steps_tab(self):
        self._steps_text = tk.Text(
            self._tab_steps, font=("Consolas", 11),
            bg=BG, fg=FG, relief="flat", state="disabled",
            wrap="word", padx=16, pady=12)
        self._steps_text.pack(fill=tk.BOTH, expand=True)

        self._steps_text.tag_configure("head",  foreground=ACCENT, font=("Consolas",11,"bold"))
        self._steps_text.tag_configure("step",  foreground=YELLOW, font=("Consolas",11,"bold"))
        self._steps_text.tag_configure("ok",    foreground=GREEN)
        self._steps_text.tag_configure("dim",   foreground=FG_DIM)
        self._steps_text.tag_configure("val",   foreground=GREEN, font=("Consolas",13,"bold"))
        self._steps_text.tag_configure("warn",  foreground=RED)
        self._steps_text.tag_configure("bar_ts",foreground="#cba6f7")
        self._steps_text.tag_configure("bar_np",foreground="#89dceb")
        self._steps_text.tag_configure("bar_kd",foreground="#f38ba8")
        self._steps_text.tag_configure("bar_bf",foreground="#fab387")

    def _animate_steps(self, q, agg_fn, agg_col_name, r_ts, t_ts, t_np, t_kd, t_bf, N):
        """Write execution steps one by one with short delays."""
        def sw(text, tag="normal"):
            self._steps_text.config(state="normal")
            self._steps_text.insert(tk.END, text, tag)
            self._steps_text.config(state="disabled")
            self._steps_text.see(tk.END)

        def swl(text="", tag="normal"):
            sw(text + "\n", tag)

        def pause(ms=180):
            time.sleep(ms / 1000)

        self._steps_text.config(state="normal")
        self._steps_text.delete("1.0", tk.END)
        self._steps_text.config(state="disabled")

        label = f"{agg_fn.upper()}({agg_col_name or '*'})"

        swl("━" * 58, "dim")
        swl(f"  Executing:  {label}", "head")
        swl("━" * 58, "dim");  swl()
        pause(200)

        # Step 1
        swl("  Step 1 ── Parse query & build filter ranges", "step");  pause(300)
        active_dims = [(i, lo, hi) for i, (lo, hi) in enumerate(q.ranges)
                       if not (lo <= -1e8 and hi >= 1e8)]
        swl(f"    {len(active_dims)} active filter dimension(s)  "
            f"(of {len(q.ranges)} total):", "dim")
        for i, lo, hi in active_dims:
            col = self._last_cols[i] if self._last_cols else f"col{i}"
            swl(f"      {col:<18}  [{lo:.4g}  →  {hi:.4g}]", "ok")
        swl();  pause(300)

        # Step 2
        swl("  Step 2 ── Grid Tree: identify overlapping regions", "step");  pause(300)
        n_reg = self._last_stats.get("n_regions", 1) if self._last_stats else 1
        hit   = r_ts.n_regions
        swl(f"    Total regions in index : {n_reg}", "dim")
        swl(f"    Regions hit by query   : {hit}", "ok" if hit < n_reg else "dim")
        if hit < n_reg:
            swl(f"    Regions pruned         : {n_reg - hit}  ✓ skipped entirely", "ok")
        else:
            swl(f"    All regions scanned    : no pruning on this dataset/query", "dim")
        swl();  pause(350)

        # Step 3
        swl("  Step 3 ── Scan rows inside overlapping regions", "step");  pause(300)
        scan_pct = r_ts.n_scanned / max(N, 1) * 100
        bar_len  = 40
        filled   = int(bar_len * scan_pct / 100)
        bar      = "█" * filled + "░" * (bar_len - filled)
        swl(f"    Rows scanned  : {r_ts.n_scanned:,} / {N:,}  ({scan_pct:.1f}%)")
        swl(f"    [{bar}]", "dim")
        swl();  pause(350)

        # Step 4
        swl("  Step 4 ── Evaluate predicates & aggregate", "step");  pause(300)
        match_pct = r_ts.n_matched / max(r_ts.n_scanned, 1) * 100
        swl(f"    Rows matched  : {r_ts.n_matched:,}  ({match_pct:.1f}% of scanned)", "ok")
        swl(f"    {label}  =  ", "dim");  sw(fmt(r_ts.value, agg_fn), "val");  swl();  swl()
        pause(300)

        # Step 5 — speed breakdown
        swl("  Step 5 ── Latency breakdown vs other methods", "step");  pause(200)
        methods = [
            ("Tsunami",     t_ts, "bar_ts"),
            ("NumPy",       t_np, "bar_np"),
            ("Brute Force", t_bf, "bar_bf"),
        ]
        if t_kd > 0:
            methods.insert(2, ("KDTree", t_kd, "bar_kd"))
        max_t   = max(m[1] for m in methods)
        bar_w   = 30
        fastest = min(m[1] for m in methods)
        for name, t, tag in methods:
            filled = max(1, int(bar_w * t / max_t))
            bar    = "█" * filled
            marker = "  ◀ fastest" if t == fastest else ""
            sw(f"    {name:<14} {t:>7.3f}ms  ", tag)
            sw(bar, tag)
            swl(marker, tag)
            pause(120)
        if t_kd == 0:
            swl("    KDTree         skipped (dataset too large)", "bar_kd")

        swl()
        bf_su = t_bf / t_ts if t_ts > 0 else 0
        swl(f"    Tsunami  {bf_su:.1f}× faster than Brute Force", "dim")
        swl("━" * 58, "dim")

    # ── Public update API ─────────────────────────────────────────────────────

    def update_query(self, data, col_names, q, agg_fn, agg_col_name,
                     r_ts, t_ts, t_np, t_kd, t_bf, idx_stats):
        self._last_q     = q
        self._last_data  = data
        self._last_cols  = col_names
        self._last_stats = idx_stats

        N     = len(data)

        # Update axis selectors if columns changed
        if list(self._x_cb["values"]) != col_names:
            self._x_cb.config(values=col_names)
            self._y_cb.config(values=col_names)
            self._x_var.set(col_names[0])
            self._y_var.set(col_names[min(1, len(col_names)-1)])

        # Scatter plot (quick, on main thread)
        self._refresh_scatter()

        # Execution steps animation (background thread so it streams in)
        _threading.Thread(
            target=self._animate_steps,
            args=(q, agg_fn, agg_col_name, r_ts, t_ts, t_np, t_kd, t_bf, N),
            daemon=True,
        ).start()

    def _show_idle(self):
        for text_widget in (self._steps_text,):
            text_widget.config(state="normal")
            text_widget.delete("1.0", tk.END)
            text_widget.insert(tk.END,
                "\n\n  Load a dataset and run a query to see the live visualisation.\n\n",
                "dim")
            text_widget.config(state="disabled")

    def _on_close(self):
        self.withdraw()   # hide instead of destroy so it can be re-shown


# ═══════════════════════════════════════════════════════════════════════════════
class TsunamiGUI(tk.Tk):
# ═══════════════════════════════════════════════════════════════════════════════

    def __init__(self):
        super().__init__()
        self.title("Tsunami Index")
        self.geometry("1200x760")
        self.minsize(900, 600)
        self.configure(bg=BG)

        self.data      = None
        self.col_names = []
        self.idx       = None
        self.tree      = None
        self.source    = None
        self._history  = []
        self._hist_pos = -1
        self._viz_win  = None        # created on first click

        self._build_ui()
        self._print_banner()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Left sidebar (scrollable) ─────────────────────────────────────────
        sb_outer = tk.Frame(self, bg=PANEL, width=236)
        sb_outer.pack(side=tk.LEFT, fill=tk.Y, padx=(6,0), pady=6)
        sb_outer.pack_propagate(False)

        sb_scroll = ttk.Scrollbar(sb_outer, orient="vertical")
        sb_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        sb_canvas = tk.Canvas(sb_outer, bg=PANEL, highlightthickness=0,
                              yscrollcommand=sb_scroll.set, width=220)
        sb_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb_scroll.config(command=sb_canvas.yview)

        sidebar = tk.Frame(sb_canvas, bg=PANEL)
        _sb_win = sb_canvas.create_window((0, 0), window=sidebar, anchor="nw")

        def _on_sidebar_configure(e):
            sb_canvas.configure(scrollregion=sb_canvas.bbox("all"))
            sb_canvas.itemconfig(_sb_win, width=sb_canvas.winfo_width())

        sidebar.bind("<Configure>", _on_sidebar_configure)
        sb_canvas.bind("<Configure>",
                       lambda e: sb_canvas.itemconfig(_sb_win, width=e.width))

        def _on_mousewheel(e):
            sb_canvas.yview_scroll(int(-1*(e.delta/120)), "units")
        sb_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self._build_sidebar(sidebar)

        # ── Main terminal area ────────────────────────────────────────────────
        main = tk.Frame(self, bg=BG)
        main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)
        self._build_terminal(main)

    def _build_sidebar(self, p):
        tk.Label(p, text="🌊 Tsunami", font=FONT_LG,
                 bg=PANEL, fg=ACCENT).pack(pady=(14,2))
        tk.Label(p, text="Learned multi-dim index",
                 font=("Consolas", 9), bg=PANEL, fg=FG_DIM).pack(pady=(0,10))

        def sep(): ttk.Separator(p, orient="horizontal").pack(fill=tk.X, padx=8, pady=6)

        sep()

        # ── Load ──────────────────────────────────────────────────────────────
        self._section(p, "Load Dataset")
        self._btn(p, "📂  Load CSV…",        self._pick_csv,   ACCENT)
        self._btn(p, "🏠  California demo",   self._load_demo,  "#313244")

        row_row = tk.Frame(p, bg=PANEL)
        row_row.pack(fill=tk.X, padx=10, pady=(4,0))
        tk.Label(row_row, text="Rows:", font=("Consolas",9),
                 bg=PANEL, fg=FG_DIM).pack(side=tk.LEFT)
        self._demo_rows = tk.StringVar(value="500000")
        ttk.Combobox(row_row, textvariable=self._demo_rows,
                     values=["200000","500000","1000000","2000000","4000000","5000000"],
                     state="readonly", width=9).pack(side=tk.LEFT, padx=(4,0))

        self._custom_workload = tk.BooleanVar(value=False)
        tk.Checkbutton(
            p, text="Custom training queries",
            variable=self._custom_workload,
            font=("Consolas", 9), bg=PANEL, fg=FG_DIM,
            selectcolor=PANEL, activebackground=PANEL,
            activeforeground=FG, relief="flat", cursor="hand2",
        ).pack(anchor="w", padx=10, pady=(2, 0))

        sep()

        # ── Index info ────────────────────────────────────────────────────────
        self._section(p, "Index Info")
        self._rows_lbl  = self._kv(p, "Rows",    "—")
        self._cols_lbl  = self._kv(p, "Columns", "—")
        self._build_lbl = self._kv(p, "Build",   "—")
        self._src_lbl   = self._kv(p, "Source",  "—")

        sep()

        # ── Quick commands ────────────────────────────────────────────────────
        self._section(p, "Quick Commands")
        self._btn(p, "columns",         lambda: self._run_cmd("columns"),        "#313244")
        self._btn(p, "tables",          lambda: self._run_cmd("tables"),         "#313244")
        self._btn(p, "places",          lambda: self._run_cmd("places"),         "#313244")
        self._btn(p, "neighbourhoods",  lambda: self._run_cmd("neighbourhoods"), "#313244")
        self._btn(p, "help",            lambda: self._run_cmd("help"),           "#313244")

        sep()

        # ── Example queries (rebuilt dynamically on dataset load) ─────────────
        self._section(p, "Example Queries")
        self._example_frame = tk.Frame(p, bg=PANEL)
        self._example_frame.pack(fill=tk.X)
        self._refresh_examples()

        sep()

        sep()

        # ── Visualise ─────────────────────────────────────────────────────────
        self._section(p, "Visualisation")
        self._btn(p, "📊  Open Viz Window", self._open_viz, "#313244")

        sep()

        # ── Clear ─────────────────────────────────────────────────────────────
        self._btn(p, "🗑  Clear terminal", self._clear_terminal, "#313244")

    def _build_terminal(self, p):
        # ── Output area ───────────────────────────────────────────────────────
        out_frame = tk.Frame(p, bg=BG)
        out_frame.pack(fill=tk.BOTH, expand=True)

        self.out = tk.Text(
            out_frame,
            font=FONT_T, bg=BG, fg=FG,
            insertbackground=FG, relief="flat",
            state="disabled", wrap="none",
        )
        vsb = ttk.Scrollbar(out_frame, orient="vertical",   command=self.out.yview)
        hsb = ttk.Scrollbar(out_frame, orient="horizontal", command=self.out.xview)
        self.out.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side=tk.RIGHT,  fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.out.pack(fill=tk.BOTH, expand=True)

        # colour tags
        self.out.tag_configure("prompt",  foreground=ACCENT, font=FONT_B)
        self.out.tag_configure("cmd",     foreground=FG,     font=FONT_B)
        self.out.tag_configure("accent",  foreground=ACCENT)
        self.out.tag_configure("green",   foreground=GREEN)
        self.out.tag_configure("red",     foreground=RED)
        self.out.tag_configure("yellow",  foreground=YELLOW)
        self.out.tag_configure("cyan",    foreground=CYAN)
        self.out.tag_configure("blue",    foreground=BLUE)
        self.out.tag_configure("dim",     foreground=FG_DIM)
        self.out.tag_configure("bold",    font=FONT_B)
        self.out.tag_configure("value",   foreground=GREEN,  font=("Consolas",13,"bold"))
        self.out.tag_configure("header",  foreground=ACCENT, font=FONT_B)
        self.out.tag_configure("normal",  foreground=FG)

        # ── Input row ─────────────────────────────────────────────────────────
        inp_frame = tk.Frame(p, bg=PANEL)
        inp_frame.pack(fill=tk.X, pady=(4,0))

        tk.Label(inp_frame, text="tsunami>", font=FONT_B,
                 bg=PANEL, fg=ACCENT).pack(side=tk.LEFT, padx=(8,4), pady=6)

        self.inp = tk.Entry(inp_frame, font=FONT_T,
                            bg="#313244", fg=FG, insertbackground=FG,
                            relief="flat")
        self.inp.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=6)
        self.inp.bind("<Return>",   self._on_enter)
        self.inp.bind("<Up>",       self._hist_up)
        self.inp.bind("<Down>",     self._hist_down)
        self.inp.bind("<Tab>",      self._autocomplete)
        self.inp.focus()

        tk.Button(inp_frame, text="Run ▶",
                  command=self._on_enter,
                  bg=GREEN, fg="#1e1e2e", font=FONT_B,
                  relief="flat", activebackground="#c3e7c0",
                  cursor="hand2", padx=10
                  ).pack(side=tk.LEFT, padx=(4,8), pady=4)

    # ── Sidebar helpers ───────────────────────────────────────────────────────

    def _section(self, p, text):
        tk.Label(p, text=text, font=("Consolas",9,"bold"),
                 bg=PANEL, fg=FG_DIM).pack(anchor="w", padx=10, pady=(4,2))

    def _btn(self, p, text, cmd, bg):
        tk.Button(p, text=text, command=cmd,
                  bg=bg, fg=FG, font=FONT, relief="flat",
                  activebackground="#45475a", cursor="hand2",
                  anchor="w", padx=8
                  ).pack(fill=tk.X, padx=8, pady=1)

    def _kv(self, p, key, val):
        row = tk.Frame(p, bg=PANEL)
        row.pack(fill=tk.X, padx=10, pady=1)
        tk.Label(row, text=f"{key}:", font=FONT, bg=PANEL,
                 fg=FG_DIM, width=8, anchor="w").pack(side=tk.LEFT)
        lbl = tk.Label(row, text=val, font=FONT_B, bg=PANEL, fg=FG, anchor="w")
        lbl.pack(side=tk.LEFT)
        return lbl

    # ── Terminal output ───────────────────────────────────────────────────────

    def _write(self, text, tag="normal"):
        if _threading.current_thread() is not _threading.main_thread():
            self.after(0, lambda t=text, tg=tag: self._write(t, tg))
            return
        self.out.config(state="normal")
        self.out.insert(tk.END, text, tag)
        self.out.config(state="disabled")
        self.out.see(tk.END)

    def _writeln(self, text="", tag="normal"):
        self._write(text + "\n", tag)

    def _print_banner(self):
        self._writeln("=" * 62, "dim")
        self._writeln("  🌊  Tsunami Index  —  Interactive Query Interface", "accent")
        self._writeln("  Type SQL queries below, or click buttons on the left.", "dim")
        self._writeln("  Type  help  for full syntax reference.", "dim")
        self._writeln("=" * 62, "dim")
        self._writeln()

    def _clear_terminal(self):
        self.out.config(state="normal")
        self.out.delete("1.0", tk.END)
        self.out.config(state="disabled")
        self._print_banner()

    # ── Input handling ────────────────────────────────────────────────────────

    def _on_enter(self, event=None):
        sql = self.inp.get().strip()
        if not sql:
            return
        self.inp.delete(0, tk.END)

        # save to history
        if not self._history or self._history[-1] != sql:
            self._history.append(sql)
        self._hist_pos = len(self._history)

        # echo command
        self._writeln()
        self._write("tsunami> ", "prompt")
        self._writeln(sql, "cmd")

        # dispatch
        _threading.Thread(target=self._dispatch, args=(sql,), daemon=True).start()

    def _run_cmd(self, cmd):
        self.inp.delete(0, tk.END)
        self.inp.insert(0, cmd)
        self._on_enter()

    def _insert_example(self, sql):
        self.inp.delete(0, tk.END)
        self.inp.insert(0, sql)
        self.inp.focus()

    def _hist_up(self, event):
        if not self._history: return
        self._hist_pos = max(0, self._hist_pos - 1)
        self.inp.delete(0, tk.END)
        self.inp.insert(0, self._history[self._hist_pos])

    def _hist_down(self, event):
        if not self._history: return
        self._hist_pos = min(len(self._history), self._hist_pos + 1)
        self.inp.delete(0, tk.END)
        if self._hist_pos < len(self._history):
            self.inp.insert(0, self._history[self._hist_pos])

    def _autocomplete(self, event):
        CMDS = ["SELECT COUNT(*) FROM data WHERE ",
                "SELECT AVG() FROM data WHERE ",
                "SELECT SUM() FROM data WHERE ",
                "SELECT MIN() FROM data WHERE ",
                "SELECT MAX() FROM data WHERE ",
                "columns", "tables", "places", "neighbourhoods",
                "help", "quit", "load "]
        cur = self.inp.get()
        matches = [c for c in CMDS if c.lower().startswith(cur.lower())]
        if len(matches) == 1:
            self.inp.delete(0, tk.END)
            self.inp.insert(0, matches[0])
        return "break"

    # ── Command dispatcher ────────────────────────────────────────────────────

    def _dispatch(self, sql):
        low = sql.strip().lower()

        if low in ("quit", "exit", "q"):
            self.after(0, self.destroy)

        elif low == "help":
            self.after(0, self._cmd_help)

        elif low == "columns":
            self.after(0, self._cmd_columns)

        elif low == "tables":
            self.after(0, self._cmd_tables)

        elif low in _LOOKUP_TABLES:
            self.after(0, lambda: self._cmd_lookup_table(low))

        elif low.startswith("load "):
            path = sql[5:].strip().strip("\"'")
            self.after(0, lambda: self._cmd_load(path))

        elif low.startswith("select"):
            self._cmd_select(sql)

        else:
            self.after(0, lambda: self._writeln(
                f"  Unknown command '{sql}'. Type  help  for syntax.", "red"))

    # ── Commands ──────────────────────────────────────────────────────────────

    def _cmd_help(self):
        lines = [
            ("", "dim"),
            ("  SQL Syntax:", "accent"),
            ("    SELECT COUNT(*)   FROM data WHERE <filters>", "normal"),
            ("    SELECT AVG(col)   FROM data WHERE <filters>", "normal"),
            ("    SELECT SUM(col)   FROM data WHERE <filters>", "normal"),
            ("    SELECT MIN(col)   FROM data WHERE <filters>", "normal"),
            ("    SELECT MAX(col)   FROM data WHERE <filters>", "normal"),
            ("", "dim"),
            ("  Filter operators (combine with AND):", "accent"),
            ("    col BETWEEN lo AND hi", "normal"),
            ("    col >= val  |  col <= val  |  col > val  |  col < val", "normal"),
            ("", "dim"),
            ("  Row-returning with ORDER BY / LIMIT:", "accent"),
            ("    SELECT * FROM data WHERE col >= 5 ORDER BY col DESC LIMIT 20", "normal"),
            ("    SELECT col1, col2 FROM data WHERE ... ORDER BY col ASC LIMIT 10", "normal"),
            ("", "dim"),
            ("  Lookup table filters:", "accent"),
            ("    Place = 'San Francisco'       (California — needs Latitude/Longitude)", "normal"),
            ("    Neighbourhood = 'Williamsburg' (NYC — needs latitude/longitude)", "normal"),
            ("", "dim"),
            ("  Commands:", "accent"),
            ("    load <file.csv>    load a new CSV and rebuild the index", "normal"),
            ("    columns            show columns + value ranges", "normal"),
            ("    tables             show all loaded tables", "normal"),
            ("    places             show California place names", "normal"),
            ("    neighbourhoods     show NYC neighbourhood names", "normal"),
            ("    help               show this message", "normal"),
            ("    quit               exit", "normal"),
            ("", "dim"),
            ("  Tips:", "accent"),
            ("    ↑ / ↓ arrows      navigate command history", "dim"),
            ("    Tab               autocomplete commands", "dim"),
            ("    Example buttons   on the left sidebar", "dim"),
            ("", "dim"),
        ]
        for text, tag in lines:
            self._writeln(text, tag)

    def _cmd_columns(self):
        if self.data is None:
            self._writeln("  No dataset loaded. Use  load <file.csv>  or the sidebar.", "red")
            return
        self._writeln()
        self._write(f"  {'#':<4} {'Column':<20} {'Min':>12} {'Max':>12}  {'Mean':>12}\n", "accent")
        self._writeln("  " + "─" * 54, "dim")
        for i, c in enumerate(self.col_names):
            col_data = self.data[:, i]
            self._writeln(
                f"  {i:<4} {c:<20} {col_data.min():>12.4f} {col_data.max():>12.4f}"
                f"  {col_data.mean():>12.4f}")
        self._writeln()

    def _cmd_tables(self):
        if self.data is None:
            self._writeln("  No dataset loaded.", "red")
            return
        N, D = self.data.shape
        self._writeln()
        self._writeln(f"  Table 1 (main)  —  {self.source}", "accent")
        self._writeln(f"    {N:,} rows × {D} columns")
        self._writeln(f"    Columns: {', '.join(self.col_names)}", "dim")
        for i, (kw, _) in enumerate(_LOOKUP_TABLES.items(), 2):
            self._writeln(f"  Table {i} (lookup)  —  {kw.title()} table", "accent")
            self._writeln(f"    Type  '{kw}'  to see contents.", "dim")
        self._writeln()

    def _cmd_lookup_table(self, keyword):
        info = _LOOKUP_TABLES.get(keyword)
        if info is None:
            self._writeln(f"  Unknown table '{keyword}'.", "red")
            return

        # fetch data from the lookup module
        self._writeln()
        if keyword == "place":
            from places import list_places
            df = list_places()
            self._write(
                f"  {'Place':<22} {'Lat min':>8} {'Lat max':>8} {'Lon min':>9} {'Lon max':>9}\n",
                "accent")
            self._writeln("  " + "─" * 62, "dim")
            for _, row in df.iterrows():
                self._writeln(
                    f"  {row.Place:<22} {row.Lat_min:>8.2f} {row.Lat_max:>8.2f}"
                    f" {row.Lon_min:>9.2f} {row.Lon_max:>9.2f}")

        elif keyword == "neighbourhood" and nyc_list:
            df = nyc_list()
            self._write(
                f"  {'Neighbourhood':<24} {'Lat min':>8} {'Lat max':>8} {'Lon min':>9} {'Lon max':>9}\n",
                "accent")
            self._writeln("  " + "─" * 64, "dim")
            for _, row in df.iterrows():
                self._writeln(
                    f"  {row.Place:<24} {row.Lat_min:>8.3f} {row.Lat_max:>8.3f}"
                    f" {row.Lon_min:>9.3f} {row.Lon_max:>9.3f}")
        self._writeln()

    def _cmd_load(self, path):
        if not os.path.exists(path):
            self._writeln(f"  File not found: {path}", "red")
            return
        mb = os.path.getsize(path) / 1_048_576 if os.path.exists(path) else 0
        self._writeln(f"  Loading '{os.path.basename(path)}' ({mb:,.0f} MB)…", "dim")
        try:
            data, cols = load_csv(path)
        except Exception as e:
            self._writeln(f"  Load error: {e}", "red")
            return

        if self._custom_workload.get():
            workload = self._ask_custom_workload(data, cols)
            if not workload:
                self._writeln("  Build cancelled.", "yellow")
                return
            self._writeln(f"  Building Tsunami index (learned) with {len(workload)} queries…", "dim")
        else:
            workload = []
            self._writeln("  Building Tsunami index…", "dim")
        t0  = time.perf_counter()
        idx = build_index(data, cols, workload)
        ms  = (time.perf_counter() - t0) * 1000

        if len(data) <= 2_000_000:
            self._writeln("  Building KDTree…", "dim")
            tree = KDTree(data)
        else:
            self._writeln(f"  KDTree skipped ({len(data):,} rows — too large).", "dim")
            tree = None

        self.data = data; self.col_names = cols
        self.idx = idx;   self.tree = tree
        self.source = os.path.basename(path)

        self.after(0, lambda: self._update_stats(ms))
        self._writeln(f"  Ready — {len(data):,} rows × {len(cols)} columns, "
                      f"index built in {ms:.0f} ms", "green")
        self._writeln()

    def _cmd_select(self, sql):
        if self.data is None:
            self.after(0, lambda: self._writeln(
                "  No dataset loaded. Use  load <file.csv>  or the sidebar.", "red"))
            return

        # Route: row-returning vs aggregate
        sel_m   = re.match(r'SELECT\s+(.+?)\s+FROM\b', sql, re.IGNORECASE)
        sel_tok = sel_m.group(1).strip() if sel_m else ""
        is_row  = (
            sel_tok == '*' or
            re.search(r'\bORDER\s+BY\b', sql, re.IGNORECASE) or
            re.search(r'\bLIMIT\b',      sql, re.IGNORECASE) or
            (sel_tok and not re.match(r'(COUNT|AVG|SUM|MIN|MAX)\s*\(', sel_tok, re.IGNORECASE))
        )

        if is_row:
            try:
                rows, dcols, ord_col, ord_dir, lim, total, ranges = \
                    parse_row_query(sql, self.data, self.col_names)
            except ParseError as e:
                self.after(0, lambda: self._writeln(f"  Parse error: {e}", "red"))
                return

            # build a Query from the parsed ranges so we can time all 4 methods
            from tsunami_index import Query as _Q
            range_list = [(ranges[c][0], ranges[c][1]) for c in self.col_names]
            q_row = _Q(range_list, agg_fn="count", agg_col=0)
            try:
                r_ts, t_ts = timed(lambda: self.idx.query(q_row))
                r_np, t_np = timed(lambda: numpy_query(self.data, q_row))
                if self.tree is not None:
                    r_kd, t_kd = timed(lambda: kdtree_query(self.tree, self.data, q_row))
                else:
                    r_kd, t_kd = None, 0.0
                r_bf, t_bf = timed(lambda: self.idx.brute_force(q_row))
            except Exception:
                r_ts = r_np = r_kd = r_bf = None
                t_ts = t_np = t_kd = t_bf = 0.0

            _rows, _dcols, _oc, _od, _lim, _tot = rows, dcols, ord_col, ord_dir, lim, total
            _times = (t_ts, t_np, t_kd, t_bf)
            def _show_row(_r=_rows, _d=_dcols, _oc=_oc, _od=_od, _l=_lim, _t=_tot, _tm=_times):
                self._print_row_table(_r, _d, _oc, _od, _l, _t)
                self._print_row_comparison(_tm)
            self.after(0, _show_row)
            return

        # parse aggregate
        try:
            q, agg_fn, agg_col_name = parse_query(sql, self.data, self.col_names)
        except ParseError as e:
            self.after(0, lambda: self._writeln(f"  Parse error: {e}", "red"))
            return

        # run all methods
        try:
            r_ts, t_ts = timed(lambda: self.idx.query(q))
            r_np, t_np = timed(lambda: numpy_query(self.data, q))
            if self.tree is not None:
                r_kd, t_kd = timed(lambda: kdtree_query(self.tree, self.data, q))
            else:
                r_kd, t_kd = None, 0.0
            r_bf, t_bf = timed(lambda: self.idx.brute_force(q))
        except Exception as e:
            self.after(0, lambda: self._writeln(f"  Runtime error: {e}", "red"))
            return

        self.after(0, lambda: self._print_result(
            q, agg_fn, agg_col_name,
            r_ts, r_np, r_kd, r_bf,
            t_ts, t_np, t_kd, t_bf,
        ))

    def _print_row_table(self, rows, col_names, order_col, order_dir, limit, total):
        if len(rows) == 0:
            self._writeln("  No rows matched.", "yellow")
            return

        # column widths
        col_w = [max(len(c), 8) for c in col_names]
        for row in rows:
            for i, v in enumerate(row):
                col_w[i] = max(col_w[i], len(f"{v:.4g}"))

        sep  = "  +" + "+".join("-" * (w + 2) for w in col_w) + "+"
        head = "  |" + "|".join(f" {c:^{w}} " for c, w in zip(col_names, col_w)) + "|"

        self._writeln()
        self._writeln(sep, "dim")
        self._writeln(head, "accent")
        self._writeln(sep, "dim")
        for row in rows:
            cells = "|".join(f" {v:>{w}.4g} " for v, w in zip(row, col_w))
            self._writeln(f"  |{cells}|")
        self._writeln(sep, "dim")

        order_note = f"  ORDER BY {order_col} {order_dir}" if order_col else ""
        self._writeln(
            f"  {len(rows)} of {total:,} matched rows shown  "
            f"(LIMIT {limit}){order_note}", "dim")
        self._writeln()

    def _print_row_comparison(self, times):
        t_ts, t_np, t_kd, t_bf = times
        if t_ts == 0.0 and t_np == 0.0:
            return
        methods = [("Tsunami", t_ts), ("NumPy", t_np)]
        if t_kd > 0:
            methods.append(("KDTree", t_kd))
        methods.append(("Brute Force", t_bf))
        fastest = min(t for _, t in methods)
        self._write(f"  {'Method':<14} {'Filter time':>12}\n", "accent")
        self._writeln("  " + "─" * 30, "dim")
        for name, t in methods:
            is_best = (t == fastest)
            tag     = "yellow" if is_best else "normal"
            marker  = "  ◀ fastest" if is_best else ""
            self._writeln(f"  {name:<14} {t:>9.3f}ms{marker}", tag)
        if t_kd == 0:
            self._writeln("  KDTree         skipped (too large)", "dim")
        bf_su = t_bf / t_ts if t_ts > 0 else 0
        self._writeln(f"\n  Tsunami  {bf_su:.1f}× faster than Brute Force", "dim")
        self._writeln()

    def _print_result(self, q, agg_fn, agg_col_name,
                      r_ts, r_np, r_kd, r_bf, t_ts, t_np, t_kd, t_bf):
        val     = r_ts.value
        val_str = fmt(val, agg_fn)
        label   = f"{agg_fn.upper()}({agg_col_name or '*'})"
        N       = len(self.data)

        # SQL-style box
        width = max(len(label), len(val_str), 22)
        bar   = "  +" + "-" * (width + 2) + "+"
        self._writeln()
        self._writeln(bar, "dim")
        self._writeln(f"  | {label:<{width}} |", "header")
        self._writeln(bar, "dim")
        self._writeln(f"  | {val_str:<{width}} |", "value")
        self._writeln(bar, "dim")
        self._writeln(
            f"  {r_ts.n_matched:,} row(s) matched  ·  "
            f"scanned {r_ts.n_scanned:,}/{N:,}  "
            f"({r_ts.n_scanned/N*100:.0f}%)", "dim")
        self._writeln()

        # Method comparison
        methods = [
            ("Tsunami",     t_ts, r_ts.value),
            ("NumPy",       t_np, r_np[0]),
        ]
        if r_kd is not None:
            methods.append(("KDTree", t_kd, r_kd[0]))
        methods.append(("Brute Force", t_bf, r_bf[1]))
        fastest = min(m[1] for m in methods)

        self._write(
            f"  {'Method':<14} {'Time':>9}  {'Result':>16}  Correct\n", "accent")
        self._writeln("  " + "─" * 52, "dim")

        for name, t, v in methods:
            is_best  = (t == fastest)
            v_str    = fmt(v, agg_fn) if not (isinstance(v, float) and np.isnan(v)) else "n/a"
            ok       = close_enough(v, val)
            tag      = "yellow" if is_best else "normal"
            marker   = "  ◀ fastest" if is_best else ""
            self._write(f"  {name:<14} {t:>7.3f}ms  {v_str:>16}  ", tag)
            self._write("PASS" if ok else "FAIL", "green" if ok else "red")
            self._writeln(marker, tag)
        if r_kd is None:
            self._writeln("  KDTree         skipped (dataset too large)", "dim")

        bf_su = t_bf / t_ts if t_ts > 0 else 0
        self._writeln(f"\n  Tsunami  {bf_su:.1f}× faster than Brute Force", "dim")
        self._writeln()

        # Push to viz window if open
        if self._viz_win and self._viz_win.winfo_exists() and self._viz_win.winfo_viewable():
            idx_stats = {
                "n_regions": getattr(self.idx, '_n_regions', 1),
                "regions": self._get_region_bounds(),
            }
            self._viz_win.update_query(
                self.data, self.col_names, q, agg_fn, agg_col_name,
                r_ts, t_ts, t_np, t_kd, t_bf, idx_stats)

    def _get_region_bounds(self):
        """Try to extract per-region bounding boxes from the index internals."""
        try:
            regions = []
            for rb in (getattr(self.idx, '_regions', None) or []):
                bounds_raw = (getattr(rb, 'bounds', None) or
                              getattr(rb, 'bbox',   None) or
                              getattr(rb, '_bounds', None))
                if bounds_raw is not None:
                    # expect shape (D, 2) or list of (lo, hi) per dim
                    arr = np.array(bounds_raw)
                    if arr.shape == (len(self.col_names), 2):
                        regions.append(arr)
            return regions
        except Exception:
            return []

    # ── Data loading (sidebar buttons) ────────────────────────────────────────

    def _pick_csv(self):
        path = filedialog.askopenfilename(
            title="Open dataset",
            filetypes=[("CSV / Parquet", "*.csv *.parquet"),
                       ("CSV files", "*.csv"),
                       ("Parquet files", "*.parquet"),
                       ("All files", "*.*")])
        if path:
            self._run_cmd(f'load "{path}"')

    def _load_demo(self):
        def _do():
            n = int(self._demo_rows.get())
            self._writeln(f"  Loading California Housing demo ({n:,} rows)…", "dim")
            data, cols = generate_california(n=n)
            if self._custom_workload.get():
                workload = self._ask_custom_workload(data, cols)
                if not workload:
                    self._writeln("  Build cancelled.", "yellow")
                    return
                self._writeln(f"  Building Tsunami index (learned) with {len(workload)} queries…", "dim")
            else:
                workload = []
                self._writeln("  Building Tsunami index…", "dim")
            t0  = time.perf_counter()
            idx = build_index(data, cols, workload)
            ms  = (time.perf_counter() - t0) * 1000
            if len(data) <= 2_000_000:
                self._writeln("  Building KDTree…", "dim")
                tree = KDTree(data)
            else:
                self._writeln(f"  KDTree skipped ({len(data):,} rows — too large).", "dim")
                tree = None
            self.data = data; self.col_names = cols
            self.idx = idx;   self.tree = tree
            self.source = "California Housing"
            self.after(0, lambda: self._update_stats(ms))
            self._writeln(
                f"  Ready — {len(data):,} rows × {len(cols)} columns, "
                f"index built in {ms:.0f} ms", "green")
            self._writeln()
        self._writeln()
        self._write("tsunami> ", "prompt")
        self._writeln("Use California Housing demo", "cmd")
        _threading.Thread(target=_do, daemon=True).start()

    def _open_viz(self):
        if self._viz_win is None or not self._viz_win.winfo_exists():
            self._viz_win = VisualizationWindow(self)
        else:
            self._viz_win.deiconify()
            self._viz_win.lift()

    def _ask_custom_workload(self, data, col_names):
        """Show a dialog for the user to enter training queries. Returns list of Query objects."""
        from query_cli import parse_query, ParseError

        result = []
        win = tk.Toplevel(self)
        win.title("Training Queries")
        win.geometry("640x420")
        win.configure(bg=BG)
        win.grab_set()

        tk.Label(win, text="Enter training queries (one per line):",
                 font=FONT_B, bg=BG, fg=ACCENT).pack(anchor="w", padx=14, pady=(14,4))
        tk.Label(win,
                 text="These teach Tsunami which columns and ranges you care about.\n"
                      "Use any SELECT aggregate query — WHERE conditions are what matter.",
                 font=("Consolas", 9), bg=BG, fg=FG_DIM, justify="left",
                 ).pack(anchor="w", padx=14)

        # file load row
        file_row = tk.Frame(win, bg=BG)
        file_row.pack(fill=tk.X, padx=14, pady=(4, 0))
        tk.Label(file_row, text="Load from file:", font=("Consolas", 9),
                 bg=BG, fg=FG_DIM).pack(side=tk.LEFT)

        def _load_file():
            path = filedialog.askopenfilename(
                title="Open query file",
                filetypes=[("Text/SQL files", "*.txt *.sql"), ("All files", "*.*")])
            if not path:
                return
            try:
                with open(path, "r") as f:
                    content = f.read()
                txt.delete("1.0", tk.END)
                txt.insert("1.0", content)
                status.config(text=f"  Loaded: {os.path.basename(path)}", fg=GREEN)
            except Exception as e:
                status.config(text=f"  Could not read file: {e}", fg=RED)

        tk.Button(file_row, text="Browse…", command=_load_file,
                  bg="#313244", fg=FG, font=("Consolas", 9), relief="flat",
                  cursor="hand2").pack(side=tk.LEFT, padx=6)

        txt = tk.Text(win, font=FONT_T, bg=PANEL, fg=FG, insertbackground=FG,
                      relief="flat", padx=10, pady=8, height=12)
        txt.pack(fill=tk.BOTH, expand=True, padx=14, pady=8)

        # pre-fill with a couple of examples using real column names
        c0, c1 = col_names[0], col_names[min(1, len(col_names)-1)]
        p25 = float(np.percentile(data[:, 0], 25))
        p75 = float(np.percentile(data[:, 0], 75))
        txt.insert("1.0",
            f"SELECT COUNT(*) FROM data WHERE {c0} >= {p75:.3g}\n"
            f"SELECT AVG({c1}) FROM data WHERE {c0} BETWEEN {p25:.3g} AND {p75:.3g}\n")

        status = tk.Label(win, text="", font=("Consolas", 9), bg=BG, fg=RED)
        status.pack(anchor="w", padx=14)

        def _ok():
            lines = [l.strip() for l in txt.get("1.0", tk.END).splitlines()
                     if l.strip() and not l.strip().startswith("#")]
            if not lines:
                status.config(text="  Enter at least one query.")
                return
            parsed = []
            for line in lines:
                try:
                    q, _, _ = parse_query(line, data, col_names)
                    parsed.append(q)
                except Exception as e:
                    status.config(text=f"  Parse error: {e}")
                    return
            result.extend(parsed)
            win.destroy()

        def _cancel():
            win.destroy()

        btn_row = tk.Frame(win, bg=BG)
        btn_row.pack(fill=tk.X, padx=14, pady=(0,12))
        tk.Button(btn_row, text="Build with these queries", command=_ok,
                  bg=ACCENT, fg=BG, font=FONT_B, relief="flat",
                  activebackground="#b4befe", cursor="hand2").pack(side=tk.LEFT, padx=(0,8))
        tk.Button(btn_row, text="Cancel", command=_cancel,
                  bg="#313244", fg=FG, font=FONT, relief="flat",
                  cursor="hand2").pack(side=tk.LEFT)

        self.wait_window(win)
        return result  # empty list if cancelled

    def _update_stats(self, build_ms):
        N, D = self.data.shape
        self._rows_lbl.config( text=f"{N:,}")
        self._cols_lbl.config( text=str(D))
        self._build_lbl.config(text=f"{build_ms:.0f} ms")
        self._src_lbl.config(  text=self.source[:18])
        self._refresh_examples()

    def _refresh_examples(self):
        # clear old buttons
        for w in self._example_frame.winfo_children():
            w.destroy()

        cols = self.col_names
        if not cols:
            # no dataset yet — show placeholder
            tk.Label(self._example_frame, text="  Load a dataset first",
                     font=FONT, bg=PANEL, fg=FG_DIM).pack(anchor="w", padx=8)
            return

        data = self.data
        # pick representative columns by role
        c0   = cols[0]                                  # first col (numeric filter)
        c_hi = cols[np.argmax(data.std(axis=0))]       # highest-variance col (good for AVG)
        c_sum = cols[np.argmax(data.max(axis=0))]      # largest-range col (good for SUM)
        c_ord = cols[np.argmin(data.std(axis=0))]      # lowest-variance col (good for ORDER BY)

        # representative threshold: 75th percentile of first col
        thresh = float(np.percentile(data[:, 0], 75))
        lo_val = float(np.percentile(data[:, 0], 25))
        hi_val = float(np.percentile(data[:, 0], 90))

        has_lat = "Latitude"  in cols and "Longitude" in cols
        has_llc = "latitude"  in cols and "longitude" in cols

        examples = [
            (f"COUNT(*) where {c0}≥ p75",
             f"SELECT COUNT(*) FROM data WHERE {c0} >= {thresh:.3g}"),
            (f"AVG {c_hi}",
             f"SELECT AVG({c_hi}) FROM data WHERE {c0} BETWEEN {lo_val:.3g} AND {hi_val:.3g}"),
            (f"SUM {c_sum}",
             f"SELECT SUM({c_sum}) FROM data WHERE {c0} >= {lo_val:.3g}"),
            (f"MIN {c_hi}",
             f"SELECT MIN({c_hi}) FROM data WHERE {c0} >= {thresh:.3g}"),
            (f"MAX {c_hi}",
             f"SELECT MAX({c_hi}) FROM data WHERE {c0} <= {hi_val:.3g}"),
            (f"SELECT * ORDER BY {c_hi} DESC",
             f"SELECT * FROM data ORDER BY {c_hi} DESC LIMIT 10"),
            (f"Top 20 by {c_hi}",
             f"SELECT * FROM data WHERE {c0} >= {thresh:.3g} ORDER BY {c_hi} DESC LIMIT 20"),
            (f"Bottom 10 by {c_ord}",
             f"SELECT * FROM data ORDER BY {c_ord} ASC LIMIT 10"),
        ]

        if has_lat:
            examples.append(("Place = San Francisco",
                "SELECT COUNT(*) FROM data WHERE Place = 'San Francisco'"))
        if has_llc:
            examples.append(("Neighbourhood = Williamsburg",
                "SELECT COUNT(*) FROM data WHERE Neighbourhood = 'Williamsburg'"))

        for label, sql in examples:
            self._btn(self._example_frame, label,
                      lambda s=sql: self._insert_example(s), "#1e1e2e")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = TsunamiGUI()
    app.mainloop()
