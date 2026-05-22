#!/usr/bin/env python3
"""graph_compare.py — compare two nav-graph builders side-by-side.

Reads:
  inputs/nav_graph_node.json
  inputs/nav_graph_node_timing.csv
  inputs/nav_graph_node_e2e.json
  inputs/nav_graph_node_e2e_timing.csv

Writes:
  outputs/plots/*.png      – every comparison plot
  outputs/summary.csv      – flat per-metric table (baseline | e2e | delta)
  outputs/report.md        – narrative report with embedded plot links
  outputs/per_node_nn.csv  – per-node "distance to nearest in other graph"

Run:
  python graph_compare.py
  python graph_compare.py --base-dir /tmp/some/other/dir
  python graph_compare.py --no-planning  # skip the (slow-ish) planning sim
"""

from __future__ import annotations

import argparse
import csv as _csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Minimal CSV → dict-of-arrays helper.  Pandas is *not* in the project
# venv so we deliberately avoid it; everything we need from a DataFrame
# here is column min/mean/median/dropna(), all trivially expressible on
# masked numpy arrays.
def read_csv_as_arrays(path: Path) -> Dict[str, np.ndarray]:
    """Returns a dict {column_name: np.ndarray}.  Numeric columns get
    parsed to float64 with non-numeric cells → NaN; everything else stays
    as a Python-object array (we only ever ask for numeric ops on the
    timing columns, so the object fallback is fine for, e.g., timestamps
    if they come through as strings)."""
    with open(path, newline='') as f:
        reader = _csv.reader(f)
        rows = list(reader)
    if not rows:
        return {}
    header = rows[0]
    cols: Dict[str, List[str]] = {h: [] for h in header}
    for r in rows[1:]:
        if len(r) != len(header):
            # Tolerate a half-written final line (Ctrl-C mid-row).
            r = (r + [''] * len(header))[:len(header)]
        for h, v in zip(header, r):
            cols[h].append(v)
    out: Dict[str, np.ndarray] = {}
    for h, raw in cols.items():
        try:
            arr = np.array([np.nan if (s == '' or s is None) else float(s)
                            for s in raw], dtype=np.float64)
            out[h] = arr
        except (TypeError, ValueError):
            out[h] = np.array(raw, dtype=object)
    return out


def col_mean(table: Dict[str, np.ndarray], col: str) -> float:
    if col not in table or table[col].size == 0:
        return float('nan')
    v = table[col]
    if v.dtype == object:
        return float('nan')
    finite = v[np.isfinite(v)]
    return float(finite.mean()) if finite.size > 0 else float('nan')


def col_median(table: Dict[str, np.ndarray], col: str) -> float:
    if col not in table or table[col].size == 0:
        return float('nan')
    v = table[col]
    if v.dtype == object:
        return float('nan')
    finite = v[np.isfinite(v)]
    return float(np.median(finite)) if finite.size > 0 else float('nan')


def table_is_empty(table: Optional[Dict[str, np.ndarray]]) -> bool:
    if table is None:
        return True
    for v in table.values():
        if v.size > 0:
            return False
    return True


def rows_to_skip_by_seconds(table_raw: Optional[Dict[str, np.ndarray]],
                            n_sec: float) -> int:
    """Translate "skip the first N seconds" into a row count for this CSV.

    Uses the ``frame_timestamp_sec`` column relative to that column's own
    first finite value, so each method's CSV maps independently — the
    same wall-clock duration is dropped regardless of image rate.

    Returns 0 when the trim is disabled or the column is missing.  If
    every row falls within the skip window, returns the row count
    (i.e. drop everything) — the rest of the pipeline tolerates an
    empty table gracefully.
    """
    if table_raw is None or n_sec <= 0.0:
        return 0
    ts = table_raw.get('frame_timestamp_sec')
    if ts is None or ts.size == 0:
        return 0
    finite = np.isfinite(ts)
    if not finite.any():
        return 0
    t0 = float(ts[finite][0])
    elapsed = ts - t0
    above = np.where(np.isfinite(elapsed) & (elapsed >= n_sec))[0]
    if above.size == 0:
        return int(ts.size)
    return int(above[0])


def warmup_node_id_threshold(table_raw: Optional[Dict[str, np.ndarray]],
                             skip_n: int) -> Optional[int]:
    """Pull ``num_nodes`` at the END of the warmup window from a raw CSV.

    The CSV's ``num_nodes`` column is the cumulative graph size at the end
    of each frame.  Both builders assign monotonically-increasing node IDs
    as new nodes are appended, so under no-removal-during-warmup
    (typical) every node with ID < the value at row ``skip_n - 1`` was
    created during the warmup span we want to discard.

    Returns ``None`` when the trim is disabled or the CSV is missing /
    short, in which case no node filtering is applied.
    """
    if table_raw is None or skip_n <= 0:
        return None
    nn = table_raw.get('num_nodes')
    if nn is None or nn.size == 0:
        return None
    idx = min(skip_n - 1, nn.size - 1)
    val = nn[idx]
    if not np.isfinite(val):
        return None
    return int(val)


def trim_first_n_rows(table: Optional[Dict[str, np.ndarray]],
                      n: int) -> Optional[Dict[str, np.ndarray]]:
    """Return a copy of ``table`` with the first ``n`` rows of every column
    sliced off.  Used to drop the e2e JIT-warmup spike (torch.compile +
    CUDA-graph capture + first cudnn-benchmark forwards) before any
    averaging or histogramming runs.  ``n <= 0`` is a no-op."""
    if table is None or n <= 0:
        return table
    return {k: (v[n:] if v.size > n else v[:0]) for k, v in table.items()}

DEFAULT_DIR = Path('/home/rohang73/Documents/odin_e2e/graph_compaRISION')

# Same RViz-rainbow lookup the saved-frame renderers use, just so the
# comparison plots feel visually consistent with what you saw at runtime.
import colorsys

def _rviz_rainbow(n: int = 256) -> np.ndarray:
    out = np.empty((n, 3), dtype=np.float32)
    for i, t in enumerate(np.linspace(0.0, 1.0, n)):
        h = (1.0 - float(t)) * 5.0 / 6.0
        out[i] = colorsys.hsv_to_rgb(h, 1.0, 1.0)
    return out

_RVIZ_LUT = _rviz_rainbow()

# Visual styling — fixed per method for figure consistency.
METHODS = ('baseline', 'e2e')
COLORS = {'baseline': '#1f77b4', 'e2e': '#d62728'}  # tab:blue, tab:red
LABELS = {
    'baseline': 'nav_graph_node (elevation + nav_graph_gpu)',
    'e2e':      'nav_graph_node_e2e (RGB → graph)',
}

# Every plot uses the same landscape footprint so the saved PDFs slot
# straight into reports / slides without aspect-ratio juggling.
LANDSCAPE_FIGSIZE = (12, 7)


def _save_fig(fig, plot_dir: Path, base_name: str) -> Path:
    """Write ``base_name.pdf`` (canonical vector output) AND
    ``base_name.png`` (for VSCode's markdown preview which doesn't render
    PDFs inline).  Returns the PDF path; the markdown builder swaps the
    extension to ``.png`` when emitting image embeds.
    """
    pdf_path = plot_dir / f'{base_name}.pdf'
    png_path = plot_dir / f'{base_name}.png'
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return pdf_path


# ─────────────────────────────────────────────────────────────────────
#  Graph dataclass + loader
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Graph:
    """In-memory view of a snapshot written by either node."""
    label: str               # 'baseline' or 'e2e'
    method: str              # method string from the JSON
    num_nodes: int
    num_edges: int
    positions: np.ndarray    # (N, 3) float32
    ids: np.ndarray          # (N,) int64
    types: np.ndarray        # (N,) int64 — 1=free, 2=frontier
    layer_names: List[str]
    scores: np.ndarray       # (N, L) float32 — empty if no layers
    edge_pairs: np.ndarray   # (E, 2) int64 of node IDs
    edge_weights: np.ndarray # (E,) float32
    frame_count: int
    saved_at: str
    source_path: Path

    # Built lazily — adjacency CSR keyed by row index, plus the inverse map.
    _id_to_idx: Optional[Dict[int, int]] = field(default=None, repr=False)
    _adj_csr: Optional[csr_matrix] = field(default=None, repr=False)

    @classmethod
    def load(cls, path: Path, label: str) -> 'Graph':
        with open(path) as f:
            data = json.load(f)
        nodes = data.get('nodes') or []
        edges = data.get('edges') or []
        layer_names = list(data.get('layer_names') or [])
        L = len(layer_names)
        N = len(nodes)
        positions = np.zeros((N, 3), dtype=np.float32)
        ids = np.zeros(N, dtype=np.int64)
        types = np.zeros(N, dtype=np.int64)
        scores = (
            np.full((N, L), np.nan, dtype=np.float32) if L > 0
            else np.empty((N, 0), dtype=np.float32)
        )
        for i, n in enumerate(nodes):
            pos = n.get('position') or [0.0, 0.0, 0.0]
            positions[i] = pos
            ids[i] = int(n.get('id', i))
            types[i] = int(n.get('type', 1))
            if L > 0:
                ns = n.get('scores') or {}
                for j, name in enumerate(layer_names):
                    if name in ns:
                        try:
                            scores[i, j] = float(ns[name])
                        except (TypeError, ValueError):
                            pass
        E = len(edges)
        edge_pairs = np.zeros((E, 2), dtype=np.int64)
        edge_weights = np.zeros(E, dtype=np.float32)
        for i, e in enumerate(edges):
            edge_pairs[i] = (int(e.get('node_id_0', -1)),
                             int(e.get('node_id_1', -1)))
            w = e.get('weight', float('nan'))
            try:
                edge_weights[i] = float(w)
            except (TypeError, ValueError):
                edge_weights[i] = float('nan')
        return cls(
            label=label,
            method=str(data.get('method', label)),
            num_nodes=N,
            num_edges=E,
            positions=positions,
            ids=ids,
            types=types,
            layer_names=layer_names,
            scores=scores,
            edge_pairs=edge_pairs,
            edge_weights=edge_weights,
            frame_count=int(data.get('frame_count', 0)),
            saved_at=str(data.get('saved_at_utc', '')),
            source_path=path,
        )

    def id_to_idx(self) -> Dict[int, int]:
        if self._id_to_idx is None:
            self._id_to_idx = {int(self.ids[i]): i for i in range(self.num_nodes)}
        return self._id_to_idx

    def drop_node_ids_below(self, min_id: int) -> 'Graph':
        """Return a copy with every node whose ID is < ``min_id`` removed.

        Edges with at least one endpoint below the threshold are also
        dropped.  Used to filter out warmup-era nodes once we know how
        many node IDs had already been assigned by the end of the trimmed
        timing window.
        """
        if min_id <= 0 or self.num_nodes == 0:
            return self
        keep = self.ids >= int(min_id)
        if bool(keep.all()):
            return self
        new_positions = self.positions[keep]
        new_ids = self.ids[keep]
        new_types = self.types[keep]
        new_scores = (
            self.scores[keep] if self.scores.shape[1] > 0 else self.scores
        )
        if self.num_edges > 0:
            em = ((self.edge_pairs[:, 0] >= int(min_id))
                  & (self.edge_pairs[:, 1] >= int(min_id)))
            new_edges = self.edge_pairs[em]
            new_weights = self.edge_weights[em]
        else:
            new_edges = self.edge_pairs
            new_weights = self.edge_weights
        return Graph(
            label=self.label,
            method=self.method,
            num_nodes=int(keep.sum()),
            num_edges=int(new_edges.shape[0]),
            positions=new_positions,
            ids=new_ids,
            types=new_types,
            layer_names=self.layer_names,
            scores=new_scores,
            edge_pairs=new_edges,
            edge_weights=new_weights,
            frame_count=self.frame_count,
            saved_at=self.saved_at,
            source_path=self.source_path,
        )

    def adjacency_csr(self) -> csr_matrix:
        """Symmetric CSR adjacency keyed by row index in `positions`.

        Weight = edge weight if finite, else the Euclidean distance between
        endpoints.  Edges referencing unknown IDs are silently dropped.
        """
        if self._adj_csr is not None:
            return self._adj_csr
        if self.num_nodes == 0 or self.num_edges == 0:
            self._adj_csr = csr_matrix((self.num_nodes, self.num_nodes), dtype=np.float32)
            return self._adj_csr
        id_to_idx = self.id_to_idx()
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []
        for k in range(self.num_edges):
            a = int(self.edge_pairs[k, 0])
            b = int(self.edge_pairs[k, 1])
            ia = id_to_idx.get(a, -1)
            ib = id_to_idx.get(b, -1)
            if ia < 0 or ib < 0:
                continue
            w = float(self.edge_weights[k])
            if not math.isfinite(w):
                w = float(np.linalg.norm(self.positions[ia] - self.positions[ib]))
            rows.extend((ia, ib))
            cols.extend((ib, ia))
            data.extend((w, w))
        self._adj_csr = csr_matrix(
            (data, (rows, cols)), shape=(self.num_nodes, self.num_nodes),
            dtype=np.float32,
        )
        return self._adj_csr


# ─────────────────────────────────────────────────────────────────────
#  Single-graph structural metrics
# ─────────────────────────────────────────────────────────────────────

def graph_stats(g: Graph) -> Dict[str, float]:
    """Per-graph scalar metrics — fed straight into the summary table."""
    stats: Dict[str, float] = {}
    stats['num_nodes'] = float(g.num_nodes)
    stats['num_edges'] = float(g.num_edges)
    stats['num_free']     = float(int((g.types == 1).sum()))
    stats['num_frontier'] = float(int((g.types == 2).sum()))
    stats['frame_count']  = float(g.frame_count)

    if g.num_nodes == 0:
        for k in ('bbox_x', 'bbox_y', 'bbox_z', 'xy_area_m2',
                  'xy_density_per_m2', 'z_mean', 'z_std',
                  'mean_nn_distance_m', 'median_nn_distance_m',
                  'avg_degree', 'mean_edge_len_m', 'max_edge_len_m',
                  'largest_cc_size', 'largest_cc_frac', 'num_cc'):
            stats[k] = float('nan')
        return stats

    p = g.positions
    bb = p.max(axis=0) - p.min(axis=0)
    stats['bbox_x'] = float(bb[0])
    stats['bbox_y'] = float(bb[1])
    stats['bbox_z'] = float(bb[2])
    stats['xy_area_m2'] = float(bb[0] * bb[1])
    stats['xy_density_per_m2'] = (
        float(g.num_nodes) / max(stats['xy_area_m2'], 1e-6)
    )
    stats['z_mean'] = float(p[:, 2].mean())
    stats['z_std']  = float(p[:, 2].std())

    # ── Node-density proxy: each node's distance to its NEAREST other
    # node in 3-D.  Mean across all nodes is a robust "how spread-out is
    # the graph?" number; lower ⇒ denser.  Independent of map extent
    # (unlike xy_density_per_m2), so the two methods compare fairly even
    # if they cover slightly different regions.
    if g.num_nodes >= 2:
        d_nn, _ = cKDTree(p).query(p, k=2)
        nn = d_nn[:, 1]
        stats['mean_nn_distance_m']   = float(nn.mean())
        stats['median_nn_distance_m'] = float(np.median(nn))
    else:
        stats['mean_nn_distance_m']   = float('nan')
        stats['median_nn_distance_m'] = float('nan')

    adj = g.adjacency_csr()
    # avg degree counts both directions, so 2E/N.
    stats['avg_degree'] = float(adj.nnz) / float(g.num_nodes)

    if adj.nnz > 0:
        edge_lens = adj.data  # one entry per directed edge — same set twice
        stats['mean_edge_len_m'] = float(edge_lens.mean())
        stats['max_edge_len_m']  = float(edge_lens.max())
    else:
        stats['mean_edge_len_m'] = float('nan')
        stats['max_edge_len_m']  = float('nan')

    n_cc, cc_labels = connected_components(adj, directed=False)
    sizes = np.bincount(cc_labels)
    largest = int(sizes.max()) if sizes.size > 0 else 0
    stats['num_cc']            = float(n_cc)
    stats['largest_cc_size']   = float(largest)
    stats['largest_cc_frac']   = float(largest) / float(g.num_nodes)

    return stats


# ─────────────────────────────────────────────────────────────────────
#  Pairwise (A↔B) similarity metrics
# ─────────────────────────────────────────────────────────────────────

def nn_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """For each row of A, Euclidean distance to nearest row of B."""
    if A.shape[0] == 0 or B.shape[0] == 0:
        return np.array([], dtype=np.float32)
    return cKDTree(B).query(A, k=1)[0].astype(np.float32)


def chamfer_distance(A: np.ndarray, B: np.ndarray) -> float:
    if A.shape[0] == 0 or B.shape[0] == 0:
        return float('nan')
    d_ab = nn_distances(A, B)
    d_ba = nn_distances(B, A)
    return float(0.5 * (d_ab.mean() + d_ba.mean()))


def hausdorff_distance(A: np.ndarray, B: np.ndarray) -> float:
    if A.shape[0] == 0 or B.shape[0] == 0:
        return float('nan')
    d_ab = nn_distances(A, B).max()
    d_ba = nn_distances(B, A).max()
    return float(max(d_ab, d_ba))


def bbox_iou_2d(A: np.ndarray, B: np.ndarray) -> float:
    if A.shape[0] == 0 or B.shape[0] == 0:
        return float('nan')
    a0, a1 = A[:, :2].min(0), A[:, :2].max(0)
    b0, b1 = B[:, :2].min(0), B[:, :2].max(0)
    inter_lo = np.maximum(a0, b0)
    inter_hi = np.minimum(a1, b1)
    inter_wh = np.maximum(inter_hi - inter_lo, 0.0)
    inter = float(inter_wh[0] * inter_wh[1])
    area_a = float((a1[0] - a0[0]) * (a1[1] - a0[1]))
    area_b = float((b1[0] - b0[0]) * (b1[1] - b0[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else float('nan')


def occupancy_iou_2d(A: np.ndarray, B: np.ndarray, cell_m: float) -> float:
    """Rasterize node positions into a shared 2D grid, compute IoU.

    Picks a common bounding box (xy union of both graphs) padded by 1 m,
    bins both node sets at ``cell_m`` resolution, then computes
    |A ∩ B| / |A ∪ B| over occupied cells.  Coarse-grained measure of
    "do these graphs cover the same area?"
    """
    if A.shape[0] == 0 or B.shape[0] == 0:
        return float('nan')
    pad = 1.0
    xy = np.vstack([A[:, :2], B[:, :2]])
    lo = xy.min(0) - pad
    hi = xy.max(0) + pad
    nx = max(1, int(math.ceil((hi[0] - lo[0]) / cell_m)))
    ny = max(1, int(math.ceil((hi[1] - lo[1]) / cell_m)))

    def _rasterize(P: np.ndarray) -> np.ndarray:
        ix = np.clip(((P[:, 0] - lo[0]) / cell_m).astype(np.int64), 0, nx - 1)
        iy = np.clip(((P[:, 1] - lo[1]) / cell_m).astype(np.int64), 0, ny - 1)
        m = np.zeros((nx, ny), dtype=bool)
        m[ix, iy] = True
        return m

    ma = _rasterize(A)
    mb = _rasterize(B)
    inter = int(np.logical_and(ma, mb).sum())
    union = int(np.logical_or(ma, mb).sum())
    return inter / union if union > 0 else float('nan')


def node_density_correlation(A: np.ndarray, B: np.ndarray,
                             cell_m: float) -> float:
    """Pearson correlation of 2D node-count grids — captures both extent
    AND density similarity (whereas occupancy IoU is binary)."""
    if A.shape[0] == 0 or B.shape[0] == 0:
        return float('nan')
    pad = 1.0
    xy = np.vstack([A[:, :2], B[:, :2]])
    lo = xy.min(0) - pad
    hi = xy.max(0) + pad
    nx = max(1, int(math.ceil((hi[0] - lo[0]) / cell_m)))
    ny = max(1, int(math.ceil((hi[1] - lo[1]) / cell_m)))

    def _hist(P: np.ndarray) -> np.ndarray:
        ix = np.clip(((P[:, 0] - lo[0]) / cell_m).astype(np.int64), 0, nx - 1)
        iy = np.clip(((P[:, 1] - lo[1]) / cell_m).astype(np.int64), 0, ny - 1)
        h, _, _ = np.histogram2d(
            ix, iy, bins=[np.arange(nx + 1), np.arange(ny + 1)],
        )
        return h.astype(np.float64).ravel()

    ha = _hist(A)
    hb = _hist(B)
    if ha.std() < 1e-9 or hb.std() < 1e-9:
        return float('nan')
    return float(np.corrcoef(ha, hb)[0, 1])


# ─────────────────────────────────────────────────────────────────────
#  Planning-quality proxy
# ─────────────────────────────────────────────────────────────────────

def planning_metrics(g: Graph, query_xy: np.ndarray,
                     n_pairs: int) -> Dict[str, float]:
    """Sample ``n_pairs`` random (start, goal) xy locations from ``query_xy``,
    snap each end to the nearest node in ``g``, run Dijkstra, summarise.

    The ``query_xy`` input is shared between methods (it's the union of
    both node sets), so the same world-locations are tested in both graphs
    — giving us an apples-to-apples planning comparison.

    Metrics returned:
      success_rate          — fraction of pairs with finite shortest-path
      mean_path_length_m    — across successful pairs
      median_path_length_m
      mean_snap_distance_m  — start/goal → nearest-node distance
    """
    out = {
        'planning_success_rate':       float('nan'),
        'planning_mean_path_length_m': float('nan'),
        'planning_median_path_length_m': float('nan'),
        'planning_mean_snap_distance_m': float('nan'),
        'planning_n_pairs':            0.0,
    }
    if g.num_nodes < 2 or query_xy.shape[0] < 2 or n_pairs <= 0:
        return out

    rng = np.random.default_rng(seed=42)
    n_q = query_xy.shape[0]
    pairs_idx = rng.integers(0, n_q, size=(n_pairs, 2))
    keep = pairs_idx[:, 0] != pairs_idx[:, 1]
    pairs_idx = pairs_idx[keep]
    if pairs_idx.shape[0] == 0:
        return out

    starts_xy = query_xy[pairs_idx[:, 0]]
    goals_xy = query_xy[pairs_idx[:, 1]]

    # Snap to nearest graph node by xy.
    tree = cKDTree(g.positions[:, :2])
    d_start, i_start = tree.query(starts_xy, k=1)
    d_goal,  i_goal  = tree.query(goals_xy, k=1)

    adj = g.adjacency_csr()
    # Run Dijkstra once per unique start to amortise.  We have many start
    # indices possibly repeated; np.unique gives us the inverse map back.
    uniq_starts, inv = np.unique(i_start, return_inverse=True)
    dist_matrix = dijkstra(adj, indices=uniq_starts, directed=False)

    path_lens = dist_matrix[inv, i_goal]
    finite = np.isfinite(path_lens)
    out['planning_n_pairs'] = float(pairs_idx.shape[0])
    out['planning_success_rate'] = float(finite.mean())
    if finite.any():
        out['planning_mean_path_length_m']   = float(path_lens[finite].mean())
        out['planning_median_path_length_m'] = float(np.median(path_lens[finite]))
    out['planning_mean_snap_distance_m'] = float(
        np.concatenate([d_start, d_goal]).mean()
    )
    return out


# ─────────────────────────────────────────────────────────────────────
#  Plot helpers — one figure per concern, saved to outputs/plots/
# ─────────────────────────────────────────────────────────────────────

def _setup_plot_dir(outdir: Path) -> Path:
    plot_dir = outdir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def plot_xy_overlay(graphs: Dict[str, Graph], plot_dir: Path) -> Path:
    """All nodes (free + frontier merged) shown as a single series per method.

    The frontier/free split is intentionally collapsed here: the e2e
    method doesn't classify frontiers, so showing them separately for the
    baseline only would be visually asymmetric.  Per-type breakdowns live
    in the summary table instead.

    Layout note: no title — the legend is anchored above the top border
    (where the title would sit) so the plotting area stays maximised.
    """
    fig, ax = plt.subplots(figsize=LANDSCAPE_FIGSIZE)
    for label, g in graphs.items():
        if g.num_nodes == 0:
            continue
        ax.scatter(
            g.positions[:, 0], g.positions[:, 1],
            s=8, alpha=0.6, color=COLORS[label],
            label=f'{label} nodes (n={g.num_nodes})',
        )
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    # Smaller axis-tick labels so the metre numbers don't crowd the
    # plotting area when the bounding box is wide.
    ax.tick_params(axis='both', labelsize=8)
    ax.grid(True, alpha=0.3)
    # Legend anchored above the axes — bbox y > 1.0 places it in the
    # space the title would normally occupy.  ncol = number of series so
    # entries lay out horizontally instead of stacking vertically.
    n_series = sum(1 for g in graphs.values() if g.num_nodes > 0)
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=max(1, n_series),
        fontsize=8,
        frameon=False,
        borderaxespad=0.0,
    )
    return _save_fig(fig, plot_dir, 'xy_overlay')


def plot_z_histograms(graphs: Dict[str, Graph], plot_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=LANDSCAPE_FIGSIZE)
    bins = 40
    for label, g in graphs.items():
        if g.num_nodes == 0:
            continue
        ax.hist(g.positions[:, 2], bins=bins, alpha=0.5,
                color=COLORS[label], label=f'{label} (n={g.num_nodes})')
    ax.set_xlabel('z (m)')
    ax.set_ylabel('node count')
    ax.set_title('Per-method Z distribution')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    return _save_fig(fig, plot_dir, 'z_histograms')


def plot_degree_histograms(graphs: Dict[str, Graph], plot_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=LANDSCAPE_FIGSIZE)
    max_deg = 0
    deg_arrays = {}
    for label, g in graphs.items():
        if g.num_nodes == 0:
            deg_arrays[label] = np.array([])
            continue
        adj = g.adjacency_csr()
        deg = np.asarray((adj != 0).sum(axis=1)).ravel()
        deg_arrays[label] = deg
        if deg.size > 0:
            max_deg = max(max_deg, int(deg.max()))
    bins = np.arange(0, max_deg + 2) - 0.5
    for label, deg in deg_arrays.items():
        if deg.size == 0:
            continue
        ax.hist(deg, bins=bins, alpha=0.55, color=COLORS[label],
                label=f'{label} (mean={deg.mean():.2f})')
    ax.set_xlabel('node degree')
    ax.set_ylabel('count')
    ax.set_title('Degree distribution')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    return _save_fig(fig, plot_dir, 'degree_histograms')


def plot_nn_histograms(nn_b_to_e: np.ndarray, nn_e_to_b: np.ndarray,
                       plot_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=LANDSCAPE_FIGSIZE)
    if nn_b_to_e.size > 0:
        ax.hist(nn_b_to_e, bins=40, alpha=0.55, color=COLORS['baseline'],
                label=f'baseline → e2e (n={nn_b_to_e.size})')
    if nn_e_to_b.size > 0:
        ax.hist(nn_e_to_b, bins=40, alpha=0.55, color=COLORS['e2e'],
                label=f'e2e → baseline (n={nn_e_to_b.size})')
    ax.set_xlabel('Euclidean distance to nearest in other graph (m)')
    ax.set_ylabel('node count')
    ax.set_title('Nearest-neighbour distance distribution')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    return _save_fig(fig, plot_dir, 'nn_distance_histograms')


TableType = Dict[str, np.ndarray]


def plot_timing_per_frame(timings: Dict[str, Optional[TableType]],
                          plot_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=LANDSCAPE_FIGSIZE)
    for label, df in timings.items():
        if table_is_empty(df) or 't_frame_total_ms' not in df:
            continue
        ax.plot(df['frame_index'], df['t_frame_total_ms'],
                color=COLORS[label], alpha=0.7,
                label=f'{label} (mean={col_mean(df, "t_frame_total_ms"):.1f} ms)')
    ax.set_xlabel('frame index')
    ax.set_ylabel('frame total time (ms)')
    ax.set_title('Per-frame total processing time')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    return _save_fig(fig, plot_dir, 'timing_per_frame')


def plot_timing_vs_nodes(timings: Dict[str, Optional[TableType]],
                         plot_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=LANDSCAPE_FIGSIZE)
    for label, df in timings.items():
        if table_is_empty(df):
            continue
        if 't_frame_total_ms' not in df or 'num_nodes' not in df:
            continue
        ax.scatter(df['num_nodes'], df['t_frame_total_ms'],
                   s=6, alpha=0.5, color=COLORS[label],
                   label=f'{label} (median={col_median(df, "t_frame_total_ms"):.1f} ms)')
    ax.set_xlabel('number of global graph nodes')
    ax.set_ylabel('frame total time (ms)')
    ax.set_title('Frame total time vs graph size')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    return _save_fig(fig, plot_dir, 'timing_vs_nodes')


def plot_timing_breakdown(timings: Dict[str, Optional[TableType]],
                          plot_dir: Path) -> Path:
    """Mean per-step time, side-by-side stacked bars.

    The two methods don't share sub-step names — baseline has
    parse/emap/local/merge/other; e2e has inference/merge/edges/other.
    We render each method's stack independently and label the segments.
    Inter-method comparison is "which stack is taller and where does the
    height come from".

    Layout details: each segment's in-bar label is a single line
    ``"<step> <ms>"`` (no internal newline) and the y-axis upper bound
    is padded above the tallest stack so the "total: X ms" annotation
    sits cleanly inside the axes — well below the top border.
    """
    fig, ax = plt.subplots(figsize=LANDSCAPE_FIGSIZE)
    step_specs = {
        'baseline': [
            ('t_parse_ms', 'parse cloud'),
            ('t_emap_ms',  'elev. map update'),
            ('t_local_ms', 'local graph (GPU)'),
            ('t_merge_ms', 'global merge'),
            ('t_other_ms', 'occupancy grid generation'),
        ],
        'e2e': [
            ('t_inference_ms', 'model inference'),
            ('t_merge_ms',     'global merge'),
            ('t_edges_ms',     'edge build'),
            ('t_other_ms',     'other'),
        ],
    }
    x_positions = {'baseline': 0, 'e2e': 1}
    bar_width = 0.6
    cmap = plt.get_cmap('tab10')

    # ── Pass 1: compute per-method (step, mean_ms) pairs + totals so we
    # can size the y-axis before drawing.  Drawing in a second pass keeps
    # the total-annotation placement consistent across methods.
    plan: Dict[str, List[Tuple[str, float]]] = {}
    totals: Dict[str, float] = {}
    for label, df in timings.items():
        if table_is_empty(df):
            continue
        rows: List[Tuple[str, float]] = []
        for col, pretty in step_specs[label]:
            if col not in df:
                continue
            v = col_mean(df, col)
            if not math.isfinite(v):
                continue
            rows.append((pretty, v))
        if rows:
            plan[label] = rows
            totals[label] = sum(v for _, v in rows)

    if not totals:
        ax.set_ylabel('mean per-frame time (ms)')
        ax.grid(True, axis='y', alpha=0.3)
        return _save_fig(fig, plot_dir, 'timing_breakdown')

    max_total = max(totals.values())
    # 22% headroom above the tallest stack: leaves a comfortable gap
    # between the bar top and the upper border, with room for the total
    # annotation in between.
    ax.set_ylim(0, max_total * 1.22)

    # ── Pass 2: draw bars, in-bar labels, total annotation.
    for label, rows in plan.items():
        cum = 0.0
        for j, (pretty, v) in enumerate(rows):
            ax.bar(x_positions[label], v, bottom=cum,
                   width=bar_width, color=cmap(j),
                   edgecolor='black', linewidth=0.4,
                   label=f'{label}: {pretty}')
            ax.text(x_positions[label], cum + v / 2.0,
                    f'{pretty} {v:.1f} ms',
                    ha='center', va='center', fontsize=8)
            cum += v
        # Total goes a fixed fraction of max_total above the bar, so it
        # always sits below the top border by ~17% of max_total.
        ax.text(x_positions[label],
                cum + max_total * 0.05,
                f'total: {cum:.1f} ms',
                ha='center', va='bottom',
                fontsize=10, weight='bold')

    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels(list(x_positions.keys()))
    ax.set_ylabel('mean per-frame time (ms)')
    ax.grid(True, axis='y', alpha=0.3)
    return _save_fig(fig, plot_dir, 'timing_breakdown')


def plot_timing_histograms(timings: Dict[str, Optional[TableType]],
                           plot_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=LANDSCAPE_FIGSIZE)
    bins = 40
    for label, df in timings.items():
        if table_is_empty(df) or 't_frame_total_ms' not in df:
            continue
        vals = df['t_frame_total_ms']
        if vals.dtype == object:
            continue
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            continue
        ax.hist(finite, bins=bins, alpha=0.55,
                color=COLORS[label],
                label=f'{label} (median={col_median(df, "t_frame_total_ms"):.1f} ms)')
    ax.set_xlabel('per-frame total time (ms)')
    ax.set_ylabel('count')
    ax.set_title('Frame-total time histogram')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    return _save_fig(fig, plot_dir, 'timing_histograms')


def plot_graph_growth(timings: Dict[str, Optional[TableType]],
                      plot_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=LANDSCAPE_FIGSIZE)
    for label, df in timings.items():
        if table_is_empty(df) or 'num_nodes' not in df:
            continue
        final_n = int(df['num_nodes'][-1]) if df['num_nodes'].size > 0 else 0
        ax.plot(df['frame_index'], df['num_nodes'],
                color=COLORS[label], alpha=0.85,
                label=f'{label} (final={final_n})')
    ax.set_xlabel('frame index')
    ax.set_ylabel('global graph node count')
    ax.set_title('Graph growth over time')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    return _save_fig(fig, plot_dir, 'graph_growth')


# ─────────────────────────────────────────────────────────────────────
#  Report assembly
# ─────────────────────────────────────────────────────────────────────

def build_summary_rows(stats: Dict[str, Dict[str, float]],
                       pairwise: Dict[str, float]) -> List[Dict[str, Any]]:
    """One row per metric — baseline value, e2e value, delta (e2e-baseline).

    Pairwise metrics (chamfer, hausdorff, etc.) appear at the bottom as
    single-valued rows with the value shown under "baseline" (the
    interpretation is symmetric — they describe the comparison itself).
    """
    rows: List[Dict[str, Any]] = []
    seen: List[str] = []
    for k in list(stats['baseline'].keys()) + list(stats['e2e'].keys()):
        if k not in seen:
            seen.append(k)
    for k in seen:
        a = stats['baseline'].get(k, float('nan'))
        b = stats['e2e'].get(k, float('nan'))
        delta = (b - a) if (math.isfinite(a) and math.isfinite(b)) else float('nan')
        rows.append({
            'metric': k, 'baseline': a, 'e2e': b,
            'delta_e2e_minus_baseline': delta,
        })
    for k, v in pairwise.items():
        rows.append({
            'metric': k, 'baseline': v, 'e2e': '',
            'delta_e2e_minus_baseline': '',
        })
    return rows


def write_summary_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    cols = ['metric', 'baseline', 'e2e', 'delta_e2e_minus_baseline']
    with open(path, 'w', newline='') as f:
        w = _csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _fmt(x) -> str:
    if isinstance(x, (int, np.integer)):
        return f'{int(x)}'
    if isinstance(x, (float, np.floating)):
        if not math.isfinite(float(x)):
            return 'NaN'
        return f'{float(x):.3f}'
    return str(x)


def build_markdown_report(summary: List[Dict[str, Any]],
                          plot_paths: Dict[str, Path],
                          graphs: Dict[str, Graph],
                          skip_first_n_seconds: float = 0.0) -> str:
    lines: List[str] = []
    lines.append('# nav-graph comparison report\n')
    lines.append('## Sources\n')
    for label, g in graphs.items():
        lines.append(f'- **{label}** — `{g.source_path}` '
                     f'(saved {g.saved_at or "n/a"}, '
                     f'{g.num_nodes} nodes, {g.num_edges} edges, '
                     f'frame_count={g.frame_count})')
    if skip_first_n_seconds > 0.0:
        lines.append('')
        lines.append(
            f'> **Note:** the first **{skip_first_n_seconds:.1f} s** of '
            f'each timing CSV (measured against that CSV\'s own first '
            f'`frame_timestamp_sec`) were dropped before any plot / mean '
            f'/ median was computed — covers the e2e JIT-warmup spike.  '
            f'The same filter is propagated to the graph snapshots: every '
            f'node whose ID was assigned during the warmup window is also '
            f'excluded from the XY overlay, structural metrics, planning '
            f'queries, and nearest-neighbour comparisons.'
        )
    lines.append('')
    lines.append('## Summary table\n')
    lines.append('| metric | baseline | e2e | delta (e2e − baseline) |')
    lines.append('|---|---:|---:|---:|')
    for row in summary:
        lines.append(
            f'| {row["metric"]} | {_fmt(row["baseline"])} '
            f'| {_fmt(row["e2e"])} | {_fmt(row["delta_e2e_minus_baseline"])} |'
        )
    lines.append('\n## Plots\n')
    lines.append(
        'Each plot is saved as a PDF (canonical, vector) **and** as a PNG '
        'twin so the markdown preview can render it inline.  Click any PDF '
        'link to open the print-quality version.\n'
    )
    for name, path in plot_paths.items():
        if path is None:
            continue
        png_name = f'{path.stem}.png'
        pdf_name = f'{path.stem}.pdf'
        lines.append(f'### {name}\n')
        lines.append(f'![{name}](plots/{png_name})\n')
        lines.append(f'_[Open as PDF](plots/{pdf_name})_\n')
    lines.append('\n## Interpretation hints\n')
    lines.append(
        '- **mean_nn_distance_m / median_nn_distance_m** — average and '
        'median distance from each node to its nearest neighbour in 3-D '
        '("node density" proxy).  Lower ⇒ denser graph.  Independent of '
        'the map extent, so the two methods compare fairly even when they '
        'cover slightly different regions.\n'
        '- **Chamfer / Hausdorff** measure how spatially close the two '
        'node sets are.  Lower = more similar spatial coverage.  Chamfer '
        'is the mean of the two-way nearest-neighbour means; Hausdorff is '
        'the worst nearest-neighbour distance — sensitive to outliers.\n'
        '- **Bbox / occupancy / density correlation** describe extent and '
        'distribution agreement.  Occupancy IoU close to 1.0 ⇒ both '
        'graphs cover the same XY region at the chosen cell size.\n'
        '- **Planning success rate** is the apples-to-apples quality '
        'proxy: random world-locations are projected to each graph, then '
        'Dijkstra checks whether a path exists.  Equal or higher e2e '
        'success rate at comparable mean-path-length argues parity; '
        'higher mean-path-length is suspicious (detours).\n'
        '- **Timing breakdown** is mean per-step time over the whole run. '
        'Stacks are not directly comparable beyond their totals because '
        'the two methods have different sub-steps — but the total height '
        'is.  Watch the *vs. node count* scatter for scaling behaviour.\n'
    )
    return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Compare two saved nav-graph snapshots.')
    ap.add_argument('--base-dir', type=Path, default=DEFAULT_DIR,
                    help='Root directory containing inputs/ and outputs/ subfolders.')
    ap.add_argument('--baseline-json', type=Path, default=None,
                    help='Override path to baseline graph JSON.')
    ap.add_argument('--e2e-json', type=Path, default=None,
                    help='Override path to e2e graph JSON.')
    ap.add_argument('--baseline-csv', type=Path, default=None,
                    help='Override path to baseline timing CSV.')
    ap.add_argument('--e2e-csv', type=Path, default=None,
                    help='Override path to e2e timing CSV.')
    ap.add_argument('--occupancy-cell-m', type=float, default=0.5,
                    help='Cell size (m) for occupancy IoU + density correlation.')
    ap.add_argument('--planning-pairs', type=int, default=200,
                    help='Number of (start, goal) pairs sampled per method.')
    ap.add_argument('--no-planning', action='store_true',
                    help='Skip the (slow-ish) planning quality proxy.')
    ap.add_argument('--skip-first-n-seconds', type=float, default=15.0,
                    help=('Drop the first N seconds of each timing CSV '
                          '(measured against that CSV\'s own first '
                          '``frame_timestamp_sec``).  Defaults to 15.0 to '
                          'cover the e2e JIT-warmup spike (torch.compile '
                          '+ cudnn.benchmark + first CUDA-graph capture) '
                          'which lasts ~14 s on Ampere/Ada GPUs.  Same '
                          'filter is also applied to the graph snapshots, '
                          'so XY overlays / structural metrics / planning '
                          'queries see only post-warmup nodes.  Set to '
                          '0.0 to disable.'))
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    base = args.base_dir.expanduser()
    in_dir = base / 'inputs'
    out_dir = base / 'outputs'
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = _setup_plot_dir(out_dir)

    baseline_json = args.baseline_json or in_dir / 'nav_graph_node.json'
    e2e_json      = args.e2e_json      or in_dir / 'nav_graph_node_e2e.json'
    baseline_csv  = args.baseline_csv  or in_dir / 'nav_graph_node_timing.csv'
    e2e_csv       = args.e2e_csv       or in_dir / 'nav_graph_node_e2e_timing.csv'

    for label, p in (('baseline', baseline_json), ('e2e', e2e_json)):
        if not p.exists():
            print(f'ERROR: missing snapshot for {label}: {p}', flush=True)
            return 2

    print(f'Loading graphs:\n  baseline ← {baseline_json}\n  e2e      ← {e2e_json}')
    graphs: Dict[str, Graph] = {
        'baseline': Graph.load(baseline_json, 'baseline'),
        'e2e':      Graph.load(e2e_json,      'e2e'),
    }

    print('Loading timing CSVs:')
    timings: Dict[str, Optional[TableType]] = {}
    warmup_thresholds: Dict[str, Optional[int]] = {}
    skip_sec = float(args.skip_first_n_seconds)
    for label, p in (('baseline', baseline_csv), ('e2e', e2e_csv)):
        if p.exists():
            try:
                df_raw = read_csv_as_arrays(p)
                n_raw = next((v.size for v in df_raw.values() if v.size > 0), 0)
                # Translate "skip first N seconds" into a per-method row
                # count using this CSV's own first ``frame_timestamp_sec``.
                # Each method runs its own trim — they may differ in row
                # count if image rates differ, but the dropped span is the
                # same wall-clock duration.
                skip_n = rows_to_skip_by_seconds(df_raw, skip_sec)
                # Capture the warmup-era node-ID cap from the *raw* table
                # before we trim it, so we can filter the graph snapshot
                # the same way.
                warmup_thresholds[label] = warmup_node_id_threshold(df_raw, skip_n)
                df = trim_first_n_rows(df_raw, skip_n) if skip_n > 0 else df_raw
                n_kept = next((v.size for v in df.values() if v.size > 0), 0)
                timings[label] = df
                if skip_n > 0:
                    thr = warmup_thresholds[label]
                    thr_str = (f'drop node_id<{thr}' if thr is not None
                               else 'no node filter')
                    print(f'  {label} ← {p} ({n_raw} rows; dropped first '
                          f'{skip_sec:.1f}s → {skip_n} rows → {n_kept} kept; '
                          f'{thr_str})')
                else:
                    print(f'  {label} ← {p} ({n_raw} rows; no trim)')
            except Exception as exc:
                print(f'  {label} ← {p} FAILED ({exc}); skipping')
                timings[label] = None
                warmup_thresholds[label] = None
        else:
            print(f'  {label} ← (missing {p}, plots will skip)')
            timings[label] = None
            warmup_thresholds[label] = None

    # ── Apply the same warmup filter to the loaded graph snapshots so
    # every downstream plot / stat sees only post-warmup nodes.
    for label, thr in warmup_thresholds.items():
        if thr is not None and thr > 0:
            before = graphs[label].num_nodes
            graphs[label] = graphs[label].drop_node_ids_below(thr)
            after = graphs[label].num_nodes
            print(f'  {label} graph: dropped {before - after} warmup nodes '
                  f'(id < {thr}); {after} nodes / {graphs[label].num_edges} '
                  f'edges remain')

    # ── Structural stats ────────────────────────────────────────────
    print('Computing structural stats...')
    stats = {label: graph_stats(g) for label, g in graphs.items()}

    # ── Pairwise similarity ─────────────────────────────────────────
    print('Computing pairwise similarity...')
    A = graphs['baseline'].positions
    B = graphs['e2e'].positions
    pairwise: Dict[str, float] = {
        'chamfer_distance_3d_m':    chamfer_distance(A, B),
        'hausdorff_distance_3d_m':  hausdorff_distance(A, B),
        'bbox_iou_2d':              bbox_iou_2d(A, B),
        'occupancy_iou_2d':         occupancy_iou_2d(A, B, args.occupancy_cell_m),
        'node_density_correlation': node_density_correlation(A, B, args.occupancy_cell_m),
    }

    # Save per-node NN distances for downstream inspection.
    nn_b_to_e = nn_distances(A, B)
    nn_e_to_b = nn_distances(B, A)
    nn_df_rows: List[Dict] = []
    for i in range(A.shape[0]):
        nn_df_rows.append({
            'method': 'baseline',
            'node_id': int(graphs['baseline'].ids[i]),
            'x': float(A[i, 0]), 'y': float(A[i, 1]), 'z': float(A[i, 2]),
            'nearest_in_other_m': float(nn_b_to_e[i]) if nn_b_to_e.size else float('nan'),
        })
    for i in range(B.shape[0]):
        nn_df_rows.append({
            'method': 'e2e',
            'node_id': int(graphs['e2e'].ids[i]),
            'x': float(B[i, 0]), 'y': float(B[i, 1]), 'z': float(B[i, 2]),
            'nearest_in_other_m': float(nn_e_to_b[i]) if nn_e_to_b.size else float('nan'),
        })
    with open(out_dir / 'per_node_nn.csv', 'w', newline='') as f:
        if nn_df_rows:
            cols = ['method', 'node_id', 'x', 'y', 'z', 'nearest_in_other_m']
            w = _csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in nn_df_rows:
                w.writerow(r)

    # ── Planning quality proxy ──────────────────────────────────────
    if not args.no_planning:
        print('Sampling planning queries...')
        # Use the union of both node XY sets as the query distribution —
        # ensures both graphs are evaluated on the same world locations.
        if A.shape[0] + B.shape[0] > 0:
            query_xy = np.vstack([A[:, :2], B[:, :2]])
        else:
            query_xy = np.empty((0, 2), dtype=np.float32)
        for label, g in graphs.items():
            p = planning_metrics(g, query_xy, args.planning_pairs)
            stats[label].update(p)
    else:
        for label in graphs:
            for k in ('planning_success_rate', 'planning_mean_path_length_m',
                      'planning_median_path_length_m', 'planning_mean_snap_distance_m',
                      'planning_n_pairs'):
                stats[label][k] = float('nan')

    # Add timing summaries (means) into stats so the summary table carries them.
    for label, df in timings.items():
        if table_is_empty(df):
            continue
        for col in df.keys():
            if col.startswith('t_') and col.endswith('_ms'):
                stats[label][f'mean_{col}'] = col_mean(df, col)

    # ── Plots ───────────────────────────────────────────────────────
    print('Rendering plots...')
    plots: Dict[str, Path] = {}
    plots['xy_overlay']            = plot_xy_overlay(graphs, plot_dir)
    plots['z_histograms']          = plot_z_histograms(graphs, plot_dir)
    plots['degree_histograms']     = plot_degree_histograms(graphs, plot_dir)
    plots['nn_distance_histograms']= plot_nn_histograms(nn_b_to_e, nn_e_to_b, plot_dir)
    if any(not table_is_empty(df) for df in timings.values()):
        plots['timing_per_frame']  = plot_timing_per_frame(timings, plot_dir)
        plots['timing_vs_nodes']   = plot_timing_vs_nodes(timings, plot_dir)
        plots['timing_breakdown']  = plot_timing_breakdown(timings, plot_dir)
        plots['timing_histograms'] = plot_timing_histograms(timings, plot_dir)
        plots['graph_growth']      = plot_graph_growth(timings, plot_dir)

    # ── Summary CSV + markdown report ───────────────────────────────
    summary_rows = build_summary_rows(stats, pairwise)
    write_summary_csv(summary_rows, out_dir / 'summary.csv')
    report = build_markdown_report(
        summary_rows, plots, graphs, skip_first_n_seconds=skip_sec,
    )
    (out_dir / 'report.md').write_text(report)

    print(f'Done. Outputs in {out_dir}')
    print(f'  - summary.csv  ({len(summary_rows)} rows)')
    print(f'  - report.md')
    print(f'  - per_node_nn.csv')
    print(f'  - plots/  ({len(plots)} PNGs)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
