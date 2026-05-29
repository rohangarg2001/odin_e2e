#!/usr/bin/env python3
"""graph_compare.py — compare nav-graph builder snapshots side-by-side.

Reads:
  inputs/nav_graph_node.json          ← baseline (fixed name)
  inputs/nav_graph_node_timing.csv
  inputs/<name>.json                  ← any number of e2e variants, auto-discovered
  inputs/<name>_timing.csv

Writes:
  outputs/plots/*.png      – every comparison plot
  outputs/summary.csv      – flat per-metric table (one column per method)
  outputs/report.md        – narrative report with embedded plot links
  outputs/per_node_nn.csv  – per-node "distance to nearest baseline node"

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
import matplotlib.colors as mcolors
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


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

# Visual styling — populated dynamically in main() for however many methods exist.
# Baseline is always tab:blue; e2e variants cycle through _AUTO_PALETTE.
COLORS: Dict[str, str] = {'baseline': '#1f77b4'}
LABELS: Dict[str, str] = {'baseline': 'nav_graph_node (elevation + nav_graph_gpu)'}

_AUTO_PALETTE = ['#d62728', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf']


def _register_methods(method_names: List[str]) -> None:
    """Populate COLORS and LABELS for all discovered methods."""
    e2e_idx = 0
    for m in method_names:
        if m == 'baseline':
            continue
        if m not in COLORS:
            COLORS[m] = _AUTO_PALETTE[e2e_idx % len(_AUTO_PALETTE)]
        if m not in LABELS:
            LABELS[m] = m
        e2e_idx += 1

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

    def keep_nodes_up_to_id(self, max_id: int) -> 'Graph':
        """Return a copy keeping only nodes with ID <= ``max_id``.

        Edges with at least one endpoint above the threshold are also dropped.
        Used to compare graphs at the same "snapshot point" in time — since IDs
        are assigned monotonically, ID <= N means "the first N assigned nodes".
        """
        if self.num_nodes == 0:
            return self
        keep = self.ids <= int(max_id)
        if bool(keep.all()):
            return self
        new_positions = self.positions[keep]
        new_ids = self.ids[keep]
        new_types = self.types[keep]
        new_scores = (
            self.scores[keep] if self.scores.shape[1] > 0 else self.scores
        )
        if self.num_edges > 0:
            em = ((self.edge_pairs[:, 0] <= int(max_id))
                  & (self.edge_pairs[:, 1] <= int(max_id)))
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

def _rasterise_graph(g: 'Graph', cell_m: float) -> np.ndarray:
    """Return (M, 2) int32 array of unique grid cell (col, row) indices
    covering all node positions + sampled edge points at ``cell_m`` resolution.
    """
    if g.num_nodes == 0:
        return np.empty((0, 2), dtype=np.int32)
    xy = g.positions[:, :2]
    pts = [xy]
    if g.num_edges > 0 and g.edge_pairs.shape[0] > 0:
        id_to_idx: Dict[int, int] = {int(nid): i for i, nid in enumerate(g.ids)}
        step = cell_m / 2.0
        seg_pts: List[np.ndarray] = []
        for k in range(g.edge_pairs.shape[0]):
            ia = id_to_idx.get(int(g.edge_pairs[k, 0]))
            ib = id_to_idx.get(int(g.edge_pairs[k, 1]))
            if ia is None or ib is None:
                continue
            a, b = xy[ia], xy[ib]
            d = float(np.linalg.norm(b - a))
            if d < 1e-6:
                continue
            n_steps = max(2, int(d / step) + 1)
            ts = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)
            seg_pts.append(a + ts[:, None] * (b - a))
        if seg_pts:
            pts.append(np.vstack(seg_pts))
    all_xy = np.vstack(pts)
    cells = np.floor(all_xy / cell_m).astype(np.int32)
    return np.unique(cells, axis=0)  # (M, 2): col=[:,0], row=[:,1]


def covered_area_m2(g: 'Graph', cell_m: float = 0.5) -> float:
    """Area covered by the graph, in m².

    Rasterises node positions AND sampled edge points onto a 2D grid at
    ``cell_m`` resolution, then counts unique occupied cells.  This fills
    in the corridors between nodes so the result reflects the actual
    traversed footprint rather than just the node-point density.
    """
    return float(_rasterise_graph(g, cell_m).shape[0]) * cell_m * cell_m


def graph_stats(g: 'Graph', cell_m: float = 0.5) -> Dict[str, float]:
    """Per-graph scalar metrics — fed straight into the summary table."""
    stats: Dict[str, float] = {}
    stats['num_nodes'] = float(g.num_nodes)
    stats['num_edges'] = float(g.num_edges)
    stats['num_free']     = float(int((g.types == 1).sum()))
    stats['num_frontier'] = float(int((g.types == 2).sum()))
    stats['frame_count']  = float(g.frame_count)

    if g.num_nodes == 0:
        for k in ('bbox_x', 'bbox_y', 'bbox_z', 'xy_area_m2', 'covered_area_m2',
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
    stats['covered_area_m2'] = covered_area_m2(g, cell_m)
    stats['xy_density_per_m2'] = (
        float(g.num_nodes) / max(stats['covered_area_m2'], 1e-6)
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


def covered_area_iou_2d(g_a: 'Graph', g_b: 'Graph', cell_m: float) -> float:
    """IoU of the two graphs' traversed footprints at ``cell_m`` resolution.

    Uses ``_rasterise_graph`` (nodes + interpolated edge points) for both
    graphs, then computes |A ∩ B| / |A ∪ B| over occupied grid cells.
    Unlike ``occupancy_iou_2d`` (node-only), this accounts for the corridors
    between nodes that edges fill in.
    """
    cells_a = _rasterise_graph(g_a, cell_m)
    cells_b = _rasterise_graph(g_b, cell_m)
    if cells_a.shape[0] == 0 or cells_b.shape[0] == 0:
        return float('nan')
    set_a = set(map(tuple, cells_a.tolist()))
    set_b = set(map(tuple, cells_b.tolist()))
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
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

def planning_run(g: Graph, query_xy: np.ndarray,
                 n_pairs: int) -> Dict[str, Any]:
    """Sample ``n_pairs`` random (start, goal) xy locations from ``query_xy``,
    snap each end to the nearest node in ``g``, run Dijkstra (with
    predecessor tracking so paths can be reconstructed), and return both
    the per-pair details *and* the summary scalars.

    Return shape::

        {
            'metrics': {'planning_success_rate': ..., ... 5 keys ... },
            'pairs':   [
                {
                    'query_start_xy':  np.ndarray (2,)
                    'query_goal_xy':   np.ndarray (2,)
                    'start_node_idx':  int
                    'goal_node_idx':   int
                    'start_snap_m':    float
                    'goal_snap_m':     float
                    'path_length_m':   float           # inf if disconnected
                    'success':         bool
                    'path_indices':    Optional[List[int]]   # None on failure
                },
                ...
            ],
        }

    ``query_xy`` is shared between methods (the union of both node sets),
    so the same world-locations are tested in both graphs — giving us an
    apples-to-apples planning comparison.
    """
    out: Dict[str, Any] = {
        'metrics': {
            'planning_success_rate':         float('nan'),
            'planning_mean_path_length_m':   float('nan'),
            'planning_median_path_length_m': float('nan'),
            'planning_mean_snap_distance_m': float('nan'),
            'planning_n_pairs':              0.0,
        },
        'pairs': [],
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
    goals_xy  = query_xy[pairs_idx[:, 1]]

    # Snap to nearest graph node by XY.
    tree = cKDTree(g.positions[:, :2])
    d_start, i_start = tree.query(starts_xy, k=1)
    d_goal,  i_goal  = tree.query(goals_xy, k=1)

    adj = g.adjacency_csr()
    # Run Dijkstra once per unique start with predecessor tracking so we
    # can reconstruct each successful pair's actual path for the example
    # plots.  ``predecessors[k, j]`` is the predecessor of node j on the
    # shortest path from unique-start k; -9999 = unreachable / start.
    uniq_starts, inv = np.unique(i_start, return_inverse=True)
    dist_matrix, predecessors = dijkstra(
        adj, indices=uniq_starts, directed=False, return_predecessors=True,
    )

    path_lens = dist_matrix[inv, i_goal]
    finite = np.isfinite(path_lens)

    # Build per-pair detail rows, including reconstructed paths.
    SENTINEL = -9999
    for k in range(pairs_idx.shape[0]):
        sidx = int(i_start[k])
        gidx = int(i_goal[k])
        ok = bool(finite[k])
        path_indices: Optional[List[int]] = None
        if ok:
            row = int(inv[k])
            pred_row = predecessors[row]
            path = [gidx]
            cur = gidx
            # Walk predecessors back to the start of THIS source.
            # ``uniq_starts[row]`` is the start node index.
            safety = int(adj.shape[0]) + 5  # never loop forever
            while cur != int(uniq_starts[row]) and safety > 0:
                cur = int(pred_row[cur])
                if cur == SENTINEL:
                    path = []
                    break
                path.append(cur)
                safety -= 1
            if path:
                path_indices = list(reversed(path))
        out['pairs'].append({
            'query_start_xy': starts_xy[k].astype(np.float64),
            'query_goal_xy':  goals_xy[k].astype(np.float64),
            'start_node_idx': sidx,
            'goal_node_idx':  gidx,
            'start_snap_m':   float(d_start[k]),
            'goal_snap_m':    float(d_goal[k]),
            'path_length_m':  float(path_lens[k]),
            'success':        ok,
            'path_indices':   path_indices,
        })

    out['metrics']['planning_n_pairs']      = float(pairs_idx.shape[0])
    out['metrics']['planning_success_rate'] = float(finite.mean())
    if finite.any():
        out['metrics']['planning_mean_path_length_m']   = float(path_lens[finite].mean())
        out['metrics']['planning_median_path_length_m'] = float(np.median(path_lens[finite]))
    out['metrics']['planning_mean_snap_distance_m'] = float(
        np.concatenate([d_start, d_goal]).mean(),
    )
    return out


def planning_metrics(g: Graph, query_xy: np.ndarray,
                     n_pairs: int) -> Dict[str, float]:
    """Backwards-compatible wrapper — returns just the scalar metrics."""
    return planning_run(g, query_xy, n_pairs)['metrics']


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

    Layout notes: no title, no legend — the figure is intentionally
    minimal.  Both axes are inverted so the figure is rotated 180° from
    the raw odom-frame view (inverting x AND y is the same as rotating
    the canvas about its centre).
    """
    _XY_COLORS = {'baseline': '#d62728', 'e2e_vits': '#1f77b4',
                  'e2e_vitb': '#1f77b4', 'e2e_vitb_518': '#1f77b4'}
    _XY_LABELS = {'baseline': 'NavGraph-GPU', 'e2e_vits': 'RGB2Graph (ViT-s)',
                  'e2e_vitb': 'RGB2Graph (ViT-b)', 'e2e_vitb_518': 'RGB2Graph (ViT-b)'}

    fig, ax = plt.subplots(figsize=LANDSCAPE_FIGSIZE)
    for label, g in graphs.items():
        if g.num_nodes == 0:
            continue
        ax.scatter(
            g.positions[:, 0], g.positions[:, 1],
            s=0.2, alpha=0.6, color=_XY_COLORS.get(label, COLORS.get(label, '#888888')),
            label=_XY_LABELS.get(label, LABELS.get(label, label)),
        )
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    # Smaller axis-tick labels so the metre numbers don't crowd the
    # plotting area when the bounding box is wide.
    ax.tick_params(axis='both', labelsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.75,
              facecolor='white', edgecolor='#aaaaaa',
              markerscale=8)
    # Rotate the whole canvas 180° about its centre.
    ax.invert_xaxis()
    ax.invert_yaxis()
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


def plot_nn_histograms(nn_pairs: Dict[str, tuple],
                       plot_dir: Path) -> Path:
    """nn_pairs: {method: (nn_baseline_to_method, nn_method_to_baseline)}"""
    fig, ax = plt.subplots(figsize=LANDSCAPE_FIGSIZE)
    for method, (b_to_m, m_to_b) in nn_pairs.items():
        col = COLORS.get(method, '#888888')
        if b_to_m.size > 0:
            ax.hist(b_to_m, bins=40, alpha=0.55, color=COLORS['baseline'],
                    label=f'baseline → {method} (n={b_to_m.size})')
        if m_to_b.size > 0:
            ax.hist(m_to_b, bins=40, alpha=0.55, color=col,
                    label=f'{method} → baseline (n={m_to_b.size})')
    ax.set_xlabel('Euclidean distance to nearest in other graph (m)')
    ax.set_ylabel('node count')
    ax.set_title('Nearest-neighbour distance distribution')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    return _save_fig(fig, plot_dir, 'nn_distance_histograms')


TableType = Dict[str, np.ndarray]

# Columns that make up the "other" portion of t_graph_total_ms.
# New CSVs have the four detailed sub-columns; old CSVs have only t_other_ms.
# These two sets are mutually exclusive — a CSV written by the new node will
# not contain t_other_ms, and vice versa — so summing whichever set is present
# never double-counts.
_OTHER_DETAIL_COLS  = ('t_elev_extract_ms', 't_update_residual_ms',
                       't_visited_time_ms',  't_layers_ms')
_OTHER_FALLBACK_COL = 't_other_ms'


def _baseline_other_cols(df: TableType) -> tuple:
    """Return the 'other' timing column(s) to use for a given CSV.

    Prefers the four detailed sub-columns (new recordings); falls back to
    the legacy t_other_ms rollup (old recordings).  Never returns both, so
    callers can safely sum the result without risk of double-counting."""
    if any(c in df for c in _OTHER_DETAIL_COLS):
        return _OTHER_DETAIL_COLS
    return (_OTHER_FALLBACK_COL,)


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
    _TIMING_LABELS = {
        'baseline':      'Geometric Pipeline',
        'e2e_vitb':      'RGB-to-Graph model (ViT-b)',
        'e2e_vitb_518':  'RGB-to-Graph model  (ViT-b)',
        'e2e_vits':      'RGB-to-Graph model  (ViT-s)',
    }
    fig, ax = plt.subplots(figsize=(LANDSCAPE_FIGSIZE[0], LANDSCAPE_FIGSIZE[1] + 5))
    # New CSVs have the four detailed sub-columns; old CSVs have only
    # t_other_ms.  Both sets are listed here — whichever columns are absent
    # in a given CSV are skipped by the `if col not in df: continue` guard
    # below, so there is never double-counting.
    _BASELINE_STEPS = [
        ('t_parse_ms',           'parse cloud'),
        ('t_emap_ms',            'elev. map update'),
        ('t_elev_extract_ms',    'elev. extract'),
        ('t_local_ms',           'local graph (GPU)'),
        ('t_merge_ms',           'global merge'),
        ('t_update_residual_ms', 'update residual'),
        ('t_visited_time_ms',    'visited-time layer'),
        ('t_layers_ms',          'layer compute'),
        # legacy fallback — present only in CSVs recorded before the
        # sub-column instrumentation was added:
        ('t_other_ms',           'map update +\nbuilder overhead'),
    ]
    _E2E_STEPS = [
        ('t_inference_ms', 'model inference'),
        ('t_merge_ms',     'global merge'),
        ('t_edges_ms',     'edge build'),
        ('t_other_ms',     'other'),
    ]
    # Assign x positions in the order methods appear in timings dict.
    x_positions = {label: i for i, label in enumerate(timings.keys())}
    bar_width = max(0.60, 0.85 - 0.05 * max(0, len(timings) - 2))
    # Fixed 5-stop palette for the stacked bar segments — used in order
    # from the bottom of each stack up.  Baseline consumes all five
    # (it has 5 segments); e2e only consumes the first four.
    _PALETTE = ['#e19c24', '#e3c05d', '#d9ecf9', '#f2e9b9', '#e3c05d']
    def _seg_color(j: int) -> str:
        return _PALETTE[j % len(_PALETTE)]

    # ── Pass 1: compute per-method (step, mean_ms) pairs + totals so we
    # can size the y-axis before drawing.  Drawing in a second pass keeps
    # the total-annotation placement consistent across methods.
    plan: Dict[str, List[Tuple[str, float]]] = {}
    totals: Dict[str, float] = {}
    for label, df in timings.items():
        if table_is_empty(df):
            continue
        steps = _BASELINE_STEPS if label == 'baseline' else _E2E_STEPS
        rows: List[Tuple[str, float]] = []
        for col, pretty in steps:
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
        ax.set_ylabel('Mean per-frame graph generation time (ms)')
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
                   width=bar_width, color=_seg_color(j),
                   edgecolor='black', linewidth=0.4,
                   label=f'{label}: {pretty}')
            ax.text(x_positions[label], cum + v / 2.0,
                    f'{pretty} {v:.1f} ms',
                    ha='center', va='center', fontsize=11)
            cum += v
        # Total goes a fixed fraction of max_total above the bar, so it
        # always sits below the top border by ~17% of max_total.
        ax.text(x_positions[label],
                cum + max_total * 0.09,
                f'Total: {cum:.1f} ms',
                ha='center', va='bottom',
                fontsize=13, weight='bold')

    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels(
        [_TIMING_LABELS.get(k, k) for k in x_positions.keys()],
        rotation=0, ha='center', fontsize=13)
    ax.set_ylabel('Mean per-frame graph generation time (ms)', fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)
    return _save_fig(fig, plot_dir, 'timing_breakdown')


def plot_timing_breakdown_combined(timings: Dict[str, Optional[TableType]],
                                   plot_dir: Path) -> Path:
    """Same style as plot_timing_breakdown, but with merged segments.

    Geometric Pipeline:
      t_parse_ms                        → "cloud parse"
      t_emap_ms + t_other_ms (or subs)  → "mapping & traversibility update"
      t_local_ms                        → "local graph"
      t_merge_ms                        → "global merge"

    E2E models:
      t_inference_ms            → "model inference"
      t_merge_ms + t_edges_ms + t_other_ms → "merging +\nedge building"

    Saved as timing_breakdown_@.pdf / .png
    """
    _TIMING_LABELS = {
        'baseline':     'Geometric\npipeline',
        'e2e_vitb':     'RGB-to-Graph\nmodel (ViT-b)',
        'e2e_vitb_518': 'RGB-to-Graph\nmodel (ViT-b)',
        'e2e_vits':     'RGB-to-Graph\nmodel (ViT-s)',
    }
    fig, ax = plt.subplots(figsize=(LANDSCAPE_FIGSIZE[0], LANDSCAPE_FIGSIZE[1] + 6))
    _PALETTE = ['#e19c24', '#e3c05d', '#d9ecf9', '#f2e9b9', '#e3c05d']

    # ── Pass 1: build combined step plan ─────────────────────────────────────
    plan: Dict[str, List[Tuple[str, float]]] = {}
    totals: Dict[str, float] = {}
    for label, df in timings.items():
        if table_is_empty(df):
            continue
        if label == 'baseline':
            # Use whichever "other" columns are present in this CSV — the
            # detailed sub-columns (new recordings) or t_other_ms (old).
            # _baseline_other_cols guarantees the two sets are never mixed.
            other_cols = _baseline_other_cols(df)
            map_trav = sum(
                col_mean(df, c) for c in ('t_emap_ms',) + other_cols
                if math.isfinite(col_mean(df, c))
            )
            rows: List[Tuple[str, float]] = []
            v_parse = col_mean(df, 't_parse_ms')
            if math.isfinite(v_parse) and v_parse > 0:
                rows.append(('cloud\nparse', v_parse))
            if map_trav > 0:
                rows.append(('mapping &\ntraversibility\nupdate', map_trav))
            for col, pretty in [('t_local_ms', 'local graph'),
                                 ('t_merge_ms', 'global merge')]:
                v = col_mean(df, col)
                if math.isfinite(v):
                    rows.append((pretty, v))
        else:
            infer = col_mean(df, 't_inference_ms')
            rest  = sum(
                col_mean(df, c) for c in ('t_merge_ms', 't_edges_ms', 't_other_ms')
                if math.isfinite(col_mean(df, c))
            )
            rows = []
            if math.isfinite(infer):
                rows.append(('model\ninference', infer))
            if rest > 0:
                rows.append(('merging +\nedge building', rest))
        if rows:
            plan[label] = rows
            totals[label] = sum(v for _, v in rows)

    if not totals:
        ax.set_ylabel('Mean per-frame graph generation time (ms)')
        ax.grid(True, axis='y', alpha=0.3)
        return _save_fig(fig, plot_dir, 'timing_breakdown_2')

    max_total = max(totals.values())
    ax.set_ylim(0, max_total * 1.12)
    x_positions = {lbl: i * 1.6 for i, lbl in enumerate(plan.keys())}
    bar_width   = max(1.35, 1.5 - 0.05 * max(0, len(plan) - 2))

    # ── Pass 2: draw ─────────────────────────────────────────────────────────
    for label, rows in plan.items():
        cum = 0.0
        for j, (pretty, v) in enumerate(rows):
            ax.bar(x_positions[label], v, bottom=cum,
                   width=bar_width, color=_PALETTE[j % len(_PALETTE)],
                   edgecolor='black', linewidth=0.4)
            ax.text(x_positions[label], cum + v / 2.0,
                    pretty,
                    ha='center', va='center', fontsize=22, linespacing=1.25)
            cum += v
        ax.text(x_positions[label],
                cum + max_total * 0.09 -1,
                f'Total: {cum:.1f} ms',
                ha='center', va='bottom',
                fontsize=25, weight='bold')

    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels(
        [_TIMING_LABELS.get(k, k) for k in x_positions.keys()],
        rotation=0, ha='center', fontsize=25)
    ax.set_ylabel('Mean per-frame graph generation time (ms)', fontsize=25)
    ax.grid(True, axis='y', alpha=0.3)
    return _save_fig(fig, plot_dir, 'timing_breakdown_2')


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


def plot_covered_area(graphs: Dict[str, Graph], plot_dir: Path,
                      cell_m: float = 0.5) -> Path:
    """One subplot per method: covered grid cells as a filled raster image,
    with node positions overlaid as dots.

    Each cell that was reached by a node or edge is filled with the method's
    colour.  The area annotation in the title shows the total covered area in m².
    """
    n = len(graphs)
    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(LANDSCAPE_FIGSIZE[0] * ncols / 2,
                                      LANDSCAPE_FIGSIZE[1] * nrows * 0.85))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for ax, (label, g) in zip(axes_flat, graphs.items()):
        col = COLORS.get(label, '#888888')
        cells = _rasterise_graph(g, cell_m)
        area = float(cells.shape[0]) * cell_m * cell_m

        if cells.shape[0] == 0:
            ax.text(0.5, 0.5, f'{label}: empty graph',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            continue

        # Build a 2D binary image from the cell indices and show with imshow.
        c_min, r_min = cells[:, 0].min(), cells[:, 1].min()
        c_max, r_max = cells[:, 0].max(), cells[:, 1].max()
        img = np.zeros((r_max - r_min + 1, c_max - c_min + 1), dtype=np.uint8)
        img[cells[:, 1] - r_min, cells[:, 0] - c_min] = 1

        # World-space extent for imshow so axis ticks show metres.
        x0 = c_min * cell_m
        x1 = (c_max + 1) * cell_m
        y0 = r_min * cell_m
        y1 = (r_max + 1) * cell_m

        # Coloured RGBA: occupied cells → method colour; empty → transparent.
        rgba = np.zeros((*img.shape, 4), dtype=np.float32)
        rgb = mcolors.to_rgb(col)
        rgba[img == 1, :3] = rgb
        rgba[img == 1,  3] = 0.55      # semi-transparent fill
        rgba[img == 0,  3] = 0.0       # fully transparent background
        ax.imshow(rgba, origin='lower',
                  extent=[x0, x1, y0, y1], aspect='equal',
                  interpolation='nearest', zorder=1)

        # Node positions on top.
        if g.num_nodes > 0:
            ax.scatter(g.positions[:, 0], g.positions[:, 1],
                       s=2, alpha=0.7, color=col, zorder=2, linewidths=0)

        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_xlabel('x (m)', fontsize=8)
        ax.set_ylabel('y (m)', fontsize=8)
        ax.set_title(
            f'{label}\ncovered area = {area:.1f} m²  '
            f'({cells.shape[0]} cells @ {cell_m} m)',
            fontsize=9,
        )
        ax.tick_params(axis='both', labelsize=7)
        ax.grid(False)

    # Hide unused axes if grid has leftover slots.
    for ax in axes_flat[len(graphs):]:
        ax.set_axis_off()

    fig.tight_layout()
    return _save_fig(fig, plot_dir, 'covered_area')


def plot_covered_area_iou(graphs: Dict[str, Graph], plot_dir: Path,
                          cell_m: float = 0.5,
                          name: str = 'covered_area_iou') -> Path:
    """For each e2e method: three-colour overlay vs baseline.

    Green  = intersection (covered by both)
    Red    = baseline only
    Blue   = e2e method only

    The IoU value is annotated in the subplot title.
    """
    baseline = graphs.get('baseline')
    e2e_methods = [m for m in graphs if m != 'baseline']
    n = len(e2e_methods)
    if n == 0 or baseline is None:
        fig, ax = plt.subplots(figsize=LANDSCAPE_FIGSIZE)
        ax.text(0.5, 0.5, 'No e2e methods', ha='center', va='center',
                transform=ax.transAxes)
        return _save_fig(fig, plot_dir, name)

    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(LANDSCAPE_FIGSIZE[0] * ncols / 2,
                                      LANDSCAPE_FIGSIZE[1] * nrows * 0.85))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    cells_base = _rasterise_graph(baseline, cell_m)
    set_base = set(map(tuple, cells_base.tolist())) if cells_base.shape[0] > 0 else set()

    for ax, m in zip(axes_flat, e2e_methods):
        g = graphs[m]
        cells_m = _rasterise_graph(g, cell_m)
        set_m = set(map(tuple, cells_m.tolist())) if cells_m.shape[0] > 0 else set()

        inter = set_base & set_m
        base_only = set_base - set_m
        m_only = set_m - set_base
        union_size = len(set_base | set_m)
        iou = len(inter) / union_size if union_size > 0 else float('nan')

        # Collect all cells to determine canvas extent.
        all_cells = set_base | set_m
        if not all_cells:
            ax.text(0.5, 0.5, f'{m}: empty', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_axis_off()
            continue

        all_arr = np.array(list(all_cells), dtype=np.int32)
        c_min, r_min = all_arr[:, 0].min(), all_arr[:, 1].min()
        c_max, r_max = all_arr[:, 0].max(), all_arr[:, 1].max()
        h = r_max - r_min + 1
        w = c_max - c_min + 1

        # RGBA image: 0=transparent background.
        rgba = np.zeros((h, w, 4), dtype=np.float32)

        def _paint(cells_set, color_rgb, alpha=0.65):
            if not cells_set:
                return
            arr = np.array(list(cells_set), dtype=np.int32)
            rs = arr[:, 1] - r_min
            cs = arr[:, 0] - c_min
            rgba[rs, cs, :3] = color_rgb
            rgba[rs, cs, 3] = alpha

        _paint(base_only, (0.84, 0.15, 0.16))   # red — baseline only
        _paint(m_only,    (0.12, 0.47, 0.71))   # blue — e2e only
        _paint(inter,     (0.17, 0.63, 0.17))   # green — intersection

        x0, x1 = c_min * cell_m, (c_max + 1) * cell_m
        y0, y1 = r_min * cell_m, (r_max + 1) * cell_m
        ax.imshow(rgba, origin='lower', extent=[x0, x1, y0, y1],
                  aspect='equal', interpolation='nearest')

        # Legend patches.
        from matplotlib.patches import Patch
        legend_els = [
            Patch(facecolor=(0.17, 0.63, 0.17), label=f'intersection ({len(inter)} cells)'),
            Patch(facecolor=(0.84, 0.15, 0.16), label=f'baseline only ({len(base_only)} cells)'),
            Patch(facecolor=(0.12, 0.47, 0.71), label=f'{m} only ({len(m_only)} cells)'),
        ]
        ax.legend(handles=legend_els, loc='best', fontsize=7, frameon=True)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_xlabel('x (m)', fontsize=8)
        ax.set_ylabel('y (m)', fontsize=8)
        iou_str = f'{iou:.3f}' if math.isfinite(iou) else 'NaN'
        ax.set_title(f'baseline vs {m}\ncovered-area IoU = {iou_str}  '
                     f'(cell = {cell_m} m)', fontsize=9)
        ax.tick_params(axis='both', labelsize=7)
        ax.grid(False)

    for ax in axes_flat[n:]:
        ax.set_axis_off()

    fig.tight_layout()
    return _save_fig(fig, plot_dir, 'covered_area_iou')


# ─────────────────────────────────────────────────────────────────────
#  Global occupancy map helpers
# ─────────────────────────────────────────────────────────────────────

def load_global_occ_map(base_dir: Path) -> Optional[Dict]:
    """Load inputs/global_occ_map.npz. Returns None if not found."""
    npz_path = base_dir / 'inputs' / 'global_occ_map.npz'
    if not npz_path.exists():
        return None
    try:
        data = np.load(str(npz_path))
        return {
            'grid':       data['grid'],              # uint8: 0=occ, 128=unk, 255=free
            'origin_x':   float(data['origin_x']),   # world x of col 0 (west edge)
            'origin_y':   float(data['origin_y']),   # world y of row 0 (south edge)
            'resolution': float(data['resolution']),
        }
    except Exception as exc:
        print(f'  [occ map] load failed: {exc}')
        return None


def crop_occ_map(occ: Dict,
                 x_min: Optional[float] = None, x_max: Optional[float] = None,
                 y_min: Optional[float] = None, y_max: Optional[float] = None) -> Dict:
    """Return a cropped view of the occ map, clamped to world-coordinate bounds."""
    grid = occ['grid']
    ox, oy, res = occ['origin_x'], occ['origin_y'], occ['resolution']
    H, W = grid.shape

    c0 = int(np.floor((x_min - ox) / res)) if x_min is not None else 0
    c1 = int(np.ceil((x_max - ox) / res))  if x_max is not None else W
    r0 = int(np.floor((y_min - oy) / res)) if y_min is not None else 0
    r1 = int(np.ceil((y_max - oy) / res))  if y_max is not None else H

    c0 = max(0, c0);  c1 = min(W, c1)
    r0 = max(0, r0);  r1 = min(H, r1)

    return {
        'grid':       grid[r0:r1, c0:c1].copy(),
        'origin_x':   ox + c0 * res,
        'origin_y':   oy + r0 * res,
        'resolution': res,
    }


def node_edge_quality(g: Graph, occ: Dict) -> Dict[str, float]:
    """Count nodes/edges whose positions fall in free vs occupied occ-map cells.

    Edges are checked at their midpoint.  Returns both raw counts and percentages
    of total nodes / valid edges (those whose both endpoints are known).
    All lookups are vectorised with numpy.
    """
    grid = occ['grid']
    ox, oy, res = occ['origin_x'], occ['origin_y'], occ['resolution']
    H, W = grid.shape
    nan = float('nan')

    if g.num_nodes == 0:
        return {k: nan for k in (
            'nodes_in_free', 'nodes_in_free_pct',
            'nodes_in_occ',  'nodes_in_occ_pct',
            'edges_mid_in_free', 'edges_mid_in_free_pct',
            'edges_mid_in_occ',  'edges_mid_in_occ_pct',
        )}

    # ── Nodes ────────────────────────────────────────────────────────
    xy = g.positions[:, :2]
    c_idx = np.floor((xy[:, 0] - ox) / res).astype(np.int32)
    r_idx = np.floor((xy[:, 1] - oy) / res).astype(np.int32)
    inb = (r_idx >= 0) & (r_idx < H) & (c_idx >= 0) & (c_idx < W)

    cell_vals = np.full(g.num_nodes, 128, dtype=np.uint8)
    if inb.any():
        cell_vals[inb] = grid[r_idx[inb], c_idx[inb]]

    n_free = int((cell_vals == 255).sum())
    n_occ  = int((cell_vals == 0).sum())
    N = g.num_nodes

    out: Dict[str, float] = {
        'nodes_in_free':     float(n_free),
        'nodes_in_free_pct': 100.0 * n_free / N,
        'nodes_in_occ':      float(n_occ),
        'nodes_in_occ_pct':  100.0 * n_occ  / N,
    }

    # ── Edges (midpoint) ─────────────────────────────────────────────
    if g.num_edges == 0:
        out.update({k: nan for k in (
            'edges_mid_in_free', 'edges_mid_in_free_pct',
            'edges_mid_in_occ',  'edges_mid_in_occ_pct',
        )})
        return out

    # Vectorised ID → index map via direct-address array.
    max_id = int(g.ids.max())
    id_map = np.full(max_id + 1, -1, dtype=np.int32)
    id_map[g.ids.astype(np.int32)] = np.arange(g.num_nodes, dtype=np.int32)

    a_raw = g.edge_pairs[:, 0].astype(np.int64)
    b_raw = g.edge_pairs[:, 1].astype(np.int64)
    in_range = (a_raw <= max_id) & (b_raw <= max_id) & (a_raw >= 0) & (b_raw >= 0)
    a_idx = np.where(in_range, id_map[np.clip(a_raw, 0, max_id)], -1).astype(np.int32)
    b_idx = np.where(in_range, id_map[np.clip(b_raw, 0, max_id)], -1).astype(np.int32)
    valid = (a_idx >= 0) & (b_idx >= 0)

    if not valid.any():
        out.update({k: nan for k in (
            'edges_mid_in_free', 'edges_mid_in_free_pct',
            'edges_mid_in_occ',  'edges_mid_in_occ_pct',
        )})
        return out

    mx = (g.positions[a_idx[valid], 0] + g.positions[b_idx[valid], 0]) / 2
    my = (g.positions[a_idx[valid], 1] + g.positions[b_idx[valid], 1]) / 2
    c_e = np.floor((mx - ox) / res).astype(np.int32)
    r_e = np.floor((my - oy) / res).astype(np.int32)
    inb_e = (r_e >= 0) & (r_e < H) & (c_e >= 0) & (c_e < W)

    cell_e = np.full(int(valid.sum()), 128, dtype=np.uint8)
    if inb_e.any():
        cell_e[inb_e] = grid[r_e[inb_e], c_e[inb_e]]

    e_free = int((cell_e == 255).sum())
    e_occ  = int((cell_e == 0).sum())
    E = int(valid.sum())
    out.update({
        'edges_mid_in_free':     float(e_free),
        'edges_mid_in_free_pct': 100.0 * e_free / E,
        'edges_mid_in_occ':      float(e_occ),
        'edges_mid_in_occ_pct':  100.0 * e_occ  / E,
    })
    return out


def _graph_bubble_cells_on_occ(g: Graph, occ: Dict, radius_m: float = 0.5) -> np.ndarray:
    """Return unique (col, row) int32 grid indices covered by disk of ``radius_m``
    centred on every node.  Fully vectorised — no Python loops over nodes.

    Returns shape (M, 2): col=[:,0], row=[:,1].
    """
    grid = occ['grid']
    ox, oy, res = occ['origin_x'], occ['origin_y'], occ['resolution']
    H, W = grid.shape

    if g.num_nodes == 0:
        return np.empty((0, 2), dtype=np.int32)

    # Disk template: all (dc, dr) offsets within radius_m.
    r_cells = max(0, int(np.ceil(radius_m / res)))
    dc_range = np.arange(-r_cells, r_cells + 1, dtype=np.int32)
    dc2d, dr2d = np.meshgrid(dc_range, dc_range)
    disk_mask = (dc2d ** 2 + dr2d ** 2) <= r_cells ** 2
    dc_disk = dc2d[disk_mask].ravel()   # (D,)
    dr_disk = dr2d[disk_mask].ravel()   # (D,)

    # Node centres in grid coordinates.
    xy = g.positions[:, :2]
    c_nodes = np.floor((xy[:, 0] - ox) / res).astype(np.int32)  # (N,)
    r_nodes = np.floor((xy[:, 1] - oy) / res).astype(np.int32)  # (N,)

    # Broadcast: (N, 1) + (1, D) → (N*D,)
    all_c = (c_nodes[:, None] + dc_disk[None, :]).ravel()
    all_r = (r_nodes[:, None] + dr_disk[None, :]).ravel()

    inb = (all_r >= 0) & (all_r < H) & (all_c >= 0) & (all_c < W)
    if not inb.any():
        return np.empty((0, 2), dtype=np.int32)

    cr = np.stack([all_c[inb], all_r[inb]], axis=1)
    return np.unique(cr, axis=0)


def free_area_coverage_pct(g: Graph, occ: Dict, radius_m: float = 0.5) -> float:
    """% of free cells in the occupancy map covered by a disk of ``radius_m`` around
    each graph node.
    """
    grid = occ['grid']
    total_free = int((grid == 255).sum())
    if total_free == 0 or g.num_nodes == 0:
        return float('nan')

    cr_unique = _graph_bubble_cells_on_occ(g, occ, radius_m)
    if cr_unique.shape[0] == 0:
        return 0.0

    free_mask = grid[cr_unique[:, 1], cr_unique[:, 0]] == 255
    return 100.0 * int(free_mask.sum()) / total_free


def _edge_segments(g: Graph) -> Optional[np.ndarray]:
    """Return (E_valid, 2, 2) float32 array of edge endpoint XY pairs, or None."""
    if g.num_edges == 0 or g.num_nodes == 0:
        return None
    max_id = int(g.ids.max())
    id_map = np.full(max_id + 1, -1, dtype=np.int32)
    id_map[g.ids.astype(np.int32)] = np.arange(g.num_nodes, dtype=np.int32)
    a_raw = g.edge_pairs[:, 0].astype(np.int64)
    b_raw = g.edge_pairs[:, 1].astype(np.int64)
    in_range = (a_raw <= max_id) & (b_raw <= max_id) & (a_raw >= 0) & (b_raw >= 0)
    a_idx = np.where(in_range, id_map[np.clip(a_raw, 0, max_id)], -1).astype(np.int32)
    b_idx = np.where(in_range, id_map[np.clip(b_raw, 0, max_id)], -1).astype(np.int32)
    valid = (a_idx >= 0) & (b_idx >= 0)
    if not valid.any():
        return None
    pa = g.positions[a_idx[valid], :2].astype(np.float32)
    pb = g.positions[b_idx[valid], :2].astype(np.float32)
    return np.stack([pa, pb], axis=1)  # (E_valid, 2, 2)


def plot_graphs_on_occ_map(graphs: Dict[str, Graph], occ: Dict,
                            plot_dir: Path,
                            name: str = 'graphs_on_occ_map',
                            show_quality: bool = False) -> Path:
    """One subplot per method: graph nodes + edges drawn on the global occ map.

    Background: light-gray=free, dark-gray=occupied, mid-gray=unknown.
    Node/edge colours follow per-method COLORS palette.
    ``name`` is the output file stem (allows multiple calls with different crops).
    When ``show_quality=True``, each subplot title includes the % of nodes and
    edges falling in free / occupied cells.
    """
    grid = occ['grid']
    ox, oy, res = occ['origin_x'], occ['origin_y'], occ['resolution']
    H, W = grid.shape

    bg = np.full((H, W, 3), 160, dtype=np.uint8)   # mid-gray = unknown
    bg[grid == 255] = [220, 220, 220]               # light gray = free
    bg[grid == 0]   = [50,  50,  50]                # dark  gray = occupied
    x_min, x_max = ox, ox + W * res
    y_min, y_max = oy, oy + H * res

    n = len(graphs)
    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(LANDSCAPE_FIGSIZE[0] * max(ncols, 1) / 2,
                 LANDSCAPE_FIGSIZE[1] * nrows * 0.9),
    )
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for ax, (label, g) in zip(axes_flat, graphs.items()):
        col = COLORS.get(label, '#888888')

        ax.imshow(bg, origin='lower',
                  extent=[x_min, x_max, y_min, y_max],
                  aspect='equal', interpolation='nearest', zorder=0)

        segs = _edge_segments(g)
        if segs is not None:
            lc = LineCollection(segs, colors=[col], linewidths=0.6,
                                alpha=0.5, zorder=1)
            ax.add_collection(lc)

        if g.num_nodes > 0:
            ax.scatter(g.positions[:, 0], g.positions[:, 1],
                       s=3, color=col, alpha=0.85, zorder=2, linewidths=0)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('x (m)', fontsize=8)
        ax.set_ylabel('y (m)', fontsize=8)

        title_top = f'{label} — {g.num_nodes} nodes, {g.num_edges} edges'
        if show_quality and g.num_nodes > 0:
            q = node_edge_quality(g, occ)
            def _pct(key: str) -> str:
                v = q.get(key, float('nan'))
                return f'{v:.1f}%' if math.isfinite(v) else 'N/A'
            title_top += (
                f'\nnodes: {_pct("nodes_in_free_pct")} free / '
                f'{_pct("nodes_in_occ_pct")} occ'
                f'   edges: {_pct("edges_mid_in_free_pct")} free / '
                f'{_pct("edges_mid_in_occ_pct")} occ'
            )
        ax.set_title(title_top, fontsize=8)
        ax.tick_params(axis='both', labelsize=7)

    for ax in axes_flat[n:]:
        ax.set_axis_off()

    fig.tight_layout()
    return _save_fig(fig, plot_dir, name)


def plot_overlay_on_occ_map(graphs: Dict[str, Graph], occ: Dict,
                             plot_dir: Path,
                             name: str = 'overlay_on_occ_map') -> Path:
    """All methods drawn on a single occ-map panel with a shared legend.

    Intended for a two-method comparison (e.g. baseline + e2e_vits), though
    it accepts any subset.  Legend is placed inside the map at the top-left.
    """
    grid = occ['grid']
    ox, oy, res = occ['origin_x'], occ['origin_y'], occ['resolution']
    H, W = grid.shape

    bg = np.full((H, W, 3), 160, dtype=np.uint8)
    bg[grid == 255] = [220, 220, 220]
    bg[grid == 0]   = [50,  50,  50]
    x_min, x_max = ox, ox + W * res
    y_min, y_max = oy, oy + H * res

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(bg, origin='lower',
              extent=[x_min, x_max, y_min, y_max],
              aspect='equal', interpolation='nearest', zorder=0)

    handles = []
    for label, g in graphs.items():
        col = COLORS.get(label, '#888888')
        disp = LABELS.get(label, label)

        segs = _edge_segments(g)
        if segs is not None:
            lc = LineCollection(segs, colors=[col], linewidths=0.7,
                                alpha=0.5, zorder=1)
            ax.add_collection(lc)

        if g.num_nodes > 0:
            ax.scatter(g.positions[:, 0], g.positions[:, 1],
                       s=4, color=col, alpha=0.9, zorder=2, linewidths=0,
                       label=f'{disp}  ({g.num_nodes} nodes, {g.num_edges} edges)')
        else:
            ax.scatter([], [], s=4, color=col, label=disp)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('x (m)', fontsize=9)
    ax.set_ylabel('y (m)', fontsize=9)
    ax.set_title('Graph overlay on occupancy map', fontsize=11)

    ax.legend(loc='upper left', fontsize=8, framealpha=0.75,
              facecolor='white', edgecolor='#aaaaaa',
              bbox_to_anchor=(0.01, 0.99), bbox_transform=ax.transAxes)

    fig.tight_layout()
    return _save_fig(fig, plot_dir, name)


def plot_largest_cc(graphs: Dict[str, Graph], plot_dir: Path) -> Path:
    """Side-by-side scatter: faint background of every node + the
    largest connected component overlaid in the method's accent colour.

    A method with a single dominant CC will look almost solidly tinted;
    a fragmented graph will show the dominant component plus large grey
    archipelagos that didn't connect.
    """
    fig, axes = plt.subplots(1, 2, figsize=(LANDSCAPE_FIGSIZE[0],
                                            LANDSCAPE_FIGSIZE[1] * 0.9))
    if not hasattr(axes, '__iter__'):
        axes = [axes]
    for ax, (label, g) in zip(axes, graphs.items()):
        if g.num_nodes == 0:
            ax.text(0.5, 0.5, f'{label}: empty graph',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            continue
        adj = g.adjacency_csr()
        n_cc, cc_labels = connected_components(adj, directed=False)
        sizes = np.bincount(cc_labels)
        largest_cc_id = int(sizes.argmax())
        in_largest = cc_labels == largest_cc_id
        n_in = int(in_largest.sum())
        # Light grey backdrop of every node.
        ax.scatter(g.positions[:, 0], g.positions[:, 1],
                   s=4, alpha=0.25, color='#888888',
                   label=f'all nodes (n={g.num_nodes})')
        # Largest CC in the method's colour, drawn on top.
        ax.scatter(g.positions[in_largest, 0],
                   g.positions[in_largest, 1],
                   s=6, alpha=0.85, color=COLORS[label],
                   label=f'largest CC (n={n_in}, {n_in / g.num_nodes:.1%})')
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(
            f'{label} — {n_cc} component'
            f'{"s" if n_cc != 1 else ""}, largest = {n_in}/{g.num_nodes}',
            fontsize=10,
        )
        ax.tick_params(axis='both', labelsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8, frameon=False)
    fig.tight_layout()
    return _save_fig(fig, plot_dir, 'largest_cc')


def _draw_planning_example(g: Graph, pair: dict, label: str,
                           outcome: str, idx: int,
                           plot_dir: Path,
                           cc_labels: np.ndarray) -> Path:
    """Render one single (start, goal) planning pair as its own figure.

    Success — the reconstructed Dijkstra path is highlighted; start/goal
    snap arrows show the gap between the query xy and the graph node it
    snapped to.

    Failure — the start's connected component is tinted green, the
    goal's connected component is tinted red, so the disconnect is
    visually obvious (Dijkstra on an undirected graph only fails when
    the endpoints live in different components).
    """
    fig, ax = plt.subplots(figsize=LANDSCAPE_FIGSIZE)
    pos = g.positions
    # Faint backdrop: every node + every edge in light grey.
    adj = g.adjacency_csr()
    rows, cols = adj.nonzero()
    # Each undirected edge appears twice in the CSR — keep only one
    # direction so we don't double-draw.
    mask = rows < cols
    for r, c in zip(rows[mask], cols[mask]):
        ax.plot([pos[r, 0], pos[c, 0]], [pos[r, 1], pos[c, 1]],
                color='#cccccc', linewidth=0.4, zorder=1)
    ax.scatter(pos[:, 0], pos[:, 1], s=3, color='#888888',
               alpha=0.5, zorder=2)

    qs = pair['query_start_xy']
    qg = pair['query_goal_xy']
    si = pair['start_node_idx']
    gi = pair['goal_node_idx']
    s_node = pos[si]
    g_node = pos[gi]

    if outcome == 'success' and pair['path_indices'] is not None:
        path = pair['path_indices']
        if len(path) >= 2:
            xs = pos[path, 0]
            ys = pos[path, 1]
            ax.plot(xs, ys, color=COLORS[label], linewidth=2.4,
                    alpha=0.95, zorder=4, label='shortest path')
            ax.scatter(xs, ys, s=12, color=COLORS[label],
                       zorder=5)
        title = (f'{label} — SUCCESS  |  path: {pair["path_length_m"]:.2f} m, '
                 f'{len(path)} nodes  |  snap (s/g): '
                 f'{pair["start_snap_m"]:.2f} / {pair["goal_snap_m"]:.2f} m')
    else:
        # Failure: shade the start and goal CCs differently so the
        # disconnect is obvious.
        start_cc = cc_labels == cc_labels[si]
        goal_cc  = cc_labels == cc_labels[gi]
        ax.scatter(pos[start_cc, 0], pos[start_cc, 1],
                   s=8, color='#2ca02c', alpha=0.5, zorder=3,
                   label=f'start CC (n={int(start_cc.sum())})')
        ax.scatter(pos[goal_cc, 0], pos[goal_cc, 1],
                   s=8, color='#d62728', alpha=0.5, zorder=3,
                   label=f'goal CC (n={int(goal_cc.sum())})')
        title = (f'{label} — FAILURE  |  start/goal in different CCs  |  '
                 f'snap (s/g): {pair["start_snap_m"]:.2f} / '
                 f'{pair["goal_snap_m"]:.2f} m')

    # Snap arrows: query xy → snapped node.  Drawn after the path so
    # they're not buried under the line.
    ax.plot([qs[0], s_node[0]], [qs[1], s_node[1]],
            color='#2ca02c', linewidth=1.0, linestyle=':',
            alpha=0.7, zorder=6)
    ax.plot([qg[0], g_node[0]], [qg[1], g_node[1]],
            color='#d62728', linewidth=1.0, linestyle=':',
            alpha=0.7, zorder=6)
    # Query points (the world-frame randomly sampled xy).
    ax.scatter([qs[0]], [qs[1]], s=80, marker='o',
               edgecolor='#2ca02c', facecolor='white',
               linewidths=1.6, zorder=7, label='query start')
    ax.scatter([qg[0]], [qg[1]], s=80, marker='*',
               edgecolor='#d62728', facecolor='white',
               linewidths=1.6, zorder=7, label='query goal')
    # Snapped graph nodes (filled).
    ax.scatter([s_node[0]], [s_node[1]], s=60, marker='o',
               color='#2ca02c', zorder=8, label='snapped start node')
    ax.scatter([g_node[0]], [g_node[1]], s=120, marker='*',
               color='#d62728', zorder=8, label='snapped goal node')

    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.tick_params(axis='both', labelsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=9)
    ax.legend(loc='best', fontsize=7, frameon=False)
    return _save_fig(
        fig, plot_dir, f'planning_{label}_{outcome}_{idx:02d}',
    )


def plot_planning_examples(graphs: Dict[str, Graph],
                           planning_details: Dict[str, dict],
                           plot_dir: Path,
                           n_each: int = 10) -> Dict[str, List[Path]]:
    """Save ``n_each`` success + ``n_each`` failure example PDFs per method.

    Lands them in ``plot_dir / 'planning_examples'`` so they don't clutter
    the top-level ``plots/`` directory.  Returns the per-method list of
    written PDF paths so the markdown report can embed a few.

    Each plot annotates: the world-frame query (open circle / star), the
    snapped graph node it routed through (filled circle / star), the snap
    arrow, and either the reconstructed Dijkstra path (success) or the
    two connected components the endpoints fell into (failure).
    """
    examples_dir = plot_dir / 'planning_examples'
    examples_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, List[Path]] = {label: [] for label in graphs.keys()}
    for label, details in planning_details.items():
        g = graphs[label]
        if g.num_nodes == 0 or not details.get('pairs'):
            continue
        adj = g.adjacency_csr()
        _, cc_labels = connected_components(adj, directed=False)
        successes = [d for d in details['pairs'] if d['success']]
        failures  = [d for d in details['pairs'] if not d['success']]
        for i, pair in enumerate(successes[:n_each]):
            written[label].append(
                _draw_planning_example(g, pair, label, 'success', i,
                                       examples_dir, cc_labels),
            )
        for i, pair in enumerate(failures[:n_each]):
            written[label].append(
                _draw_planning_example(g, pair, label, 'failure', i,
                                       examples_dir, cc_labels),
            )
    return written


def plot_free_area_coverage_map(graphs: Dict[str, Graph], occ: Dict,
                                plot_dir: Path,
                                radius_m: float = 0.5,
                                name: str = 'free_area_coverage_map') -> Path:
    """One subplot per method: occ map background + covered disk cells highlighted.

    Each node contributes a filled disk of ``radius_m``.  No nodes or edges drawn.
    """
    grid = occ['grid']
    ox, oy, res = occ['origin_x'], occ['origin_y'], occ['resolution']
    H, W = grid.shape

    # North-up RGB base image (row 0 = north = large y).
    base = np.zeros((H, W, 3), dtype=np.uint8)
    base[grid == 255] = [240, 240, 240]
    base[grid == 0]   = [30,  30,  30]
    base[grid == 128] = [160, 160, 160]
    base_flip = np.flipud(base)

    methods = list(graphs.keys())
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 7), squeeze=False)

    COLORS = {'baseline': (0.2, 0.6, 1.0),
              'e2e_vitb': (1.0, 0.4, 0.1),
              'e2e_vitb_518': (1.0, 0.4, 0.1),
              'e2e_vits': (0.2, 0.8, 0.3)}

    total_free = int((grid == 255).sum())

    for ax, label in zip(axes[0], methods):
        ax.imshow(base_flip, origin='upper', aspect='equal',
                  extent=[ox, ox + W * res, oy, oy + H * res])

        cr = _graph_bubble_cells_on_occ(graphs[label], occ, radius_m)
        pct = float('nan')
        if cr.shape[0] > 0 and total_free > 0:
            free_mask = grid[cr[:, 1], cr[:, 0]] == 255
            pct = 100.0 * int(free_mask.sum()) / total_free

            cov_rgba = np.zeros((H, W, 4), dtype=np.float32)
            col = COLORS.get(label, (1.0, 0.8, 0.0))
            cov_rgba[cr[:, 1], cr[:, 0]] = [col[0], col[1], col[2], 0.65]
            ax.imshow(np.flipud(cov_rgba), origin='upper', aspect='equal',
                      extent=[ox, ox + W * res, oy, oy + H * res])

        pct_str = f'{pct:.1f}%' if math.isfinite(pct) else '—'
        ax.set_title(f'{LABELS.get(label, label)}\n'
                     f'Free area covered: {pct_str}  (r={radius_m} m disk)',
                     fontsize=11)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

    fig.suptitle('Graph free-area coverage  (highlighted = disk cells around nodes)',
                 fontsize=12)
    fig.tight_layout()
    return _save_fig(fig, plot_dir, name)


def plot_summary_table(methods: List[str],
                       stats: Dict[str, Dict[str, float]],
                       pairwise: Dict[str, Dict[str, float]],
                       timings: Dict[str, Any],
                       until_ids: Dict[str, int],
                       plot_dir: Path) -> Path:
    """Render a concise at-a-glance summary table as a PNG.

    Columns: one per method.
    Rows (all quantities are clearly labelled with scope):
      Full-run metrics
        Mean frame time (ms)
        Covered area IoU (vs baseline)
        Planning success rate (%)
      Until-N metrics (each method at its own cutoff N)
        Planning success rate (%)
        Nodes in free %
        Nodes in occ %
        Edges in free %
        Edges in occ %
    """
    nan = float('nan')

    def _fmt(v, scale: float = 1.0, decimals: int = 1) -> str:
        try:
            f = float(v) * scale
        except (TypeError, ValueError):
            return '—'
        return '—' if not math.isfinite(f) else f'{f:.{decimals}f}'

    # ── Row definitions: (display label, value_fn(method) → str) ─────
    rows: List[Tuple[str, List[str]]] = []

    def _add_row(label: str, fn) -> None:
        rows.append((label, [fn(m) for m in methods]))

    # Full-run: mean frame time
    _add_row('Mean frame time (ms) [full]',
             lambda m: _fmt(col_mean(timings.get(m) or {}, 't_frame_total_ms')))

    # Full-run: covered area IoU (baseline → '—')
    _add_row('Covered area IoU [full]',
             lambda m: '—' if m == 'baseline'
             else _fmt(pairwise.get(m, {}).get('covered_area_iou_2d', nan), decimals=2))

    # Full-run: % of free area covered by graph
    _add_row('Free area covered % [full]',
             lambda m: _fmt(stats.get(m, {}).get('free_area_coverage_pct', nan)))

    # Full-run: avg inter-node distance (mean edge length)
    _add_row('Avg inter-node dist (m) [full]',
             lambda m: _fmt(stats.get(m, {}).get('mean_edge_len_m', nan), decimals=2))

    # Full-run: planning success rate (scaled to %)
    _add_row('Planning success % [full]',
             lambda m: _fmt(stats.get(m, {}).get('planning_success_rate', nan), 100.0))

    # Full-run: planning mean + median path length
    _add_row('Planning mean path length (m) [full]',
             lambda m: _fmt(stats.get(m, {}).get('planning_mean_path_length_m', nan), decimals=2))
    _add_row('Planning median path length (m) [full]',
             lambda m: _fmt(stats.get(m, {}).get('planning_median_path_length_m', nan), decimals=2))

    # Full-run: mean path length on common success pairs (all methods succeeded)
    n_common = stats.get(methods[0], {}).get('planning_n_common_success_pairs', nan)
    n_common_str = f'{int(n_common)} pairs' if math.isfinite(n_common) else ''
    _add_row(f'Planning mean path (m) [common successes{(" " + n_common_str) if n_common_str else ""}]',
             lambda m: _fmt(stats.get(m, {}).get('planning_mean_path_length_common_m', nan), decimals=2))

    n_full_rows = len(rows)  # separator goes after all full-run rows

    # Until-N rows — only if until_ids was provided
    if until_ids:
        def _until(m: str, key: str, scale: float = 1.0, decimals: int = 1) -> str:
            n = until_ids.get(m)
            if n is None:
                return '—'
            return _fmt(stats.get(m, {}).get(f'until{n}_{key}', nan), scale, decimals)

        # Build a compact suffix showing per-method N values
        def _n_suffix() -> str:
            pairs = sorted({(m, until_ids[m]) for m in methods if m in until_ids})
            return '  (' + ', '.join(f'{m}={n}' for m, n in pairs) + ')'

        sfx = _n_suffix()
        _add_row(f'Covered area IoU [until N]{sfx}',
                 lambda m: '—' if m == 'baseline'
                 else _until(m, 'covered_area_iou_2d', decimals=2))
        _add_row(f'Planning success % [until N]{sfx}',
                 lambda m: _until(m, 'planning_success_rate', 100.0))
        _add_row(f'Planning mean path length (m) [until N]{sfx}',
                 lambda m: _until(m, 'planning_mean_path_length_m', decimals=2))
        _add_row(f'Planning median path length (m) [until N]{sfx}',
                 lambda m: _until(m, 'planning_median_path_length_m', decimals=2))
        _add_row(f'Nodes in free % [until N]{sfx}',
                 lambda m: _until(m, 'nodes_in_free_pct'))
        _add_row(f'Nodes in occ % [until N]{sfx}',
                 lambda m: _until(m, 'nodes_in_occ_pct'))
        _add_row(f'Edges in free % [until N]{sfx}',
                 lambda m: _until(m, 'edges_mid_in_free_pct'))
        _add_row(f'Edges in occ % [until N]{sfx}',
                 lambda m: _until(m, 'edges_mid_in_occ_pct'))

    # ── Build table arrays ────────────────────────────────────────────
    row_labels = [r[0] for r in rows]
    cell_text  = [r[1] for r in rows]
    col_labels = [LABELS.get(m, m) for m in methods]

    n_r = len(row_labels)
    n_c = len(col_labels)

    fig_w = max(8, 2.8 * n_c + 3.5)
    fig_h = max(3, 0.55 * n_r + 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')

    tbl = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.7)

    # Header row: blue background, white bold text
    for j in range(n_c):
        c = tbl[0, j]
        c.set_facecolor('#1f77b4')
        c.set_text_props(color='white', weight='bold')

    # Row-label column: light gray
    for i in range(n_r):
        c = tbl[i + 1, -1]
        c.set_facecolor('#e8e8e8')
        c.set_text_props(weight='bold')
        c.auto_set_font_size = False

    # Alternating row shading for data cells
    for i in range(n_r):
        shade = '#f7f7f7' if i % 2 == 0 else '#ffffff'
        for j in range(n_c):
            tbl[i + 1, j].set_facecolor(shade)

    # Separator line between full-run and until-N sections
    if until_ids:
        for j in range(n_c):
            tbl[n_full_rows, j].set_edgecolor('#888888')

    fig.tight_layout(pad=0.5)
    return _save_fig(fig, plot_dir, 'summary_table')


# ─────────────────────────────────────────────────────────────────────
#  Report assembly
# ─────────────────────────────────────────────────────────────────────

def build_summary_rows(stats: Dict[str, Dict[str, float]],
                       pairwise: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    """One row per metric, one column per method + delta vs baseline.

    pairwise: {method: {metric: value}} for each non-baseline method.
    Pairwise metrics appear at the bottom, showing their value under the
    relevant non-baseline method column.
    """
    methods = list(stats.keys())
    e2e_methods = [m for m in methods if m != 'baseline']
    rows: List[Dict[str, Any]] = []
    seen: List[str] = []
    for m in methods:
        for k in stats[m].keys():
            if k not in seen:
                seen.append(k)
    for k in seen:
        row: Dict[str, Any] = {'metric': k}
        base_val = stats.get('baseline', {}).get(k, float('nan'))
        row['baseline'] = base_val
        for m in e2e_methods:
            v = stats[m].get(k, float('nan'))
            row[m] = v
            delta = (v - base_val) if (math.isfinite(v) and math.isfinite(base_val)) else float('nan')
            row[f'delta_{m}_minus_baseline'] = delta
        rows.append(row)
    # Pairwise metrics (chamfer etc.) — one block per e2e method.
    for m, pw in pairwise.items():
        for k, v in pw.items():
            row = {'metric': f'{k} [baseline vs {m}]', 'baseline': ''}
            for om in e2e_methods:
                row[om] = v if om == m else ''
                row[f'delta_{om}_minus_baseline'] = ''
            rows.append(row)
    return rows


def write_summary_csv(rows: List[Dict[str, Any]], path: Path,
                      methods: List[str]) -> None:
    e2e_methods = [m for m in methods if m != 'baseline']
    cols = (['metric', 'baseline']
            + e2e_methods
            + [f'delta_{m}_minus_baseline' for m in e2e_methods])
    with open(path, 'w', newline='') as f:
        w = _csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
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
                          skip_first_n_seconds: float = 0.0,
                          planning_example_paths: Optional[Dict[str, List[Path]]] = None) -> str:
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
    e2e_methods = [m for m in graphs if m != 'baseline']
    header_cols = ['baseline'] + e2e_methods + [f'δ {m}' for m in e2e_methods]
    lines.append('| metric | ' + ' | '.join(header_cols) + ' |')
    lines.append('|---' + '|---:' * len(header_cols) + '|')
    for row in summary:
        vals = [_fmt(row.get('baseline', ''))]
        for m in e2e_methods:
            vals.append(_fmt(row.get(m, '')))
        for m in e2e_methods:
            vals.append(_fmt(row.get(f'delta_{m}_minus_baseline', '')))
        lines.append(f'| {row["metric"]} | ' + ' | '.join(vals) + ' |')
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

    # Planning examples are saved as PDFs in a sub-folder.  Not embedded
    # because there can be 40+ of them per run; we just point at the
    # folder and embed the first success + first failure per method as
    # quick previews so the report has *something* on the page.
    if planning_example_paths:
        lines.append('\n## Planning examples\n')
        lines.append(
            f'**{sum(len(v) for v in planning_example_paths.values())} '
            f'individual planning-example PDFs** (3 successes + 3 '
            f'failures per method) live in '
            f'`plots/planning_examples/` — each one renders a single '
            f'(start, goal) pair, the snapped graph nodes, and either '
            f'the reconstructed Dijkstra path (success) or the two '
            f'disconnected components the endpoints landed in (failure).'
        )
        for label, paths in planning_example_paths.items():
            first_success = next(
                (p for p in paths if '_success_' in p.stem), None,
            )
            first_failure = next(
                (p for p in paths if '_failure_' in p.stem), None,
            )
            if first_success or first_failure:
                lines.append(f'\n**{label}** — quick previews:\n')
            if first_success is not None:
                png = f'planning_examples/{first_success.stem}.png'
                pdf = f'planning_examples/{first_success.stem}.pdf'
                lines.append(f'![{label} success 00](plots/{png})\n')
                lines.append(f'_[Open as PDF](plots/{pdf})_\n')
            if first_failure is not None:
                png = f'planning_examples/{first_failure.stem}.png'
                pdf = f'planning_examples/{first_failure.stem}.pdf'
                lines.append(f'![{label} failure 00](plots/{png})\n')
                lines.append(f'_[Open as PDF](plots/{pdf})_\n')

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
    ap.add_argument('--baseline-csv', type=Path, default=None,
                    help='Override path to baseline timing CSV.')
    ap.add_argument('--occupancy-cell-m', type=float, default=0.5,
                    help='Cell size (m) for occupancy IoU + density correlation.')
    ap.add_argument('--planning-pairs', type=int, default=200,
                    help='Number of (start, goal) pairs sampled per method.')
    ap.add_argument('--no-planning', action='store_true',
                    help='Skip the (slow-ish) planning quality proxy.')
    ap.add_argument('--until-node-id', nargs='+', default=None,
                    metavar='[METHOD=]N',
                    help='Produce an occ-map overlay plot and node/edge quality '
                         'metrics for each graph truncated to nodes with ID <= N. '
                         'Accepts either a single integer applied to all methods '
                         '(--until-node-id 1000) or per-method pairs '
                         '(--until-node-id baseline=500 e2e_vitb=800).  '
                         'Methods not listed are skipped.  IDs are monotonically '
                         'assigned so ID <= N captures the graph at a point in time.')
    ap.add_argument('--save-planning-examples', action='store_true', default=False,
                    help='Save per-pair planning-example PDFs/PNGs (3 success + 3 '
                         'failure per method) into plots/planning_examples/. '
                         'Off by default — can produce many files.')
    ap.add_argument('--coverage-radius-m', type=float, default=0.5,
                    help='Radius (m) of the filled disk placed around each node when '
                         'computing and visualising free-area coverage. Default 0.5 m.')
    ap.add_argument('--occ-x-min', type=float, default=None,
                    help='Crop occ map: keep only cells with world x >= this value (m).')
    ap.add_argument('--occ-x-max', type=float, default=None,
                    help='Crop occ map: keep only cells with world x <= this value (m).')
    ap.add_argument('--occ-y-min', type=float, default=None,
                    help='Crop occ map: keep only cells with world y >= this value (m).')
    ap.add_argument('--occ-y-max', type=float, default=None,
                    help='Crop occ map: keep only cells with world y <= this value (m).')
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
    baseline_csv  = args.baseline_csv  or in_dir / 'nav_graph_node_timing.csv'

    if not baseline_json.exists():
        print(f'ERROR: missing baseline snapshot: {baseline_json}', flush=True)
        return 2

    # Auto-discover every *.json in inputs/ that isn't the baseline file.
    e2e_jsons: Dict[str, Path] = {}
    for p in sorted(in_dir.glob('*.json')):
        if p.resolve() == baseline_json.resolve():
            continue
        e2e_jsons[p.stem] = p

    if not e2e_jsons:
        print('ERROR: no e2e snapshots found in inputs/. '
              'Run the e2e node with -p name:=<name> first.', flush=True)
        return 2

    all_methods = ['baseline'] + list(e2e_jsons.keys())
    _register_methods(all_methods)

    print(f'Loading graphs ({len(all_methods)} methods):')
    print(f'  baseline ← {baseline_json}')
    graphs: Dict[str, Graph] = {
        'baseline': Graph.load(baseline_json, 'baseline'),
    }
    for name, p in e2e_jsons.items():
        print(f'  {name} ← {p}')
        graphs[name] = Graph.load(p, name)

    # CSV paths: baseline is fixed; e2e methods use {stem}_timing.csv convention.
    csv_paths: Dict[str, Path] = {'baseline': baseline_csv}
    for name in e2e_jsons:
        csv_paths[name] = in_dir / f'{name}_timing.csv'

    print('Loading timing CSVs:')
    timings: Dict[str, Optional[TableType]] = {}
    warmup_thresholds: Dict[str, Optional[int]] = {}
    skip_sec = float(args.skip_first_n_seconds)
    for label, p in csv_paths.items():
        if p.exists():
            try:
                df_raw = read_csv_as_arrays(p)
                n_raw = next((v.size for v in df_raw.values() if v.size > 0), 0)
                skip_n = rows_to_skip_by_seconds(df_raw, skip_sec)
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
            print(f'  {label} ← (missing {p}, timing plots will skip)')
            timings[label] = None
            warmup_thresholds[label] = None

    # Apply warmup filter to graph snapshots.
    for label, thr in warmup_thresholds.items():
        if thr is not None and thr > 0:
            before = graphs[label].num_nodes
            graphs[label] = graphs[label].drop_node_ids_below(thr)
            after = graphs[label].num_nodes
            print(f'  {label} graph: dropped {before - after} warmup nodes '
                  f'(id < {thr}); {after} nodes / {graphs[label].num_edges} '
                  f'edges remain')

    # ── Global occupancy map (optional — written by nav_graph_node on exit) ──
    print('Loading global occupancy map...')
    occ_map = load_global_occ_map(base)
    if occ_map is not None:
        any_crop = any(v is not None for v in (
            args.occ_x_min, args.occ_x_max, args.occ_y_min, args.occ_y_max))
        if any_crop:
            occ_map = crop_occ_map(occ_map,
                                   x_min=args.occ_x_min, x_max=args.occ_x_max,
                                   y_min=args.occ_y_min, y_max=args.occ_y_max)
            print(f'  cropped to x=[{args.occ_x_min}, {args.occ_x_max}] '
                  f'y=[{args.occ_y_min}, {args.occ_y_max}]')
        H_occ, W_occ = occ_map['grid'].shape
        print(f'  global_occ_map: {W_occ}×{H_occ} cells @ '
              f'{occ_map["resolution"]:.3f} m  '
              f'origin=({occ_map["origin_x"]:.1f}, {occ_map["origin_y"]:.1f})')
    else:
        print('  global_occ_map.npz not found — occ-map plots and quality '
              'metrics will be skipped')

    # ── Structural stats ────────────────────────────────────────────
    print('Computing structural stats...')
    stats = {label: graph_stats(g, args.occupancy_cell_m) for label, g in graphs.items()}

    # ── Node / edge quality on occ map ─────────────────────────────
    if occ_map is not None:
        print('Computing node/edge quality on occ map...')
        for label, g in graphs.items():
            stats[label].update(node_edge_quality(g, occ_map))
            stats[label]['free_area_coverage_pct'] = free_area_coverage_pct(
                g, occ_map, args.coverage_radius_m)

    # ── Pairwise similarity (baseline vs each other method) ─────────
    print('Computing pairwise similarity...')
    A = graphs['baseline'].positions
    e2e_method_names = [m for m in graphs if m != 'baseline']
    pairwise: Dict[str, Dict[str, float]] = {}
    nn_pairs: Dict[str, tuple] = {}
    nn_df_rows: List[Dict] = []
    for m in e2e_method_names:
        B = graphs[m].positions
        pairwise[m] = {
            'chamfer_distance_3d_m':    chamfer_distance(A, B),
            'hausdorff_distance_3d_m':  hausdorff_distance(A, B),
            'bbox_iou_2d':              bbox_iou_2d(A, B),
            'occupancy_iou_2d':         occupancy_iou_2d(A, B, args.occupancy_cell_m),
            'covered_area_iou_2d':      covered_area_iou_2d(graphs['baseline'], graphs[m], args.occupancy_cell_m),
            'node_density_correlation': node_density_correlation(A, B, args.occupancy_cell_m),
        }
        nn_b_to_m = nn_distances(A, B)
        nn_m_to_b = nn_distances(B, A)
        nn_pairs[m] = (nn_b_to_m, nn_m_to_b)
        for i in range(B.shape[0]):
            nn_df_rows.append({
                'method': m,
                'node_id': int(graphs[m].ids[i]),
                'x': float(B[i, 0]), 'y': float(B[i, 1]), 'z': float(B[i, 2]),
                'nearest_baseline_m': float(nn_m_to_b[i]) if nn_m_to_b.size else float('nan'),
            })
    for i in range(A.shape[0]):
        nn_df_rows.insert(i, {
            'method': 'baseline',
            'node_id': int(graphs['baseline'].ids[i]),
            'x': float(A[i, 0]), 'y': float(A[i, 1]), 'z': float(A[i, 2]),
            'nearest_baseline_m': float('nan'),
        })
    with open(out_dir / 'per_node_nn.csv', 'w', newline='') as f:
        if nn_df_rows:
            cols = ['method', 'node_id', 'x', 'y', 'z', 'nearest_baseline_m']
            w = _csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in nn_df_rows:
                w.writerow(r)

    # ── Planning quality proxy ──────────────────────────────────────
    planning_details: Dict[str, dict] = {}
    if not args.no_planning:
        print('Sampling planning queries...')
        # Use the union of all node XY sets as the query distribution —
        # ensures all graphs are evaluated on the same world locations.
        all_pos = [g.positions[:, :2] for g in graphs.values() if g.num_nodes > 0]
        if all_pos:
            query_xy = np.vstack(all_pos)
        else:
            query_xy = np.empty((0, 2), dtype=np.float32)
        for label, g in graphs.items():
            run = planning_run(g, query_xy, args.planning_pairs)
            planning_details[label] = run
            stats[label].update(run['metrics'])
    else:
        for label in graphs:
            for k in ('planning_success_rate', 'planning_mean_path_length_m',
                      'planning_median_path_length_m', 'planning_mean_snap_distance_m',
                      'planning_n_pairs'):
                stats[label][k] = float('nan')

    # ── Common-success path lengths (full graphs) ───────────────────
    # Pairs are generated with seed=42 and the same query_xy for all
    # methods, so pair index k corresponds to the same (start, goal)
    # world query across every method.  Find the indices where every
    # method succeeded and report mean path length per method for that subset.
    if planning_details and len(planning_details) > 1:
        pair_counts = [len(d['pairs']) for d in planning_details.values()]
        n_p = min(pair_counts) if pair_counts else 0
        if n_p > 0 and len(set(pair_counts)) == 1:
            common_ok = np.ones(n_p, dtype=bool)
            for d in planning_details.values():
                for k in range(n_p):
                    if not d['pairs'][k]['success']:
                        common_ok[k] = False
            n_common = int(common_ok.sum())
            print(f'  Common success pairs (all methods): {n_common} / {n_p}')
            for label, details in planning_details.items():
                common_lens = [details['pairs'][k]['path_length_m']
                               for k in range(n_p) if common_ok[k]]
                stats[label]['planning_mean_path_length_common_m'] = (
                    float(np.mean(common_lens)) if common_lens else float('nan'))
                stats[label]['planning_n_common_success_pairs'] = float(n_common)
        else:
            for label in graphs:
                stats[label]['planning_mean_path_length_common_m'] = float('nan')
                stats[label]['planning_n_common_success_pairs'] = float('nan')
    else:
        for label in graphs:
            stats[label]['planning_mean_path_length_common_m'] = float('nan')
            stats[label]['planning_n_common_success_pairs'] = float('nan')

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
    plots['xy_overlay']            = plot_xy_overlay(
        {k: graphs[k] for k in ('baseline', 'e2e_vits') if k in graphs},
        plot_dir)
    plots['z_histograms']          = plot_z_histograms(graphs, plot_dir)
    plots['degree_histograms']     = plot_degree_histograms(graphs, plot_dir)
    plots['nn_distance_histograms']= plot_nn_histograms(nn_pairs, plot_dir)
    if any(not table_is_empty(df) for df in timings.values()):
        plots['timing_per_frame']  = plot_timing_per_frame(timings, plot_dir)
        plots['timing_vs_nodes']   = plot_timing_vs_nodes(timings, plot_dir)
        plots['timing_breakdown']  = plot_timing_breakdown(timings, plot_dir)
        plots['timing_breakdown_2'] = plot_timing_breakdown_combined(timings, plot_dir)
        plots['timing_histograms'] = plot_timing_histograms(timings, plot_dir)
        plots['graph_growth']      = plot_graph_growth(timings, plot_dir)

    # Largest connected component overlay (side-by-side per method).
    plots['largest_cc']        = plot_largest_cc(graphs, plot_dir)
    plots['covered_area']      = plot_covered_area(graphs, plot_dir, args.occupancy_cell_m)
    plots['covered_area_iou']  = plot_covered_area_iou(graphs, plot_dir, args.occupancy_cell_m)

    # Graphs overlaid on the global occupancy map (if available).
    if occ_map is not None:
        plots['graphs_on_occ_map'] = plot_graphs_on_occ_map(graphs, occ_map, plot_dir)
        plots['free_area_coverage_map'] = plot_free_area_coverage_map(
            graphs, occ_map, plot_dir, radius_m=args.coverage_radius_m)

        # Single-panel overlay: baseline + e2e_vits only.
        overlay_pair = {k: graphs[k] for k in ('baseline', 'e2e_vits') if k in graphs}
        if len(overlay_pair) >= 2:
            plots['overlay_baseline_vits'] = plot_overlay_on_occ_map(
                overlay_pair, occ_map, plot_dir, name='overlay_baseline_vits')

    # ── Until-node-id analysis ──────────────────────────────────────
    # Parse --until-node-id tokens into a {method: N} dict.
    # Accepts a bare integer ("1000" → all methods) or METHOD=N pairs.
    until_ids: Dict[str, int] = {}
    if args.until_node_id is not None:
        for token in args.until_node_id:
            if '=' in token:
                method_tok, n_tok = token.split('=', 1)
                until_ids[method_tok.strip()] = int(n_tok.strip())
            else:
                # Bare integer → apply to every loaded method.
                n_all = int(token)
                until_ids = {label: n_all for label in graphs}
                break

    if until_ids and occ_map is not None:
        print('Computing until-node-id analysis...')
        graphs_trunc: Dict[str, Graph] = {}
        for label, g in graphs.items():
            n = until_ids.get(label)
            if n is None:
                continue
            g_t = g.keep_nodes_up_to_id(n)
            graphs_trunc[label] = g_t
            print(f'  {label} (until id={n}): '
                  f'{g_t.num_nodes} nodes, {g_t.num_edges} edges')
            if occ_map is not None:
                quality = node_edge_quality(g_t, occ_map)
                for k, v in quality.items():
                    stats[label][f'until{n}_{k}'] = v

        if graphs_trunc:
            plots['graphs_on_occ_map_until'] = plot_graphs_on_occ_map(
                graphs_trunc, occ_map, plot_dir,
                name='graphs_on_occ_map_until', show_quality=True,
            )

            # Covered area IoU for truncated graphs (needs baseline in trunc).
            if 'baseline' in graphs_trunc:
                plots['covered_area_iou_until'] = plot_covered_area_iou(
                    graphs_trunc, plot_dir, args.occupancy_cell_m,
                    name='covered_area_iou_until',
                )
                for m, g_t in graphs_trunc.items():
                    if m == 'baseline':
                        continue
                    n = until_ids[m]
                    iou = covered_area_iou_2d(graphs_trunc['baseline'], g_t,
                                              args.occupancy_cell_m)
                    stats[m][f'until{n}_covered_area_iou_2d'] = iou

            # Planning quality on truncated graphs.
            if not args.no_planning:
                print('  Sampling planning queries for until-N graphs...')
                all_pos_t = [g_t.positions[:, :2] for g_t in graphs_trunc.values()
                             if g_t.num_nodes > 0]
                query_xy_t = (np.vstack(all_pos_t) if all_pos_t
                              else np.empty((0, 2), dtype=np.float32))
                for label, g_t in graphs_trunc.items():
                    n = until_ids[label]
                    run_t = planning_run(g_t, query_xy_t, args.planning_pairs)
                    for k, v in run_t['metrics'].items():
                        stats[label][f'until{n}_{k}'] = v

    # ── Summary table PNG ───────────────────────────────────────────
    plots['summary_table'] = plot_summary_table(
        all_methods, stats, pairwise, timings, until_ids, plot_dir,
    )

    # Per-pair planning example plots — off by default (use --save-planning-examples).
    planning_example_paths: Dict[str, List[Path]] = {}
    if planning_details and args.save_planning_examples:
        try:
            planning_example_paths = plot_planning_examples(
                graphs, planning_details, plot_dir, n_each=3,
            )
            n_written = sum(len(v) for v in planning_example_paths.values())
            print(f'Planning examples → {plot_dir / "planning_examples"} '
                  f'({n_written} PDFs written)')
        except Exception as exc:
            print(f'plot_planning_examples failed: {exc}')

    # ── Summary CSV + markdown report ───────────────────────────────
    summary_rows = build_summary_rows(stats, pairwise)
    write_summary_csv(summary_rows, out_dir / 'summary.csv', all_methods)
    report = build_markdown_report(
        summary_rows, plots, graphs,
        skip_first_n_seconds=skip_sec,
        planning_example_paths=planning_example_paths or None,
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
