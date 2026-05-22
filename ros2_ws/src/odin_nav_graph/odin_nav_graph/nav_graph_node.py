#!/usr/bin/env python3
"""ROS2 node that builds a navigation graph from /odin1/cloud_raw using nav_graph_gpu.

Pipeline per cloud message:
    1. Parse PointCloud2 -> (N, 3) xyz in odin1_base_link.
    2. Find the closest /odin1/odometry_highfreq message in time and use its
       pose to transform points into the odom frame.
    3. Move the rolling elevation map to the robot's current xy and project
       the transformed points into it (max-z aggregation).
    4. Hand the elevation map to NavigationGraphBuilder
       (input_type="elevation_map") which internally:
         - computes traversability,
         - converts it to occupancy,
         - rotates into the bottom-left occupancy convention,
         - runs the GPU frontier kernel + local waypoint generator,
         - merges into the persistent global graph.
    5. Publish elevation cloud + graph nodes / edges / frontiers in odom.

All published topics live in the ``odom`` frame so RViz only needs the
fixed frame set to ``odom``.

ros2 run odin_nav_graph nav_graph_node   --ros-args   -p out_directory:=/home/rohang73/Documents/odin_e2e/saved_outputs_odin   -p cam_fx:=800.0   -p cam_fy:=800.0   -p cam_cx:=800.0   -p cam_cy:=648.0 -p map_length_xy:=30.0 -p cloud_max_range:=15.0

"""

from __future__ import annotations

import math
import os
import sys
import time
from collections import deque
from contextlib import suppress
from typing import Optional, Tuple

import numpy as np
import rclpy
import torch
import torch.nn.functional as F
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
from nav_msgs.msg import Odometry, OccupancyGrid, MapMetaData
from std_msgs.msg import Header
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose, Quaternion
import base64
import colorsys
import cv2
import json
import queue
import threading
from pathlib import Path
from scipy.spatial.transform import Rotation as _ScipyR
from scipy.ndimage import binary_dilation

# Allow running without pip-installing nav_graph by adding the sibling
# ``nav_graph_gpu`` checkout to sys.path.  Pip-installed nav_graph wins.
def _ensure_nav_graph_on_path() -> None:
    try:
        import nav_graph  # noqa: F401
        return
    except ImportError:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.environ.get('ODIN_NAV_GRAPH_GPU_PATH', ''),
        # Source layout: <repo>/ros2_ws/src/odin_nav_graph/odin_nav_graph
        os.path.abspath(os.path.join(here, '..', '..', '..', '..', 'nav_graph_gpu')),
        # Installed layout (best-effort, may not match every distro)
        os.path.abspath(os.path.join(here, '..', '..', '..', '..', '..', '..', 'nav_graph_gpu')),
    ]
    for path in candidates:
        if path and os.path.isdir(os.path.join(path, 'nav_graph')):
            sys.path.insert(0, path)
            return


_ensure_nav_graph_on_path()


def _ensure_explorfm_on_path() -> None:
    """Make ``explorfm`` and its ``nvidia_radio`` sibling importable.

    Both live inside the ``nebula2-wildos`` submodule at repo root.  We add
    the submodule root (so ``explorfm`` and ``nvidia_radio`` resolve as
    top-level packages, matching how ExploRFMInference imports them).
    """
    try:
        import explorfm  # noqa: F401
        import nvidia_radio  # noqa: F401
        return
    except ImportError:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.environ.get('NEBULA2_WILDOS_PATH', ''),
        os.path.abspath(os.path.join(here, '..', '..', '..', '..', 'nebula2-wildos')),
        os.path.abspath(os.path.join(here, '..', '..', '..', '..', '..', '..', 'nebula2-wildos')),
    ]
    for path in candidates:
        if path and os.path.isdir(os.path.join(path, 'explorfm')):
            sys.path.insert(0, path)
            return


from nav_graph import (  # noqa: E402
    NavigationGraphBuilder,
    NavGraphConfig,
    FrontierConfig,
    ExplorationConfig,
    ElevationMapConfig,
)
from nav_graph.core.graph_layer import (  # noqa: E402
    ComputeLayer,
    ExternalLayer,
    GraphContext,
    MergePolicy,
)

from odin_nav_graph.elevation_map import ElevationMapWrapper  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────

def parse_xyz_points(msg: PointCloud2) -> np.ndarray:
    """Extract (N, 3) float32 xyz from a PointCloud2 with arbitrary
    (possibly unaligned) point_step.  Faster than read_points(list)."""
    n = msg.width * msg.height
    if n == 0:
        return np.empty((0, 3), dtype=np.float32)
    offs = {f.name: f.offset for f in msg.fields}
    if not all(k in offs for k in ('x', 'y', 'z')):
        raise ValueError(f'PointCloud2 missing x/y/z fields: {list(offs)}')
    ox, oy, oz = offs['x'], offs['y'], offs['z']
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, msg.point_step)
    out = np.empty((n, 3), dtype=np.float32)
    out[:, 0] = np.ascontiguousarray(raw[:, ox:ox + 4]).view(np.float32).ravel()
    out[:, 1] = np.ascontiguousarray(raw[:, oy:oy + 4]).view(np.float32).ravel()
    out[:, 2] = np.ascontiguousarray(raw[:, oz:oz + 4]).view(np.float32).ravel()
    return out


def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Right-handed rotation matrix from a unit quaternion (xyzw)."""
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )


def stamp_to_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


# ─────────────────────────────────────────────────────────────────────
#  Builder patches
#
#  Two wrappers for real-world (vs. sim) robustness.  Both are
#  independent of elevation-Z handling — nav_graph now does that
#  correctly in-core: the waypoint generator stamps per-node Z from the
#  elevation map at graph-generation time, so local and global Z agree
#  and the 3D merge works without help.
#
#    - _make_traversability_smoother: majority-vote median filter on the
#      categorical traversability map, killing speckle.
#    - _make_ext_map_fixup: lets confirmed observations clear cells a
#      noisy scan marked stuck-occupied in the extended map.
#
#  The patches operate on builder instances so we don't have to touch
#  the nav_graph submodule.  Keep them in sync with the upstream
#  ``GlobalGraphGenerator`` invariants documented in nav_graph's
#  CLAUDE.md (parallel-array invariant, ID-vs-index discipline).
#
#  Removed 2026-05: the 2D-merge and in-bounds-Z patches (the latter
#  overrode ``_assign_z_from_elevation``, which no longer exists).  The
#  per-node-Z min-filter moved into nav_graph's WaypointGraphGeneratorConfig
#  as ``elevation_min_filter_radius`` — fed below by the
#  ``z_lookup_min_filter_radius_cells`` ROS param.
# ─────────────────────────────────────────────────────────────────────

def _make_traversability_smoother(builder, median_size: int):
    """Wrap ``_compute_traversability`` with a post-pass median-filter smoother.

    Encoding before filtering:  free=1.0, unknown=NaN→0.5, occupied=0.0.
    After ``scipy.ndimage.median_filter`` on this float map, re-threshold:
      >=0.75 → free, <=0.25 → occupied, else → NaN.

    A cell needs a majority of same-type neighbours to keep its class.
    Isolated speckles of any type (free island, occupied dot, unknown gap)
    get replaced by the value of the cells around them — exactly a
    majority-vote / smoothing operation on the categorical map.

    ``median_size`` must be odd and >= 3; 0 or 1 disables.
    """
    if median_size < 3 or median_size % 2 == 0:
        return  # nothing to install

    orig_fn = builder._compute_traversability  # static method → callable

    def _patched(elevation: np.ndarray, config):
        from scipy.ndimage import median_filter as _mf
        trav = orig_fn(elevation, config)

        # NaN (unknown) → 0.5 so it participates in the median vote.
        filled = np.where(np.isnan(trav), 0.5, trav).astype(np.float32)
        smoothed = _mf(filled, size=median_size, mode='nearest')

        result = np.full_like(trav, np.nan, dtype=np.float32)
        result[smoothed >= 0.75] = 1.0   # majority free
        result[smoothed <= 0.25] = 0.0   # majority occupied
        # 0.25 < smoothed < 0.75 stays NaN (genuinely mixed neighbourhood)
        return result

    builder._compute_traversability = _patched


def _make_obstacle_unknown_dilator(builder, obs_iters: int, unknown_iters: int):
    """Wrap ``_compute_traversability`` to dilate obstacles, then unknown.

    Runs *after* the median smoother (install this last).  Two sequential
    passes on the categorical traversability map (free=1.0, unknown=NaN,
    occupied=0.0):

      1. Dilate occupied cells outward by ``obs_iters`` iterations of a
         3×3 structuring element.  Free cells abutting obstacles become
         occupied → no narrow free strips hugging walls.
      2. Dilate unknown cells outward by ``unknown_iters`` iterations,
         but never overwrite obstacles.  Free cells abutting unknown
         become unknown → no narrow free strips between obstacles and
         unknown space (the case the sampler keeps grabbing).

    At 0.10 m/cell, ``iters=N`` ≈ N·10 cm of growth.  Either count ≤ 0
    disables that pass; both ≤ 0 is a no-op.
    """
    if obs_iters <= 0 and unknown_iters <= 0:
        return

    orig_fn = builder._compute_traversability  # callable

    def _patched(elevation: np.ndarray, config):
        trav = orig_fn(elevation, config)
        if trav is None:
            return trav
        out = trav.copy()
        if obs_iters > 0:
            occ = (out == 0.0)
            if occ.any():
                occ_d = binary_dilation(occ, iterations=obs_iters)
                out[occ_d] = 0.0
        if unknown_iters > 0:
            unk = np.isnan(out)
            if unk.any():
                unk_d = binary_dilation(unk, iterations=unknown_iters)
                # Don't overwrite obstacles; only convert free → unknown.
                out[unk_d & (out == 1.0)] = np.nan
        return out

    builder._compute_traversability = _patched


def _make_ext_map_fixup(gg, original_fn):
    """Wrap ``_update_extended_map`` so that confirmed observations always win.

    The default merge rule makes occupied cells sticky: once a cell is written
    as 0 (occupied) it can never be cleared to free (255).  In simulation this
    is fine — walls are walls.  In real-world data, traversability
    misclassification from sensor noise permanently blocks the extended map so
    no edges can be collision-checked through it.

    This wrapper calls the original function (which handles rolling, padding,
    and first-observation logic) and then does a second pass over the local
    footprint with the rule:

        any confirmed (non-unknown) observation → overwrite

    That means a free observation from the current scan can clear a cell that
    was previously marked occupied by a noisy scan.  Cells outside the local
    footprint are unaffected — their history is preserved.

    Only applies when ``occ_grid_int8`` is provided (3-state path), because
    the binary path cannot distinguish unknown from occupied.
    """
    def _patched(new_local_grid, center_x, center_y, resolution, occ_grid_int8=None):
        original_fn(new_local_grid, center_x, center_y, resolution, occ_grid_int8=occ_grid_int8)

        if gg._ext_map is None or occ_grid_int8 is None:
            return

        H, W = new_local_grid.shape
        ext_H, ext_W = gg._ext_shape
        row_off = (ext_H - H) // 2
        col_off = (ext_W - W) // 2
        device = gg._ext_map.device

        # Reconstruct local_ext exactly as _update_extended_map does (occ_grid_int8
        # has already been rot90'd twice by add_local_graph_gpu before being passed here).
        int8_t = torch.from_numpy(np.ascontiguousarray(occ_grid_int8)).to(device)
        local_ext = torch.full((H, W), 128, dtype=torch.uint8, device=device)
        local_ext[int8_t == 0]   = 255   # free
        local_ext[int8_t == 100] = 0     # occupied

        # Any confirmed observation overwrites — overrides the sticky-occupied rule.
        ext_slice = gg._ext_map[row_off:row_off + H, col_off:col_off + W]
        confirmed = local_ext != 128
        ext_slice[confirmed] = local_ext[confirmed]

    return _patched


def _install_builder_patches(builder) -> None:
    """Install instance-level patches on a ``NavigationGraphBuilder``."""
    gg = builder.global_builder
    # Wrap _update_extended_map so confirmed observations can clear stuck-occupied cells.
    gg._update_extended_map = _make_ext_map_fixup(gg, gg._update_extended_map)


def _install_builder_timing_patches(builder) -> dict:
    """Wrap the local-graph generator + global-merge calls to record their
    per-call durations.  Returns a dict that the caller can read after each
    ``builder.update(...)``:

      * ``local_ms`` — last ``local_generator.build_graph_from_grid_map`` call.
      * ``merge_ms`` — last ``global_builder.add_local_graph_gpu`` call.

    Both keys absent until the first wrapped call.  The patch is idempotent —
    re-applying it would re-stack timings; only call once per builder.
    """
    timings: dict = {}

    lg = builder.local_generator
    orig_build = lg.build_graph_from_grid_map
    def _timed_build(*args, **kwargs):
        t0 = time.perf_counter()
        out = orig_build(*args, **kwargs)
        timings['local_ms'] = (time.perf_counter() - t0) * 1000.0
        return out
    lg.build_graph_from_grid_map = _timed_build

    gg = builder.global_builder
    orig_merge = gg.add_local_graph_gpu
    def _timed_merge(*args, **kwargs):
        t0 = time.perf_counter()
        out = orig_merge(*args, **kwargs)
        timings['merge_ms'] = (time.perf_counter() - t0) * 1000.0
        return out
    gg.add_local_graph_gpu = _timed_merge

    return timings


def _pq_to_se3(translation: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    """4×4 float64 SE3 from translation (3,) and unit quaternion (x,y,z,w)."""
    se3 = np.eye(4, dtype=np.float64)
    se3[:3, :3] = _ScipyR.from_quat(quaternion_xyzw).as_matrix()
    se3[:3, 3] = translation
    return se3


def _rgb_to_b64png(rgb: np.ndarray) -> str:
    """Encode an HxWx3 RGB array as a base64-encoded PNG string."""
    _, buf = cv2.imencode('.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf.tobytes()).decode()


def _build_rviz_rainbow_lut(n: int = 256) -> np.ndarray:
    """RViz default intensity colormap, sampled to ``n`` entries.

    Reproduces ``rviz_default_plugins`` ``IntensityPCTransformer`` with
    "Use rainbow" enabled: h = (1 - v) * 5/6, s = 1, v = 1.  Intensity 0
    is blue, 1 is red, with the usual cyan/green/yellow stops in between.
    Returns an (n, 3) float32 array of RGB in [0, 1].
    """
    out = np.empty((n, 3), dtype=np.float32)
    for i, t in enumerate(np.linspace(0.0, 1.0, n)):
        h = (1.0 - float(t)) * 5.0 / 6.0
        out[i] = colorsys.hsv_to_rgb(h, 1.0, 1.0)
    return out


_RVIZ_RAINBOW_LUT = _build_rviz_rainbow_lut()


def _put_outlined_text(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    scale: float = 0.5,
    fg: Tuple[int, int, int] = (255, 255, 255),
    bg: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    """putText with a 3-px black stroke under the foreground glyph.

    The thick black pass acts as an outline so the foreground stays legible
    against any image background.  Used by the frontier debug legend.
    """
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, bg, 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, fg, 1, cv2.LINE_AA)


def _save_svg_plain(path: Path, rgb: np.ndarray) -> None:
    h, w = rgb.shape[:2]
    b64 = _rgb_to_b64png(rgb)
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">\n'
        f'<image href="data:image/png;base64,{b64}" width="{w}" height="{h}"/>\n'
        f'</svg>\n'
    )


def _node_circles(nodes: list) -> list:
    parts = []
    for node in nodes:
        u, v = node['pixel']
        color = '#ffff00' if node['type'] == 'frontier' else '#0000ff'
        r = 7 if node['type'] == 'frontier' else 5
        parts.append(f'<circle cx="{u}" cy="{v}" r="{r}" fill="{color}" opacity="0.85"/>')
    return parts


def _save_svg_overlay(path: Path, rgb: np.ndarray, nodes: list) -> None:
    h, w = rgb.shape[:2]
    b64 = _rgb_to_b64png(rgb)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
        f'<image href="data:image/png;base64,{b64}" width="{w}" height="{h}"/>',
    ] + _node_circles(nodes) + ['</svg>']
    path.write_text('\n'.join(parts))


def _save_svg_edges(path: Path, rgb: np.ndarray, nodes: list, edges: list) -> None:
    h, w = rgb.shape[:2]
    b64 = _rgb_to_b64png(rgb)
    id_to_px = {n['id']: n['pixel'] for n in nodes}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
        f'<image href="data:image/png;base64,{b64}" width="{w}" height="{h}"/>',
    ]
    for edge in edges:
        p0 = id_to_px.get(edge['node_id_0'])
        p1 = id_to_px.get(edge['node_id_1'])
        if p0 and p1:
            parts.append(
                f'<line x1="{p0[0]}" y1="{p0[1]}" x2="{p1[0]}" y2="{p1[1]}"'
                f' stroke="#ffffff" stroke-width="1.5" opacity="0.6"/>'
            )
    parts += _node_circles(nodes)
    parts.append('</svg>')
    path.write_text('\n'.join(parts))


# ─────────────────────────────────────────────────────────────────────
#  Layers
# ─────────────────────────────────────────────────────────────────────

class VisitedTimeLayer(ExternalLayer):
    """Tracks how often the robot has been near each node.

    Each frame the node closest to the robot gets +1 ingested via the
    CUSTOM merge policy (``new = old + delta``).  ``read()`` divides the
    accumulated count by ``total_frames`` so the per-node value is the
    fraction of time spent near that node — bounded in [0, 1] regardless
    of run length.
    """

    name = 'visited_time'

    def __init__(self, layer_name: str = 'visited_time') -> None:
        super().__init__(
            name=layer_name,
            ingest_policy=MergePolicy.CUSTOM,
            custom_ingest_fn=lambda old, new: old + new,
        )
        self.total_frames: int = 0

    def tick(self) -> None:
        """Advance the frame counter; call once per cloud_callback."""
        self.total_frames += 1

    def read(self, node_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
        counts = super().read(node_ids, device)
        if self.total_frames == 0:
            return counts
        return counts / float(self.total_frames)


class PathProximityLayer(ComputeLayer):
    """Scores every node by closeness to the robot's travelled path.

    Each ``compute()`` appends the robot's current xy to an internal path
    buffer (gated by ``min_step`` so the buffer stays sparse, capped at
    ``max_points``), then scores every node ``1 / (1 + falloff * d)`` where
    ``d`` is the distance to the nearest path point.  All on the GPU.
    """

    name = 'path_proximity'

    def __init__(
        self,
        layer_name: str = 'path_proximity',
        falloff: float = 0.5,
        min_step: float = 0.2,
        max_points: int = 8000,
    ) -> None:
        self.name = layer_name
        self.falloff = float(falloff)
        self._min_step_sq = float(min_step) ** 2
        self._max_points = int(max_points)
        self._path: list = []  # list of (x, y) in the odom frame

    def compute(self, ctx: GraphContext) -> torch.Tensor:
        # Append the current robot xy, gated by the minimum step.
        if ctx.robot_position is not None:
            x, y = ctx.robot_position
            if not self._path:
                self._path.append((x, y))
            else:
                lx, ly = self._path[-1]
                if (x - lx) ** 2 + (y - ly) ** 2 >= self._min_step_sq:
                    self._path.append((x, y))
                    if len(self._path) > self._max_points:
                        del self._path[:-self._max_points]

        n = ctx.num_nodes
        if n == 0 or not self._path:
            return torch.zeros(n, dtype=torch.float32, device=ctx.device)

        node_xy = ctx.node_positions[:, :2].contiguous()
        path_xy = torch.tensor(self._path, dtype=torch.float32, device=ctx.device)
        dmin = torch.cdist(node_xy, path_xy).amin(dim=1)
        return 1.0 / (1.0 + self.falloff * dmin)


# ─────────────────────────────────────────────────────────────────────
#  Node
# ─────────────────────────────────────────────────────────────────────

class OdinNavGraphNode(Node):
    def __init__(self) -> None:
        super().__init__('odin_nav_graph_node')

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter('cloud_topic', '/odin1/cloud_raw')
        self.declare_parameter('odom_topic', '/odin1/odometry_highfreq')
        self.declare_parameter('frame_id', 'odom')
        self.declare_parameter('robot_frame', 'odin1_base_link')

        # Elevation map geometry
        self.declare_parameter('map_length_xy', 12.0)
        self.declare_parameter('map_resolution', 0.10)
        self.declare_parameter('cloud_max_range', 7.0)
        self.declare_parameter('sensor_noise_factor', 0.05)
        # Drop elevation cells whose absolute height differs from the robot's
        # current odom z by more than this many metres (catches roofs/ceilings
        # the lidar sweeps in).  0 disables the filter.
        self.declare_parameter('elevation_z_clip', 0.9)
        self.declare_parameter('em_position_noise', 0.0)
        self.declare_parameter('em_orientation_noise', 0.0)

        # Time sync
        self.declare_parameter('odom_buffer_seconds', 1.0)
        self.declare_parameter('odom_match_max_dt', 0.1)

        # nav_graph builder params
        self.declare_parameter('safety_distance', 0.05)
        self.declare_parameter('merge_node_distance', 0.6)
        self.declare_parameter('global_merge_distance', 0.6)
        self.declare_parameter('global_max_candidate_edge_distance', 3.0)
        self.declare_parameter('free_space_sampling_threshold', 0.50)
        self.declare_parameter('boundary_inflation_factor', 1.2)

        self.declare_parameter('frontier_kernel_size', 2)
        self.declare_parameter('frontier_odom_threshold', 1.0)
        self.declare_parameter('frontier_max_edge_connectivity', 14)
        self.declare_parameter('minimum_distance_between_frontiers', 0.01)
        self.declare_parameter('minimum_points_in_cluster', 1)
        # Fraction of the neighbourhood window (relative to its area) that must
        # be free / unknown for a cell to qualify as a frontier.
        self.declare_parameter('frontier_min_free_fraction', 0.25)
        self.declare_parameter('frontier_min_unknown_fraction', 0.25)

        # Elevation -> traversability params
        self.declare_parameter('elev_max_height_diff', 0.7)
        self.declare_parameter('elev_max_slope', 1.0)
        self.declare_parameter('elev_gaussian_sigma', 0.5)
        self.declare_parameter('elev_window_size', 5)
        self.declare_parameter('elev_border_cells', 0)
        # Traversability smoothing: median filter on the float map after height/slope
        # computation but before node generation.  NaN (unknown) is treated as 0.5
        # (midpoint between free=1.0 and occupied=0.0) so it participates in the vote.
        # Re-threshold: >=0.75 → free, <=0.25 → occupied, else → NaN (unknown).
        # This means a cell needs a majority of same-type neighbours to keep its class;
        # isolated speckles of any type get replaced by their surroundings.
        # Must be an odd integer >= 3.  0 or 1 disables.  At 0.10 m/cell, size=3
        # removes single-cell speckles (10 cm); size=5 removes up to 20 cm features.
        self.declare_parameter('trav_median_filter_size', 5)
        # Categorical dilation pass that runs after the median smoother.
        # Counters narrow free strips next to obstacles / unknown space that the
        # sampler keeps picking up.  At 0.10 m/cell, iters=N ≈ N·10 cm of growth.
        # Either ≤ 0 disables that pass.
        self.declare_parameter('trav_obstacle_dilate_iters', 1)
        self.declare_parameter('trav_unknown_dilate_iters', 1)

        # Throttling / publishing
        self.declare_parameter('process_every_n', 1)
        self.declare_parameter('max_cloud_frames', 10000)   # 0 = unlimited; stop graph updates after N frames  #200 when starting from 200s
        self.declare_parameter('publish_elevation_cloud', True)
        # 2D top-down heatmap of the local elevation map (nav_msgs/OccupancyGrid).
        # Heights in [z_min, z_max] are linearly quantised to 0..100; cells
        # outside the range are clamped; unknown cells become -1.
        self.declare_parameter('publish_elevation_grid', True)
        self.declare_parameter('elevation_grid_z_min', -4.0)
        self.declare_parameter('elevation_grid_z_max', 10.0)
        self.declare_parameter('publish_edges', True)
        self.declare_parameter('max_edges_published', 100000)
        # Safety: if the global graph blows up past this many nodes,
        # log a warning and reset.  Healthy operation should stay well
        # under this — typically a few thousand nodes for a reasonable
        # exploration area.
        self.declare_parameter('max_graph_nodes_before_reset', 30000)
        # Radius (in cells) of the min-filter applied before reading
        # per-node Z off the elevation map.  Counters far-range /
        # grazing-angle bias that puffs up boundary nodes' Z.
        # 0 disables; default 2 → 5×5-cell window (50 cm at 0.1 m res).
        self.declare_parameter('z_lookup_min_filter_radius_cells', 2)
        # Visualisation-only offset added to every published graph node
        # (graph_nodes, frontier_cloud, graph_edges) so they sit clearly
        # above the SLAM cloud / elevation cloud in RViz.  Stored Z in
        # the global graph is unchanged.
        self.declare_parameter('viz_z_offset', 0.45)

        # RGB + graph saver
        self.declare_parameter('out_directory', '')
        self.declare_parameter('save_every_n_frames', 5)
        self.declare_parameter('save_frame_start', 7000)   # first _rgb_count to save (80*20)
        self.declare_parameter('save_frame_end',   7000)   # last  _rgb_count to save (120*20)
        self.declare_parameter('cam_image_topic', '/odin1/image/undistorted')
        self.declare_parameter('cam_info_topic', '/odin1/camera_info')
        self.declare_parameter('cam_frame', 'camera_optical')  # label for JSON only
        self.declare_parameter('cam_fx', 0.0)
        self.declare_parameter('cam_fy', 0.0)
        self.declare_parameter('cam_cx', 0.0)
        self.declare_parameter('cam_cy', 0.0)
        # Static camera-to-base_link extrinsic (Odin Nav Stack defaults)
        self.declare_parameter('cam_base_tx', -0.0042)
        self.declare_parameter('cam_base_ty',  0.0328)
        self.declare_parameter('cam_base_tz',  0.0005)
        self.declare_parameter('cam_base_qx', -0.4951)
        self.declare_parameter('cam_base_qy',  0.5048)
        self.declare_parameter('cam_base_qz', -0.4996)
        self.declare_parameter('cam_base_qw',  0.5005)

        # ExploRFM (per-image traversability + frontier-score model) → node layers
        self.declare_parameter('enable_explorfm_layers', True)
        self.declare_parameter('explorfm_every_n_images', 1)
        self.declare_parameter('explorfm_frontier_ckpt', '')
        self.declare_parameter('explorfm_trav_ckpt', '')
        self.declare_parameter('explorfm_radio_version', 'c-radio_v3-b')
        self.declare_parameter('explorfm_adaptor_version', '')
        self.declare_parameter('explorfm_adaptor_ckpt_path', '')
        self.declare_parameter('explorfm_use_naclip', True)
        self.declare_parameter('explorfm_radio_dim', 768)
        self.declare_parameter('explorfm_static_scale_factor', 0.5)
        self.declare_parameter('explorfm_precision', 'FP16')
        self.declare_parameter('explorfm_trav_layer', 'traversability')
        self.declare_parameter('explorfm_frontier_layer', 'frontier_score')
        self.declare_parameter('explorfm_car_detector_layer', 'car')
        # 3D-distance gating for the car layer: a node within
        # car_distance_threshold metres of the car's 3D centroid is scored
        # 1 - d/threshold; nodes outside are forced to 0.  The 3D centroid
        # is the nearest projected graph node to the centroid pixel of the
        # car high-sim region.
        self.declare_parameter('car_distance_threshold', 8.0)
        # Debug-image saver for the car layer.  Writes a PNG every N image
        # callbacks with: RGB + projected graph nodes + the high-sim car
        # region tinted + the centroid pixel marked.  Independent of the
        # main RGB+graph saver (which is gated by save_frame_start/end).
        self.declare_parameter('car_debug_save_every_n', 10000)
        self.declare_parameter(
            'car_debug_dir',
            '/home/rohang73/Documents/odin_e2e/car_layer_degub_images',
        )
        # Optional parallel PNG output — same rendered frame, PNG alongside
        # the PDF.  The PDF is the canonical archive; PNG is for quick
        # previews / video stitching.  Empty string disables.
        self.declare_parameter(
            'car_debug_png_dir',
            '/home/rohang73/Documents/odin_e2e/car_layer_degub_images_png',
        )
        # Debug-image saver for the frontier_score layer.  Same async PDF
        # pipeline as the car saver — writes RGB + per-pixel ExploRFM
        # frontier heatmap overlay + visible frontier nodes coloured by
        # the persistent frontier_score layer (matches the on-screen
        # /frontier_score_cloud's RViz rainbow + autobounds).
        self.declare_parameter('frontier_debug_save_every_n', 10000)
        self.declare_parameter(
            'frontier_debug_dir',
            '/home/rohang73/Documents/odin_e2e/frontier_layer_debug_images',
        )
        self.declare_parameter(
            'frontier_debug_png_dir',
            '/home/rohang73/Documents/odin_e2e/frontier_layer_debug_images_png',
        )
        # Heatmap-overlay threshold for the frontier debug image.  Only
        # pixels with front_map > mean + k·std are colored — everything
        # else stays as the raw RGB.  Higher k → smaller, brighter patches.
        self.declare_parameter('frontier_debug_high_std_k', 2.5)
        # Whether to overlay graph edges (white lines between connected
        # visible nodes) in the car / frontier debug images.  Off by
        # default — set true to inspect graph connectivity in-frame.
        self.declare_parameter('debug_image_draw_edges', False)

        # ── Graph-comparison artefacts ───────────────────────────────────
        # Written into the same directory the comparison script reads.  The
        # JSON is the final-state graph snapshot (called from main() on
        # Ctrl-C or normal exit); the CSV is a per-frame timing breakdown
        # of the core graph computation, appended every cloud_callback.
        self.declare_parameter(
            'graph_snapshot_path',
            '/home/rohang73/Documents/odin_e2e/graph_compaRISION/inputs/nav_graph_node.json',
        )
        self.declare_parameter(
            'timing_csv_path',
            '/home/rohang73/Documents/odin_e2e/graph_compaRISION/inputs/nav_graph_node_timing.csv',
        )
        self.declare_parameter('enable_graph_snapshot_save', True)
        self.declare_parameter('enable_timing_csv', True)

        # Image-space car detection.  A pixel is "high-sim car" iff it passes
        # BOTH thresholds — the adaptive one is necessary (the loudest tail
        # always exists) but not sufficient on no-car frames, so the absolute
        # floor catches frames where the loudest tail still has low cosine
        # similarity overall.
        #   explorfm_object_std_k        — adaptive: sim > mean + k·std of
        #       this frame's sim map.  Higher k → fewer high pixels per frame.
        #   car_absolute_sim_threshold   — absolute floor on the cosine
        #       similarity itself.  Set above the typical no-car ``sim_max``
        #       you see in the log to suppress phantom detections (e.g. 0.15
        #       or 0.20 for SigLIP2 raw cosine sim; tune from the per-frame
        #       sim_max printed in the [explorfm rgb=…] log line).  0.0 →
        #       absolute floor disabled, only the adaptive cutoff applies.
        #   car_min_high_pixels          — final guardrail: if fewer than
        #       this many pixels survive both thresholds, the detection is
        #       treated as noise (no centroid, all car scores → 0).
        #   car_max_centroid_node_pixel_distance — the nearest projected
        #       graph node to the car centroid must be within this many
        #       pixels.  If no node is that close, the centroid has no
        #       trustworthy depth anchor and the detection is dropped (all
        #       node scores → 0).  Set <= 0 to disable the check.
        #   explorfm_object_threshold    — legacy floor; folded into the same
        #       check as car_absolute_sim_threshold via max(). Kept for
        #       backward compatibility with existing launch configs.
        #   explorfm_object_uv_radius    — deprecated, no longer used (car
        #       scores are now driven by 3D distance, not pixel dilation).
        self.declare_parameter('explorfm_object_std_k', 3.0)
        self.declare_parameter('car_absolute_sim_threshold', 0.1)
        self.declare_parameter('car_min_high_pixels', 15)
        self.declare_parameter('car_max_centroid_node_pixel_distance', 70.0)
        self.declare_parameter('explorfm_object_threshold', 0.0)
        self.declare_parameter('explorfm_object_uv_radius', 25)

        # Visited-time layer (per-node fraction of frames the robot was nearest to it).
        self.declare_parameter('enable_visited_time_layer', True)
        self.declare_parameter('visited_time_layer', 'visited_time')

        # Path-proximity layer (per-node closeness to the robot's travelled path).
        self.declare_parameter('enable_path_proximity_layer', True)
        self.declare_parameter('path_proximity_layer', 'path_proximity')

        # Per-node proximity falloff for the path layer.
        # score = 1 / (1 + falloff * distance_to_nearest_source) → 1 on top of a
        # source, decaying with distance.  0.5 ⇒ score ≈ 0.5 at 2 m away.
        self.declare_parameter('layer_proximity_falloff', 0.5)

        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        cloud_topic = gp('cloud_topic')
        odom_topic = gp('odom_topic')
        self.frame_id = gp('frame_id')
        self.robot_frame = gp('robot_frame')

        length_xy = float(gp('map_length_xy'))
        res = float(gp('map_resolution'))
        self.cloud_max_range = float(gp('cloud_max_range'))
        sensor_noise_factor = float(gp('sensor_noise_factor'))
        self.em_position_noise = float(gp('em_position_noise'))
        self.em_orientation_noise = float(gp('em_orientation_noise'))
        self.odom_buffer_seconds = float(gp('odom_buffer_seconds'))
        self.odom_match_max_dt = float(gp('odom_match_max_dt'))
        self.process_every_n = int(gp('process_every_n'))
        self.max_cloud_frames = int(gp('max_cloud_frames'))
        self.publish_elevation_cloud = bool(gp('publish_elevation_cloud'))
        self.publish_elevation_grid = bool(gp('publish_elevation_grid'))
        self.elevation_grid_z_min = float(gp('elevation_grid_z_min'))
        self.elevation_grid_z_max = float(gp('elevation_grid_z_max'))
        self.publish_edges_flag = bool(gp('publish_edges'))
        self.max_graph_nodes_before_reset = int(gp('max_graph_nodes_before_reset'))
        self._z_lookup_min_filter_radius_cells = int(gp('z_lookup_min_filter_radius_cells'))
        self.viz_z_offset = float(gp('viz_z_offset'))
        self._layer_falloff = float(gp('layer_proximity_falloff'))
        self.max_edges_published = int(gp('max_edges_published'))

        # ── nav_graph config ──────────────────────────────────────────
        cfg = NavGraphConfig(
            free_space_sampling_threshold=float(gp('free_space_sampling_threshold')),
            boundary_inflation_factor=float(gp('boundary_inflation_factor')),
            safety_distance=float(gp('safety_distance')),
            merge_node_distance=float(gp('merge_node_distance')),
            global_merge_distance=float(gp('global_merge_distance')),
            global_max_candidate_edge_distance=float(gp('global_max_candidate_edge_distance')),
            global_max_candidate_edge_search_distance = 100,
            global_max_connections = 10,
            elevation_min_filter_radius=self._z_lookup_min_filter_radius_cells,
            frontier=FrontierConfig(
                kernel_size=int(gp('frontier_kernel_size')),
                odom_proximity_threshold=float(gp('frontier_odom_threshold')),
                max_edge_connectivity=int(gp('frontier_max_edge_connectivity')),
                minimum_distance_between_frontiers=float(gp('minimum_distance_between_frontiers')),
                minimum_points_in_cluster=int(gp('minimum_points_in_cluster')),
                min_free_fraction=float(gp('frontier_min_free_fraction')),
                min_unknown_fraction=float(gp('frontier_min_unknown_fraction')),
                angular_gap_min_gap_deg = 100.0,
            ),
            elevation_map=ElevationMapConfig(
                gaussian_sigma=float(gp('elev_gaussian_sigma')),
                window_size=int(gp('elev_window_size')),
                max_height_diff=float(gp('elev_max_height_diff')),
                max_slope=float(gp('elev_max_slope')),
                border_cells=int(gp('elev_border_cells')),
            ),
        )

        self.get_logger().info('Initialising NavigationGraphBuilder (GPU)...')
        self.builder = NavigationGraphBuilder(cfg)
        # Real-world robustness wrappers (per-node Z is handled in-core now).
        _install_builder_patches(self.builder)
        # Sub-step timings (local-graph + global-merge) for the comparison
        # CSV — wraps the same builder before any other patches stack on top.
        self._builder_timings: dict = _install_builder_timing_patches(self.builder)
        self._trav_median_filter_size = int(gp('trav_median_filter_size'))
        self._trav_obstacle_dilate_iters = int(gp('trav_obstacle_dilate_iters'))
        self._trav_unknown_dilate_iters = int(gp('trav_unknown_dilate_iters'))
        _make_traversability_smoother(self.builder, self._trav_median_filter_size)
        _make_obstacle_unknown_dilator(
            self.builder,
            self._trav_obstacle_dilate_iters,
            self._trav_unknown_dilate_iters,
        )
        self.get_logger().info(
            f'NavigationGraphBuilder ready '
            f'(ext-map fixup + trav-smoother size={self._trav_median_filter_size} '
            f'+ obs_dilate={self._trav_obstacle_dilate_iters} '
            f'unk_dilate={self._trav_unknown_dilate_iters} '
            f'patches installed).'
        )

        # Visited-time layer: increments the closest-to-robot node by 1 each
        # cloud frame, then normalises by total frames on read.  Registered
        # before update() so its scores appear in compute_layers() output.
        self.visited_time_layer: Optional[VisitedTimeLayer] = None
        if bool(gp('enable_visited_time_layer')):
            self.visited_time_layer = VisitedTimeLayer(
                layer_name=str(gp('visited_time_layer')),
            )
            self.builder.add_layer(self.visited_time_layer, weight=0.0)
            self.get_logger().info(
                f'Visited-time layer registered as {self.visited_time_layer.name!r} '
                '(weight=0.0 — ingest-only).'
            )

        # Path-proximity layer: a ComputeLayer that appends the robot xy each
        # compute_layers() pass and scores nodes by closeness to that path.
        self.path_proximity_layer: Optional[PathProximityLayer] = None
        if bool(gp('enable_path_proximity_layer')):
            self.path_proximity_layer = PathProximityLayer(
                layer_name=str(gp('path_proximity_layer')),
                falloff=self._layer_falloff,
            )
            self.builder.add_layer(self.path_proximity_layer, weight=0.0)
            self.get_logger().info(
                f'Path-proximity layer registered as {self.path_proximity_layer.name!r} '
                '(weight=0.0 — GPU compute, viz-only).'
            )

        # ── Rolling elevation map (elevation_mapping_cupy) ────────────
        self.get_logger().info('Initialising ElevationMap (elevation_mapping_cupy)...')
        self.emap = ElevationMapWrapper(
            map_length=length_xy,
            resolution=res,
            sensor_noise_factor=sensor_noise_factor,
            z_clip_threshold=float(gp('elevation_z_clip')),
        )
        self.get_logger().info(
            f'ElevationMap ready (cell_n={self.emap._param.cell_n}, '
            f'usable shape={self.emap.grid_shape()}).'
        )
        self._em_initialised = False

        # ── State ─────────────────────────────────────────────────────
        self.odom_buf: deque[Tuple[float, Odometry]] = deque()
        self.frame_count = 0
        self.last_robot_xy: Optional[Tuple[float, float]] = None
        self.last_robot_yaw: float = 0.0

        # ── Subscribers ───────────────────────────────────────────────
        self.create_subscription(Odometry, odom_topic, self.odom_callback, 100)
        self.create_subscription(PointCloud2, cloud_topic, self.cloud_callback, 5)

        # ── Publishers ────────────────────────────────────────────────
        self.elev_pub = self.create_publisher(PointCloud2, '~/elevation_cloud', 1)
        # 2D heatmap of the local elevation map; view with an RViz Map display.
        self.elev_grid_pub = self.create_publisher(OccupancyGrid, '~/elevation_grid', 1)
        self.graph_pub = self.create_publisher(PointCloud2, '~/graph_nodes', 1)
        self.frontier_pub = self.create_publisher(PointCloud2, '~/frontier_cloud', 1)
        # Raw (pre-clustering) frontier cells straight from FrontierDetector.
        self.raw_frontier_pub = self.create_publisher(PointCloud2, '~/raw_frontiers', 1)
        self.edges_pub = self.create_publisher(Marker, '~/graph_edges', 1)
        # Layer visualisation clouds — intensity in [0,1], colour with RViz intensity colormap.
        #   ~/frontier_score_cloud  — frontier nodes, ExploRFM frontier score.
        #   ~/car_layer             — all nodes, 3D closeness to a detected car.
        #   ~/path_proximity_layer  — all nodes, closeness to the robot path.
        self.frontier_score_pub = self.create_publisher(PointCloud2, '~/frontier_score_cloud', 1)
        self.car_layer_pub = self.create_publisher(PointCloud2, '~/car_layer', 1)
        self.path_prox_pub = self.create_publisher(PointCloud2, '~/path_proximity_layer', 1)
        # Extended global occupancy grid used for edge collision checking inside GlobalGraphGenerator.
        self.global_occ_pub = self.create_publisher(OccupancyGrid, '~/global_occ_grid', 1)

        # ── RGB + graph saver ─────────────────────────────────────────
        self._save_every_n = int(gp('save_every_n_frames'))
        self._save_frame_start = int(gp('save_frame_start'))
        self._save_frame_end   = int(gp('save_frame_end'))
        self._rgb_count = 0
        self._last_result = None

        out_dir_str = str(gp('out_directory')).strip()
        self._out_dir: Optional[Path] = Path(out_dir_str) if out_dir_str else None
        if self._out_dir:
            self._out_dir.mkdir(parents=True, exist_ok=True)
            self.get_logger().info(f'RGB+graph saver enabled → {self._out_dir}')

        cam_fx = float(gp('cam_fx'))
        cam_fy = float(gp('cam_fy'))
        cam_cx = float(gp('cam_cx'))
        cam_cy = float(gp('cam_cy'))
        if cam_fx > 0.0 and cam_fy > 0.0:
            self._cam_K: Optional[np.ndarray] = np.array(
                [[cam_fx, 0.0, cam_cx], [0.0, cam_fy, cam_cy], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            self.get_logger().info(
                f'Camera K from params: fx={cam_fx} fy={cam_fy} cx={cam_cx} cy={cam_cy}'
            )
        else:
            self._cam_K = None

        self._cam_frame: str = str(gp('cam_frame')).strip()

        # Static cam→base_link extrinsic (T_base_from_cam).  Same pattern as the cloud
        # pipeline which uses odometry directly instead of TF.
        t_bc = np.array([gp('cam_base_tx'), gp('cam_base_ty'), gp('cam_base_tz')], dtype=np.float64)
        q_bc = np.array([gp('cam_base_qx'), gp('cam_base_qy'), gp('cam_base_qz'), gp('cam_base_qw')], dtype=np.float64)
        self._T_base_from_cam = _pq_to_se3(t_bc, q_bc)
        self.get_logger().info(f'T_base_from_cam: t={t_bc.tolist()} q={q_bc.tolist()}')

        self._bridge = None  # cv_bridge skipped: compiled against NumPy 1.x, segfaults with 2.x

        # ── ExploRFM model + per-node layers ──────────────────────────
        self._explorfm = None
        self._explorfm_every_n = max(1, int(gp('explorfm_every_n_images')))
        self._trav_layer_name = str(gp('explorfm_trav_layer'))
        self._front_layer_name = str(gp('explorfm_frontier_layer'))
        self._car_layer_name = str(gp('explorfm_car_detector_layer'))
        self._object_threshold = float(gp('explorfm_object_threshold'))
        self._object_std_k = float(gp('explorfm_object_std_k'))
        self._car_abs_sim_threshold = float(gp('car_absolute_sim_threshold'))
        self._car_min_high_pixels = max(0, int(gp('car_min_high_pixels')))
        self._car_max_centroid_node_px = float(gp('car_max_centroid_node_pixel_distance'))
        self._car_distance_threshold = float(gp('car_distance_threshold'))
        # Only one object query now — kept as a list so the SigLIP2 forward
        # path (which expects a list of prompts and returns (Q, D) features)
        # is unchanged.
        self.object_queries = ['car']
        # Cached text embeddings for self.object_queries (set in _init_explorfm).
        self._object_text_emb: Optional[torch.Tensor] = None

        # Car-layer debug image saver — runs on a background thread so the
        # PDF render doesn't block the ROS executor (which would starve the
        # cloud callback and stop /graph_nodes from publishing).  The queue
        # is bounded with a drop-oldest policy so a slow worker can never
        # backlog memory or hold up the producer.
        self._car_debug_every_n = max(1, int(gp('car_debug_save_every_n')))
        car_debug_dir_str = str(gp('car_debug_dir')).strip()
        self._car_debug_dir: Optional[Path] = (
            Path(car_debug_dir_str) if car_debug_dir_str else None
        )
        car_debug_png_str = str(gp('car_debug_png_dir')).strip()
        self._car_debug_png_dir: Optional[Path] = (
            Path(car_debug_png_str) if car_debug_png_str else None
        )
        if self._car_debug_png_dir is not None:
            self._car_debug_png_dir.mkdir(parents=True, exist_ok=True)
        if self._car_debug_dir is not None:
            self._car_debug_dir.mkdir(parents=True, exist_ok=True)
        self._car_debug_queue: Optional[queue.Queue] = None
        self._car_debug_thread: Optional[threading.Thread] = None
        if self._car_debug_dir is not None or self._car_debug_png_dir is not None:
            self._car_debug_queue = queue.Queue(maxsize=8)
            self._car_debug_thread = threading.Thread(
                target=self._car_debug_worker,
                name='CarDebugSaver',
                daemon=True,
            )
            self._car_debug_thread.start()
            self.get_logger().info(
                f'Car-layer debug images → '
                f'PDF={self._car_debug_dir or "off"}, '
                f'PNG={self._car_debug_png_dir or "off"} '
                f'(every {self._car_debug_every_n} image frames, '
                f'async worker, queue max=8)'
            )

        # Frontier-layer debug saver — same async PDF pipeline as the car
        # saver, just feeding a different layer and adding a per-pixel
        # ExploRFM frontier-score heatmap overlay so you can see what the
        # model considers visually important.
        self._frontier_debug_every_n = max(1, int(gp('frontier_debug_save_every_n')))
        self._frontier_debug_high_std_k = float(gp('frontier_debug_high_std_k'))
        self._debug_image_draw_edges = bool(gp('debug_image_draw_edges'))
        frontier_debug_dir_str = str(gp('frontier_debug_dir')).strip()
        self._frontier_debug_dir: Optional[Path] = (
            Path(frontier_debug_dir_str) if frontier_debug_dir_str else None
        )
        frontier_debug_png_str = str(gp('frontier_debug_png_dir')).strip()
        self._frontier_debug_png_dir: Optional[Path] = (
            Path(frontier_debug_png_str) if frontier_debug_png_str else None
        )
        if self._frontier_debug_png_dir is not None:
            self._frontier_debug_png_dir.mkdir(parents=True, exist_ok=True)
        if self._frontier_debug_dir is not None:
            self._frontier_debug_dir.mkdir(parents=True, exist_ok=True)
        self._frontier_debug_queue: Optional[queue.Queue] = None
        self._frontier_debug_thread: Optional[threading.Thread] = None
        if (self._frontier_debug_dir is not None
                or self._frontier_debug_png_dir is not None):
            self._frontier_debug_queue = queue.Queue(maxsize=8)
            self._frontier_debug_thread = threading.Thread(
                target=self._frontier_debug_worker,
                name='FrontierDebugSaver',
                daemon=True,
            )
            self._frontier_debug_thread.start()
            self.get_logger().info(
                f'Frontier-layer debug images → '
                f'PDF={self._frontier_debug_dir or "off"}, '
                f'PNG={self._frontier_debug_png_dir or "off"} '
                f'(every {self._frontier_debug_every_n} image frames, '
                f'async worker, queue max=8)'
            )

        if bool(gp('enable_explorfm_layers')):
            self._init_explorfm(gp)

        # Subscribe to image+info if EITHER the saver OR the model is enabled.
        if self._out_dir or self._explorfm is not None:
            img_topic = str(gp('cam_image_topic'))
            info_topic = str(gp('cam_info_topic'))
            self.create_subscription(CameraInfo, info_topic, self._camera_info_cb, 1)
            self.create_subscription(Image, img_topic, self._image_cb, 10)
            self.get_logger().info(
                f'Image subscriptions: image={img_topic}  camera_info={info_topic} '
                f'(saver={"on" if self._out_dir else "off"}, '
                f'explorfm={"on" if self._explorfm is not None else "off"})'
            )

        H, W = self.emap.grid_shape()
        self.get_logger().info(
            f'Ready | cloud={cloud_topic} odom={odom_topic} '
            f'map={W}x{H}@{res:.3f}m frame={self.frame_id}'
        )

        # ── Graph-comparison artefacts: final-snapshot path + timing CSV ──
        self._save_snapshot_enabled = bool(gp('enable_graph_snapshot_save'))
        self._graph_snapshot_path: Optional[Path] = (
            Path(str(gp('graph_snapshot_path'))).expanduser()
            if self._save_snapshot_enabled and str(gp('graph_snapshot_path')).strip()
            else None
        )
        if self._graph_snapshot_path is not None:
            self._graph_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            self.get_logger().info(
                f'Final-graph snapshot will write to {self._graph_snapshot_path}'
            )
        self._timing_csv_writer = None
        self._timing_csv_file = None
        self._timing_csv_cols = [
            'frame_index', 'frame_timestamp_sec',
            'num_input_points', 'num_valid_cells',
            'num_nodes', 'num_edges', 'num_frontiers',
            't_parse_ms', 't_emap_ms',
            't_local_ms', 't_merge_ms', 't_other_ms', 't_graph_total_ms',
            't_frame_total_ms',
        ]
        if bool(gp('enable_timing_csv')):
            csv_path_str = str(gp('timing_csv_path')).strip()
            if csv_path_str:
                import csv as _csv
                csv_path = Path(csv_path_str).expanduser()
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                # Line-buffered so the file is durable across Ctrl-C / crashes.
                self._timing_csv_file = open(csv_path, 'w', newline='', buffering=1)
                self._timing_csv_writer = _csv.DictWriter(
                    self._timing_csv_file, fieldnames=self._timing_csv_cols,
                )
                self._timing_csv_writer.writeheader()
                self.get_logger().info(f'Per-frame timing CSV → {csv_path}')

    # ──────────────────────────────────────────────────
    #  Callbacks
    # ──────────────────────────────────────────────────

    def odom_callback(self, msg: Odometry) -> None:
        t = stamp_to_sec(msg.header.stamp)
        self.odom_buf.append((t, msg))
        cutoff = t - self.odom_buffer_seconds
        while self.odom_buf and self.odom_buf[0][0] < cutoff:
            self.odom_buf.popleft()

    def _find_pose_at(self, t_target: float) -> Optional[Odometry]:
        if not self.odom_buf:
            return None
        best_dt = float('inf')
        best_msg: Optional[Odometry] = None
        for t, msg in self.odom_buf:
            dt = abs(t - t_target)
            if dt < best_dt:
                best_dt = dt
                best_msg = msg
        if best_msg is None or best_dt > self.odom_match_max_dt:
            return None
        return best_msg

    def cloud_callback(self, msg: PointCloud2) -> None:
        self.frame_count += 1
        if self.max_cloud_frames > 0 and self.frame_count > self.max_cloud_frames:
            return
        if self._out_dir is not None and self._rgb_count > self._save_frame_end:
            return
        if self.process_every_n > 1 and (self.frame_count % self.process_every_n) != 0:
            return

        if msg.header.frame_id != self.robot_frame:
            self.get_logger().warning(
                f'cloud frame_id={msg.header.frame_id!r} != expected '
                f'{self.robot_frame!r}; transform may be wrong.',
                throttle_duration_sec=5.0,
            )

        t_cloud = stamp_to_sec(msg.header.stamp)
        odom = self._find_pose_at(t_cloud)
        if odom is None:
            self.get_logger().warning(
                f'No odometry within {self.odom_match_max_dt}s of cloud t={t_cloud:.3f} '
                f'(buf size {len(self.odom_buf)})',
                throttle_duration_sec=2.0,
            )
            return

        if odom.header.frame_id and odom.header.frame_id != self.frame_id:
            # We assume odom is published in the configured world frame.  Just warn.
            self.get_logger().warning(
                f'odom header frame_id={odom.header.frame_id!r} != '
                f'{self.frame_id!r}; treating odom pose as world-frame anyway.',
                throttle_duration_sec=10.0,
            )

        try:
            self._process(msg, odom)
        except Exception as e:  # pragma: no cover  - defensive
            import traceback

            self.get_logger().error(
                f'cloud_callback failed: {e}\n{traceback.format_exc()}',
                throttle_duration_sec=2.0,
            )

    # ──────────────────────────────────────────────────
    #  Core processing
    # ──────────────────────────────────────────────────

    def _process(self, msg: PointCloud2, odom: Odometry) -> None:
        t_frame_start = time.perf_counter()
        # Pose: world (odom) <- base_link.  Apply to base-frame points.
        p = odom.pose.pose.position
        q = odom.pose.pose.orientation
        rot = quat_to_rot(q.x, q.y, q.z, q.w)
        trans = np.array([p.x, p.y, p.z], dtype=np.float32)

        # Yaw in odom: project the base x-axis onto the world XY plane.
        # Robust to tilt — better than the flat-quat formula when there's
        # significant pitch/roll.
        forward = rot @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
        yaw = math.atan2(float(forward[1]), float(forward[0]))

        # Parse cloud and reject NaNs / OOR points in the sensor frame first
        # (cheap; cloud has is_dense=False so NaNs are present).
        t0 = time.perf_counter()
        xyz_base = parse_xyz_points(msg)
        finite = np.isfinite(xyz_base).all(axis=1)
        xyz_base = xyz_base[finite]

        if self.cloud_max_range > 0 and xyz_base.shape[0] > 0:
            r2 = (xyz_base ** 2).sum(axis=1)
            xyz_base = xyz_base[r2 <= self.cloud_max_range ** 2]

        if xyz_base.shape[0] == 0:
            self.get_logger().warning('Cloud has zero finite in-range points',
                                   throttle_duration_sec=2.0)
            return
        t_parse = (time.perf_counter() - t0) * 1000.0

        # ── Update elevation map ─────────────────────────────────────
        # 1) Recenter the map to the robot's xy in odom (z fixed at 0 so the
        #    map's reference plane stays horizontal in world coords).
        # 2) Integrate the cloud — pass body-frame points + the
        #    base->odom transform from odometry.  elevation_mapping_cupy
        #    transforms the points internally.
        t0 = time.perf_counter()
        self.emap.move_to(np.array([trans[0], trans[1], 0.0], dtype=np.float32))
        # Reference height for the z-clip filter — the robot's current odom z.
        self.emap.set_reference_z(float(trans[2]))
        self.emap.integrate(
            xyz_base,
            t_sensor_in_odom=trans,
            R_sensor_to_odom=rot,
            position_noise=self.em_position_noise,
            orientation_noise=self.em_orientation_noise,
        )
        self.emap.tick()
        t_emap = (time.perf_counter() - t0) * 1000.0

        self.last_robot_xy = (float(trans[0]), float(trans[1]))
        self.last_robot_yaw = yaw

        # ── Build / update graph ─────────────────────────────────────
        # The wrapper applies the rot180 fix-up so the grid handed to
        # nav_graph already matches its expected (NE-at-[0,0]) convention.
        t0 = time.perf_counter()
        elev_grid = self.emap.get_elevation_for_navgraph()
        cx, cy = self.emap.center_xy()
        result = self.builder.update(
            elev_grid,
            resolution=self.emap.resolution,
            origin_x=cx,
            origin_y=cy,
            robot_xy=self.last_robot_xy,
            robot_yaw=self.last_robot_yaw,
            detect_frontiers=True,
            input_type='elevation_map',
            compute_layers=False,
        )

        # Visited-time: tick the frame counter, then add +1 to the node closest
        # to the robot's current xy.  Done after update() so the closest-node
        # query sees the freshly-merged graph from this frame.
        if self.visited_time_layer is not None:
            self.visited_time_layer.tick()
            if result.num_nodes > 0:
                robot_pos = torch.tensor(
                    [[self.last_robot_xy[0], self.last_robot_xy[1], 0.0]],
                    dtype=torch.float32,
                )
                closest_ids = self.builder.global_builder.return_closest_node_ids(robot_pos)
                if closest_ids.numel() > 0 and (closest_ids >= 0).any():
                    valid = closest_ids[closest_ids >= 0]
                    self.builder.ingest_layer_scores(
                        self.visited_time_layer.name,
                        node_ids=valid,
                        values=torch.ones_like(valid, dtype=torch.float32),
                    )

        # Compute layer scores after ingest so visited_time reflects this frame.
        node_scores, score_layer_names = self.builder.compute_layers()
        result.node_scores = node_scores
        result.score_layer_names = score_layer_names

        t_graph = (time.perf_counter() - t0) * 1000.0
        self._last_result = result

        n_frontiers = (
            int((result.node_types == 2).sum().item()) if result.num_nodes > 0 else 0
        )
        n_valid_cells = int(np.isfinite(elev_grid).sum())
        self.get_logger().info(
            f'frame {self.frame_count} pts_in={xyz_base.shape[0]} '
            f'valid_cells={n_valid_cells} parse={t_parse:.1f}ms '
            f'emap={t_emap:.1f}ms graph={t_graph:.1f}ms '
            f'nodes={result.num_nodes} frontiers={n_frontiers} edges={result.num_edges}'
        )

        # ── Per-frame timing CSV row.  Sub-step timings come from the
        # builder-internal patches installed in __init__.  ``t_other_ms`` is
        # whatever's left of t_graph after the two main steps — frontier
        # detection, clustering, marking, layer compute, result extract.
        if self._timing_csv_writer is not None:
            t_local_ms = float(self._builder_timings.get('local_ms', float('nan')))
            t_merge_ms = float(self._builder_timings.get('merge_ms', float('nan')))
            t_other_ms = float('nan')
            if math.isfinite(t_local_ms) and math.isfinite(t_merge_ms):
                t_other_ms = max(0.0, t_graph - t_local_ms - t_merge_ms)
            t_frame_total_ms = (time.perf_counter() - t_frame_start) * 1000.0
            row = {
                'frame_index':         int(self.frame_count),
                'frame_timestamp_sec': stamp_to_sec(msg.header.stamp),
                'num_input_points':    int(xyz_base.shape[0]),
                'num_valid_cells':     n_valid_cells,
                'num_nodes':           int(result.num_nodes),
                'num_edges':           int(result.num_edges),
                'num_frontiers':       n_frontiers,
                't_parse_ms':          float(t_parse),
                't_emap_ms':           float(t_emap),
                't_local_ms':          t_local_ms,
                't_merge_ms':          t_merge_ms,
                't_other_ms':          t_other_ms,
                't_graph_total_ms':    float(t_graph),
                't_frame_total_ms':    float(t_frame_total_ms),
            }
            try:
                self._timing_csv_writer.writerow(row)
            except Exception as exc:
                self.get_logger().error(
                    f'timing-csv writerow failed: {exc}', throttle_duration_sec=5.0,
                )

        # Safety throttle: if the graph runs away (typically because of
        # a frame/convention bug), reset the global graph so we don't
        # crash RViz with multi-million-point clouds.
        if (self.max_graph_nodes_before_reset > 0
                and result.num_nodes > self.max_graph_nodes_before_reset):
            self.get_logger().error(
                f'Graph node count {result.num_nodes} exceeded '
                f'max_graph_nodes_before_reset={self.max_graph_nodes_before_reset}; '
                'resetting builder. Investigate convention/merge issues before re-running.'
            )
            self.builder.reset()
            _install_builder_patches(self.builder)
            self._builder_timings = _install_builder_timing_patches(self.builder)
            _make_traversability_smoother(self.builder, self._trav_median_filter_size)
            _make_obstacle_unknown_dilator(
                self.builder,
                self._trav_obstacle_dilate_iters,
                self._trav_unknown_dilate_iters,
            )
            return

        stamp = msg.header.stamp
        if self.publish_elevation_cloud:
            self._publish_elevation_cloud(stamp)
        if self.publish_elevation_grid:
            self._publish_elevation_grid(stamp)
        self._publish_graph_nodes(result, stamp)
        self._publish_frontier_cloud(result, stamp)
        self._publish_raw_frontiers(result, stamp)
        if self.publish_edges_flag:
            self._publish_edges(result, stamp)
        self._publish_score_clouds(result, stamp)
        self._publish_global_occ_grid(stamp)

    # ──────────────────────────────────────────────────
    #  Publishing helpers
    # ──────────────────────────────────────────────────

    def _make_xyz_intensity_cloud(
        self,
        points: np.ndarray,
        intensities: np.ndarray,
        stamp,
    ) -> PointCloud2:
        n = int(points.shape[0])
        packed = np.empty((n, 4), dtype=np.float32)
        packed[:, :3] = points
        packed[:, 3] = intensities
        cloud = PointCloud2()
        cloud.header = Header(stamp=stamp, frame_id=self.frame_id)
        cloud.height = 1
        cloud.width = n
        cloud.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        cloud.is_bigendian = False
        cloud.point_step = 16
        cloud.row_step = 16 * n
        cloud.data = packed.tobytes()
        cloud.is_dense = True
        return cloud

    def _publish_elevation_cloud(self, stamp) -> None:
        # Native elevation_mapping_cupy layout: rows index X, cols index Y.
        #   elev[r, c] -> world (cx + (r - H/2)*res, cy + (c - W/2)*res)
        elev = self.emap.get_elevation_emcupy()
        valid = np.isfinite(elev)
        if not valid.any():
            return
        H, W = elev.shape
        cx, cy = self.emap.center_xy()
        res = self.emap.resolution
        rs, cs = np.where(valid)
        xs = cx + (rs - H / 2.0) * res
        ys = cy + (cs - W / 2.0) * res
        zs = elev[rs, cs].astype(np.float32)
        pts = np.stack([xs.astype(np.float32), ys.astype(np.float32), zs], axis=1)
        self.elev_pub.publish(self._make_xyz_intensity_cloud(pts, zs, stamp))

    def _publish_elevation_grid(self, stamp) -> None:
        """Publish the local elevation map as a 2D OccupancyGrid heatmap.

        Heights are linearly quantised: ``elevation_grid_z_min`` -> 0,
        ``elevation_grid_z_max`` -> 100 (clamped); unknown cells -> -1.
        View in RViz with a Map display — Color Scheme 'costmap' renders a
        blue->red heatmap, so cells on a roof show up as red outliers.
        """
        elev = self.emap.get_elevation_emcupy()  # elev[r, c]: r->x, c->y
        valid = np.isfinite(elev)
        if not valid.any():
            return
        H, W = elev.shape
        cx, cy = self.emap.center_xy()
        res = self.emap.resolution

        z_min = self.elevation_grid_z_min
        z_max = self.elevation_grid_z_max
        span = max(z_max - z_min, 1e-6)
        norm = (elev - z_min) / span * 100.0
        data = np.full((H, W), -1, dtype=np.int8)
        data[valid] = np.clip(norm[valid], 0.0, 100.0).astype(np.int8)

        # OccupancyGrid layout: data[row*width + col], col along +x, row
        # along +y.  elev rows index x and cols index y, so width=H (x),
        # height=W (y), and the ROS grid is elev transposed.
        ros_data = np.ascontiguousarray(data.T)

        msg = OccupancyGrid()
        msg.header = Header(stamp=stamp, frame_id=self.frame_id)
        msg.info = MapMetaData()
        msg.info.resolution = float(res)
        msg.info.width = int(H)
        msg.info.height = int(W)
        msg.info.origin = Pose()
        msg.info.origin.position.x = float(cx - H * res / 2.0)
        msg.info.origin.position.y = float(cy - W * res / 2.0)
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        msg.data = ros_data.flatten().tolist()
        self.elev_grid_pub.publish(msg)

        # Log raw height stats so the actual elevation values are visible
        # in the console and the colour range can be tuned accordingly.
        vz = elev[valid]
        self.get_logger().info(
            f'elevation_grid: valid={vz.size} '
            f'z[min/med/max]={vz.min():.2f}/{np.median(vz):.2f}/{vz.max():.2f} '
            f'colour_range=[{z_min:.2f},{z_max:.2f}]',
            throttle_duration_sec=2.0,
        )

    def _publish_graph_nodes(self, result, stamp) -> None:
        if result.num_nodes == 0:
            return
        # Only free-space nodes go on /graph_nodes — frontier nodes are
        # published separately to /frontier_cloud so they can be styled
        # differently in RViz without overlap.
        free_mask = result.node_types == 1
        if not bool(free_mask.any().item()):
            return
        positions = result.node_positions[free_mask].cpu().numpy().astype(np.float32, copy=True)
        positions[:, 2] += self.viz_z_offset
        types = result.node_types[free_mask].cpu().numpy().astype(np.float32)
        self.graph_pub.publish(self._make_xyz_intensity_cloud(positions, types, stamp))

    def _publish_frontier_cloud(self, result, stamp) -> None:
        if result.num_nodes == 0:
            return
        mask = result.node_types == 2
        if not bool(mask.any().item()):
            return
        positions = result.node_positions[mask].cpu().numpy().astype(np.float32, copy=True)
        positions[:, 2] += self.viz_z_offset
        # Flat colour — constant intensity so RViz renders one solid colour.
        flat = np.ones(positions.shape[0], dtype=np.float32)
        self.frontier_pub.publish(self._make_xyz_intensity_cloud(positions, flat, stamp))

    def _publish_raw_frontiers(self, result, stamp) -> None:
        """Publish raw (pre-clustering) frontier cells from FrontierDetector.

        ``result.frontiers`` is an (F, 2) CPU tensor of world [x, y] cell
        centres.  The detector emits no Z, so cells are placed at the
        ``viz_z_offset`` height to sit clearly above the elevation cloud.
        """
        raw = result.frontiers
        if raw is None or raw.shape[0] == 0:
            return
        xy = raw.detach().cpu().numpy().astype(np.float32)
        n = xy.shape[0]
        pts = np.empty((n, 3), dtype=np.float32)
        pts[:, :2] = xy
        pts[:, 2] = self.viz_z_offset
        flat = np.ones(n, dtype=np.float32)
        self.raw_frontier_pub.publish(self._make_xyz_intensity_cloud(pts, flat, stamp))

    def _publish_edges(self, result, stamp) -> None:
        if result.num_nodes == 0 or result.num_edges == 0:
            return
        positions = result.node_positions.cpu().numpy().astype(np.float32, copy=True)
        positions[:, 2] += self.viz_z_offset
        ids = result.node_ids.cpu().numpy()
        edge_index = result.edge_index.cpu().numpy()  # (E, 2) of node IDs

        # Build id -> row index lookup once per frame (vectorized).
        max_id = int(ids.max()) + 1 if ids.size > 0 else 0
        id_to_idx = -np.ones(max_id, dtype=np.int64)
        id_to_idx[ids.astype(np.int64)] = np.arange(ids.shape[0], dtype=np.int64)

        if edge_index.shape[0] > self.max_edges_published:
            stride = max(1, edge_index.shape[0] // self.max_edges_published)
            edge_index = edge_index[::stride]

        src_idx = id_to_idx[edge_index[:, 0].astype(np.int64)]
        tgt_idx = id_to_idx[edge_index[:, 1].astype(np.int64)]
        keep = (src_idx >= 0) & (tgt_idx >= 0)
        src_idx = src_idx[keep]
        tgt_idx = tgt_idx[keep]
        if src_idx.size == 0:
            return

        m = Marker()
        m.header = Header(stamp=stamp, frame_id=self.frame_id)
        m.ns = 'nav_graph_edges'
        m.id = 0
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD
        m.scale.x = 0.04
        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 1.0
        m.color.a = 1.0
        m.pose.orientation.w = 1.0
        for s, t in zip(src_idx.tolist(), tgt_idx.tolist()):
            ps = positions[s]
            pe = positions[t]
            m.points.append(Point(x=float(ps[0]), y=float(ps[1]), z=float(ps[2])))
            m.points.append(Point(x=float(pe[0]), y=float(pe[1]), z=float(pe[2])))
        self.edges_pub.publish(m)

    def _publish_score_clouds(self, result, stamp) -> None:
        """Publish the per-layer visualisation clouds.

        ~/frontier_score_cloud  — frontier nodes, ExploRFM frontier score.
        ~/car_layer             — all nodes, 3D closeness to the per-frame car centroid.
        ~/path_proximity_layer  — all nodes, closeness to the robot path.

        Car/path scores are produced upstream (ExploRFM ingest and the
        path-proximity ComputeLayer) — here we just read the columns out of
        ``result.node_scores`` and pack them into clouds.
        """
        if result.num_nodes == 0:
            return

        pos = result.node_positions                       # (N, 3) CUDA
        # One device→host transfer for the shared xyz buffer.
        pos_np = pos.detach().cpu().numpy().astype(np.float32, copy=True)
        pos_np[:, 2] += self.viz_z_offset

        names = result.score_layer_names or []
        scores = result.node_scores

        # ── scored frontiers — ExploRFM frontier layer, frontier nodes only ──
        front_mask = result.node_types == 2
        if (scores is not None and self._front_layer_name in names
                and bool(front_mask.any())):
            col = names.index(self._front_layer_name)
            fm_np = front_mask.detach().cpu().numpy()
            fscore = scores[front_mask, col].detach().cpu().float().numpy()
            self.frontier_score_pub.publish(
                self._make_xyz_intensity_cloud(pos_np[fm_np], fscore, stamp),
            )

        # ── car layer — read the ingested score column directly ─────────────
        if scores is not None and self._car_layer_name in names:
            col = names.index(self._car_layer_name)
            vals = scores[:, col].detach().cpu().float().numpy()
            self.car_layer_pub.publish(
                self._make_xyz_intensity_cloud(pos_np, vals, stamp),
            )

        # ── path-proximity layer — read straight from the registered layer ──
        if (self.path_proximity_layer is not None and scores is not None
                and self.path_proximity_layer.name in names):
            col = names.index(self.path_proximity_layer.name)
            path_score = scores[:, col].detach().cpu().float().numpy()
            self.path_prox_pub.publish(
                self._make_xyz_intensity_cloud(pos_np, path_score, stamp),
            )

    def _publish_global_occ_grid(self, stamp) -> None:
        """Publish the extended sliding-window occupancy grid from GlobalGraphGenerator.

        This is the exact grid used for edge collision checking.  It is larger
        than the local elevation map (ext_scale_factor × local size) and
        accumulates observations across frames as the robot moves.

        Values follow the ROS convention: 0=free, 100=occupied, -1=unknown.
        Frame: same as all other published topics (odom).

        Internal _ext_map convention:
            col 0 = east, col W-1 = west,  row 0 = south, row H-1 = north.
        After flipping columns the origin sits at the south-west corner, which
        is the standard nav_msgs/OccupancyGrid layout.
        """
        gg = self.builder.global_builder
        if gg._ext_map is None:
            return

        ext_H, ext_W = gg._ext_shape
        cx, cy = gg._ext_center
        res = gg._ext_resolution

        # Move to CPU once; avoid multiple round-trips.
        ext_np = gg._ext_map.cpu().numpy()  # uint8: 0=blocked, 128=unobserved, 255=free

        # Convert to ROS int8: 0=free, 100=occupied, -1=unknown.
        ros_data = np.full((ext_H, ext_W), -1, dtype=np.int8)
        ros_data[ext_np == 255] = 0
        ros_data[ext_np == 0] = 100
        # 128 (unobserved) stays -1.

        # Flip columns: internal col 0 = east → ROS col 0 = west (standard).
        # Row order is unchanged: row 0 = south in both conventions.
        ros_data = np.ascontiguousarray(ros_data[:, ::-1])

        msg = OccupancyGrid()
        msg.header = Header(stamp=stamp, frame_id=self.frame_id)
        msg.info = MapMetaData()
        msg.info.resolution = float(res)
        msg.info.width = int(ext_W)
        msg.info.height = int(ext_H)
        msg.info.origin = Pose()
        # South-west corner (bottom-left) after column flip.
        msg.info.origin.position.x = float(cx - ext_W * res / 2.0)
        msg.info.origin.position.y = float(cy - ext_H * res / 2.0)
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        msg.data = ros_data.flatten().tolist()
        self.global_occ_pub.publish(msg)

    # ──────────────────────────────────────────────────
    #  RGB + graph saver
    # ──────────────────────────────────────────────────

    def _init_explorfm(self, gp) -> None:
        """Build the ExploRFM model and register its two output layers.

        Layers are :class:`ExternalLayer` (no compute logic) using the default
        REPLACE policy — each new image overwrites prior values for any nodes
        it observes.  Untouched nodes keep their last ingested score.
        """
        _ensure_explorfm_on_path()
        try:
            from explorfm import ExploRFMInference  # noqa: WPS433  (runtime import)
        except Exception as exc:  # pragma: no cover  - missing submodule
            self.get_logger().error(
                f'enable_explorfm_layers=True but explorfm import failed: {exc}; '
                'check that nebula2-wildos is on disk and dependencies installed.'
            )
            return

        frontier_ckpt = str(gp('explorfm_frontier_ckpt')).strip()
        trav_ckpt = str(gp('explorfm_trav_ckpt')).strip()
        if not frontier_ckpt or not trav_ckpt:
            self.get_logger().error(
                'explorfm_frontier_ckpt and explorfm_trav_ckpt must both be set.'
            )
            return

        adaptor_version = str(gp('explorfm_adaptor_version')).strip() or None
        adaptor_ckpt_path = str(gp('explorfm_adaptor_ckpt_path')).strip() or None

        self.get_logger().info(
            f'Loading ExploRFM (radio={gp("explorfm_radio_version")}, '
            f'precision={gp("explorfm_precision")}, '
            f'scale={gp("explorfm_static_scale_factor")}, '
            f'adaptor={adaptor_version})...'
        )
        try:
            self._explorfm = ExploRFMInference(
                frontier_ckpt=frontier_ckpt,
                traversability_ckpt=trav_ckpt,
                model_version=str(gp('explorfm_radio_version')),
                adaptor_version=adaptor_version,
                adaptor_ckpt_path=adaptor_ckpt_path,
                use_naclip=bool(gp('explorfm_use_naclip')),
                radio_dim=int(gp('explorfm_radio_dim')),
                static_scale_factor=float(gp('explorfm_static_scale_factor')),
                model_precision=str(gp('explorfm_precision')),
            )
        except Exception as exc:  # pragma: no cover
            import traceback
            self.get_logger().error(
                f'ExploRFM init failed: {exc}\n{traceback.format_exc()}'
            )
            self._explorfm = None
            return

        # Register two persistent layers (REPLACE on each ingest) so every
        # cloud-frame compute_layers() pass sees the most recent observation.
        # Traversability layer disabled.
        # self.builder.add_layer(
        #     ExternalLayer(self._trav_layer_name, ingest_policy=MergePolicy.REPLACE),
        #     weight=0.0,
        # )
        self.builder.add_layer(
            ExternalLayer(self._front_layer_name, ingest_policy=MergePolicy.REPLACE),
            weight=0.0,
        )
        # Car layer uses a CUSTOM max merge so colours stick — once a node
        # has been painted by a nearby car, it keeps that score (or gets
        # brighter if the robot later passes even closer).  Combined with
        # the subset-only ingest in `_infer_and_ingest` (we only ingest IDs
        # whose new score > 0), distant / no-detection frames leave the
        # existing colours alone instead of resetting them.
        self.builder.add_layer(
            ExternalLayer(
                self._car_layer_name,
                ingest_policy=MergePolicy.CUSTOM,
                custom_ingest_fn=lambda old, new: torch.maximum(old, new),
            ),
            weight=0.0,
        )
        self.get_logger().info(
            f'ExploRFM ready. Layers registered: '
            f'{self._trav_layer_name!r}, {self._front_layer_name!r}, {self._car_layer_name} '
            f'(weight=0.0 — ingest-only; raise via set_layer_weight to feed combined score).'
        )

        # Pre-compute SigLIP2 text embeddings for the object queries so we don't
        # re-encode them on every image.  Shape: (Q, D), L2-normalized along D.
        try:
            with torch.inference_mode():
                text_emb = self._explorfm.forward_on_text(self.object_queries)
            self._object_text_emb = F.normalize(text_emb.float(), dim=-1).contiguous()
            self.get_logger().info(
                f'Object text embeddings cached for queries={self.object_queries} '
                f'(shape={tuple(self._object_text_emb.shape)}, '
                f'threshold={self._object_threshold:.3f}).'
            )
        except Exception as exc:
            self.get_logger().error(
                f'Failed to encode object queries {self.object_queries}: {exc}; '
                'car layer will stay empty.'
            )
            self._object_text_emb = None

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        if self._cam_K is not None:
            return
        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        if K[0, 0] <= 0.0:
            return
        self._cam_K = K
        if not self._cam_frame:
            self._cam_frame = msg.header.frame_id
        self.get_logger().info(
            f'Camera intrinsics from {msg.header.frame_id}: '
            f'fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}'
        )

    def _image_cb(self, msg: Image) -> None:
        self._rgb_count += 1

        do_infer = (
            self._explorfm is not None
            and (self._rgb_count % self._explorfm_every_n == 0)
        )
        in_save_window = (
            self._out_dir is not None
            and self._save_frame_start <= self._rgb_count <= self._save_frame_end
            and (self._rgb_count - self._save_frame_start) % self._save_every_n == 0
        )
        do_car_debug = (
            (self._car_debug_dir is not None or self._car_debug_png_dir is not None)
            and self._explorfm is not None
            and (self._rgb_count % self._car_debug_every_n == 0)
        )
        do_frontier_debug = (
            (self._frontier_debug_dir is not None
             or self._frontier_debug_png_dir is not None)
            and self._explorfm is not None
            and (self._rgb_count % self._frontier_debug_every_n == 0)
        )
        if not (do_infer or in_save_window or do_car_debug or do_frontier_debug):
            return

        if not self._cam_frame:
            self._cam_frame = msg.header.frame_id
        if self._cam_K is None:
            self.get_logger().warning(
                'No camera intrinsics — set cam_fx/fy/cx/cy params or publish camera_info.',
                throttle_duration_sec=10.0,
            )
            return
        if self._last_result is None or self._last_result.num_nodes == 0:
            self.get_logger().info(
                'Skipping image: no graph nodes yet.', throttle_duration_sec=5.0,
            )
            return

        rgb = self._decode_image(msg)
        if rgb is None:
            return

        t_sec = stamp_to_sec(msg.header.stamp)
        proj = self._project_visible_nodes(rgb.shape[:2], t_sec)
        if proj is None:
            return

        # Per-node values produced by the model (same length/order as proj['visible_nodes']).
        trav_per_node: Optional[np.ndarray] = None
        front_per_node: Optional[np.ndarray] = None
        car_debug: Optional[dict] = None
        frontier_debug: Optional[dict] = None
        if do_infer:
            try:
                trav_per_node, front_per_node, car_debug, frontier_debug = (
                    self._infer_and_ingest(rgb, proj)
                )
            except Exception as exc:  # pragma: no cover  - defensive
                import traceback
                self.get_logger().error(
                    f'ExploRFM inference failed: {exc}\n{traceback.format_exc()}',
                    throttle_duration_sec=10.0,
                )

        if in_save_window:
            try:
                self._save_frame_files(
                    msg, rgb, proj, t_sec, trav_per_node, front_per_node,
                )
            except Exception as exc:
                import traceback
                self.get_logger().error(
                    f'_save_frame_files failed: {exc}\n{traceback.format_exc()}'
                )

        if do_car_debug:
            try:
                self._enqueue_car_debug_save(rgb, proj, car_debug, self._rgb_count)
            except Exception as exc:
                import traceback
                self.get_logger().error(
                    f'enqueue car-debug save failed: {exc}\n{traceback.format_exc()}'
                )

        if do_frontier_debug:
            try:
                self._enqueue_frontier_debug_save(
                    rgb, proj, frontier_debug, self._rgb_count,
                )
            except Exception as exc:
                import traceback
                self.get_logger().error(
                    f'enqueue frontier-debug save failed: {exc}\n{traceback.format_exc()}'
                )

    def _decode_image(self, msg: Image) -> Optional[np.ndarray]:
        """ROS Image → HxWx3 uint8 RGB. Returns None on decode failure."""
        try:
            arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            enc = msg.encoding.lower()
            ch = 4 if enc in ('bgra8', 'rgba8') else 3
            arr = arr.reshape(msg.height, msg.width, ch)
            if enc == 'bgr8':
                return arr[:, :, ::-1].copy()
            if enc == 'bgra8':
                return arr[:, :, [2, 1, 0]].copy()
            if enc == 'rgba8':
                return arr[:, :, :3].copy()
            return arr.copy()  # rgb8 / mono passthrough
        except Exception as exc:
            self.get_logger().error(
                f'Image decode failed (encoding={msg.encoding}): {exc}'
            )
            return None

    def _project_visible_nodes(
        self,
        img_shape_hw: Tuple[int, int],
        t_sec: float,
    ) -> Optional[dict]:
        """Project ``_last_result`` global nodes into the camera image.

        Returns a dict bundling everything downstream consumers need:
        the per-node arrays for inference + ingest, and the list-of-dicts
        form for SVG/JSON.  Returns None if no matching odometry exists.
        """
        odom_msg = self._find_pose_at(t_sec)
        if odom_msg is None:
            self.get_logger().warning(
                f'No odometry near image t={t_sec:.3f} (buf={len(self.odom_buf)}); skipping.',
                throttle_duration_sec=5.0,
            )
            return None

        p_o = odom_msg.pose.pose.position
        q_o = odom_msg.pose.pose.orientation
        T_odom_from_base = _pq_to_se3(
            np.array([p_o.x, p_o.y, p_o.z], dtype=np.float64),
            np.array([q_o.x, q_o.y, q_o.z, q_o.w], dtype=np.float64),
        )
        T_odom_from_cam = T_odom_from_base @ self._T_base_from_cam
        T_opt_from_odom = np.linalg.inv(T_odom_from_cam)

        result = self._last_result
        pos_odom = result.node_positions.cpu().numpy().astype(np.float64)  # (N, 3)
        node_types = result.node_types.cpu().numpy()
        node_ids = result.node_ids.cpu().numpy()

        # Optical frame: x-right, y-down, z-forward (OpenCV convention).
        R_opt = T_opt_from_odom[:3, :3]
        t_opt_vec = T_opt_from_odom[:3, 3]
        pos_opt = (R_opt @ pos_odom.T).T + t_opt_vec
        # JSON-friendly cam frame: x-fwd, y-down, z-left.
        _R_opt_to_cam = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64)
        pos_cam = (_R_opt_to_cam @ pos_opt.T).T

        front = pos_opt[:, 2] > 0.1
        pos_opt_f = pos_opt[front]
        pos_cam_f = pos_cam[front]
        pos_odom_f = pos_odom[front]
        types_f = node_types[front]
        ids_f = node_ids[front]

        img_h, img_w = img_shape_hw
        K = self._cam_K
        u_in = np.empty(0, dtype=np.int64)
        v_in = np.empty(0, dtype=np.int64)
        pos_cam_in = np.empty((0, 3), dtype=np.float64)
        pos_odom_in = np.empty((0, 3), dtype=np.float64)
        types_in = np.empty(0, dtype=node_types.dtype)
        ids_in = np.empty(0, dtype=node_ids.dtype)

        if len(pos_opt_f) > 0:
            z_vals = pos_opt_f[:, 2]
            u_all = K[0, 0] * pos_opt_f[:, 0] / z_vals + K[0, 2]
            v_all = K[1, 1] * pos_opt_f[:, 1] / z_vals + K[1, 2]
            inside = (u_all >= 0) & (u_all < img_w) & (v_all >= 0) & (v_all < img_h)
            u_in = u_all[inside].astype(np.int64)
            v_in = v_all[inside].astype(np.int64)
            pos_cam_in = pos_cam_f[inside]
            pos_odom_in = pos_odom_f[inside]
            types_in = types_f[inside]
            ids_in = ids_f[inside]

        _TYPE_STR = {1: 'free_space', 2: 'frontier'}
        visible_nodes = [
            {
                'id':           int(ids_in[i]),
                'type':         _TYPE_STR.get(int(types_in[i]), 'unknown'),
                'position_cam': pos_cam_in[i].tolist(),
                'pixel':        [int(u_in[i]), int(v_in[i])],
            }
            for i in range(len(ids_in))
        ]

        visible_id_set = {n['id'] for n in visible_nodes}
        visible_edges: list = []
        # Parallel (E, 4) int32 array of [u0, v0, u1, v1] endpoint pixels —
        # cheaper to push through worker queues than a list of dicts and
        # ready for direct cv2.line drawing in the debug renderers.
        edge_uvs_list: list = []
        if result.num_edges > 0:
            id_to_uv = {
                int(ids_in[i]): (int(u_in[i]), int(v_in[i]))
                for i in range(len(ids_in))
            }
            edge_arr = result.edge_index.cpu().numpy()
            for e in edge_arr:
                id0, id1 = int(e[0]), int(e[1])
                if id0 in visible_id_set and id1 in visible_id_set:
                    visible_edges.append({'node_id_0': id0, 'node_id_1': id1})
                    u0, v0 = id_to_uv[id0]
                    u1, v1 = id_to_uv[id1]
                    edge_uvs_list.append((u0, v0, u1, v1))
        edge_uvs = (
            np.array(edge_uvs_list, dtype=np.int32)
            if edge_uvs_list else np.empty((0, 4), dtype=np.int32)
        )

        return {
            'visible_nodes':    visible_nodes,
            'visible_edges':    visible_edges,
            'visible_edge_uv':  edge_uvs,
            'visible_ids':      ids_in.astype(np.int64),
            'visible_types':    types_in.astype(np.int64),
            'visible_u':        u_in,
            'visible_v':        v_in,
            'visible_pos_odom': pos_odom_in.astype(np.float64),
            'T_opt_from_odom':  T_opt_from_odom,
        }

    def _infer_and_ingest(
        self,
        rgb: np.ndarray,
        proj: dict,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], dict, dict]:
        """Run ExploRFM, sample at projected pixels, ingest into layers.

        Returns ``(trav_per_node, front_per_node, car_debug, frontier_debug)``:

          * ``trav_per_node``    — aligned with ``proj['visible_nodes']``.
          * ``front_per_node``   — same, NaN for non-frontier rows.
          * ``car_debug``        — info the car-debug saver needs (high-sim
            mask, centroid pixel, picked-node pixel, 3D centroid position).
          * ``frontier_debug``   — info the frontier-debug saver needs: the
            per-pixel frontier-score heatmap (``front_map_np``) plus its
            min/max.
        """
        ids_np = proj['visible_ids']
        device = self.builder.global_builder._global_pos.device

        t0 = time.perf_counter()
        trav_t, front_t, ad_feats_t = self._explorfm.forward_on_numpy(rgb)
        # trav_t and front_t are (1, 1, H, W) on cuda, sized to the input rgb. ad_feats_t is (1, D, hp, wp)
        trav_np = trav_t[0, 0].detach().float().cpu().numpy()
        front_np = front_t[0, 0].detach().float().cpu().numpy()
        infer_ms = (time.perf_counter() - t0) * 1000.0

        # Frontier debug payload is just the per-pixel score map — the
        # saver normalises it itself.  Kept as float32 to save bandwidth
        # across the worker queue.
        frontier_debug: dict = {
            'front_map_np': front_np.astype(np.float32, copy=False),
            'front_min':    float(front_np.min()) if front_np.size else 0.0,
            'front_max':    float(front_np.max()) if front_np.size else 1.0,
        }

        h, w = trav_np.shape

        # ── Traversability + frontier-score sampling (needs visible nodes) ──
        trav_vals: Optional[np.ndarray] = None
        front_vals_full: Optional[np.ndarray] = None
        n_front = 0
        if ids_np.size > 0:
            u = np.clip(proj['visible_u'], 0, w - 1)
            v = np.clip(proj['visible_v'], 0, h - 1)
            trav_vals = trav_np[v, u].astype(np.float32)

            types_np = proj['visible_types']
            front_mask = types_np == 2
            front_vals_full = np.full(ids_np.shape, np.nan, dtype=np.float32)
            if front_mask.any():
                front_vals_full[front_mask] = front_np[
                    v[front_mask], u[front_mask],
                ].astype(np.float32)

            n_front = int(front_mask.sum())
            if n_front > 0:
                front_ids_t = torch.from_numpy(ids_np[front_mask]).to(
                    device=device, dtype=torch.long,
                )
                front_vals_t = torch.from_numpy(front_vals_full[front_mask]).to(
                    device=device, dtype=torch.float32,
                )
                self.builder.ingest_layer_scores(
                    self._front_layer_name, front_ids_t, front_vals_t,
                )

        # ── Car layer: 3D-distance gating from the car centroid pixel ──────
        # Pipeline (per frame):
        #   1. SigLIP2 text-patch cosine similarity → per-pixel car heatmap.
        #   2. Adaptive threshold (mean + k·std, clamped above an optional
        #      absolute floor) → binary high-sim mask.
        #   3. Pixel centroid of the high-sim mask = the car's image-space
        #      centre.  Find the visible projected graph node closest to
        #      that pixel and take its 3D odom position as the car's 3D
        #      centroid (cheap depth substitute — no extra ray-casting).
        #   4. For each graph node within `car_distance_threshold` of the
        #      3D centroid, score = 1 - d / threshold (always > 0).  Ingest
        #      only that subset.  The car layer's CUSTOM merge takes the
        #      elementwise max with the previously-stored score, so painted
        #      nodes keep their brightest-ever colour — driving past, then
        #      away, doesn't erase them.  Nodes never seen near a car stay
        #      at the layer's default 0.
        # If no car pixels survive thresholding, no node is close enough in
        # image space to anchor the centroid, or no node is within the 3D
        # range, the frame is a no-op and previous colours persist.
        car_debug: dict = {
            'high_mask':       None,   # (H, W) bool np
            'centroid_uv':     None,   # (cu, cv) float tuple or None
            'nearest_node_uv': None,   # (u, v) float tuple or None
            'car_xyz_odom':   None,    # (3,) np.float64 or None
            'n_high_pixels':   0,
            'sim_max':         float('nan'),
            'thr':             float('nan'),
        }
        obj_hits_log = ''
        if self._object_text_emb is not None:
            patch = F.normalize(ad_feats_t.float(), dim=1)                       # (1, D, h, w)
            sim = torch.einsum('nd,bdhw->bnhw', self._object_text_emb, patch)    # (1, 1, h, w)
            sim_full = F.interpolate(
                sim, size=rgb.shape[:2], mode='bilinear', align_corners=False,
            )                                                                    # (1, 1, H, W)
            q_mean = sim_full.mean(dim=(2, 3), keepdim=True)
            q_std = sim_full.std(dim=(2, 3), keepdim=True)
            # Adaptive cutoff (per-frame): sim > mean + k·std.
            adaptive_thr = q_mean + self._object_std_k * q_std
            # Absolute floor: sim > max(car_absolute_sim_threshold, legacy
            # explorfm_object_threshold).  Folding via max() means the
            # effective per-pixel threshold is the elementwise max of the
            # adaptive cutoff and the absolute floor — equivalent to
            # requiring sim to beat BOTH thresholds.
            abs_floor = max(self._car_abs_sim_threshold, self._object_threshold)
            thr = torch.clamp(adaptive_thr, min=abs_floor)
            high_mask_np = (sim_full > thr)[0, 0].detach().cpu().numpy()
            car_debug['high_mask'] = high_mask_np
            car_debug['sim_max'] = float(sim_full.amax().item())
            car_debug['thr'] = float(thr.item())
            car_debug['abs_floor'] = float(abs_floor)
            n_high = int(high_mask_np.sum())
            car_debug['n_high_pixels'] = n_high

            # Noise guardrail: a handful of stray pixels is almost certainly
            # not a car.  Drop the detection so all node scores reset to 0.
            detection_valid = n_high >= self._car_min_high_pixels
            if not detection_valid:
                # Wipe the mask so the debug overlay also reflects "no detection".
                high_mask_np[:] = False
                car_debug['high_mask'] = high_mask_np
                n_high = 0

            car_xyz_odom: Optional[np.ndarray] = None
            nearest_pixel_dist = float('inf')
            if n_high > 0:
                ys, xs = np.where(high_mask_np)
                cu = float(xs.mean())
                cv_c = float(ys.mean())
                car_debug['centroid_uv'] = (cu, cv_c)

                if proj['visible_u'].size > 0:
                    du = proj['visible_u'].astype(np.float64) - cu
                    dv = proj['visible_v'].astype(np.float64) - cv_c
                    d2 = du * du + dv * dv
                    idx_near = int(np.argmin(d2))
                    nearest_pixel_dist = float(np.sqrt(d2[idx_near]))
                    car_debug['nearest_node_uv'] = (
                        float(proj['visible_u'][idx_near]),
                        float(proj['visible_v'][idx_near]),
                    )

                    # Pixel-distance guardrail: if the nearest projected node
                    # is farther than the configured threshold from the
                    # centroid, the anchor is untrustworthy → drop detection.
                    if (self._car_max_centroid_node_px <= 0.0
                            or nearest_pixel_dist <= self._car_max_centroid_node_px):
                        car_xyz_odom = np.asarray(
                            proj['visible_pos_odom'][idx_near], dtype=np.float64,
                        )
                        car_debug['car_xyz_odom'] = car_xyz_odom
            car_debug['nearest_node_pixel_dist'] = nearest_pixel_dist

            # Per-frame car scores — subset-only ingest with MAX merge.
            # We only push scores for nodes within the distance threshold.
            # The layer's CUSTOM merge takes the elementwise max with the
            # previously-stored score, so:
            #   * a node that has never been near a car stays at 0,
            #   * a node painted once keeps its colour forever,
            #   * a node painted twice keeps the brighter colour.
            # No-detection frames simply skip the ingest call.
            all_pos_t = self._last_result.node_positions                         # (N, 3) on device
            all_ids_t = self._last_result.node_ids                               # (N,) on device
            n_total = int(all_pos_t.shape[0]) if all_pos_t is not None else 0
            n_near = 0
            if n_total > 0 and car_xyz_odom is not None:
                car_pos_t = torch.as_tensor(
                    car_xyz_odom, dtype=all_pos_t.dtype, device=all_pos_t.device,
                )
                dist_t = torch.linalg.norm(all_pos_t - car_pos_t, dim=1)
                thr_m = float(self._car_distance_threshold)
                in_range = dist_t < thr_m
                n_near = int(in_range.sum().item())
                if n_near > 0:
                    score_t = (1.0 - dist_t[in_range] / thr_m).to(dtype=torch.float32)
                    ids_long_t = all_ids_t[in_range].to(dtype=torch.long)
                    self.builder.ingest_layer_scores(
                        self._car_layer_name, ids_long_t, score_t,
                    )

            nn_px = car_debug['nearest_node_pixel_dist']
            nn_px_str = f'{nn_px:.1f}' if math.isfinite(nn_px) else 'inf'
            obj_hits_log = (
                f' car[sim_max={car_debug["sim_max"]:.3f} '
                f'thr={car_debug["thr"]:.3f} '
                f'abs_floor={car_debug["abs_floor"]:.3f} '
                f'high_px={n_high}/{self._car_min_high_pixels}min '
                f'nn_px={nn_px_str}/{self._car_max_centroid_node_px:.0f}max '
                f'within_{self._car_distance_threshold:.1f}m={n_near}]'
            )

        self.get_logger().info(
            f'[explorfm rgb={self._rgb_count}] infer={infer_ms:.1f}ms '
            f'visible={ids_np.size} frontiers={n_front} '
            f'trav[min={trav_np.min():.2f} max={trav_np.max():.2f}] '
            f'front[min={front_np.min():.2f} max={front_np.max():.2f}]'
            f'{obj_hits_log}'
        )
        return trav_vals, front_vals_full, car_debug, frontier_debug

    def _enqueue_car_debug_save(
        self,
        rgb: np.ndarray,
        proj: dict,
        car_debug: Optional[dict],
        idx: int,
    ) -> None:
        """Snapshot inputs on the executor thread, queue for the worker.

        Reading persistent car-layer scores happens here (synchronously, on
        the executor thread) so the worker stays pure CPU and never touches
        the GPU layer registry — that avoids any read/write races with the
        next ``ingest_layer_scores`` call from a later image callback.
        """
        if self._car_debug_queue is None:
            return

        visible_ids_np = proj['visible_ids']
        n_vis = int(visible_ids_np.size)
        visible_norm = np.zeros(n_vis, dtype=np.float32)

        car_layer = self.builder._layer_registry._external_layers.get(
            self._car_layer_name,
        )
        if (car_layer is not None
                and getattr(car_layer, '_scores', None) is not None
                and self._last_result is not None
                and self._last_result.num_nodes > 0):
            device = car_layer._scores.device
            all_ids_t = self._last_result.node_ids.to(device=device, dtype=torch.long)
            all_scores = car_layer.read(all_ids_t, device).detach().cpu().numpy()
            score_min = float(all_scores.min()) if all_scores.size else 0.0
            score_max = float(all_scores.max()) if all_scores.size else 1.0
            if score_max <= score_min:
                score_max = score_min + 1e-6
            if n_vis > 0:
                vis_ids_t = torch.from_numpy(visible_ids_np).to(
                    device=device, dtype=torch.long,
                )
                vis_scores = car_layer.read(vis_ids_t, device).detach().cpu().numpy()
                visible_norm = np.clip(
                    (vis_scores - score_min) / (score_max - score_min), 0.0, 1.0,
                )

        centroid_uv = None
        if car_debug is not None and car_debug.get('centroid_uv') is not None:
            cu, cv_c = car_debug['centroid_uv']
            centroid_uv = (float(cu), float(cv_c))

        # rgb is a fresh decode in image_cb (no other reader), so we pass
        # the reference directly — no extra megabyte-scale copy per frame.
        payload = {
            'rgb':         rgb,
            'visible_u':   np.ascontiguousarray(proj['visible_u']),
            'visible_v':   np.ascontiguousarray(proj['visible_v']),
            'visible_norm': visible_norm,
            'edge_uvs': (
                np.ascontiguousarray(proj['visible_edge_uv'])
                if self._debug_image_draw_edges
                else np.empty((0, 4), dtype=np.int32)
            ),
            'centroid_uv': centroid_uv,
            'idx':         int(idx),
        }

        # Drop oldest if the worker is behind, then enqueue. Putters never block.
        while True:
            try:
                self._car_debug_queue.put_nowait(payload)
                return
            except queue.Full:
                try:
                    self._car_debug_queue.get_nowait()
                except queue.Empty:
                    pass

    def _car_debug_worker(self) -> None:
        """Background thread: render queued debug snapshots to PDF.

        Uses cv2 for in-memory raster drawing and PIL for the PDF wrap.
        Both release the GIL during their main I/O, so the ROS executor
        keeps spinning while we save.
        """
        # Lazy imports keep node-import cost low if the debug dir is unset.
        try:
            from PIL import Image  # Pillow
        except ImportError as exc:
            self.get_logger().error(
                f'Pillow (PIL) not installed; car-debug PDF saving disabled: {exc}',
            )
            return
        while True:
            payload = self._car_debug_queue.get()
            try:
                self._render_and_save_car_debug(payload, Image)
            except Exception as exc:  # pragma: no cover  - defensive
                import traceback
                self.get_logger().error(
                    f'car-debug save failed: {exc}\n{traceback.format_exc()}',
                    throttle_duration_sec=5.0,
                )
            finally:
                self._car_debug_queue.task_done()

    def _render_and_save_car_debug(self, payload: dict, pil_image_cls) -> None:
        """CPU-only render + single-page raster-PDF save.

        Inverted RViz rainbow so high score (close to the car) is blue and
        low score (far / never seen) is red — opposite of the on-screen
        cloud, which is what the user asked for in the saved PDFs.
        """
        rgb = payload['rgb']
        visible_u = payload['visible_u']
        visible_v = payload['visible_v']
        visible_norm = payload['visible_norm']
        edge_uvs = payload['edge_uvs']
        centroid_uv = payload['centroid_uv']
        idx = payload['idx']

        img = rgb.copy()  # cv2 draws in-place; preserve original

        # ── Edges first so the node dots sit on top of the lines.
        for u0, v0, u1, v1 in edge_uvs:
            cv2.line(img, (int(u0), int(v0)), (int(u1), int(v1)),
                     (255, 255, 255), 2)

        if visible_u.size > 0:
            lut = _RVIZ_RAINBOW_LUT
            # Invert: high score → low LUT index → blue; low score → red.
            lut_idx = np.clip(
                ((1.0 - visible_norm) * (lut.shape[0] - 1)).round().astype(int),
                0, lut.shape[0] - 1,
            )
            colors_rgb = (lut[lut_idx] * 255.0).astype(np.uint8)  # (N, 3)
            for u, v, c in zip(visible_u, visible_v, colors_rgb):
                center = (int(u), int(v))
                cv2.circle(img, center, 14, (int(c[0]), int(c[1]), int(c[2])), -1)
                cv2.circle(img, center, 14, (0, 0, 0), 1)  # thin black outline

        if centroid_uv is not None:
            cv2.drawMarker(
                img, (int(centroid_uv[0]), int(centroid_uv[1])),
                (255, 255, 0), markerType=cv2.MARKER_CROSS,
                markerSize=40, thickness=4,
            )

        if self._car_debug_dir is not None:
            pdf_path = self._car_debug_dir / f'car_debug_{idx:06d}.pdf'
            pil_image_cls.fromarray(img).save(str(pdf_path), 'PDF', resolution=100.0)
        if self._car_debug_png_dir is not None:
            png_path = self._car_debug_png_dir / f'car_debug_{idx:06d}.png'
            cv2.imwrite(str(png_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # ── Frontier-layer debug saver (parallel to the car saver) ──────────

    def _enqueue_frontier_debug_save(
        self,
        rgb: np.ndarray,
        proj: dict,
        frontier_debug: Optional[dict],
        idx: int,
    ) -> None:
        """Snapshot frontier-score state on the executor thread, enqueue.

        Sends:
          * the per-pixel heatmap (only used to find *high* pixels in the
            renderer — everything outside the high mask stays raw RGB),
          * non-frontier visible node pixels (drawn as solid red dots),
          * frontier visible node pixels + normalised scores (drawn as
            hollow circles coloured by the persistent ``frontier_score``
            layer, matching the on-screen ``/frontier_score_cloud``),
          * ``score_min`` / ``score_max`` — the RViz autobounds across all
            current frontier nodes (used by both the heatmap colour-mapping
            and the legend so the colours are unified).
        """
        if self._frontier_debug_queue is None:
            return

        front_map_np = None
        if frontier_debug is not None:
            front_map_np = frontier_debug.get('front_map_np')

        visible_u_all = np.ascontiguousarray(proj['visible_u'])
        visible_v_all = np.ascontiguousarray(proj['visible_v'])
        visible_types_np = proj['visible_types']
        front_mask = visible_types_np == 2

        nonfront_u = visible_u_all[~front_mask]
        nonfront_v = visible_v_all[~front_mask]
        front_u = visible_u_all[front_mask]
        front_v = visible_v_all[front_mask]
        n_vis_front = int(front_mask.sum())

        score_min = 0.0
        score_max = 1.0
        frontier_norm = np.empty(0, dtype=np.float32)

        front_layer = self.builder._layer_registry._external_layers.get(
            self._front_layer_name,
        )
        if (front_layer is not None
                and getattr(front_layer, '_scores', None) is not None
                and self._last_result is not None
                and self._last_result.num_nodes > 0):
            device = front_layer._scores.device

            # Autocompute bounds across ALL current frontier nodes — same
            # rule RViz applies to the /frontier_score_cloud display.
            all_types = self._last_result.node_types
            all_ids = self._last_result.node_ids
            all_front_mask = all_types == 2
            if bool(all_front_mask.any().item()):
                front_ids_all = all_ids[all_front_mask].to(
                    device=device, dtype=torch.long,
                )
                all_front_scores = (
                    front_layer.read(front_ids_all, device).detach().cpu().numpy()
                )
                valid = np.isfinite(all_front_scores)
                if valid.any():
                    score_min = float(all_front_scores[valid].min())
                    score_max = float(all_front_scores[valid].max())
            if score_max <= score_min:
                score_max = score_min + 1e-6

            if n_vis_front > 0:
                vis_ids_front = proj['visible_ids'][front_mask]
                vis_ids_t = torch.from_numpy(vis_ids_front).to(
                    device=device, dtype=torch.long,
                )
                vis_scores = front_layer.read(vis_ids_t, device).detach().cpu().numpy()
                vis_scores = np.where(np.isfinite(vis_scores), vis_scores, score_min)
                frontier_norm = np.clip(
                    (vis_scores - score_min) / (score_max - score_min), 0.0, 1.0,
                ).astype(np.float32)

        payload = {
            'rgb':            rgb,
            'front_map_np':   front_map_np,
            'front_u':        np.ascontiguousarray(front_u),
            'front_v':        np.ascontiguousarray(front_v),
            'frontier_norm':  frontier_norm,
            'nonfront_u':     np.ascontiguousarray(nonfront_u),
            'nonfront_v':     np.ascontiguousarray(nonfront_v),
            'edge_uvs':       (
                np.ascontiguousarray(proj['visible_edge_uv'])
                if self._debug_image_draw_edges
                else np.empty((0, 4), dtype=np.int32)
            ),
            'score_min':      float(score_min),
            'score_max':      float(score_max),
            'high_std_k':     float(self._frontier_debug_high_std_k),
            'idx':            int(idx),
        }

        while True:
            try:
                self._frontier_debug_queue.put_nowait(payload)
                return
            except queue.Full:
                try:
                    self._frontier_debug_queue.get_nowait()
                except queue.Empty:
                    pass

    def _frontier_debug_worker(self) -> None:
        """Background thread: render queued frontier debug snapshots to PDF."""
        try:
            from PIL import Image  # Pillow
        except ImportError as exc:
            self.get_logger().error(
                f'Pillow (PIL) not installed; frontier-debug PDF saving disabled: {exc}',
            )
            return
        while True:
            payload = self._frontier_debug_queue.get()
            try:
                self._render_and_save_frontier_debug(payload, Image)
            except Exception as exc:  # pragma: no cover  - defensive
                import traceback
                self.get_logger().error(
                    f'frontier-debug save failed: {exc}\n{traceback.format_exc()}',
                    throttle_duration_sec=5.0,
                )
            finally:
                self._frontier_debug_queue.task_done()

    def _render_and_save_frontier_debug(self, payload: dict, pil_image_cls) -> None:
        """CPU-only render + single-page raster-PDF save.

        Layers (bottom→top):
          * raw RGB (default — most pixels stay untouched),
          * patch-only ExploRFM heatmap: only pixels with score above
            mean + k·std (this frame's distribution) get the rainbow tint,
          * non-frontier visible nodes: solid red dots (small),
          * frontier visible nodes: HOLLOW rings, colour = persistent
            ``frontier_score`` mapped via the same RViz rainbow +
            autobounds the on-screen ``/frontier_score_cloud`` uses,
          * legend in the top-right: rainbow bar with the numeric
            ``score_min`` and ``score_max`` so the colour↔score mapping
            is readable from the saved frame alone.

        Direction matches the on-screen cloud (high score → red), unlike
        the car saver which is inverted.
        """
        rgb = payload['rgb']
        front_map_np = payload['front_map_np']
        front_u = payload['front_u']
        front_v = payload['front_v']
        frontier_norm = payload['frontier_norm']
        nonfront_u = payload['nonfront_u']
        nonfront_v = payload['nonfront_v']
        edge_uvs = payload['edge_uvs']
        score_min = payload['score_min']
        score_max = payload['score_max']
        high_std_k = payload['high_std_k']
        idx = payload['idx']

        img = rgb.copy()
        H, W = img.shape[:2]
        lut = _RVIZ_RAINBOW_LUT
        n_lut = lut.shape[0]
        span = max(score_max - score_min, 1e-6)

        # ── Heatmap: ONLY high-scoring pixels get tinted.  Threshold is
        # the per-frame mean + k·std of the raw ExploRFM map, so the
        # "high" set scales with image content rather than a fixed cutoff.
        # Colour inside the patch is normalised to the PATCH's own
        # [min, max] (the dimmest high pixel → blue, the brightest → red)
        # so the rainbow spans the full LUT instead of clamping the whole
        # patch to red.  The two scales (patch heatmap vs. persistent node
        # score) get separate legend bars below.
        patch_min: float = float('nan')
        patch_max: float = float('nan')
        if front_map_np is not None:
            if front_map_np.shape != (H, W):
                front_map_np = cv2.resize(
                    front_map_np, (W, H), interpolation=cv2.INTER_LINEAR,
                )
            f_mean = float(front_map_np.mean())
            f_std = float(front_map_np.std())
            thr = f_mean + high_std_k * f_std
            high_mask = front_map_np > thr
            if high_mask.any():
                high_vals = front_map_np[high_mask]
                patch_min = float(high_vals.min())
                patch_max = float(high_vals.max())
                patch_span = max(patch_max - patch_min, 1e-6)
                norm_map = np.clip(
                    (front_map_np - patch_min) / patch_span, 0.0, 1.0,
                )
                lut_idx_map = np.clip(
                    (norm_map * (n_lut - 1)).round().astype(np.int32),
                    0, n_lut - 1,
                )
                heat_u8 = (lut[lut_idx_map] * 255.0).astype(np.uint8)
                alpha = 0.6
                masked = high_mask[..., None]
                img = np.where(
                    masked,
                    (img.astype(np.float32) * (1.0 - alpha)
                     + heat_u8.astype(np.float32) * alpha).astype(np.uint8),
                    img,
                )

        # ── Edges between visible nodes — drawn above the heatmap but
        # below the node dots so the lines don't cover the markers.
        for u0, v0, u1, v1 in edge_uvs:
            cv2.line(img, (int(u0), int(v0)), (int(u1), int(v1)),
                     (255, 255, 255), 2)

        # ── Non-frontier (free-space) nodes: solid red small dots.
        for u, v in zip(nonfront_u, nonfront_v):
            center = (int(u), int(v))
            cv2.circle(img, center, 8, (255, 0, 0), -1)
            cv2.circle(img, center, 8, (0, 0, 0), 1)

        # ── Frontier nodes: HOLLOW rings coloured by persistent score.
        # Thick coloured outline + thin black contrast rings so the ring
        # reads against both bright and dark backgrounds.
        if front_u.size > 0:
            lut_idx = np.clip(
                (frontier_norm * (n_lut - 1)).round().astype(int),
                0, n_lut - 1,
            )
            colors_rgb = (lut[lut_idx] * 255.0).astype(np.uint8)
            for u, v, c in zip(front_u, front_v, colors_rgb):
                center = (int(u), int(v))
                color_t = (int(c[0]), int(c[1]), int(c[2]))
                cv2.circle(img, center, 14, color_t, thickness=3)
                cv2.circle(img, center, 16, (0, 0, 0), 1)
                cv2.circle(img, center, 11, (0, 0, 0), 1)

        # ── Legend: rainbow bar with score_min / score_max labels.
        bar_w = 240
        bar_h = 18
        pad = 20
        x1 = max(0, W - pad - bar_w)
        y1 = pad + 22  # leave room for title above
        gradient_idx = np.linspace(0, n_lut - 1, bar_w).astype(np.int32)
        bar_strip = (lut[gradient_idx] * 255.0).astype(np.uint8)
        bar = np.tile(bar_strip[None, :, :], (bar_h, 1, 1))
        img[y1:y1 + bar_h, x1:x1 + bar_w] = bar
        cv2.rectangle(
            img, (x1 - 1, y1 - 1), (x1 + bar_w, y1 + bar_h), (0, 0, 0), 1,
        )
        _put_outlined_text(img, 'frontier_score', (x1, y1 - 6), scale=0.55)
        y2 = y1 + bar_h + 18
        _put_outlined_text(img, f'{score_min:.2f}', (x1, y2))
        max_str = f'{score_max:.2f}'
        text_w = cv2.getTextSize(
            max_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1,
        )[0][0]
        _put_outlined_text(img, max_str, (x1 + bar_w - text_w, y2))

        if self._frontier_debug_dir is not None:
            pdf_path = self._frontier_debug_dir / f'frontier_debug_{idx:06d}.pdf'
            pil_image_cls.fromarray(img).save(str(pdf_path), 'PDF', resolution=100.0)
        if self._frontier_debug_png_dir is not None:
            png_path = self._frontier_debug_png_dir / f'frontier_debug_{idx:06d}.png'
            cv2.imwrite(str(png_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def _save_frame_files(
        self,
        msg: Image,
        rgb: np.ndarray,
        proj: dict,
        t_sec: float,
        trav_per_node: Optional[np.ndarray],
        front_per_node: Optional[np.ndarray],
    ) -> None:
        """Write the four-file frame: rgb / nodes / edges / json."""
        idx = self._rgb_count
        img_h, img_w = rgb.shape[:2]
        K = self._cam_K
        cam_frame = self._cam_frame or (msg.header.frame_id or 'camera_optical')

        # Augment visible_nodes with model values for verification in the JSON.
        visible_nodes = proj['visible_nodes']
        if trav_per_node is not None:
            for i, node in enumerate(visible_nodes):
                node['traversability'] = float(trav_per_node[i])
        if front_per_node is not None:
            for i, node in enumerate(visible_nodes):
                val = front_per_node[i]
                if np.isfinite(val):
                    node['frontier_score'] = float(val)

        _save_svg_plain(self._out_dir / f'rgb_{idx:06d}.svg', rgb)
        _save_svg_overlay(self._out_dir / f'nodes_{idx:06d}.svg', rgb, visible_nodes)
        _save_svg_edges(
            self._out_dir / f'edges_{idx:06d}.svg', rgb, visible_nodes, proj['visible_edges'],
        )

        graph_data = {
            'frame_index':         idx,
            'rgb_frame_number':    self._rgb_count,
            'timestamp':           t_sec,
            'position_convention': 'camera_frame_x_fwd_y_down_z_left',
            'camera': {
                'frame':           cam_frame,
                'odom_frame':      self.frame_id,
                'K':               K.tolist(),
                'width':           img_w,
                'height':          img_h,
                'T_opt_from_odom': proj['T_opt_from_odom'].tolist(),
            },
            'nodes': visible_nodes,
            'edges': proj['visible_edges'],
        }
        with open(self._out_dir / f'graph_{idx:06d}.json', 'w') as fh:
            json.dump(graph_data, fh, indent=2)

        self.get_logger().info(
            f'[save frame={idx}] t={t_sec:.3f}s '
            f'visible={len(visible_nodes)} nodes  {len(proj["visible_edges"])} edges'
        )

    # ──────────────────────────────────────────────────
    #  Final-graph snapshot (Ctrl-C aware)
    # ──────────────────────────────────────────────────

    def save_final_graph_snapshot(self) -> None:
        """Dump the current global graph + per-layer scores to JSON.

        Schema (matched in nav_graph_node_e2e.save_final_graph_snapshot):
          {
            "method": "nav_graph_node",
            "saved_at_utc": ISO8601,
            "frame_count": int,
            "num_nodes": int, "num_edges": int,
            "layer_names": [str],
            "nodes": [{id, type, position:[x,y,z], scores:{layer:value}}],
            "edges": [{node_id_0, node_id_1, weight}],
          }

        Robust to being called when no frame has been processed (empty graph).
        Best-effort: errors here must not block the rest of shutdown.
        """
        if self._graph_snapshot_path is None:
            return
        try:
            gb = self.builder.global_builder
            n_total = int(gb._global_ids.shape[0])
            ids = gb._global_ids.detach().cpu().numpy() if n_total > 0 else np.empty((0,), dtype=np.int64)
            pos = (gb._global_pos.detach().cpu().numpy()
                   if n_total > 0 else np.empty((0, 3), dtype=np.float32))
            types = (gb._global_node_types.detach().cpu().numpy()
                     if n_total > 0 else np.empty((0,), dtype=np.int64))
            # Edges: parallel arrays of ID pairs + weights.
            edge_ids = (gb._global_edge_ids.detach().cpu().numpy()
                        if gb._global_edge_ids.numel() > 0 else np.empty((0, 2), dtype=np.int64))
            edge_w = (gb._global_edge_weights.detach().cpu().numpy()
                      if gb._global_edge_weights.numel() > 0 else np.empty((0,), dtype=np.float32))

            # Per-node layer scores.  Recompute so the snapshot reflects the
            # very last state of every registered layer (compute_layers is
            # called every frame anyway, but re-running is cheap and means
            # we never miss a between-frame ingest).
            try:
                scores_t, layer_names = self.builder.compute_layers()
            except Exception:
                scores_t, layer_names = None, []
            scores_np = (scores_t.detach().cpu().numpy()
                         if scores_t is not None else np.empty((n_total, 0), dtype=np.float32))

            nodes_out: list = []
            for i in range(n_total):
                row_scores = {}
                if scores_np.shape[1] > 0:
                    row_scores = {
                        str(layer_names[j]): float(scores_np[i, j])
                        for j in range(scores_np.shape[1])
                    }
                nodes_out.append({
                    'id':       int(ids[i]),
                    'type':     int(types[i]),
                    'position': [float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2])],
                    'scores':   row_scores,
                })

            edges_out: list = []
            for k in range(edge_ids.shape[0]):
                edges_out.append({
                    'node_id_0': int(edge_ids[k, 0]),
                    'node_id_1': int(edge_ids[k, 1]),
                    'weight':    float(edge_w[k]) if k < edge_w.shape[0] else float('nan'),
                })

            from datetime import datetime, timezone
            payload = {
                'method':      'nav_graph_node',
                'saved_at_utc': datetime.now(timezone.utc).isoformat(),
                'frame_count': int(self.frame_count),
                'num_nodes':   int(n_total),
                'num_edges':   int(edge_ids.shape[0]),
                'layer_names': [str(n) for n in layer_names],
                'nodes':       nodes_out,
                'edges':       edges_out,
            }
            with open(self._graph_snapshot_path, 'w') as f:
                json.dump(payload, f, indent=2)
            self.get_logger().info(
                f'Final graph snapshot written: {self._graph_snapshot_path} '
                f'({n_total} nodes, {edge_ids.shape[0]} edges)'
            )
        except Exception as exc:
            import traceback
            self.get_logger().error(
                f'save_final_graph_snapshot failed: {exc}\n{traceback.format_exc()}'
            )

    def close_timing_csv(self) -> None:
        """Flush + close the timing-CSV file if it's open.  No-op otherwise."""
        try:
            if self._timing_csv_file is not None and not self._timing_csv_file.closed:
                self._timing_csv_file.flush()
                self._timing_csv_file.close()
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = OdinNavGraphNode()
    try:
        with suppress(KeyboardInterrupt):
            rclpy.spin(node)
    finally:
        # Always try to persist artefacts before shutting down — even on
        # KeyboardInterrupt or exception inside spin().
        try:
            node.save_final_graph_snapshot()
        except Exception:
            pass
        try:
            node.close_timing_csv()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
