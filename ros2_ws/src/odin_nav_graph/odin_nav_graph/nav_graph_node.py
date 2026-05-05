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
"""

from __future__ import annotations

import math
import os
import sys
import time
import types
from collections import deque
from contextlib import suppress
from typing import Optional, Tuple

import numpy as np
import rclpy
import torch
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

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

from nav_graph import (  # noqa: E402
    NavigationGraphBuilder,
    NavGraphConfig,
    FrontierConfig,
    ExplorationConfig,
    ElevationMapConfig,
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
#  When ``input_type='elevation_map'`` is used with a *rolling* local
#  elevation window (as we do with elevation_mapping_cupy), nav_graph
#  has two latent issues that combine to break the global graph:
#
#    A. ``GlobalGraphGenerator.tensor_merge_local_nodes_gpu`` does
#       ``torch.cdist(local_worlds, global_worlds)`` over **3D** XYZ.
#       The waypoint generator emits local nodes at Z=0 always, but
#       ``_assign_z_from_elevation`` overwrites the *global* Z with
#       elevation values.  Next frame, fresh local Z=0 nodes meet
#       global Z=elevation nodes → 3D distance ≈ |Z| > merge_distance,
#       merge fails, duplicate spawns at Z=0.  Stack frames and you get
#       a tower at every (x, y).
#
#    B. ``_assign_z_from_elevation`` calls ``np.clip(px, 0, W-1)`` on
#       every global node — even those far outside the current local
#       window.  Their Z is then read off the edge of the elevation
#       array (often NaN border cells), corrupting persistent state.
#
#  We patch both:
#    - merge in 2D (XY only) so Z mismatch can't break it,
#    - assign Z only to nodes actually inside the current window
#      *and* whose elevation is finite — out-of-window nodes keep their
#      previously-stored Z forever.
#
#  The patches operate on builder instances so we don't have to touch
#  the nav_graph submodule.  Keep them in sync with the upstream
#  ``GlobalGraphGenerator`` invariants documented in nav_graph's
#  CLAUDE.md (parallel-array invariant, ID-vs-index discipline).
# ─────────────────────────────────────────────────────────────────────

def _patched_merge_2d(
    self_gg,
    local_worlds: torch.Tensor,
    local_types: torch.Tensor,
    global_worlds: torch.Tensor,
    global_ids_tensor: torch.Tensor,
    next_node_id: int,
    merge_distance: float,
    device: str = 'cuda',
):
    """2D-only replacement for ``GlobalGraphGenerator.tensor_merge_local_nodes_gpu``.

    Identical to the upstream implementation, except the cdist for
    matching local→global is over (x, y) not (x, y, z).  The full local
    XYZ position is still stored when a new node is added, so per-node
    Z (set later by the patched ``_assign_z_from_elevation``) survives.
    """
    N = local_worlds.shape[0]
    if N == 0:
        return (
            torch.empty((0,), dtype=torch.long, device=device),
            torch.empty((0, 3), dtype=torch.float32, device=device),
            next_node_id,
        )

    if global_worlds.numel() == 0:
        new_ids = torch.arange(
            next_node_id, next_node_id + N, device=device, dtype=torch.long
        )
        final_ids = new_ids
        new_local_mask = torch.ones(N, dtype=torch.bool, device=device)
        updated_next_id = next_node_id + N
        local_id_positions = local_worlds
    else:
        # The only line that differs from upstream: 2D cdist.
        dists = torch.cdist(local_worlds[:, :2], global_worlds[:, :2])
        min_dists, min_idx = torch.min(dists, dim=1)

        merge_mask = min_dists < merge_distance
        merged_ids = global_ids_tensor[min_idx]

        new_local_mask = ~merge_mask
        num_new = int(new_local_mask.sum().item())
        new_ids = torch.arange(
            next_node_id, next_node_id + num_new, device=device, dtype=torch.long
        )

        final_ids = merged_ids.clone()
        final_ids[new_local_mask] = new_ids
        updated_next_id = next_node_id + num_new

        # Merged nodes snap to the existing global position (preserves
        # the persistent Z that out-of-window nodes still carry).
        local_id_positions = local_worlds.clone()
        local_id_positions[~new_local_mask] = global_worlds[min_idx[~new_local_mask]]

        if merge_mask.any():
            self_gg._global_node_types[min_idx[merge_mask]] = local_types[merge_mask]

    if new_local_mask.any():
        self_gg._global_pos = torch.cat(
            [self_gg._global_pos, local_worlds[new_local_mask]], dim=0
        )
        self_gg._global_ids = torch.cat([self_gg._global_ids, new_ids], dim=0)
        self_gg._global_node_types = torch.cat(
            [self_gg._global_node_types, local_types[new_local_mask]], dim=0
        )

    return final_ids, local_id_positions, updated_next_id


def _patched_assign_z_inbounds(
    self_builder,
    elevation_flipped: np.ndarray,
    origin_x: float,
    origin_y: float,
    resolution: float,
    width: int,
    height: int,
):
    """Replacement for ``NavigationGraphBuilder._assign_z_from_elevation``.

    Updates Z **only** for nodes that fall inside the current local
    elevation window and have a finite elevation reading.  Out-of-window
    nodes keep their stored Z forever, so once the robot drives past a
    region the height info there persists in the global graph.

    Z is sampled from a NaN-aware **min-filtered** copy of the elevation
    grid.  Without this, frontier nodes at the perimeter of the local
    map pick up grazing-angle / single-beam noise — the LiDAR's far
    range hits objects at flat angles, biasing the per-cell Z way above
    the actual floor — and end up sitting suspiciously high.  Taking
    the min over a small (configurable) neighbourhood snaps each node
    to the lowest plausible surface in that neighbourhood, which is
    almost always the floor for traversable cells.

    Window size comes from the optional builder attribute
    ``_z_lookup_min_filter_radius_cells`` (radius in cells; full kernel
    is ``2*r + 1``).  Default 2 → 5×5 cells.  Set to 0 to disable.
    """
    gg = self_builder.global_builder
    if gg._global_pos.shape[0] == 0:
        return

    radius = int(getattr(self_builder, '_z_lookup_min_filter_radius_cells', 2))
    if radius > 0:
        from scipy.ndimage import minimum_filter
        finite = np.isfinite(elevation_flipped)
        if not finite.any():
            return
        # Replace NaN with +inf so unobserved cells lose the min comparison.
        ef = np.where(finite, elevation_flipped, np.inf).astype(np.float32, copy=False)
        ef = minimum_filter(ef, size=2 * radius + 1, mode='nearest')
        # Cells whose entire neighbourhood was NaN remain +inf — restore NaN.
        elevation_for_lookup = np.where(np.isfinite(ef), ef, np.nan)
    else:
        elevation_for_lookup = elevation_flipped

    xy_cpu = gg._global_pos[:, :2].cpu().numpy()
    px = np.rint((xy_cpu[:, 0] - origin_x) / resolution + width / 2.0).astype(np.int64)
    py = np.rint((xy_cpu[:, 1] - origin_y) / resolution + height / 2.0).astype(np.int64)

    in_bounds = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    if not in_bounds.any():
        return

    px_in = px[in_bounds]
    py_in = py[in_bounds]
    z_vals = elevation_for_lookup[py_in, px_in]
    valid = np.isfinite(z_vals)
    if not valid.any():
        return

    # Indices of global nodes whose Z we're updating this frame.
    update_idx = np.where(in_bounds)[0][valid]
    z_new = z_vals[valid].astype(np.float32)

    device = gg._global_pos.device
    idx_t = torch.from_numpy(update_idx).to(device=device, dtype=torch.long)
    z_t = torch.from_numpy(z_new).to(device=device, dtype=torch.float32)
    gg._global_pos[idx_t, 2] = z_t


def _install_builder_patches(builder) -> None:
    """Install both patches on a ``NavigationGraphBuilder`` instance."""
    builder.global_builder.tensor_merge_local_nodes_gpu = types.MethodType(
        _patched_merge_2d, builder.global_builder
    )
    builder._assign_z_from_elevation = types.MethodType(
        _patched_assign_z_inbounds, builder
    )


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
        self.declare_parameter('cloud_max_range', 8.0)
        self.declare_parameter('sensor_noise_factor', 0.05)
        self.declare_parameter('em_position_noise', 0.0)
        self.declare_parameter('em_orientation_noise', 0.0)

        # Time sync
        self.declare_parameter('odom_buffer_seconds', 1.0)
        self.declare_parameter('odom_match_max_dt', 0.1)

        # nav_graph builder params
        self.declare_parameter('safety_distance', 0.05)
        self.declare_parameter('merge_node_distance', 0.5)
        self.declare_parameter('global_merge_distance', 0.5)
        self.declare_parameter('global_max_candidate_edge_distance', 1.2)
        self.declare_parameter('free_space_sampling_threshold', 0.5)

        self.declare_parameter('frontier_kernel_size', 5)
        self.declare_parameter('frontier_odom_threshold', 1.0)
        self.declare_parameter('frontier_max_edge_connectivity', 14)
        self.declare_parameter('minimum_distance_between_frontiers', 0.05)
        self.declare_parameter('minimum_points_in_cluster', 1)

        # Elevation -> traversability params
        self.declare_parameter('elev_max_height_diff', 0.2)
        self.declare_parameter('elev_max_slope', 0.5)
        self.declare_parameter('elev_gaussian_sigma', 0.5)
        self.declare_parameter('elev_window_size', 5)
        self.declare_parameter('elev_border_cells', 5)

        # Throttling / publishing
        self.declare_parameter('process_every_n', 1)
        self.declare_parameter('publish_elevation_cloud', True)
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
        self.publish_elevation_cloud = bool(gp('publish_elevation_cloud'))
        self.publish_edges_flag = bool(gp('publish_edges'))
        self.max_graph_nodes_before_reset = int(gp('max_graph_nodes_before_reset'))
        self._z_lookup_min_filter_radius_cells = int(gp('z_lookup_min_filter_radius_cells'))
        self.viz_z_offset = float(gp('viz_z_offset'))
        self.max_edges_published = int(gp('max_edges_published'))

        # ── nav_graph config ──────────────────────────────────────────
        cfg = NavGraphConfig(
            free_space_sampling_threshold=float(gp('free_space_sampling_threshold')),
            safety_distance=float(gp('safety_distance')),
            merge_node_distance=float(gp('merge_node_distance')),
            global_merge_distance=float(gp('global_merge_distance')),
            global_max_candidate_edge_distance=float(gp('global_max_candidate_edge_distance')),
            frontier=FrontierConfig(
                kernel_size=int(gp('frontier_kernel_size')),
                odom_proximity_threshold=float(gp('frontier_odom_threshold')),
                max_edge_connectivity=int(gp('frontier_max_edge_connectivity')),
                minimum_distance_between_frontiers=float(gp('minimum_distance_between_frontiers')),
                minimum_points_in_cluster=int(gp('minimum_points_in_cluster')),
                angular_gap_min_gap_deg = 20.0,
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
        # Patch the builder for rolling-elevation use:
        #   - 2D-only merge so Z mismatch can't break it
        #   - in-bounds-only, min-filtered Z assignment so out-of-window
        #     nodes keep their stored elevation forever and boundary
        #     nodes don't pick up grazing-angle / wall-top noise.
        self.builder._z_lookup_min_filter_radius_cells = (
            self._z_lookup_min_filter_radius_cells
        )
        _install_builder_patches(self.builder)
        self.get_logger().info(
            'NavigationGraphBuilder ready (2D-merge + in-bounds Z patches installed).'
        )

        # ── Rolling elevation map (elevation_mapping_cupy) ────────────
        self.get_logger().info('Initialising ElevationMap (elevation_mapping_cupy)...')
        self.emap = ElevationMapWrapper(
            map_length=length_xy,
            resolution=res,
            sensor_noise_factor=sensor_noise_factor,
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
        self.graph_pub = self.create_publisher(PointCloud2, '~/graph_nodes', 1)
        self.frontier_pub = self.create_publisher(PointCloud2, '~/frontier_cloud', 1)
        self.edges_pub = self.create_publisher(Marker, '~/graph_edges', 1)

        H, W = self.emap.grid_shape()
        self.get_logger().info(
            f'Ready | cloud={cloud_topic} odom={odom_topic} '
            f'map={W}x{H}@{res:.3f}m frame={self.frame_id}'
        )

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
        if self.process_every_n > 1 and (self.frame_count % self.process_every_n) != 0:
            return

        if msg.header.frame_id != self.robot_frame:
            self.get_logger().warn(
                f'cloud frame_id={msg.header.frame_id!r} != expected '
                f'{self.robot_frame!r}; transform may be wrong.',
                throttle_duration_sec=5.0,
            )

        t_cloud = stamp_to_sec(msg.header.stamp)
        odom = self._find_pose_at(t_cloud)
        if odom is None:
            self.get_logger().warn(
                f'No odometry within {self.odom_match_max_dt}s of cloud t={t_cloud:.3f} '
                f'(buf size {len(self.odom_buf)})',
                throttle_duration_sec=2.0,
            )
            return

        if odom.header.frame_id and odom.header.frame_id != self.frame_id:
            # We assume odom is published in the configured world frame.  Just warn.
            self.get_logger().warn(
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
            self.get_logger().warn('Cloud has zero finite in-range points',
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
            compute_layers=True,
        )
        t_graph = (time.perf_counter() - t0) * 1000.0

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
            self.builder._z_lookup_min_filter_radius_cells = (
                self._z_lookup_min_filter_radius_cells
            )
            _install_builder_patches(self.builder)
            return

        stamp = msg.header.stamp
        if self.publish_elevation_cloud:
            self._publish_elevation_cloud(stamp)
        self._publish_graph_nodes(result, stamp)
        self._publish_frontier_cloud(result, stamp)
        if self.publish_edges_flag:
            self._publish_edges(result, stamp)

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
        scores = np.zeros(positions.shape[0], dtype=np.float32)
        if (
            result.node_scores is not None
            and result.score_layer_names
            and 'combined' in result.score_layer_names
        ):
            col = result.score_layer_names.index('combined')
            scores = result.node_scores[mask, col].cpu().float().numpy()
        self.frontier_pub.publish(self._make_xyz_intensity_cloud(positions, scores, stamp))

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


def main(args=None):
    rclpy.init(args=args)
    node = OdinNavGraphNode()
    with suppress(KeyboardInterrupt):
        rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
