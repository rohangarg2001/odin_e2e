#!/usr/bin/env python3
"""ROS2 node that builds a navigation graph from RGB images using an
end-to-end model — no nav_graph_gpu, no elevation map, no LiDAR.

Pipeline per image message:
    1. Decode the camera image -> HxWx3 RGB.
    2. Pre-process exactly like training (top-crop -> resize -> ImageNet
       normalise) and run the e2e NavGraphDETR model -> per-query
       (objectness logit, 3-D position).
    3. Keep predictions above ``score_threshold``, denormalise with the
       checkpoint's pos-stats.  These "local nodes" are in the model's
       camera frame: (x_fwd, y_down, z_left).
    4. Transform local nodes camera -> optical -> base_link -> odom using
       the static camera extrinsic and the closest-in-time odometry pose.
    5. Merge the odom-frame local nodes into a persistent global node
       list: a local node within ``merge_node_distance`` of an existing
       global node is dropped, otherwise it is appended.
    6. Publish all global nodes as a PointCloud2 in the ``odom`` frame.

Edges are intentionally not built — nodes only.

ros2 run odin_nav_graph nav_graph_node_e2e --ros-args \
    -p model_checkpoint:=/home/rohang73/ASL/e2e_rgb_nav_graph/four_longrange_size280_overfit/epoch_999.pth \
    -p e2e_repo_path:=/home/rohang73/ASL/e2e_rgb_nav_graph
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from collections import deque
from contextlib import suppress
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
import torch
import torch.nn.functional as F
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
from nav_msgs.msg import Odometry
from std_msgs.msg import Header, Empty
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from scipy.spatial.transform import Rotation as _ScipyR


# ImageNet normalisation constants — must match the training pipeline.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Model camera frame is (x_fwd, y_down, z_left).  Optical frame (used by the
# camera extrinsic below) is (x_right, y_down, z_fwd).  This rotation maps a
# point from the model camera frame into the optical frame:
#     x_opt = right = -z_cam,  y_opt = down = y_cam,  z_opt = fwd = x_cam
_R_CAM_TO_OPT = np.array(
    [[0.0, 0.0, -1.0],
     [0.0, 1.0,  0.0],
     [1.0, 0.0,  0.0]],
    dtype=np.float64,
)


# ─────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────

def stamp_to_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def _finite_or_none(v: float):
    """Return v if finite, else None (serialises to JSON null instead of NaN)."""
    return v if math.isfinite(v) else None


def _pq_to_se3(translation: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    """4x4 float64 SE3 from translation (3,) and unit quaternion (x,y,z,w)."""
    se3 = np.eye(4, dtype=np.float64)
    se3[:3, :3] = _ScipyR.from_quat(quaternion_xyzw).as_matrix()
    se3[:3, 3] = translation
    return se3


# ─────────────────────────────────────────────────────────────────────
#  Node
# ─────────────────────────────────────────────────────────────────────

class OdinNavGraphE2ENode(Node):
    def __init__(self) -> None:
        super().__init__('odin_nav_graph_e2e_node')

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter('image_topic', '/odin1/image/undistorted')
        self.declare_parameter('odom_topic', '/odin1/odometry_highfreq')
        self.declare_parameter('frame_id', 'odom')

        # Which model to run:
        #   'detr'    -> NavGraphDETR (per-query objectness + 3-D regression)
        #   'heatmap' -> HeatmapNavModel (dense traversability heatmap +
        #                Poisson-disk sampling + per-pixel 3-D map)
        self.declare_parameter('model_type', 'detr')

        # DETR model location.
        self.declare_parameter(
            'model_checkpoint',
            '/home/rohang73/ASL/e2e_rgb_nav_graph/four_longrange_size280_overfit/epoch_999.pth',
        )
        self.declare_parameter(
            'e2e_repo_path', '/home/rohang73/ASL/e2e_rgb_nav_graph',
        )

        # Heatmap model location.
        self.declare_parameter(
            'heatmap_checkpoint',
            '/home/rohang73/ASL/e2e_rgb_nav_graph/heatmap_model_training_v5_wall_neg_more_vitb518/best.pth',
        )
        self.declare_parameter(
            'heatmap_repo_path',
            '/home/rohang73/ASL/e2e_rgb_nav_graph/heatmap_model',
        )
        # Heatmap-mode inference knobs.  -1 => take from checkpoint args.
        self.declare_parameter('heatmap_sample_threshold', 0.5)
        self.declare_parameter('heatmap_sample_min_dist', -1.0)
        self.declare_parameter('heatmap_sample_window', -1)

        # 0.0 => take image_size / crop_top_frac from the checkpoint args.
        self.declare_parameter('image_size', 0)
        self.declare_parameter('crop_top_frac', -1.0)
        # DETR objectness threshold; unused in heatmap mode (it uses
        # heatmap_sample_threshold instead).
        self.declare_parameter('score_threshold', 0.97)

        # A local node within this distance (m, XYZ) of an existing global
        # node is treated as the same node and dropped.
        self.declare_parameter('merge_node_distance', 0.4)

        # Visualisation-only z offset (m) added to every published node so
        # they sit clearly above the floor in RViz.  Stored global node
        # positions are unchanged — this only affects PointCloud2 + markers.
        self.declare_parameter('viz_z_offset', 0.4)

        # Forward-range gate: drop model predictions farther than this many
        # metres ahead (model camera x = forward depth).  The monocular RGB
        # model has no depth / wall awareness and hallucinates nodes through
        # walls and beyond dead ends; this clips them before they reach the
        # graph.  0.0 disables the gate.
        self.declare_parameter('max_node_range', 30.0)

        # ── Edges (heatmap mode only) ────────────────────────────────
        # Edges are built per-frame from this frame's local nodes:
        #   1. take every pair within max_edge_distance (3-D),
        #   2. sample the line in camera frame, project to the heatmap,
        #   3. keep if min(heatmap_prob) >= edge_traversability_threshold,
        #   4. add (sorted) global-ID pair to a write-once global set,
        #      respecting max_edges_per_node on each endpoint.
        # DETR mode has no heatmap, so it builds no edges.
        self.declare_parameter('max_edge_distance', 2.3)
        self.declare_parameter('edge_traversability_threshold', 0.4)
        self.declare_parameter('edge_line_samples', 8)
        self.declare_parameter('max_edges_per_node', 9)

        # ── Graph pruning: drop nodes that ended up with zero edges ──
        # Three independent triggers, each off by default:
        #   prune_isolated_nodes  -> periodic, every prune_every_n_frames
        #   prune_on_shutdown     -> one final pass before destroy_node
        #   manual trigger topic  -> always-on; publish std_msgs/Empty to fire
        self.declare_parameter('prune_isolated_nodes', False)
        self.declare_parameter('prune_every_n_frames', 200)
        self.declare_parameter('prune_on_shutdown', False)
        self.declare_parameter('prune_trigger_topic', '~/prune_now')
        # Edge passes if at least this fraction of the line's *in-bounds*
        # samples clear edge_traversability_threshold.  Out-of-heatmap
        # samples are dropped from the count entirely (don't vote either
        # way).  Lower => more lenient; 1.0 == old strict-min behaviour.
        self.declare_parameter('edge_line_min_pass_fraction', 0.85)
        # An edge also needs at least this many in-bounds samples to be
        # considered — protects against accepting an edge from a single
        # marginal sample when most of the line is off-heatmap.
        self.declare_parameter('edge_line_min_in_bounds', 3)

        # Time sync between image and odometry.
        self.declare_parameter('odom_buffer_seconds', 2.0)
        self.declare_parameter('odom_match_max_dt', 0.1)
        self.declare_parameter('process_every_n', 1)

        # Debug: project this frame's predicted local nodes back onto the
        # camera image and save the overlay every Nth processed frame.
        # Camera intrinsics come from cam_fx/fy/cx/cy if set, else from the
        # first CameraInfo message on cam_info_topic.
        self.declare_parameter('cam_info_topic', '/odin1/camera_info')
        self.declare_parameter('cam_fx', 0.0)
        self.declare_parameter('cam_fy', 0.0)
        self.declare_parameter('cam_cx', 0.0)
        self.declare_parameter('cam_cy', 0.0)
        self.declare_parameter(
            'debug_overlay_dir',
            '/home/rohang73/Documents/odin_e2e/e2e_debug_overlays',
        )
        self.declare_parameter('debug_overlay_every_n', 1)
        # Margin (px) added around the image in the overlay so nodes that
        # project outside the image (wide / far predictions) stay visible.
        self.declare_parameter('debug_overlay_pad', 500)
        # Master switch — set to true to start saving overlay PNGs.
        self.declare_parameter('save_overlay', False)

        # Global-graph visualisation: project ALL global nodes/edges onto the
        # current camera image and save two PNGs per trigger frame.
        #   save_global_viz   — master switch (off by default)
        #   global_viz_dir    — output directory
        #   global_viz_every_n — save every Nth processed frame
        self.declare_parameter('save_global_viz', True)
        self.declare_parameter(
            'global_viz_dir',
            '/home/rohang73/Documents/odin_e2e/e2e_debug_viz',
        )
        self.declare_parameter('global_viz_every_n', 1)
        # Node visibility filter for saved images.
        #   'none'     — draw all in-image nodes (original behaviour)
        #   'distance' — only nodes within viz_max_distance metres of the camera
        #   'recency'  — only nodes added in the last viz_recency_frames frames
        self.declare_parameter('viz_filter_mode', 'distance')
        self.declare_parameter('viz_max_distance', 35.0)
        self.declare_parameter('viz_recency_frames', 10)
        # Pixel-space NMS for saved images. Nodes whose projected pixel centres
        # are closer than this many pixels are deduplicated: the node with the
        # most conflicts is removed first (greedy), breaking ties by removing
        # the lower-ID node. 0.0 disables.
        self.declare_parameter('viz_min_pixel_dist', 0.0)

        # Static camera(optical)->base_link extrinsic (Odin Nav Stack defaults).
        self.declare_parameter('cam_base_tx', -0.0042)
        self.declare_parameter('cam_base_ty',  0.0328)
        self.declare_parameter('cam_base_tz',  0.0005)
        self.declare_parameter('cam_base_qx', -0.4951)
        self.declare_parameter('cam_base_qy',  0.5048)
        self.declare_parameter('cam_base_qz', -0.4996)
        self.declare_parameter('cam_base_qw',  0.5005)

        # ── Graph-comparison artefacts ───────────────────────────────────
        # Final-graph snapshot (Ctrl-C aware) + per-frame timing CSV.
        # ``name`` controls the file stems so multiple runs can coexist in
        # the same inputs/ directory (e.g. name:=e2e_vitb, name:=e2e_detr).
        # If graph_snapshot_path / timing_csv_path are set explicitly they
        # take precedence; otherwise they are derived as
        #   {snapshot_dir}/{name}.json  and  {snapshot_dir}/{name}_timing.csv
        self.declare_parameter('name', 'e2e')
        self.declare_parameter(
            'snapshot_dir',
            '/home/rohang73/Documents/odin_e2e/graph_compaRISION/inputs',
        )
        self.declare_parameter('graph_snapshot_path', '')
        self.declare_parameter('timing_csv_path', '')
        self.declare_parameter('enable_graph_snapshot_save', True)

        # ── Inference-time tuning knobs ──────────────────────────────────
        # All three default ON — they're independently revertable if any
        # one causes numerical issues or compile failures on a given GPU.
        #   enable_cudnn_benchmark — toggle torch.backends.cudnn.benchmark.
        #       Lets cuDNN pick its fastest kernel for our fixed input
        #       shape after a 1-2 frame warmup.
        #   enable_amp_autocast    — wrap the model forward in
        #       torch.amp.autocast(dtype=float16).  Usually 1.5-2× speedup
        #       on Ampere+ with no quality loss for ViT-based encoders.
        #   enable_torch_compile   — torch.compile with mode='reduce-overhead'.
        #       Pays a 10-30 s JIT cost on the first forward; afterwards
        #       another 1.2-2× speedup typical.  Set false if compile
        #       errors / shape-instability is observed.
        self.declare_parameter('enable_cudnn_benchmark', True)
        self.declare_parameter('enable_amp_autocast',    True)
        self.declare_parameter('enable_torch_compile',   True)
        self.declare_parameter('enable_timing_csv', True)

        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        image_topic = str(gp('image_topic'))
        odom_topic = str(gp('odom_topic'))
        self.frame_id = str(gp('frame_id'))
        self.score_threshold = float(gp('score_threshold'))
        self.merge_node_distance = float(gp('merge_node_distance'))
        self.max_node_range = float(gp('max_node_range'))
        self.viz_z_offset = float(gp('viz_z_offset'))
        self.max_edge_distance = float(gp('max_edge_distance'))
        self.edge_traversability_threshold = float(gp('edge_traversability_threshold'))
        self.edge_line_samples = max(2, int(gp('edge_line_samples')))
        self.max_edges_per_node = max(0, int(gp('max_edges_per_node')))
        self.edge_line_min_pass_fraction = float(gp('edge_line_min_pass_fraction'))
        self.edge_line_min_in_bounds = max(1, int(gp('edge_line_min_in_bounds')))
        self._prune_enabled = bool(gp('prune_isolated_nodes'))
        self._prune_every_n = max(1, int(gp('prune_every_n_frames')))
        self.prune_on_shutdown = bool(gp('prune_on_shutdown'))
        self.odom_buffer_seconds = float(gp('odom_buffer_seconds'))
        self.odom_match_max_dt = float(gp('odom_match_max_dt'))
        self.process_every_n = max(1, int(gp('process_every_n')))
        self._debug_every_n = max(1, int(gp('debug_overlay_every_n')))
        self._overlay_pad = max(0, int(gp('debug_overlay_pad')))
        self._save_overlay_enabled = bool(gp('save_overlay'))

        # ── Static camera extrinsic: T_base_from_optical ──────────────
        t_bc = np.array(
            [gp('cam_base_tx'), gp('cam_base_ty'), gp('cam_base_tz')],
            dtype=np.float64,
        )
        q_bc = np.array(
            [gp('cam_base_qx'), gp('cam_base_qy'),
             gp('cam_base_qz'), gp('cam_base_qw')],
            dtype=np.float64,
        )
        self._T_base_from_opt = _pq_to_se3(t_bc, q_bc)
        self.get_logger().info(
            f'T_base_from_opt: t={t_bc.tolist()} q={q_bc.tolist()}'
        )

        # ── Load the e2e model (DETR or heatmap) ──────────────────────
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Latch the inference-tuning knobs onto the instance; the model
        # loaders + inference paths read them directly.
        self._enable_cudnn_benchmark = bool(gp('enable_cudnn_benchmark'))
        self._enable_amp_autocast    = bool(gp('enable_amp_autocast'))
        self._enable_torch_compile   = bool(gp('enable_torch_compile'))
        self._on_cuda = (self._device.type == 'cuda')
        # cuDNN auto-tunes its kernel choice for the first 1-2 forwards
        # then sticks with the fastest one — only safe when input shape
        # is fixed, which it is here (always (1, 3, image_size, image_size)).
        if self._on_cuda and self._enable_cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
            self.get_logger().info('torch.backends.cudnn.benchmark = True')

        # GPU-resident ImageNet normalization tensors so _preprocess can
        # do the whole pipeline on-device without touching numpy.  Built
        # once at init since the model device never changes.
        self._mean_t = torch.as_tensor(
            IMAGENET_MEAN, dtype=torch.float32, device=self._device,
        ).view(1, 3, 1, 1)
        self._std_t = torch.as_tensor(
            IMAGENET_STD, dtype=torch.float32, device=self._device,
        ).view(1, 3, 1, 1)

        self._model_type = str(gp('model_type')).lower().strip()
        if self._model_type == 'detr':
            self._load_detr_model(
                str(gp('e2e_repo_path')), str(gp('model_checkpoint')))
        elif self._model_type == 'heatmap':
            self._load_heatmap_model(
                str(gp('heatmap_repo_path')), str(gp('heatmap_checkpoint')),
                sample_threshold=float(gp('heatmap_sample_threshold')),
                sample_min_dist=float(gp('heatmap_sample_min_dist')),
                sample_window=int(gp('heatmap_sample_window')),
            )
        else:
            raise ValueError(
                f'model_type={self._model_type!r} not understood; '
                'use "detr" or "heatmap".'
            )

        # image_size / crop_top_frac: param override, else checkpoint args.
        img_size_param = int(gp('image_size'))
        crop_param = float(gp('crop_top_frac'))
        self._image_size = img_size_param if img_size_param > 0 else self._ckpt_image_size
        self._crop_top_frac = crop_param if crop_param >= 0.0 else self._ckpt_crop_top_frac
        self.get_logger().info(
            f'Pre-processing: image_size={self._image_size} '
            f'crop_top_frac={self._crop_top_frac} threshold={self.score_threshold}'
        )

        # ── Debug overlay: camera intrinsics + output directory ───────
        cam_fx = float(gp('cam_fx'))
        cam_fy = float(gp('cam_fy'))
        cam_cx = float(gp('cam_cx'))
        cam_cy = float(gp('cam_cy'))
        if cam_fx > 0.0 and cam_fy > 0.0:
            self._cam_K: Optional[np.ndarray] = np.array(
                [[cam_fx, 0.0, cam_cx],
                 [0.0, cam_fy, cam_cy],
                 [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            self.get_logger().info(
                f'Camera K from params: fx={cam_fx} fy={cam_fy} '
                f'cx={cam_cx} cy={cam_cy}'
            )
        else:
            self._cam_K = None  # filled by the first CameraInfo message
        self._debug_dir = Path(str(gp('debug_overlay_dir')))
        if self._save_overlay_enabled:
            self._debug_dir.mkdir(parents=True, exist_ok=True)
            self.get_logger().info(
                f'Debug overlays -> {self._debug_dir} '
                f'(every {self._debug_every_n} processed frames)'
            )
        else:
            self.get_logger().info(
                'Debug overlays disabled (pass -p save_overlay:=true to enable).'
            )

        self._save_global_viz_enabled = bool(gp('save_global_viz'))
        self._global_viz_dir = Path(str(gp('global_viz_dir')))
        self._global_viz_every_n = max(1, int(gp('global_viz_every_n')))
        self._viz_filter_mode    = str(gp('viz_filter_mode')).strip()
        self._viz_max_distance   = float(gp('viz_max_distance'))
        self._viz_recency_frames = max(1, int(gp('viz_recency_frames')))
        self._viz_min_pixel_dist = float(gp('viz_min_pixel_dist'))
        if self._save_global_viz_enabled:
            self._global_viz_dir.mkdir(parents=True, exist_ok=True)
            self.get_logger().info(
                f'Global-graph viz -> {self._global_viz_dir} '
                f'(every {self._global_viz_every_n} processed frames)'
            )
        else:
            self.get_logger().info(
                'Global-graph viz disabled (pass -p save_global_viz:=true to enable).'
            )

        # ── State ─────────────────────────────────────────────────────
        self.odom_buf: deque[Tuple[float, Odometry]] = deque()
        self.frame_count = 0
        # Persistent global graph nodes — (N, 3) device tensor in the odom
        # frame, with a parallel (N,) tensor of stable per-node IDs.
        self.global_nodes = torch.empty(
            (0, 3), dtype=torch.float32, device=self._device)
        self.global_ids = torch.empty(
            (0,), dtype=torch.long, device=self._device)
        self._next_node_id = 0
        self._node_birth_frame: dict = {}  # node_id -> frame_count when first added
        # Edge set: write-once, dedup'd by sorted global-ID pair.
        # _node_degrees enforces the max_edges_per_node cap.
        from collections import defaultdict as _defaultdict
        self.global_edges: set = set()
        self._node_degrees: dict = _defaultdict(int)
        # Cached probability heatmap from the most recent forward — used by
        # the per-frame edge builder.  None when running DETR mode.
        self._last_hm_prob: Optional[torch.Tensor] = None

        # ── Subscribers / publishers ──────────────────────────────────
        self.create_subscription(Odometry, odom_topic, self.odom_callback, 100)
        self.create_subscription(Image, image_topic, self.image_callback, 10)
        # BEST_EFFORT QoS so the subscription is compatible with both
        # RELIABLE and BEST_EFFORT (sensor-data) CameraInfo publishers.
        self.create_subscription(
            CameraInfo, str(gp('cam_info_topic')), self._camera_info_cb,
            qos_profile_sensor_data)
        self.nodes_pub = self.create_publisher(PointCloud2, '~/graph_nodes', 1)
        # One TEXT marker per node showing its ID — IDs match the overlay.
        self.ids_pub = self.create_publisher(MarkerArray, '~/graph_node_ids', 1)
        # Marker LINE_LIST of all global edges (heatmap mode only).
        self.edges_pub = self.create_publisher(Marker, '~/graph_edges', 1)
        # Manual prune trigger — publish std_msgs/Empty to fire on demand.
        self.create_subscription(
            Empty, str(gp('prune_trigger_topic')), self._prune_trigger_cb, 1)

        self.get_logger().info(
            f'Ready | image={image_topic} odom={odom_topic} '
            f'frame={self.frame_id} device={self._device}'
        )

        # ── Graph-comparison artefacts ───────────────────────────────────
        self._run_name: str = str(gp('name')).strip() or 'e2e'
        _snap_dir = str(gp('snapshot_dir')).strip()
        _snap_path_str = str(gp('graph_snapshot_path')).strip()
        _csv_path_str  = str(gp('timing_csv_path')).strip()
        if not _snap_path_str and _snap_dir:
            _snap_path_str = str(Path(_snap_dir) / f'{self._run_name}.json')
        if not _csv_path_str and _snap_dir:
            _csv_path_str  = str(Path(_snap_dir) / f'{self._run_name}_timing.csv')

        self._save_snapshot_enabled = bool(gp('enable_graph_snapshot_save'))
        self._graph_snapshot_path: Optional[Path] = (
            Path(_snap_path_str).expanduser()
            if self._save_snapshot_enabled and _snap_path_str
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
            'num_local_nodes_pre', 'num_local_nodes_kept', 'num_new_nodes',
            'num_nodes', 'num_edges', 'num_new_edges',
            't_inference_ms', 't_merge_ms', 't_edges_ms', 't_other_ms',
            't_frame_total_ms',
        ]
        if bool(gp('enable_timing_csv')):
            if _csv_path_str:
                csv_path = Path(_csv_path_str).expanduser()
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                self._timing_csv_file = open(csv_path, 'w', newline='', buffering=1)
                self._timing_csv_writer = csv.DictWriter(
                    self._timing_csv_file, fieldnames=self._timing_csv_cols,
                )
                self._timing_csv_writer.writeheader()
                self.get_logger().info(f'Per-frame timing CSV → {csv_path}')

    # ──────────────────────────────────────────────────
    #  Model loading / inference
    # ──────────────────────────────────────────────────

    def _maybe_compile_model(self) -> None:
        """Wrap ``self._model`` with ``torch.compile`` if enabled.

        ``mode='reduce-overhead'`` uses CUDA graphs for low-latency
        repeated calls — safe because our input shape is locked to
        ``(1, 3, image_size, image_size)`` for the lifetime of the node.
        Falls back silently to eager mode if compile errors out (e.g.
        the model has dynamic Python control flow ``torch.compile`` can't
        trace).  First inference pays a 10-30 s JIT cost; subsequent
        calls amortise it.
        """
        if not self._enable_torch_compile or not self._on_cuda:
            return
        # DETR's backbone (osp_model) is loaded via importlib and never
        # registered in sys.modules, so TorchDynamo can't re-import it
        # during tracing.  Skip compile for DETR to avoid the error.
        if self._model_type == 'detr':
            self.get_logger().info(
                'torch.compile skipped for DETR model (osp_model not in sys.modules).'
            )
            return
        try:
            self._model = torch.compile(self._model, mode='reduce-overhead')
            self.get_logger().info(
                "torch.compile enabled (mode='reduce-overhead'); "
                'first inference will JIT (~10-30 s).'
            )
        except Exception as exc:  # pragma: no cover  - defensive
            self.get_logger().warning(
                f'torch.compile unavailable / failed: {exc}; running eager.'
            )

    def _load_detr_model(self, repo_path: str, ckpt_path: str) -> None:
        """Build NavGraphDETR and load the checkpoint.

        Inlines the e2e repo's ``train.load_model`` so we only depend on
        ``model.py`` — importing ``train`` would also drag in training-only
        deps (tensorboard, matplotlib) that this node never needs.
        """
        if repo_path and os.path.isdir(repo_path) and repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        try:
            from model import NavGraphDETR  # noqa: WPS433  (runtime import)
        except Exception as exc:  # pragma: no cover  - defensive
            raise RuntimeError(
                f'Could not import NavGraphDETR from e2e repo at {repo_path!r}: {exc}'
            ) from exc

        self.get_logger().info(f'Loading e2e model from {ckpt_path} ...')
        ckpt = torch.load(ckpt_path, map_location=self._device, weights_only=False)
        a = ckpt.get('args', {}) or {}

        self._model = NavGraphDETR(
            variant=a.get('encoder', 'vits'),
            num_queries=a.get('num_queries', 400),
            d_model=a.get('d_model', 256),
            nhead=a.get('nhead', 8),
            num_decoder_layers=a.get('num_decoder_layers', 4),
            dim_feedforward=a.get('dim_feedforward', 1024),
            dropout=a.get('dropout', 0.1),
            freeze_encoder=False,
        ).to(self._device)

        # Remap pos_head indices: checkpoints trained without dropout used
        # layer indices (2, 4); the current model has dropout layers (3, 6).
        _ph_remap = {
            'pos_head.2.weight': 'pos_head.3.weight',
            'pos_head.2.bias':   'pos_head.3.bias',
            'pos_head.4.weight': 'pos_head.6.weight',
            'pos_head.4.bias':   'pos_head.6.bias',
        }
        state = {_ph_remap.get(k, k): v for k, v in ckpt['model'].items()}
        self._model.load_state_dict(state)
        self._model.eval()
        self._maybe_compile_model()

        # Kept as device tensors so denormalisation stays on the GPU.
        pos_stats = ckpt['pos_stats']
        self._pos_mean_t = torch.as_tensor(
            pos_stats['mean'], dtype=torch.float32, device=self._device)
        self._pos_std_t = torch.as_tensor(
            pos_stats['std'], dtype=torch.float32, device=self._device)

        self._ckpt_image_size = int(a.get('image_size', 280))
        self._ckpt_crop_top_frac = float(a.get('crop_top_frac', 0.0))
        self.get_logger().info(
            f'DETR model loaded. pos_mean={self._pos_mean_t.tolist()} '
            f'pos_std={self._pos_std_t.tolist()}'
        )

    def _load_heatmap_model(
        self,
        repo_path: str,
        ckpt_path: str,
        sample_threshold: float,
        sample_min_dist: float,
        sample_window: int,
    ) -> None:
        """Build HeatmapNavModel and load the checkpoint.

        Mirrors ``heatmap_model.train.load_model`` inline so we don't pull in
        the training-only dependencies.  Also caches the two helper functions
        used at inference: ``sample_nodes_from_heatmap`` (Poisson-disk pixel
        sampler) and ``sample_pos_at_pixels`` (bilinear sampler on the per-
        pixel 3-D position map).
        """
        if repo_path and os.path.isdir(repo_path) and repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        try:
            from model import HeatmapNavModel, sample_pos_at_pixels  # noqa: WPS433
            from heatmap_utils import sample_nodes_from_heatmap  # noqa: WPS433
        except Exception as exc:  # pragma: no cover  - defensive
            raise RuntimeError(
                f'Could not import HeatmapNavModel from heatmap repo at '
                f'{repo_path!r}: {exc}'
            ) from exc

        self.get_logger().info(f'Loading heatmap model from {ckpt_path} ...')
        ckpt = torch.load(ckpt_path, map_location=self._device, weights_only=False)
        a = ckpt.get('args', {}) or {}

        self._model = HeatmapNavModel(
            variant=a.get('encoder', 'vits'),
            freeze_encoder=False,
            dropout=a.get('dropout', 0.0),
            head_mid=a.get('head_mid', 128),
        ).to(self._device)
        self._model.load_state_dict(ckpt['model'])
        self._model.eval()
        self._maybe_compile_model()

        pos_stats = ckpt['pos_stats']
        self._pos_mean_t = torch.as_tensor(
            pos_stats['mean'], dtype=torch.float32, device=self._device)
        self._pos_std_t = torch.as_tensor(
            pos_stats['std'], dtype=torch.float32, device=self._device)

        self._ckpt_image_size = int(a.get('image_size', 518))
        self._ckpt_crop_top_frac = float(a.get('crop_top_frac', 0.0))

        # Sampler config: param overrides win, else use the checkpoint defaults.
        self._heatmap_size = int(a.get('heatmap_size', 296))
        self._sample_threshold = float(sample_threshold)
        self._sample_min_dist = (
            sample_min_dist if sample_min_dist > 0.0
            else float(a.get('sample_min_dist', 8.0))
        )
        self._sample_window = (
            sample_window if sample_window > 0
            else int(a.get('sample_window', 1))
        )
        self._sample_nodes_fn = sample_nodes_from_heatmap
        self._sample_pos_fn = sample_pos_at_pixels

        self.get_logger().info(
            f'Heatmap model loaded. pos_mean={self._pos_mean_t.tolist()} '
            f'pos_std={self._pos_std_t.tolist()} '
            f'heatmap_size={self._heatmap_size} '
            f'sample[thr={self._sample_threshold} '
            f'min_dist={self._sample_min_dist} window={self._sample_window}]'
        )

    def _preprocess(self, rgb: np.ndarray) -> torch.Tensor:
        """HxWx3 uint8 RGB -> (1,3,S,S) normalised tensor.

        All steps (upload, top-crop, bilinear resize, ImageNet normalize)
        run on the GPU.  Only the raw uint8 RGB buffer crosses host→device,
        which is half the cost of uploading a normalized float32 tensor.

        Output is bilinearly interpolated and ImageNet-normalized using
        the same constants as the original PIL pipeline, so the model
        sees the same statistics it was trained on.
        """
        t = torch.from_numpy(rgb).to(
            self._device, dtype=torch.uint8, non_blocking=True,
        )  # (H, W, 3) uint8
        t = t.permute(2, 0, 1).contiguous().float().unsqueeze(0).div_(255.0)
        if self._crop_top_frac > 0.0:
            crop_top_px = int(self._crop_top_frac * t.shape[2])
            if crop_top_px > 0:
                t = t[:, :, crop_top_px:, :]
        t = F.interpolate(
            t, size=(self._image_size, self._image_size),
            mode='bilinear', align_corners=False,
        )
        return (t - self._mean_t) / self._std_t

    def _infer_local_nodes(self, rgb: np.ndarray) -> torch.Tensor:
        """Run the active model and return (M, 3) node positions as a device
        tensor in the model camera frame (x_fwd, y_down, z_left).

        Dispatches on ``model_type``; the forward-range gate is applied here
        so it covers both branches uniformly.  The monocular model has no
        depth / wall awareness and hallucinates nodes through walls / beyond
        dead ends; ``max_node_range`` clips them before the graph sees them.
        """
        image_t = self._preprocess(rgb)
        if self._model_type == 'detr':
            nodes = self._infer_detr(image_t)
        else:
            nodes = self._infer_heatmap(image_t)

        # Forward-range gate (x = forward depth in the model camera frame).
        if self.max_node_range > 0.0:
            nodes = nodes[nodes[:, 0] <= self.max_node_range]
        return nodes

    def _infer_detr(self, image_t: torch.Tensor) -> torch.Tensor:
        """NavGraphDETR forward: per-query (logit, 3-D pos).  Keeps queries
        above ``score_threshold`` and denormalises in metric camera units.

        Forward is wrapped in ``torch.amp.autocast(dtype=float16)`` on CUDA
        — typically 1.5-2× speedup with no visible quality loss on the
        ViT-based encoders we use here.  Disable with
        ``-p enable_amp_autocast:=false`` if numerical issues appear.
        """
        # DETR has no heatmap — the edge builder will see this and skip.
        self._last_hm_prob = None
        use_amp = self._enable_amp_autocast and self._on_cuda
        with torch.no_grad(), torch.amp.autocast(
            device_type='cuda', dtype=torch.float16, enabled=use_amp,
        ):
            logits, pos = self._model(image_t)
        # Cast results back to FP32 before downstream numerics so the
        # denormalisation (* std + mean) stays well-conditioned.
        logits = logits.float()
        pos = pos.float()
        keep = torch.sigmoid(logits[0]) >= self.score_threshold
        return pos[0][keep] * self._pos_std_t + self._pos_mean_t

    def _infer_heatmap(self, image_t: torch.Tensor) -> torch.Tensor:
        """HeatmapNavModel forward:
          1. Predict a dense traversability heatmap + a per-pixel 3-D map.
          2. Upsample the heatmap to ``heatmap_size`` and Poisson-disk sample
             evenly-spaced pixels with score >= sample_threshold.
          3. Bilinearly sample the 3-D position map at those pixels and
             denormalise.  Returns the (M, 3) device tensor of metric nodes.
        Matches the heatmap repo's validate.py inference path exactly.

        Only the model forward is autocast'd — the F.interpolate +
        sigmoid below stays in FP32 to keep the heatmap precise for the
        sampler.
        """
        use_amp = self._enable_amp_autocast and self._on_cuda
        with torch.no_grad():
            with torch.amp.autocast(
                device_type='cuda', dtype=torch.float16, enabled=use_amp,
            ):
                hm_logits, pos_map, _, _ = self._model(image_t)
            hm_prob = torch.sigmoid(F.interpolate(
                hm_logits.float(),
                size=(self._heatmap_size, self._heatmap_size),
                mode='bilinear', align_corners=False,
            ))
            pos_map = pos_map.float()
        # Cache for the per-frame edge builder.
        self._last_hm_prob = hm_prob
        # Sampler returns (x, y) pixel coords in the upsampled heatmap.
        hm_np = hm_prob[0, 0].detach().cpu().numpy()
        pix = self._sample_nodes_fn(
            hm_np,
            threshold=self._sample_threshold,
            min_dist=self._sample_min_dist,
            window=self._sample_window,
        )
        if len(pix) == 0:
            return torch.empty((0, 3), dtype=torch.float32, device=self._device)

        # (x, y) px -> [-1, 1] grid_sample coords, then bilinear sample.
        pix_t = torch.from_numpy(pix).to(self._device)
        grid = (2.0 * pix_t / float(self._heatmap_size) - 1.0).float()
        with torch.no_grad():
            pred_norm = self._sample_pos_fn(pos_map[0].float(), grid)  # (M, 3)
        return pred_norm * self._pos_std_t + self._pos_mean_t

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
        """Closest-in-time odometry message within ``odom_match_max_dt``."""
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

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        """Latch the camera intrinsics from the first valid CameraInfo
        message (ignored if cam_fx/fy/cx/cy params already set K)."""
        if self._cam_K is not None:
            return
        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        if K[0, 0] <= 0.0:
            return
        self._cam_K = K
        self.get_logger().info(
            f'Camera intrinsics from {msg.header.frame_id}: '
            f'fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}'
        )

    def _decode_image(self, msg: Image) -> Optional[np.ndarray]:
        """ROS Image -> HxWx3 uint8 RGB. Returns None on decode failure."""
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

    def image_callback(self, msg: Image) -> None:
        self.frame_count += 1
        if (self.frame_count % self.process_every_n) != 0:
            return

        t_img = stamp_to_sec(msg.header.stamp)
        odom = self._find_pose_at(t_img)
        if odom is None:
            self.get_logger().warn(
                f'No odometry within {self.odom_match_max_dt}s of image '
                f't={t_img:.3f} (buf size {len(self.odom_buf)})',
                throttle_duration_sec=2.0,
            )
            return

        rgb = self._decode_image(msg)
        if rgb is None:
            return

        try:
            self._process(rgb, odom, msg.header.stamp)
        except Exception as exc:  # pragma: no cover  - defensive
            import traceback
            self.get_logger().error(
                f'image_callback failed: {exc}\n{traceback.format_exc()}',
                throttle_duration_sec=2.0,
            )

    # ──────────────────────────────────────────────────
    #  Core processing
    # ──────────────────────────────────────────────────

    def _process(self, rgb: np.ndarray, odom: Odometry, stamp) -> None:
        t_frame_start = time.perf_counter()

        # 1) Local nodes from the model — model camera frame (x_fwd,y_down,z_left).
        t0 = time.perf_counter()
        local_cam = self._infer_local_nodes(rgb)  # (M, 3) device tensor
        # GPU sync so the inference timing reflects the actual GPU work and
        # not just a kernel-launch return.  Cheap (≈1 cudaEvent) and only
        # firing on the inference path.
        if local_cam.is_cuda:
            torch.cuda.synchronize()
        t_inference_ms = (time.perf_counter() - t0) * 1000.0
        n_local_pre = int(local_cam.shape[0])

        # 2) camera -> optical -> base_link -> odom.  The 4x4 chain is tiny;
        #    build it on the host, then apply it to all nodes in one matmul.
        p = odom.pose.pose.position
        q = odom.pose.pose.orientation
        T_odom_from_base = _pq_to_se3(
            np.array([p.x, p.y, p.z], dtype=np.float64),
            np.array([q.x, q.y, q.z, q.w], dtype=np.float64),
        )
        T_odom_from_cam = T_odom_from_base @ self._T_base_from_opt @ _se3_rot(_R_CAM_TO_OPT)
        R = torch.as_tensor(
            T_odom_from_cam[:3, :3], dtype=torch.float32, device=self._device)
        t = torch.as_tensor(
            T_odom_from_cam[:3, 3], dtype=torch.float32, device=self._device)
        local_odom = local_cam @ R.T + t  # (M, 3) device tensor

        # 3) Merge local nodes into the persistent global graph.
        t0 = time.perf_counter()
        local_cam_kept, local_ids, n_added = self._merge_local_nodes(
            local_cam, local_odom)
        if local_odom.is_cuda:
            torch.cuda.synchronize()
        t_merge_ms = (time.perf_counter() - t0) * 1000.0
        n_local_kept = int(local_cam_kept.shape[0])

        # 4) Build per-frame edges among the surviving local nodes (heatmap
        #    mode only) and fold them into the global edge set.
        t0 = time.perf_counter()
        n_new_edges = self._build_edges_for_frame(
            local_cam_kept, local_ids, rgb.shape[:2])
        t_edges_ms = (time.perf_counter() - t0) * 1000.0

        self.get_logger().info(
            f'frame {self.frame_count} local={local_odom.shape[0]} '
            f'added={n_added} global={self.global_nodes.shape[0]} '
            f'edges+={n_new_edges} total_edges={len(self.global_edges)}'
        )

        # ── Per-frame timing CSV row ─────────────────────────────────────
        if self._timing_csv_writer is not None:
            t_frame_total_ms = (time.perf_counter() - t_frame_start) * 1000.0
            core_sum = t_inference_ms + t_merge_ms + t_edges_ms
            t_other_ms = max(0.0, t_frame_total_ms - core_sum)
            row = {
                'frame_index':          int(self.frame_count),
                'frame_timestamp_sec':  stamp_to_sec(stamp),
                'num_local_nodes_pre':  n_local_pre,
                'num_local_nodes_kept': n_local_kept,
                'num_new_nodes':        int(n_added),
                'num_nodes':            int(self.global_nodes.shape[0]),
                'num_edges':            int(len(self.global_edges)),
                'num_new_edges':        int(n_new_edges),
                't_inference_ms':       float(t_inference_ms),
                't_merge_ms':           float(t_merge_ms),
                't_edges_ms':           float(t_edges_ms),
                't_other_ms':           float(t_other_ms),
                't_frame_total_ms':     float(t_frame_total_ms),
            }
            try:
                self._timing_csv_writer.writerow(row)
            except Exception as exc:
                self.get_logger().error(
                    f'timing-csv writerow failed: {exc}', throttle_duration_sec=5.0,
                )

        # 5) Optional periodic pruning: drop nodes that have zero edges.
        if (self._prune_enabled
                and (self.frame_count % self._prune_every_n) == 0):
            n_pruned = self._prune_isolated_nodes()
            if n_pruned > 0:
                self.get_logger().info(
                    f'[prune periodic] removed {n_pruned} isolated nodes; '
                    f'global={self.global_nodes.shape[0]} '
                    f'edges={len(self.global_edges)}'
                )

        # 6) Publish all global nodes (+ ID markers + edges for RViz).
        self._publish_nodes(stamp)
        self._publish_node_ids(stamp)
        self._publish_edges(stamp)

        # 7) Debug: every Nth frame, project this frame's local predictions
        #    onto the camera image and save a labelled overlay.  Off by
        #    default — enable with -p save_overlay:=true.
        if (self._save_overlay_enabled
                and (self.frame_count % self._debug_every_n) == 0):
            self._save_overlay(rgb, local_cam)

        # 8) Global-graph viz: project all global nodes/edges onto the current
        #    camera image and save two PNGs.  Off by default.
        if (self._save_global_viz_enabled
                and (self.frame_count % self._global_viz_every_n) == 0):
            self._save_global_viz(rgb, T_odom_from_cam)

    def _merge_local_nodes(
        self,
        local_cam: torch.Tensor,
        local_odom: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Vectorised GPU merge — no Python loops, mirrors the cdist-based
        merge in ``nav_graph``'s ``tensor_merge_local_nodes_gpu``.

        Two passes, both as a single ``torch.cdist``:
          1. intra-frame dedup — drop a local node if an earlier-indexed
             local node lies within ``merge_node_distance`` of it;
          2. global merge — for each (deduped) local node, snap to the
             nearest existing global node if within ``merge_node_distance``,
             otherwise assign a fresh ID and append to the global graph.

        Returns ``(local_cam_kept, local_ids, n_added)``:
          * ``local_cam_kept`` — surviving local-node camera-frame positions
            (same intra-frame dedup mask applied as to ``local_odom``).
          * ``local_ids`` — global ID each surviving local node maps to
            (existing ID if merged, fresh ID if new).
          * ``n_added`` — number of fresh nodes appended this frame.
        """
        if local_odom.shape[0] == 0:
            empty_ids = torch.empty((0,), dtype=torch.long, device=self._device)
            return local_cam, empty_ids, 0

        md = self.merge_node_distance

        # Pass 1: intra-frame dedup.  close.tril(-1)[i, j] is True iff j < i
        # and node j is within md of node i -> node i is a duplicate.
        d_self = torch.cdist(local_odom, local_odom)
        is_dup = (d_self < md).tril(-1).any(dim=1)
        keep = ~is_dup
        local_odom = local_odom[keep]
        local_cam = local_cam[keep]
        m_total = int(local_odom.shape[0])

        # Pass 2: per-node assignment vs. the existing global graph.
        if self.global_nodes.shape[0] > 0:
            d_glob = torch.cdist(local_odom, self.global_nodes)
            min_vals, min_idx = d_glob.min(dim=1)
            merged_mask = min_vals < md
        else:
            merged_mask = torch.zeros(m_total, dtype=torch.bool, device=self._device)
            min_idx = torch.zeros(m_total, dtype=torch.long, device=self._device)

        # Allocate the per-local-node ID tensor and fill the merged half.
        local_ids = torch.empty(m_total, dtype=torch.long, device=self._device)
        if merged_mask.any():
            local_ids[merged_mask] = self.global_ids[min_idx[merged_mask]]

        # Fresh IDs for the new-half; append positions + IDs to the global graph.
        new_mask = ~merged_mask
        n_added = int(new_mask.sum().item())
        if n_added > 0:
            new_ids = torch.arange(
                self._next_node_id, self._next_node_id + n_added,
                device=self._device, dtype=torch.long)
            self._next_node_id += n_added
            local_ids[new_mask] = new_ids
            self.global_nodes = torch.cat(
                [self.global_nodes, local_odom[new_mask]], dim=0)
            self.global_ids = torch.cat([self.global_ids, new_ids], dim=0)
            for nid in new_ids.cpu().tolist():
                self._node_birth_frame[int(nid)] = self.frame_count

        return local_cam, local_ids, n_added

    def _build_proximity_edges(
        self,
        local_cam: torch.Tensor,
        local_ids: torch.Tensor,
    ) -> int:
        """DETR-mode proximity edge builder.

        Connects all pairs of this frame's nodes within ``max_edge_distance``
        (3-D camera frame), shortest first, respecting ``max_edges_per_node``.
        No traversability check — pure proximity.

        Returns the number of newly added global edges.
        """
        m = int(local_cam.shape[0])
        d = torch.cdist(local_cam, local_cam)
        iu, ju = torch.triu_indices(m, m, offset=1, device=self._device)
        pair_d = d[iu, ju]
        keep = pair_d <= self.max_edge_distance
        if not keep.any():
            return 0
        pi = iu[keep]
        pj = ju[keep]
        pair_d = pair_d[keep]
        order = torch.argsort(pair_d)
        pi = pi[order].cpu().numpy()
        pj = pj[order].cpu().numpy()
        ids_np = local_ids.cpu().numpy()

        cap = self.max_edges_per_node
        n_new = 0
        for li, lj in zip(pi.tolist(), pj.tolist()):
            a_id = int(ids_np[li])
            b_id = int(ids_np[lj])
            if a_id == b_id:
                continue
            key = (a_id, b_id) if a_id < b_id else (b_id, a_id)
            if key in self.global_edges:
                continue
            if cap > 0 and (self._node_degrees[a_id] >= cap
                            or self._node_degrees[b_id] >= cap):
                continue
            self.global_edges.add(key)
            self._node_degrees[a_id] += 1
            self._node_degrees[b_id] += 1
            n_new += 1
        return n_new

    def _build_edges_for_frame(
        self,
        local_cam: torch.Tensor,
        local_ids: torch.Tensor,
        img_hw: Tuple[int, int],
    ) -> int:
        """Edge builder for both model modes.

        Heatmap mode: samples N points along each candidate line, projects to
        the heatmap, and keeps the pair only if the fraction of in-bounds
        samples clearing ``edge_traversability_threshold`` meets
        ``edge_line_min_pass_fraction``.

        DETR mode: no traversability map — connects pairs within
        ``max_edge_distance`` by proximity only, shortest candidates first,
        respecting ``max_edges_per_node``.

        Returns the number of *newly* added global edges.
        """
        m = int(local_cam.shape[0])
        if m < 2:
            return 0

        if self._model_type == 'detr':
            return self._build_proximity_edges(local_cam, local_ids)

        if self._last_hm_prob is None or self._cam_K is None:
            return 0

        # ── 1. candidate pairs within max_edge_distance, upper triangle only.
        d = torch.cdist(local_cam, local_cam)
        iu, ju = torch.triu_indices(m, m, offset=1, device=self._device)
        pair_d = d[iu, ju]
        keep = pair_d <= self.max_edge_distance
        if not keep.any():
            return 0
        pi = iu[keep]
        pj = ju[keep]
        pair_d = pair_d[keep]
        K = int(pi.shape[0])

        # ── 2. sample N points along each line in the camera frame.
        N = self.edge_line_samples
        a = local_cam[pi]                                       # (K, 3)
        b = local_cam[pj]                                       # (K, 3)
        ts = torch.linspace(0.0, 1.0, N, device=self._device).view(1, N, 1)
        line = a.unsqueeze(1) * (1.0 - ts) + b.unsqueeze(1) * ts  # (K, N, 3)
        flat = line.reshape(-1, 3)                              # (K*N, 3)
        x = flat[:, 0]
        y = flat[:, 1]
        z = flat[:, 2]
        behind = x <= 0.1
        x_safe = x.clamp(min=1e-3)

        # ── 3. project camera-frame line samples to heatmap pixel coords.
        K_mat = self._cam_K
        fx = float(K_mat[0, 0]); fy = float(K_mat[1, 1])
        cx = float(K_mat[0, 2]); cy = float(K_mat[1, 2])
        H_img, W_img = img_hw
        crop_top_px = int(self._crop_top_frac * H_img)
        H_visible = max(1, H_img - crop_top_px)
        hm_size = self._heatmap_size

        u = cx + fx * (-z) / x_safe                             # original-image px
        v_after_crop = (cy + fy * y / x_safe) - float(crop_top_px)
        # Re-scale to model-input (= heatmap_size pixels at heatmap_size grid).
        u_hm = u * (hm_size / float(W_img))
        v_hm = v_after_crop * (hm_size / float(H_visible))
        gx = 2.0 * u_hm / float(hm_size) - 1.0
        gy = 2.0 * v_hm / float(hm_size) - 1.0
        # In-bounds = in front of camera *and* projection lands inside the
        # heatmap.  Out-of-bounds samples are dropped from the vote (they
        # don't count for or against the edge) so near-image-edge lines
        # whose perspective bends a sample just off the heatmap aren't
        # silently rejected.
        in_bounds = (~behind) & (gx >= -1.0) & (gx <= 1.0) \
                              & (gy >= -1.0) & (gy <= 1.0)
        grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)

        # ── 4. one grid_sample for all K*N points; reshape, vote per line.
        vals = F.grid_sample(
            self._last_hm_prob, grid,
            mode='bilinear', padding_mode='zeros', align_corners=False,
        ).view(K, N)
        in_bounds = in_bounds.view(K, N)
        # Fraction-passing: of the *in-bounds* samples, what fraction clear
        # threshold?  Strict-min (the old behaviour) is recovered by
        # setting edge_line_min_pass_fraction=1.0.  This is far more robust
        # to the natural heatmap softness at the image boundary.
        passes = in_bounds & (vals >= self.edge_traversability_threshold)
        n_in = in_bounds.sum(dim=1)
        n_pass = passes.sum(dim=1)
        frac = n_pass.float() / n_in.clamp(min=1).float()
        edge_ok = (n_in >= self.edge_line_min_in_bounds) \
                  & (frac >= self.edge_line_min_pass_fraction)
        if not edge_ok.any():
            return 0

        # ── 5. sort surviving candidates by length (shortest first),
        #       then add to the global set respecting per-node degree caps.
        pi_ok = pi[edge_ok]
        pj_ok = pj[edge_ok]
        d_ok = pair_d[edge_ok]
        order = torch.argsort(d_ok)
        pi_ok = pi_ok[order].cpu().numpy()
        pj_ok = pj_ok[order].cpu().numpy()
        ids_np = local_ids.cpu().numpy()

        cap = self.max_edges_per_node
        n_new = 0
        for li, lj in zip(pi_ok.tolist(), pj_ok.tolist()):
            a_id = int(ids_np[li])
            b_id = int(ids_np[lj])
            if a_id == b_id:
                continue  # both locals snapped to the same global node
            key = (a_id, b_id) if a_id < b_id else (b_id, a_id)
            if key in self.global_edges:
                continue
            if cap > 0 and (self._node_degrees[a_id] >= cap
                            or self._node_degrees[b_id] >= cap):
                continue
            self.global_edges.add(key)
            self._node_degrees[a_id] += 1
            self._node_degrees[b_id] += 1
            n_new += 1
        return n_new

    def _publish_edges(self, stamp) -> None:
        """Publish the global edge set as a Marker LINE_LIST (odom frame).
        Edge endpoints are looked up from ``global_nodes`` via the parallel
        ``global_ids`` tensor and lifted by ``viz_z_offset`` (viz-only)."""
        if not self.global_edges or self.global_nodes.shape[0] == 0:
            # Always publish a DELETE so RViz clears stale edges when the
            # graph is reset upstream.  Cheap.
            return
        ids_np = self.global_ids.detach().cpu().numpy().astype(np.int64)
        pos_np = self.global_nodes.detach().cpu().numpy().astype(np.float32, copy=True)
        pos_np[:, 2] += self.viz_z_offset

        # ID -> row-index lookup over the dense ID range.
        max_id = int(ids_np.max()) + 1
        id_to_idx = -np.ones(max_id, dtype=np.int64)
        id_to_idx[ids_np] = np.arange(ids_np.shape[0], dtype=np.int64)

        m = Marker()
        m.header = Header(stamp=stamp, frame_id=self.frame_id)
        m.ns = 'graph_edges'
        m.id = 0
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD
        m.scale.x = 0.04
        m.color.r = 0.2
        m.color.g = 1.0
        m.color.b = 0.2
        m.color.a = 0.8
        m.pose.orientation.w = 1.0

        for a_id, b_id in self.global_edges:
            ia = id_to_idx[a_id]
            ib = id_to_idx[b_id]
            if ia < 0 or ib < 0:
                continue
            pa = pos_np[ia]
            pb = pos_np[ib]
            m.points.append(Point(x=float(pa[0]), y=float(pa[1]), z=float(pa[2])))
            m.points.append(Point(x=float(pb[0]), y=float(pb[1]), z=float(pb[2])))
        self.edges_pub.publish(m)

    def _prune_isolated_nodes(self) -> int:
        """Remove every global node whose edge degree is 0.

        Stable IDs let us prune from the storage tensors without breaking
        the edge set — surviving nodes keep their IDs, and the edge set
        only ever references nodes with degree >= 1.  ``_next_node_id``
        keeps marching forward, so pruned IDs are never reused.

        Returns the number of nodes removed.
        """
        n = int(self.global_nodes.shape[0])
        if n == 0:
            return 0
        ids_list = self.global_ids.cpu().tolist()
        keep_py = [self._node_degrees.get(int(nid), 0) > 0 for nid in ids_list]
        n_pruned = keep_py.count(False)
        if n_pruned == 0:
            return 0
        # Drop the per-node degree and birth-frame entries for pruned IDs.
        for nid, k in zip(ids_list, keep_py):
            if not k:
                self._node_degrees.pop(int(nid), None)
                self._node_birth_frame.pop(int(nid), None)
        keep_mask = torch.tensor(
            keep_py, dtype=torch.bool, device=self._device)
        self.global_nodes = self.global_nodes[keep_mask]
        self.global_ids = self.global_ids[keep_mask]
        return n_pruned

    def _prune_trigger_cb(self, _msg: Empty) -> None:
        """Manual-prune callback — always active regardless of the periodic
        switch.  Trigger from any terminal with::

            ros2 topic pub --once \
              /odin_nav_graph_e2e_node/prune_now std_msgs/msg/Empty {}
        """
        n_pruned = self._prune_isolated_nodes()
        self.get_logger().info(
            f'[prune trigger] removed {n_pruned} isolated nodes; '
            f'global={self.global_nodes.shape[0]} '
            f'edges={len(self.global_edges)}'
        )

    def _save_overlay(self, rgb: np.ndarray, local_cam: torch.Tensor) -> None:
        """Project this frame's *local* predictions onto the camera image and
        save a labelled overlay (debug only).

        ``local_cam`` is the raw per-frame model output in the model camera
        frame (x_fwd, y_down, z_left) — projected directly with the camera
        intrinsics, no odom transform involved, so this is a clean check of
        the model output itself.  Projection:
            u = cx + fx * (-z) / x        v = cy + fy * y / x

        The image is drawn on a padded canvas so predictions that project
        *outside* the image stay visible in the grey margin instead of being
        silently dropped.  Each node is labelled with its forward depth x.
        """
        if self._cam_K is None:
            self.get_logger().warn(
                'No camera intrinsics yet — set cam_fx/fy/cx/cy params or wait '
                'for CameraInfo; skipping overlay.',
                throttle_duration_sec=5.0,
            )
            return
        n = int(local_cam.shape[0])
        if n == 0:
            return

        K = self._cam_K
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        h, w = rgb.shape[:2]
        pad = self._overlay_pad

        # Padded canvas: original image centred in a grey margin.
        canvas = np.full((h + 2 * pad, w + 2 * pad, 3), 40, dtype=np.uint8)
        canvas[pad:pad + h, pad:pad + w] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        H_c, W_c = canvas.shape[:2]

        # Local nodes are already in the model camera frame — project directly.
        pc = local_cam.detach().cpu().numpy().astype(np.float64)

        # Project only nodes in front of the camera (x = forward depth).
        front = pc[:, 0] > 0.1
        xf, yf, zf = pc[front, 0], pc[front, 1], pc[front, 2]
        u = cx + fx * (-zf) / xf + pad
        v = cy + fy * yf / xf + pad
        within = (u >= 0) & (u < W_c) & (v >= 0) & (v < H_c)

        n_in_image = 0
        # cv2 has no batched draw — loop over the in-canvas nodes only.
        for uu, vv, xx in zip(u[within], v[within], xf[within]):
            ui, vi = int(round(uu)), int(round(vv))
            in_img = (pad <= ui < pad + w) and (pad <= vi < pad + h)
            n_in_image += int(in_img)
            # Yellow inside the real image, orange in the off-image margin.
            colour = (0, 255, 255) if in_img else (0, 165, 255)
            cv2.circle(canvas, (ui, vi), 7, colour, -1)
            cv2.circle(canvas, (ui, vi), 7, (0, 0, 0), 1)
            label = f'{xx:.1f}m'
            org = (ui + 9, vi + 4)
            cv2.putText(canvas, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1, cv2.LINE_AA)

        path = self._debug_dir / f'overlay_{self.frame_count:06d}.png'
        cv2.imwrite(str(path), canvas)
        self.get_logger().info(
            f'[overlay] frame {self.frame_count}: local={n} '
            f'in_front={int(front.sum())} in_canvas={int(within.sum())} '
            f'in_image={n_in_image} -> {path}'
        )

    def _save_global_viz(self, rgb: np.ndarray, T_odom_from_cam: np.ndarray) -> None:
        """Project ALL global nodes/edges into the current camera image and
        save two PNGs to ``global_viz_dir``:

            nodes_{frame:06d}.png  — RGB + every global node inside the image
            edges_{frame:06d}.png  — same nodes + edges where at least one
                                     endpoint is inside the image (line is
                                     clipped at the canvas boundary)

        Nodes that project outside the image boundary are not drawn.
        Edges between two out-of-image nodes are also skipped; edges that
        connect an in-image node to an out-of-image node are drawn and
        clipped at the image edge by cv2.line.

        ``T_odom_from_cam`` is the 4×4 SE3 from the current frame's
        ``_process()`` call; its inverse maps global (odom) positions back
        into the model camera frame (x_fwd, y_down, z_left).
        """
        if self._cam_K is None:
            self.get_logger().warn(
                'No camera intrinsics yet — skipping global viz.',
                throttle_duration_sec=5.0,
            )
            return
        n = int(self.global_nodes.shape[0])
        if n == 0:
            return

        # Odom -> model camera frame.
        T_cam_from_odom = np.linalg.inv(T_odom_from_cam)
        R_co = T_cam_from_odom[:3, :3]
        t_co = T_cam_from_odom[:3, 3]

        pos_odom = self.global_nodes.detach().cpu().numpy().astype(np.float64)  # (N,3)
        pos_cam = pos_odom @ R_co.T + t_co                                       # (N,3)

        K = self._cam_K
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        h, w = rgb.shape[:2]

        # Project (model cam: x=fwd, y=down, z=left).
        xc = pos_cam[:, 0]
        yc = pos_cam[:, 1]
        zc = pos_cam[:, 2]
        in_front = xc > 0.1
        x_safe = np.where(in_front, xc, 1.0)
        u_px = cx + fx * (-zc) / x_safe   # (N,)
        v_px = cy + fy * yc  / x_safe     # (N,)

        # A node is "in image" only if it's in front AND projects inside [0,w)x[0,h).
        u_i = np.round(u_px).astype(np.int32)
        v_i = np.round(v_px).astype(np.int32)
        in_image = in_front & (u_i >= 0) & (u_i < w) & (v_i >= 0) & (v_i < h)
        in_bounds = in_image.copy()  # frustum-only mask, unaffected by later filters

        # ── Optional visibility filter ────────────────────────────────────────
        if self._viz_filter_mode == 'distance':
            # Camera position in odom frame is the translation column of T_odom_from_cam.
            cam_pos = T_odom_from_cam[:3, 3]
            dist = np.linalg.norm(pos_odom - cam_pos, axis=1)
            in_image = in_image & (dist <= self._viz_max_distance)
        elif self._viz_filter_mode == 'recency':
            ids_np_local = self.global_ids.detach().cpu().numpy().astype(np.int64)
            cutoff = self.frame_count - self._viz_recency_frames
            recent = np.array(
                [self._node_birth_frame.get(int(nid), 0) >= cutoff
                 for nid in ids_np_local],
                dtype=bool,
            )
            in_image = in_image & recent

        # ── Pixel-space NMS ───────────────────────────────────────────────────
        # Greedily remove the node with the most pixel-space conflicts until no
        # two surviving nodes are closer than viz_min_pixel_dist pixels.
        # Tie-break: remove the lower global-ID node (older node loses).
        if self._viz_min_pixel_dist > 0.0:
            cands = np.where(in_image)[0]           # indices into global arrays
            nc = len(cands)
            if nc > 1:
                cu = u_i[cands].astype(np.float32)
                cv_ = v_i[cands].astype(np.float32)
                # Pairwise pixel distances — O(nc^2) but nc is small in practice
                du = cu[:, None] - cu[None, :]      # (nc, nc)
                dv = cv_[:, None] - cv_[None, :]
                pdist = np.sqrt(du * du + dv * dv)
                thr = self._viz_min_pixel_dist
                # conflict[i] = set of local indices j != i with pdist[i,j] < thr
                conflicts = [
                    set(np.where((pdist[i] < thr) & (np.arange(nc) != i))[0])
                    for i in range(nc)
                ]
                ids_np_cands = self.global_ids.detach().cpu().numpy().astype(np.int64)[cands]
                removed = set()
                while True:
                    active_deg = {
                        i: len(conflicts[i] - removed)
                        for i in range(nc) if i not in removed
                    }
                    max_deg = max(active_deg.values(), default=0)
                    if max_deg == 0:
                        break
                    # Among nodes with the highest conflict count, remove the
                    # one with the smallest global ID (older node).
                    worst = min(
                        (i for i, d in active_deg.items() if d == max_deg),
                        key=lambda i: int(ids_np_cands[i]),
                    )
                    removed.add(worst)
                keep_local = np.array(
                    [i for i in range(nc) if i not in removed], dtype=np.int64)
                surviving = cands[keep_local] if len(keep_local) else np.array([], dtype=np.int64)
                in_image = np.zeros(len(in_image), dtype=bool)
                in_image[surviving] = True

        base_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        canvas_n = base_bgr.copy()
        canvas_e = base_bgr.copy()

        ids_np = self.global_ids.detach().cpu().numpy().astype(np.int64)
        max_id = int(ids_np.max()) + 1
        id_to_idx = -np.ones(max_id, dtype=np.int64)
        id_to_idx[ids_np] = np.arange(n, dtype=np.int64)

        # Hide in-image nodes that have no edge connecting to another in-image node.
        if len(self.global_edges) > 0:
            has_visible_edge = np.zeros(n, dtype=bool)
            for a_id, b_id in self.global_edges:
                if a_id >= max_id or b_id >= max_id:
                    continue
                ia = int(id_to_idx[a_id])
                ib = int(id_to_idx[b_id])
                if ia < 0 or ib < 0:
                    continue
                if in_image[ia] and in_image[ib]:
                    has_visible_edge[ia] = True
                    has_visible_edge[ib] = True
            in_image = in_image & has_visible_edge

        _NODE_BGR    = (255, 170, 0)    # #00AAFF in BGR
        _BORDER_BGR  = (0, 0, 0)
        _NODE_RADIUS = 16
        _BORDER      = 4
        _EDGE_COLOR  = (255, 255, 255)  # white
        _EDGE_ALPHA  = 0.6

        # Draw edges onto a scratch layer, then blend at _EDGE_ALPHA so
        # they appear semi-transparent.  An edge is drawn only if at least one
        # endpoint is a visible (in_image) node.  If the other endpoint also
        # projects into the image (in_bounds) it must also be in_image;
        # otherwise it's simply off-screen and cv2.line clips it naturally.
        edge_layer = canvas_e.copy()
        for a_id, b_id in self.global_edges:
            if a_id >= max_id or b_id >= max_id:
                continue
            ia = int(id_to_idx[a_id])
            ib = int(id_to_idx[b_id])
            if ia < 0 or ib < 0:
                continue
            # At least one endpoint must be a drawn node.
            if not in_image[ia] and not in_image[ib]:
                continue
            # If the other endpoint projects into the image frame but its node
            # was filtered out, skip — it would produce a dangling edge.
            if in_image[ia] and in_bounds[ib] and not in_image[ib]:
                continue
            if in_image[ib] and in_bounds[ia] and not in_image[ia]:
                continue
            if not in_front[ia] or not in_front[ib]:
                continue
            cv2.line(edge_layer,
                     (int(u_i[ia]), int(v_i[ia])),
                     (int(u_i[ib]), int(v_i[ib])),
                     _EDGE_COLOR, 2)
        canvas_e = cv2.addWeighted(edge_layer, _EDGE_ALPHA, canvas_e, 1.0 - _EDGE_ALPHA, 0)

        # Draw only in-image nodes on both canvases (nodes sit on top of edges).
        for i in np.where(in_image)[0]:
            ui, vi = int(u_i[i]), int(v_i[i])
            for canvas in (canvas_n, canvas_e):
                cv2.circle(canvas, (ui, vi), _NODE_RADIUS + _BORDER, _BORDER_BGR, -1)
                cv2.circle(canvas, (ui, vi), _NODE_RADIUS, _NODE_BGR, -1)

        path_n = self._global_viz_dir / f'nodes_{self.frame_count:06d}.png'
        path_e = self._global_viz_dir / f'edges_{self.frame_count:06d}.png'
        cv2.imwrite(str(path_n), canvas_n)
        cv2.imwrite(str(path_e), canvas_e)
        self.get_logger().info(
            f'[global_viz] frame {self.frame_count}: '
            f'{int(in_image.sum())} nodes in image / {n} total, '
            f'{len(self.global_edges)} edges -> {self._global_viz_dir}'
        )

    def _publish_node_ids(self, stamp) -> None:
        """Publish one TEXT_VIEW_FACING marker per global node showing its ID,
        so the RViz 3-D view can be cross-referenced with the debug overlay."""
        n = int(self.global_nodes.shape[0])
        if n == 0:
            return
        glob = self.global_nodes.detach().cpu().numpy()
        ids = self.global_ids.detach().cpu().numpy()

        arr = MarkerArray()
        for pos, nid in zip(glob, ids):
            m = Marker()
            m.header = Header(stamp=stamp, frame_id=self.frame_id)
            m.ns = 'node_ids'
            m.id = int(nid)
            m.type = Marker.TEXT_VIEW_FACING
            m.action = Marker.ADD
            m.pose.position.x = float(pos[0])
            m.pose.position.y = float(pos[1])
            # viz_z_offset matches the PointCloud2 lift; +0.25 floats the text
            # marker above the dot in the same lifted plane.
            m.pose.position.z = float(pos[2]) + self.viz_z_offset + 0.25
            m.pose.orientation.w = 1.0
            m.scale.z = 0.25  # text height (m)
            m.color.r = 1.0
            m.color.g = 1.0
            m.color.b = 1.0
            m.color.a = 1.0
            m.text = str(int(nid))
            arr.markers.append(m)
        self.ids_pub.publish(arr)

    def _publish_nodes(self, stamp) -> None:
        n = int(self.global_nodes.shape[0])
        if n == 0:
            return
        # Single device->host transfer of the whole graph.
        packed = np.empty((n, 4), dtype=np.float32)
        packed[:, :3] = self.global_nodes.detach().cpu().numpy()
        packed[:, 2] += self.viz_z_offset  # viz-only lift above the floor
        packed[:, 3] = 1.0  # constant intensity

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
        self.nodes_pub.publish(cloud)


def _se3_rot(R: np.ndarray) -> np.ndarray:
    """4x4 SE3 with rotation R and zero translation."""
    se3 = np.eye(4, dtype=np.float64)
    se3[:3, :3] = R
    return se3


# ─────────────────────────────────────────────────────────────────────
#  Graph-comparison artefacts (e2e side)
# ─────────────────────────────────────────────────────────────────────

def _save_final_graph_snapshot_e2e(node: 'OdinNavGraphE2ENode') -> None:
    """Dump the e2e node's current global graph to JSON, schema-compatible
    with ``nav_graph_node.save_final_graph_snapshot``.

    Node ``type`` is always 1 (free_space) — the e2e pipeline doesn't
    classify frontiers.  Edge ``weight`` is the Euclidean distance between
    endpoints in odom, computed at save time (the edge set itself stores
    only ID pairs).
    """
    if node._graph_snapshot_path is None:
        return
    try:
        n = int(node.global_nodes.shape[0])
        positions = (
            node.global_nodes.detach().cpu().numpy()
            if n > 0 else np.empty((0, 3), dtype=np.float32)
        )
        ids = (
            node.global_ids.detach().cpu().numpy()
            if n > 0 else np.empty((0,), dtype=np.int64)
        )
        id_to_idx = {int(ids[i]): i for i in range(n)}

        nodes_out: list = []
        for i in range(n):
            nodes_out.append({
                'id':       int(ids[i]),
                'type':     1,  # free_space — no frontier classification
                'position': [
                    float(positions[i, 0]),
                    float(positions[i, 1]),
                    float(positions[i, 2]),
                ],
                'scores':   {},
            })

        edges_out: list = []
        for pair in node.global_edges:
            a, b = int(pair[0]), int(pair[1])
            ia = id_to_idx.get(a, -1)
            ib = id_to_idx.get(b, -1)
            if ia < 0 or ib < 0:
                w = None  # endpoint pruned but pair lingered
            else:
                w = _finite_or_none(float(np.linalg.norm(positions[ia] - positions[ib])))
            edges_out.append({'node_id_0': a, 'node_id_1': b, 'weight': w})

        from datetime import datetime, timezone
        payload = {
            'method':       node._run_name,
            'saved_at_utc': datetime.now(timezone.utc).isoformat(),
            'frame_count':  int(node.frame_count),
            'num_nodes':    n,
            'num_edges':    len(edges_out),
            'layer_names':  [],
            'nodes':        nodes_out,
            'edges':        edges_out,
        }
        tmp = node._graph_snapshot_path.with_suffix('.tmp')
        with open(tmp, 'w') as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, node._graph_snapshot_path)
        node.get_logger().info(
            f'Final graph snapshot written: {node._graph_snapshot_path} '
            f'({n} nodes, {len(edges_out)} edges)'
        )
    except Exception as exc:
        import traceback
        node.get_logger().error(
            f'save_final_graph_snapshot failed: {exc}\n{traceback.format_exc()}'
        )


def _close_timing_csv_e2e(node: 'OdinNavGraphE2ENode') -> None:
    try:
        if node._timing_csv_file is not None and not node._timing_csv_file.closed:
            node._timing_csv_file.flush()
            node._timing_csv_file.close()
    except Exception:
        pass


def main(args=None):
    rclpy.init(args=args)
    node = OdinNavGraphE2ENode()
    try:
        with suppress(KeyboardInterrupt):
            rclpy.spin(node)
    finally:
        if node.prune_on_shutdown:
            try:
                n_pruned = node._prune_isolated_nodes()
                node.get_logger().info(
                    f'[prune shutdown] removed {n_pruned} isolated nodes; '
                    f'global={node.global_nodes.shape[0]} '
                    f'edges={len(node.global_edges)}'
                )
            except Exception:
                pass
        try:
            _save_final_graph_snapshot_e2e(node)
        except Exception:
            pass
        try:
            _close_timing_csv_e2e(node)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
