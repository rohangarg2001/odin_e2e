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

import os
import sys
from collections import deque
from contextlib import suppress
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
import torch
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
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

        self.declare_parameter(
            'model_checkpoint',
            '/home/rohang73/ASL/e2e_rgb_nav_graph/four_longrange_size280_overfit/epoch_999.pth',
        )
        self.declare_parameter(
            'e2e_repo_path', '/home/rohang73/ASL/e2e_rgb_nav_graph',
        )
        # 0.0 => take image_size / crop_top_frac from the checkpoint args.
        self.declare_parameter('image_size', 0)
        self.declare_parameter('crop_top_frac', -1.0)
        self.declare_parameter('score_threshold', 0.7)

        # A local node within this distance (m, XYZ) of an existing global
        # node is treated as the same node and dropped.
        self.declare_parameter('merge_node_distance', 0.5)

        # Forward-range gate: drop model predictions farther than this many
        # metres ahead (model camera x = forward depth).  The monocular RGB
        # model has no depth / wall awareness and hallucinates nodes through
        # walls and beyond dead ends; this clips them before they reach the
        # graph.  0.0 disables the gate.
        self.declare_parameter('max_node_range', 6.0)

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

        # Static camera(optical)->base_link extrinsic (Odin Nav Stack defaults).
        self.declare_parameter('cam_base_tx', -0.0042)
        self.declare_parameter('cam_base_ty',  0.0328)
        self.declare_parameter('cam_base_tz',  0.0005)
        self.declare_parameter('cam_base_qx', -0.4951)
        self.declare_parameter('cam_base_qy',  0.5048)
        self.declare_parameter('cam_base_qz', -0.4996)
        self.declare_parameter('cam_base_qw',  0.5005)

        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        image_topic = str(gp('image_topic'))
        odom_topic = str(gp('odom_topic'))
        self.frame_id = str(gp('frame_id'))
        self.score_threshold = float(gp('score_threshold'))
        self.merge_node_distance = float(gp('merge_node_distance'))
        self.max_node_range = float(gp('max_node_range'))
        self.odom_buffer_seconds = float(gp('odom_buffer_seconds'))
        self.odom_match_max_dt = float(gp('odom_match_max_dt'))
        self.process_every_n = max(1, int(gp('process_every_n')))
        self._debug_every_n = max(1, int(gp('debug_overlay_every_n')))
        self._overlay_pad = max(0, int(gp('debug_overlay_pad')))

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

        # ── Load the e2e model ────────────────────────────────────────
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt_path = str(gp('model_checkpoint'))
        repo_path = str(gp('e2e_repo_path'))
        self._load_e2e_model(repo_path, ckpt_path)

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
        self._debug_dir.mkdir(parents=True, exist_ok=True)
        self.get_logger().info(
            f'Debug overlays -> {self._debug_dir} '
            f'(every {self._debug_every_n} processed frames)'
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

        self.get_logger().info(
            f'Ready | image={image_topic} odom={odom_topic} '
            f'frame={self.frame_id} device={self._device}'
        )

    # ──────────────────────────────────────────────────
    #  Model loading / inference
    # ──────────────────────────────────────────────────

    def _load_e2e_model(self, repo_path: str, ckpt_path: str) -> None:
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

        # Kept as device tensors so denormalisation stays on the GPU.
        pos_stats = ckpt['pos_stats']
        self._pos_mean_t = torch.as_tensor(
            pos_stats['mean'], dtype=torch.float32, device=self._device)
        self._pos_std_t = torch.as_tensor(
            pos_stats['std'], dtype=torch.float32, device=self._device)

        self._ckpt_image_size = int(a.get('image_size', 280))
        self._ckpt_crop_top_frac = float(a.get('crop_top_frac', 0.0))
        self.get_logger().info(
            f'Model loaded. pos_mean={self._pos_mean_t.tolist()} '
            f'pos_std={self._pos_std_t.tolist()}'
        )

    def _preprocess(self, rgb: np.ndarray) -> torch.Tensor:
        """HxWx3 uint8 RGB -> (1,3,S,S) normalised tensor, matching training."""
        from PIL import Image as _PILImage

        img = _PILImage.fromarray(rgb)
        w, h = img.size
        crop_top_px = int(self._crop_top_frac * h)
        if crop_top_px > 0:
            img = img.crop((0, crop_top_px, w, h))
        img = img.resize((self._image_size, self._image_size), _PILImage.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0)
        return tensor.to(self._device)

    def _infer_local_nodes(self, rgb: np.ndarray) -> torch.Tensor:
        """Run the model and return (M, 3) node positions as a device tensor
        in the model camera frame (x_fwd, y_down, z_left).

        Predictions are gated by score, then by forward range
        (``max_node_range``) — the model has no depth awareness and predicts
        nodes through walls / beyond dead ends, so anything too far ahead is
        dropped before it can reach the graph.
        """
        image_t = self._preprocess(rgb)
        with torch.no_grad():
            logits, pos = self._model(image_t)
        keep = torch.sigmoid(logits[0]) >= self.score_threshold
        nodes = pos[0][keep].float() * self._pos_std_t + self._pos_mean_t

        # Forward-range gate (x = forward depth in the model camera frame).
        if self.max_node_range > 0.0:
            nodes = nodes[nodes[:, 0] <= self.max_node_range]
        return nodes

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
            self.get_logger().warning(
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
        # 1) Local nodes from the model — model camera frame (x_fwd,y_down,z_left).
        local_cam = self._infer_local_nodes(rgb)  # (M, 3) device tensor

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
        n_added = self._merge_local_nodes(local_odom)

        self.get_logger().info(
            f'frame {self.frame_count} local={local_odom.shape[0]} '
            f'added={n_added} global={self.global_nodes.shape[0]}'
        )

        # 4) Publish all global nodes (+ ID markers for RViz).
        self._publish_nodes(stamp)
        self._publish_node_ids(stamp)

        # 5) Debug: every Nth frame, project this frame's local predictions
        #    onto the camera image and save a labelled overlay.
        if (self.frame_count % self._debug_every_n) == 0:
            self._save_overlay(rgb, local_cam)

    def _merge_local_nodes(self, local_odom: torch.Tensor) -> int:
        """Vectorised GPU merge — no Python loops, mirrors the cdist-based
        merge in ``nav_graph``'s ``tensor_merge_local_nodes_gpu``.

        Two passes, both as a single ``torch.cdist``:
          1. intra-frame dedup — drop a local node if an earlier-indexed
             local node lies within ``merge_node_distance`` of it;
          2. global merge — drop a (deduped) local node if any existing
             global node lies within ``merge_node_distance``.
        Survivors are appended to ``self.global_nodes`` in one ``cat``.
        """
        if local_odom.shape[0] == 0:
            return 0

        md = self.merge_node_distance

        # Pass 1: intra-frame dedup.  close.tril(-1)[i, j] is True iff j < i
        # and node j is within md of node i -> node i is a duplicate.
        d_self = torch.cdist(local_odom, local_odom)
        is_dup = (d_self < md).tril(-1).any(dim=1)
        local_odom = local_odom[~is_dup]

        # Pass 2: merge against the existing global graph.
        if self.global_nodes.shape[0] > 0:
            d_glob = torch.cdist(local_odom, self.global_nodes)
            new_mask = d_glob.min(dim=1).values >= md
            new_nodes = local_odom[new_mask]
        else:
            new_nodes = local_odom

        # Assign a stable ID to each surviving node.
        m = int(new_nodes.shape[0])
        new_ids = torch.arange(
            self._next_node_id, self._next_node_id + m,
            device=self._device, dtype=torch.long)
        self._next_node_id += m

        self.global_nodes = torch.cat([self.global_nodes, new_nodes], dim=0)
        self.global_ids = torch.cat([self.global_ids, new_ids], dim=0)
        return m

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
            self.get_logger().warning(
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
            m.pose.position.z = float(pos[2]) + 0.25  # float text above node
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


def main(args=None):
    rclpy.init(args=args)
    node = OdinNavGraphE2ENode()
    with suppress(KeyboardInterrupt):
        rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
