#!/usr/bin/env python3
"""nav_graph_node_e2e_unfiltered.py

Identical to nav_graph_node_e2e but with every filter that can reduce
the node set unconditionally disabled:

  max_node_range      → 0.0  forward-range gate removed
  score_threshold     → 0.0  all DETR queries kept regardless of confidence
  _sample_threshold   → 0.0  all heatmap pixels sampled
  merge_node_distance → 0.0  no intra-frame dedup, no global-graph merging;
                             every local prediction becomes a fresh global node
  pruning             → off  no periodic or shutdown pruning

Both e2e_debug_overlays (local) and e2e_debug_viz (global) are always saved.
All node rendering uses the same large-circle style. Saves run on a background
thread so they never block the ROS executor and cause frame drops.

ros2 run odin_nav_graph nav_graph_node_e2e_unfiltered --ros-args \
    -p model_checkpoint:=<path> \
    -p e2e_repo_path:=<path> \
    -p name:=e2e_unfiltered
"""

from __future__ import annotations

import queue
import threading
import traceback
from contextlib import suppress

import cv2
import numpy as np
import rclpy

from odin_nav_graph.nav_graph_node_e2e import (
    OdinNavGraphE2ENode,
    _close_timing_csv_e2e,
    _save_final_graph_snapshot_e2e,
)

_NODE_BGR    = (255, 170, 0)
_BORDER_BGR  = (0, 0, 0)
_NODE_RADIUS = 16
_BORDER      = 4
_EDGE_COLOR  = (255, 255, 255)
_EDGE_ALPHA  = 0.6


class OdinNavGraphE2EUnfilteredNode(OdinNavGraphE2ENode):
    """E2E nav graph node with all node-count-limiting filters disabled.

    Every method is inherited from OdinNavGraphE2ENode.  After the parent
    __init__ runs (which reads ROS2 parameters and loads the model), this
    __init__ unconditionally overrides the filtering thresholds to their
    pass-everything values.

    Both _save_overlay (local) and _save_global_viz (global) are always
    enabled and run on a single background worker thread so they never block
    the ROS executor.  The main callback only copies the necessary arrays and
    enqueues a payload dict — the cv2 drawing and imwrite happen off-thread.
    If the worker falls behind, the oldest queued item is dropped to keep
    memory bounded.
    """

    def __init__(self) -> None:
        super().__init__()

        # ── Disable every filter that can reduce the node set ────────────

        self.max_node_range = 0.0
        self.score_threshold = 0.0
        # These are intentionally NOT overridden — they respect whatever
        # -p values were passed:
        #   heatmap_sample_threshold, viz_min_pixel_dist,
        #   max_edge_distance, max_edges_per_node,
        #   edge_traversability_threshold, edge_line_samples,
        #   edge_line_min_pass_fraction, edge_line_min_in_bounds
        self.merge_node_distance = 0.0

        self._prune_enabled = False
        self.prune_on_shutdown = False

        # Disable distance/recency viz filtering only — pixel NMS stays
        # configurable via viz_min_pixel_dist param.
        self._viz_filter_mode = 'none'

        # Always save both local overlay and global viz.
        self._save_overlay_enabled = True

        # ── Background save worker ───────────────────────────────────────
        # Queue holds payload dicts; maxsize=8 so we only keep a small burst.
        self._save_queue: queue.Queue = queue.Queue(maxsize=8)
        self._save_thread = threading.Thread(
            target=self._save_worker, daemon=True, name='UnfilteredSave')
        self._save_thread.start()

        self.get_logger().info(
            '[unfiltered] All node filters disabled — '
            'max_node_range=0 score_threshold=0 '
            'merge_node_distance=0 pruning=off '
            'viz_filter=none has_visible_edge=off '
            'async_save=on'
        )

    # ── Public overrides: snapshot + enqueue (called on executor thread) ─────

    def _save_global_viz(self, rgb: np.ndarray, T_odom_from_cam: np.ndarray) -> None:
        if self._cam_K is None or int(self.global_nodes.shape[0]) == 0:
            return
        payload = {
            'type':              'global',
            'rgb':               rgb.copy(),
            'pos_odom':          self.global_nodes.detach().cpu().numpy().astype(np.float64),
            'ids':               self.global_ids.detach().cpu().numpy().astype(np.int64),
            'edges':             list(self.global_edges),
            'T':                 T_odom_from_cam.copy(),
            'K':                 self._cam_K.copy(),
            'frame':             self.frame_count,
            'out_dir':           self._global_viz_dir,
            'viz_min_pixel_dist': self._viz_min_pixel_dist,
        }
        self._enqueue(payload)

    def _save_overlay(self, rgb: np.ndarray, local_cam) -> None:
        if self._cam_K is None:
            return
        pc = local_cam.detach().cpu().numpy().astype(np.float64)
        if pc.shape[0] == 0:
            return
        payload = {
            'type':    'overlay',
            'rgb':     rgb.copy(),
            'pc':      pc,
            'K':       self._cam_K.copy(),
            'frame':   self.frame_count,
            'out_dir': self._debug_dir,
        }
        self._enqueue(payload)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _enqueue(self, payload: dict) -> None:
        """Non-blocking enqueue — drop oldest item if the queue is full."""
        while True:
            try:
                self._save_queue.put_nowait(payload)
                return
            except queue.Full:
                try:
                    self._save_queue.get_nowait()
                except queue.Empty:
                    pass

    def _save_worker(self) -> None:
        """Background thread: render and write images from queued payloads."""
        while True:
            payload = self._save_queue.get()
            if payload is None:  # shutdown sentinel
                self._save_queue.task_done()
                break
            try:
                if payload['type'] == 'global':
                    self._render_global_viz(payload)
                elif payload['type'] == 'overlay':
                    self._render_overlay(payload)
            except Exception as exc:
                try:
                    self.get_logger().error(
                        f'[unfiltered save] render failed: {exc}\n'
                        f'{traceback.format_exc()}',
                        throttle_duration_sec=5.0,
                    )
                except Exception:
                    pass
            finally:
                self._save_queue.task_done()

    def _render_global_viz(self, p: dict) -> None:
        pos_odom          = p['pos_odom']   # (N, 3)
        ids_np            = p['ids']        # (N,)
        edges             = p['edges']
        T                 = p['T']
        K                 = p['K']
        rgb               = p['rgb']
        frame             = p['frame']
        out_dir           = p['out_dir']
        viz_min_pixel_dist = p['viz_min_pixel_dist']
        n = pos_odom.shape[0]

        T_cam_from_odom = np.linalg.inv(T)
        R_co = T_cam_from_odom[:3, :3]
        t_co = T_cam_from_odom[:3, 3]
        pos_cam = pos_odom @ R_co.T + t_co

        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        h, w = rgb.shape[:2]

        xc, yc, zc = pos_cam[:, 0], pos_cam[:, 1], pos_cam[:, 2]
        in_front = xc > 0.1
        x_safe = np.where(in_front, xc, 1.0)
        u_i = np.round(cx + fx * (-zc) / x_safe).astype(np.int32)
        v_i = np.round(cy + fy *   yc  / x_safe).astype(np.int32)
        in_image = in_front & (u_i >= 0) & (u_i < w) & (v_i >= 0) & (v_i < h)

        # ── Pixel-space NMS (mirrors parent, only active when param > 0) ──
        if viz_min_pixel_dist > 0.0:
            cands = np.where(in_image)[0]
            nc = len(cands)
            if nc > 1:
                cu  = u_i[cands].astype(np.float32)
                cv_ = v_i[cands].astype(np.float32)
                du = cu[:, None] - cu[None, :]
                dv = cv_[:, None] - cv_[None, :]
                pdist = np.sqrt(du * du + dv * dv)
                thr = viz_min_pixel_dist
                conflicts = [
                    set(np.where((pdist[i] < thr) & (np.arange(nc) != i))[0])
                    for i in range(nc)
                ]
                ids_np_cands = ids_np[cands]
                removed: set = set()
                while True:
                    active_deg = {
                        i: len(conflicts[i] - removed)
                        for i in range(nc) if i not in removed
                    }
                    max_deg = max(active_deg.values(), default=0)
                    if max_deg == 0:
                        break
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

        max_id = int(ids_np.max()) + 1
        id_to_idx = -np.ones(max_id, dtype=np.int64)
        id_to_idx[ids_np] = np.arange(n, dtype=np.int64)

        base_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        canvas_n = base_bgr.copy()
        canvas_e = base_bgr.copy()

        edge_layer = canvas_e.copy()
        for a_id, b_id in edges:
            if a_id >= max_id or b_id >= max_id:
                continue
            ia = int(id_to_idx[a_id])
            ib = int(id_to_idx[b_id])
            if ia < 0 or ib < 0:
                continue
            if not in_image[ia] and not in_image[ib]:
                continue
            if not in_front[ia] or not in_front[ib]:
                continue
            cv2.line(edge_layer,
                     (int(u_i[ia]), int(v_i[ia])),
                     (int(u_i[ib]), int(v_i[ib])),
                     _EDGE_COLOR, 2)
        canvas_e = cv2.addWeighted(edge_layer, _EDGE_ALPHA, canvas_e, 1.0 - _EDGE_ALPHA, 0)

        for i in np.where(in_image)[0]:
            ui, vi = int(u_i[i]), int(v_i[i])
            for canvas in (canvas_n, canvas_e):
                cv2.circle(canvas, (ui, vi), _NODE_RADIUS + _BORDER, _BORDER_BGR, -1)
                cv2.circle(canvas, (ui, vi), _NODE_RADIUS, _NODE_BGR, -1)

        cv2.imwrite(str(out_dir / f'nodes_{frame:06d}.png'), canvas_n)
        cv2.imwrite(str(out_dir / f'edges_{frame:06d}.png'), canvas_e)
        try:
            self.get_logger().info(
                f'[global_viz] frame {frame}: '
                f'{int(in_image.sum())} nodes in image / {n} total, '
                f'{len(edges)} edges -> {out_dir}'
            )
        except Exception:
            pass

    def _render_overlay(self, p: dict) -> None:
        pc      = p['pc']   # (M, 3) local nodes in camera frame
        K       = p['K']
        rgb     = p['rgb']
        frame   = p['frame']
        out_dir = p['out_dir']
        n = pc.shape[0]

        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        h, w = rgb.shape[:2]

        xc, yc, zc = pc[:, 0], pc[:, 1], pc[:, 2]
        in_front = xc > 0.1
        x_safe = np.where(in_front, xc, 1.0)
        u_i = np.round(cx + fx * (-zc) / x_safe).astype(np.int32)
        v_i = np.round(cy + fy *   yc  / x_safe).astype(np.int32)
        in_image = in_front & (u_i >= 0) & (u_i < w) & (v_i >= 0) & (v_i < h)

        canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for i in np.where(in_image)[0]:
            ui, vi = int(u_i[i]), int(v_i[i])
            cv2.circle(canvas, (ui, vi), _NODE_RADIUS + _BORDER, _BORDER_BGR, -1)
            cv2.circle(canvas, (ui, vi), _NODE_RADIUS, _NODE_BGR, -1)

        cv2.imwrite(str(out_dir / f'nodes_{frame:06d}.png'), canvas)
        try:
            self.get_logger().info(
                f'[overlay] frame {frame}: '
                f'{int(in_image.sum())} local nodes in image / {n} total '
                f'-> {out_dir}'
            )
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = OdinNavGraphE2EUnfilteredNode()
    try:
        with suppress(KeyboardInterrupt):
            rclpy.spin(node)
    finally:
        # Signal save worker to flush remaining items and exit.
        node._save_queue.put(None)
        node._save_thread.join(timeout=15.0)

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
