#!/usr/bin/env python3
"""Example 10: batch-process a list of frame numbers and save
  - original RGB image
  - RGB image with nav-graph overlay (frontiers=yellow, free=blue)
  - JSON with visible node positions (camera frame: x-fwd, y-down, z-left),
    node IDs, types, pixel coords, and edges between visible nodes

All outputs go to /home/yash/ASL/rgb_to_graph/rgb_nav_graph_dataset
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path

import numpy as np
import cv2
import tqdm
from scipy.spatial.transform import Rotation as R
import zarr
import imageio
import open3d as o3d
from elevation_mapping_cupy import ElevationMap, Parameter
import elevation_mapping_cupy

from nav_graph import NavigationGraphBuilder, NavGraphConfig, ElevationMapConfig, FrontierConfig
from nav_graph.core.config import NODE_TYPE_FREE_SPACE, NODE_TYPE_FRONTIER



"""
MISSIONS


2024-10-01-11-29-55  2024-11-11-12-07-40  2024-11-15-16-41-14
2024-10-01-11-47-44  2024-11-11-12-42-47  2024-11-18-12-05-01
2024-10-01-12-00-49  2024-11-11-14-29-44  2024-11-18-13-22-14
2024-11-02-17-10-25  2024-11-11-16-14-23  2024-11-18-13-48-19
2024-11-02-17-18-32  2024-11-14-11-17-02  2024-11-18-15-46-05
2024-11-02-17-43-10  2024-11-14-12-01-26  2024-11-18-16-59-23
2024-11-02-21-12-51  2024-11-14-13-45-37  2024-11-18-17-13-09
2024-11-03-07-52-45  2024-11-14-14-36-02  2024-11-18-17-31-36
2024-11-03-07-57-34  2024-11-14-15-22-43  2024-11-25-14-57-08
2024-11-03-08-17-23  2024-11-14-16-04-09  2024-11-25-16-36-19
2024-11-03-13-51-43  2024-11-15-10-16-35  2024-12-03-13-15-38
2024-11-03-13-59-54  2024-11-15-11-18-14  2024-12-03-13-26-40
2024-11-04-10-57-34  2024-11-15-11-37-15  2024-12-09-09-34-43
2024-11-04-12-55-59  2024-11-15-12-06-03  2024-12-09-09-41-46
2024-11-04-13-07-13  2024-11-15-14-14-12  2024-12-09-11-28-28
2024-11-04-16-05-00  2024-11-15-14-43-52  2024-12-09-11-53-11

"""

mission_ids = [
    # "2024-12-09-11-53-11",
    # "2024-11-18-15-46-05",
    # "2024-11-04-10-57-34",
    # "2024-11-11-12-07-40",
    # "2024-11-15-16-41-14",
    # "2024-10-01-11-47-44",
    # "2024-11-11-12-42-47",
    # "2024-11-18-12-05-01"
    # "2024-10-01-12-00-49",
    # "2024-11-11-14-29-44"
    # "2024-11-18-13-22-14",
    # "2024-11-02-17-10-25",
    # "2024-11-11-16-14-23"
    # "2024-11-18-13-48-19",
    # "2024-11-02-17-18-32",
    # "2024-11-14-11-17-02", #TODO (clear and redownload)
    # "2024-11-02-17-43-10",
    # "2024-11-14-12-01-26"
    # "2024-11-18-16-59-23" #done
    # "2024-11-02-21-12-51", #done
    # "2024-11-14-13-45-37" #done
    # "2024-11-18-17-13-09", #done
    # "2024-11-03-07-52-45" #done
    # "2024-11-14-14-36-02", #done
    #   "2024-11-18-17-31-36" #done






]

# logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")


def pq_to_se3(p, q):
    se3 = np.eye(4, dtype=np.float32)
    if isinstance(q, dict) or (hasattr(q, "dtype") and q.dtype.names):
        se3[:3, :3] = R.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
        se3[:3, 3] = [p["x"], p["y"], p["z"]]
    else:
        se3[:3, :3] = R.from_quat(q).as_matrix()
        se3[:3, 3] = p
    return se3


def attrs_to_se3(attrs):
    return pq_to_se3(attrs["transform"]["translation"], attrs["transform"]["rotation"])


class FastGetClosestTf:
    def __init__(self, odom: zarr.Group):
        self.timestamps = odom["timestamp"][:]
        self.pose_pos = odom["pose_pos"][:]
        self.pose_orien = odom["pose_orien"][:]

    def __call__(self, timestamp: float) -> np.ndarray:
        idx = np.argmin(np.abs(self.timestamps - timestamp))
        return pq_to_se3(self.pose_pos[idx], self.pose_orien[idx])


class ElevationMapWrapper:
    def __init__(self, map_length: float = 16.0, resolution: float = 0.04):
        root = Path(elevation_mapping_cupy.__file__).parent
        param = Parameter(
            use_chainer=False,
            weight_file=root / "config/core/weights.dat",
            plugin_config_file=root / "config/core/plugin_config.yaml",
        )
        param.enable_drift_compensation = False
        param.subscriber = {
            "front_upper_depth": {
                "topic_name": "/integrated_depth",
                "data_type": "pointcloud",
            }
        }
        param.map_length = map_length
        param.resolution = resolution
        param.update()
        self._map = ElevationMap(param)
        self.resolution = resolution

    def integrate(self, pts, trans, rot, position_noise=0.0, orientation_noise=0.0):
        self._map.input_pointcloud(pts, ["x", "y", "z"], rot, trans, position_noise, orientation_noise)

    def move_to(self, trans, rot):
        self._map.move_to(trans, rot)

    def tick(self):
        self._map.update_variance()
        self._map.update_time()

    def get_elevation(self) -> np.ndarray:
        elevation = self._map.get_layer("elevation").get()
        is_valid = self._map.get_layer("is_valid").get()
        elevation[is_valid == 0] = np.nan
        return elevation[1:-1, 1:-1]


def hdr_front_optical_from_hdr_front() -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array(
        [[0.0, 0.0, 1.0],
         [0.0, 1.0, 0.0],
         [-1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    return T


def compute_odom_to_base(mission_root, tf_lookup, timestamp):
    base_to_box_base = pq_to_se3(
        mission_root["tf"].attrs["tf"]["box_base"]["translation"],
        mission_root["tf"].attrs["tf"]["box_base"]["rotation"],
    )
    dlio_world_to_hesai = tf_lookup(timestamp)
    odom_to_box_base = dlio_world_to_hesai @ attrs_to_se3(
        mission_root["hesai_points_undistorted"].attrs
    )
    odom_to_base = odom_to_box_base @ np.linalg.inv(base_to_box_base)
    return odom_to_base, base_to_box_base


def compute_odom_to_cam(mission_root, tf_lookup, timestamp):
    odom_to_base, base_to_box_base = compute_odom_to_base(mission_root, tf_lookup, timestamp)
    cam_to_box_base = attrs_to_se3(mission_root["hdr_front"].attrs)
    return odom_to_base @ base_to_box_base @ np.linalg.inv(cam_to_box_base)


def compute_odom_to_sensor(mission_root, tag, odom_to_base, base_to_box_base):
    if "depth_camera" in tag:
        sensor_to_base = attrs_to_se3(mission_root[tag].attrs)
        return odom_to_base @ np.linalg.inv(sensor_to_base)
    else:
        sensor_to_box_base = attrs_to_se3(mission_root[tag].attrs)
        return odom_to_base @ base_to_box_base @ np.linalg.inv(sensor_to_box_base)



_COLOUR_FRONTIER  = (0, 255, 255)   # yellow
_COLOUR_FREE      = (255, 0, 0)     # blue

def project_and_draw_nodes_fisheye(
    rgb_image,
    pos_cam,        # (N,3) in camera frame: x-fwd, y-down, z-left
    node_types,     # (N,)
    node_ids,       # (N,)
    K, D,
    img_w, img_h,
    output_path=None,
    radius=6,
    draw_text=False,
    text_stride=10,
):
    """Project nav-graph nodes onto fisheye image with type-based colouring.

    Returns
    -------
    vis_img : np.ndarray
    visible_nodes : list[dict]  – only nodes that land inside the image
        keys: id, type, position_cam, pixel
    """
    vis_img = rgb_image.copy()

    if pos_cam is None or len(pos_cam) == 0:
        return vis_img, []

    # Camera frame (x-fwd, y-down, z-left) → OpenCV frame (x-right, y-down, z-fwd)
    R_cam_to_cv = np.array([
        [0,  0, -1],
        [0,  1,  0],
        [1,  0,  0]
    ], dtype=np.float64)
    pos_cv = (R_cam_to_cv @ pos_cam.T).T

    # Keep points in front of camera
    z = pos_cv[:, 2]
    front_mask = z > 0.1
    pts_cv_front    = pos_cv[front_mask]
    pos_cam_front   = pos_cam[front_mask]
    types_front     = node_types[front_mask]
    ids_front       = node_ids[front_mask]
    orig_idx_front  = np.where(front_mask)[0]

    if len(pts_cv_front) == 0:
        return vis_img, []

    # Project
    rvec = np.zeros((3, 1))
    tvec = np.zeros((3, 1))
    img_pts, _ = cv2.fisheye.projectPoints(
        pts_cv_front.reshape(-1, 1, 3).astype(np.float64), rvec, tvec, K, D
    )
    img_pts = img_pts.reshape(-1, 2)

    u = img_pts[:, 0]
    v = img_pts[:, 1]
    inside = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)

    u_vis = u[inside].astype(int)
    v_vis = v[inside].astype(int)
    pos_cam_vis  = pos_cam_front[inside]
    types_vis    = types_front[inside]
    ids_vis      = ids_front[inside]

    # Draw
    for x, y, t in zip(u_vis, v_vis, types_vis):
        colour = _COLOUR_FRONTIER if t == NODE_TYPE_FRONTIER else _COLOUR_FREE
        radius = 7 if t == NODE_TYPE_FRONTIER else 5
        cv2.circle(vis_img, (x, y), int(radius), colour, -1)

    if draw_text:
        for i in range(0, len(u_vis), text_stride):
            x, y = u_vis[i], v_vis[i]
            X, Y, Z = pos_cam_vis[i]
            cv2.putText(
                vis_img, f"{X:.1f},{Y:.1f},{Z:.1f}",
                (x + 3, y - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA,
            )

    if output_path is not None:
        imageio.imwrite(output_path, vis_img)

    # Assemble visible-node list
    type_str = {NODE_TYPE_FREE_SPACE: "free_space", NODE_TYPE_FRONTIER: "frontier"}
    visible_nodes = [
        {
            "id":           int(node_id),
            "type":         type_str.get(int(t), "unknown"),
            "position_cam": pos_cam_vis[i].tolist(),   # [x-fwd, y-down, z-left]
            "pixel":        [int(u_vis[i]), int(v_vis[i])],
        }
        for i, (node_id, t) in enumerate(zip(ids_vis, types_vis))
    ]
    return vis_img, visible_nodes


# ─────────────────────────────────────────────────────────────────────────────
#  Per-frame processing
# ─────────────────────────────────────────────────────────────────────────────

def process_frame(
    img_idx,
    mission_root,
    tf_lookup,
    base_to_box_base,
    hdr_ts,
    K, D,
    img_w, img_h,
    mission_folder,
    output_dir,
    args,
):
    print(len(hdr_ts))
    if img_idx >= len(hdr_ts):
        print(f"  SKIP frame {img_idx}: out of range (max {len(hdr_ts) - 1})")
        return

    img_timestamp = hdr_ts[img_idx]
    img_path = mission_folder / "images" / "hdr_front" / f"{img_idx:06d}.jpeg"
    if not img_path.exists():
        print(f"  SKIP frame {img_idx}: image not found at {img_path}")
        return

    print(f"\n── Frame {img_idx} (t={img_timestamp:.3f}) ──")
    rgb_image = imageio.imread(img_path)

    # Save original RGB
    rgb_out = output_dir / f"rgb_{img_idx:06d}.png"
    imageio.imwrite(rgb_out, rgb_image)

    # Camera extrinsics
    odom_to_hdr_front = compute_odom_to_cam(mission_root, tf_lookup, img_timestamp)
    hdr_front_from_odom = np.linalg.inv(odom_to_hdr_front).astype(np.float64)
    cam_from_odom = (hdr_front_optical_from_hdr_front() @ hdr_front_from_odom).astype(np.float64)
    odom_to_cam = np.linalg.inv(cam_from_odom)
    cam_pos_odom = odom_to_cam[:3, 3]

    # Build elevation map fixed at camera position
    lidar_tags = ["livox_points_undistorted", "hesai_points_undistorted"]
    integration_settings = {
        "hesai_points_undistorted": (0.005, 95),
        "livox_points_undistorted": (0.015, 95),
    }

    resolution = args.resolution
    map_length = args.map_length
    emw = ElevationMapWrapper(map_length=map_length, resolution=resolution)
    cam_origin = np.array([cam_pos_odom[0], cam_pos_odom[1], 0.0], dtype=np.float32)
    emw.move_to(cam_origin, np.eye(3))
    emw.tick()

    all_frames = []
    for tag in lidar_tags:
        ts = mission_root[tag]["timestamp"][:]
        valid_counts = mission_root[tag]["valid"][:, 0]
        points_data = mission_root[tag]["points"]  # lazy zarr array — do NOT slice with [:] here
        if args.single_lidar_frame:
            # Use only the single closest frame to the image timestamp
            closest_i = int(np.argmin(np.abs(ts - img_timestamp)))
            all_frames.append((ts[closest_i], tag, closest_i, valid_counts[closest_i], points_data))
        else:
            for i in range(len(ts)):
                if abs(ts[i] - img_timestamp) <= args.time_window:
                    all_frames.append((ts[i], tag, i, valid_counts[i], points_data))

    all_frames.sort(key=lambda x: x[0])
    max_frames = min(args.max_frames, len(all_frames))
    mode_str = "single closest frame per sensor" if args.single_lidar_frame else f"window ±{args.time_window}s"
    print(f"  LiDAR frames ({mode_str}): {len(all_frames)} selected, processing up to {max_frames}")

    for timestamp, tag, elem_idx, valid_count, points_data in tqdm.tqdm(
        all_frames[:max_frames], desc="  Integrating LiDAR", leave=False
    ):
        if valid_count <= 0:
            continue
        points = points_data[elem_idx][:valid_count]
        max_range = integration_settings[tag][1]
        valid = np.linalg.norm(points[:, :2], axis=1) < max_range
        points = points[valid]
        if points.shape[0] == 0:
            continue
        dlio_tf = tf_lookup(timestamp)
        odom_to_box_base = dlio_tf @ attrs_to_se3(
            mission_root["hesai_points_undistorted"].attrs
        )
        odom_to_base = odom_to_box_base @ np.linalg.inv(base_to_box_base)
        odom_to_sensor = compute_odom_to_sensor(
            mission_root, tag, odom_to_base, base_to_box_base
        )
        emw.integrate(
            points,
            odom_to_sensor[:3, 3],
            odom_to_sensor[:3, :3],
            integration_settings[tag][0],
        )
        emw.tick()

    origin_x = float(cam_origin[0])
    origin_y = float(cam_origin[1])
    elevation = emw.get_elevation()

    # Crop elevation map to a smaller region around the camera.
    # origin_x/y is the center of the map, so a symmetric crop keeps it valid.
    if args.crop_size is not None:
        H, W = elevation.shape
        half = int(args.crop_size / args.resolution / 2)
        ch, cw = H // 2, W // 2
        r0, r1 = max(0, ch - half), min(H, ch + half)
        c0, c1 = max(0, cw - half), min(W, cw + half)
        elevation = elevation[r0:r1, c0:c1]
        print(f"  Elevation map cropped to {elevation.shape} "
              f"({args.crop_size}m x {args.crop_size}m)")

    # Build nav graph
    nav_config = NavGraphConfig(
        safety_distance=0.03,
        free_space_sampling_threshold=0.14,
        merge_node_distance=0.14,
        global_merge_distance=0.14,
        global_max_candidate_edge_distance=1.0,
        boundary_inflation_factor = 2.0,
        global_max_connections = 10,
        frontier=FrontierConfig(
            kernel_size=5,
            odom_proximity_threshold=0.1,
            max_edge_connectivity=9,
            minimum_points_in_cluster = 1,
            angular_gap_min_gap_deg = 140,
            use_angular_gap_filter=True
        ),
        elevation_map=ElevationMapConfig(
            gaussian_sigma=0.2,
            window_size=8,
            max_height_diff=0.25,
            max_slope=0.45,
            border_cells=30,
        ),
        device="cuda",
    )
    builder = NavigationGraphBuilder(config=nav_config)
    elevation_for_graph = np.flipud(np.fliplr(elevation))
    result = builder.update(
        grid=elevation_for_graph,
        resolution=resolution,
        origin_x=origin_x,
        origin_y=origin_y,
        input_type="elevation_map",
        detect_frontiers=True,
    )
    print(f"  Graph: {result.num_nodes} nodes, {result.num_edges} edges, "
          f"{result.num_frontiers} frontiers")

    if result.num_nodes == 0:
        print(f"  SKIP frame {img_idx}: no nav graph nodes generated")
        return

    # Extract and fix-up node positions (same rotation fix as example 09)
    np_data = result.to_numpy()
    pos   = np_data["node_positions"].copy()  # (N, 3) in odom frame
    types = np_data["node_types"]
    ids   = np_data["node_ids"]
    edges = np_data["edge_index"]             # (E, 2) — indices into pos

    pos[:, 2] *= -1
    Rz = R.from_euler('z', np.pi / 2).as_matrix()
    Ry = R.from_euler('y', np.pi).as_matrix()
    Rx = R.from_euler('x', 0).as_matrix()
    R_fix = Rx @ Ry @ Rz
    grid_origin = np.array([origin_x - origin_y, origin_y - origin_x, 0.0], dtype=np.float32)
    pos = (R_fix @ pos.T).T + grid_origin   # now in odom frame

    # Transform to camera frame (x-fwd, y-down, z-left)
    R_cam_odom = cam_from_odom[:3, :3]
    t_cam      = cam_from_odom[:3, 3]
    pos_cam = (R_cam_odom @ pos.T).T + t_cam

    # Overlay image
    overlay_path = output_dir / f"overlay_{img_idx:06d}.png"
    _, visible_nodes = project_and_draw_nodes_fisheye(
        rgb_image=rgb_image,
        pos_cam=pos_cam,
        node_types=types,
        node_ids=ids,
        K=K, D=D,
        img_w=img_w, img_h=img_h,
        output_path=overlay_path,
    )
    print(f"  Projected {len(visible_nodes)} nodes visible in image")

    # Build edge list restricted to visible nodes
    visible_id_set = {n["id"] for n in visible_nodes}
    id_to_idx = {int(nid): i for i, nid in enumerate(ids)}
    visible_edges = []
    for e in edges:
        id0 = int(ids[int(e[0])])
        id1 = int(ids[int(e[1])])
        if id0 in visible_id_set and id1 in visible_id_set:
            visible_edges.append({"node_id_0": id0, "node_id_1": id1})

    # Overlay image with edges
    overlay_edges_path = output_dir / f"overlay_edges_{img_idx:06d}.png"
    node_pixel = {n["id"]: tuple(n["pixel"]) for n in visible_nodes}
    overlay_edges_img = imageio.imread(overlay_path).copy()
    for e in visible_edges:
        p0 = node_pixel.get(e["node_id_0"])
        p1 = node_pixel.get(e["node_id_1"])
        if p0 and p1:
            cv2.line(overlay_edges_img, p0, p1, (0, 255, 0), 1, cv2.LINE_AA)
    imageio.imwrite(overlay_edges_path, overlay_edges_img)

    # Save JSON
    graph_data = {
        "frame_index": img_idx,
        "timestamp":   float(img_timestamp),
        "position_convention": "camera_frame_x_fwd_y_down_z_left",
        "config": {
            "mission_id": mission_root.store.path.split("/")[-2] if hasattr(mission_root.store, "path") else "unknown",
            "elevation_map": {
                "map_length":  args.map_length,
                "resolution":  args.resolution,
                "crop_size":   args.crop_size,
                "time_window": args.time_window,
                "max_frames":  args.max_frames,
                "single_lidar_frame": args.single_lidar_frame,
            },
            "lidar": {
                tag: {"position_noise": integration_settings[tag][0], "max_range": integration_settings[tag][1]}
                for tag in lidar_tags
            },
            "camera": {
            "K": K.tolist(),
            "D": D.tolist(),
            "width":  img_w,
            "height": img_h,
        },
        "nav_graph": {
                "safety_distance":                  nav_config.safety_distance,
                "free_space_sampling_threshold":    nav_config.free_space_sampling_threshold,
                "merge_node_distance":              nav_config.merge_node_distance,
                "global_merge_distance":            nav_config.global_merge_distance,
                "global_max_candidate_edge_distance": nav_config.global_max_candidate_edge_distance,
                "elevation_map": {
                    "gaussian_sigma":   nav_config.elevation_map.gaussian_sigma,
                    "window_size":      nav_config.elevation_map.window_size,
                    "max_height_diff":  nav_config.elevation_map.max_height_diff,
                    "max_slope":        nav_config.elevation_map.max_slope,
                    "border_cells":     nav_config.elevation_map.border_cells,
                },
                "frontier": {
                    "kernel_size":               nav_config.frontier.kernel_size,
                    "odom_proximity_threshold":  nav_config.frontier.odom_proximity_threshold,
                    "max_edge_connectivity":     nav_config.frontier.max_edge_connectivity,
                    "minimum_points_in_cluster": nav_config.frontier.minimum_points_in_cluster,
                    "angular_gap_min_gap_deg":   nav_config.frontier.angular_gap_min_gap_deg,
                    "use_angular_gap_filter":    nav_config.frontier.use_angular_gap_filter,
                },
            },
        },
        "nodes": visible_nodes,
        "edges": visible_edges,
    }
    json_path = output_dir / f"graph_{img_idx:06d}.json"
    with open(json_path, "w") as f:
        json.dump(graph_data, f, indent=2)

    print(f"  Saved: {rgb_out.name}, {overlay_path.name}, {overlay_edges_path.name}, {json_path.name}")

    # Explicitly free GPU memory before the next frame's allocations
    del emw, builder
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset download
# ─────────────────────────────────────────────────────────────────────────────

_DOWNLOAD_TOPICS = [
    "hdr_front",
    "livox_points_undistorted",
    "hesai_points_undistorted",
    "tf",
    "dlio_map_odometry",
]


def ensure_mission_downloaded(mission: str, dataset_folder: Path) -> None:
    """Download mission data from HuggingFace if not already present."""
    import shutil
    import tarfile
    data_dir = dataset_folder / mission / "data"
    # Consider downloaded if the zarr store has at least one group beyond .zgroup
    if data_dir.exists() and any(p for p in data_dir.iterdir() if not p.name.startswith(".")):
        return

    print(f"Mission {mission} not found locally — downloading from HuggingFace...")
    from huggingface_hub import snapshot_download

    allow_patterns = [f"{mission}/*.yaml", "*/.zgroup"]
    allow_patterns += [f"{mission}/*{topic}*" for topic in _DOWNLOAD_TOPICS]

    cache = snapshot_download(
        repo_id="leggedrobotics/grand_tour_dataset",
        allow_patterns=allow_patterns,
        repo_type="dataset",
    )

    print(f"Extracting to {dataset_folder}...")
    cache_path = Path(cache)
    for source_path in cache_path.rglob("*"):
        if not source_path.is_file():
            continue
        dest_path = dataset_folder / source_path.relative_to(cache_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.suffix == ".tar":
            with tarfile.open(source_path, "r") as tar:
                tar.extractall(path=dest_path.parent)
        else:
            shutil.copy2(source_path, dest_path)

    print("Download complete.\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def warmup_gpu():
    print("Warming up GPU...")
    try:
        import cupy as cp
        x = cp.zeros((512, 512), dtype=cp.float32)
        _ = (x @ x).sum()
        cp.cuda.Stream.null.synchronize()
    except Exception as e:
        print(f"  CuPy warmup skipped: {e}")
    try:
        import torch
        if torch.cuda.is_available():
            x = torch.zeros(512, 512, device="cuda")
            _ = (x @ x).sum()
            torch.cuda.synchronize()
    except Exception as e:
        print(f"  Torch warmup skipped: {e}")
    print("GPU warmup done.\n")


def run_mission(mission_id, args):
    dataset_folder = Path(args.data_dir).expanduser()
    ensure_mission_downloaded(mission_id, dataset_folder)
    mission_folder = dataset_folder / mission_id
    if not (mission_folder / "data").exists():
        print(f"ERROR: Mission data not found at {mission_folder}/data")
        return

    output_dir = Path("/home/yash/ASL/rgb_to_graph/trial_2_rgb_nav_graph_dataset_" + mission_id + "_mission")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Opening mission: {mission_id}")

    mission_root = zarr.open_group(mission_folder / "data", mode="r")
    tf_lookup = FastGetClosestTf(mission_root["dlio_map_odometry"])

    hdr_ts = mission_root["hdr_front"]["timestamp"][:]

    cam_info = mission_root["hdr_front"].attrs["camera_info"]
    K = np.array(cam_info["K"], dtype=np.float64).reshape(3, 3)
    D = np.array(cam_info["D"], dtype=np.float64)
    img_w, img_h = cam_info["width"], cam_info["height"]

    base_to_box_base = pq_to_se3(
        mission_root["tf"].attrs["tf"]["box_base"]["translation"],
        mission_root["tf"].attrs["tf"]["box_base"]["rotation"],
    )

    # frame_numbers = args.frame_numbers
    frame_numbers = [x for x in range(0, len(hdr_ts), 50)]
    print(f"Processing {len(frame_numbers)} frames: {frame_numbers}\n")

    for img_idx in frame_numbers:
        try:
            process_frame(
                img_idx=img_idx,
                mission_root=mission_root,
                tf_lookup=tf_lookup,
                base_to_box_base=base_to_box_base,
                hdr_ts=hdr_ts,
                K=K, D=D,
                img_w=img_w, img_h=img_h,
                mission_folder=mission_folder,
                output_dir=output_dir,
                args=args,
            )
        except Exception as e:
            print(f"  ERROR frame {img_idx}: {e}")

    print(f"Done with mission {mission_id}.\n")


def run_pipeline(args):
    warmup_gpu()
    for mission_id in mission_ids:
        print(f"\n{'='*60}\nMission: {mission_id}\n{'='*60}")
        run_mission(mission_id, args)




def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch RGB + LiDAR -> Nav Graph dataset saver"
    )
    
    parser.add_argument(
        "--data-dir", default="~/grand_tour_dataset",
        help="Path to GrandTour dataset folder",
    )
    parser.add_argument(
        "--frame-numbers", type=int, nargs="+", default=[750, 800, 850],
        help="List of hdr_front frame indices to process",
    )
    parser.add_argument(
        "--max-frames", type=int, default=2000,
        help="Max LiDAR frames to integrate per image",
    )
    parser.add_argument(
        "--time-window", type=float, default=0.4,
        help="Seconds around image timestamp to gather LiDAR data",
    )
    parser.add_argument(
        "--map-length", type=float, default=15.0,  #keep lidar range huge above in the code and this low, so that the unknown border cell actually works
        help="Elevation map side length in meters",
    )
    parser.add_argument(
        "--resolution", type=float, default=0.04,
        help="Elevation map resolution in meters/cell",
    )
    parser.add_argument(
        "--crop-size", type=float, default=None,
        help="Crop the elevation map to this side length (meters) centred on the "
             "camera before nav graph generation. None: use the full map.",
    )
    parser.add_argument(
        "--single-lidar-frame", action="store_true",
        help="Use only the single closest LiDAR frame per sensor instead of "
             "accumulating all frames within --time-window",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())