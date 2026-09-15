# Odin debug recording on Curiosity

## Create one debug bag

Run from this repository. Processing is synchronous: every raw cloud is handled once,
without depending on DDS queues or replay speed. Original Odin messages are copied
without changing their serialized payloads or receipt timestamps.

```bash
bash scripts/run_offline.sh \
  /home/rohang73/odin_recordings/all_topics_20260915T060959Z_combined \
  /home/rohang73/odin_recordings/odin_debug_20260915T060959Z \
  --native-rows-axis y --debug --seed 0
source setup_env.sh
.venv/bin/python scripts/verify_and_plot.py \
  /home/rohang73/odin_recordings/odin_debug_20260915T060959Z
.venv/bin/python scripts/package_debug_report.py \
  /home/rohang73/odin_recordings/odin_debug_20260915T060959Z
bash scripts/view_bag.sh \
  /home/rohang73/odin_recordings/odin_debug_20260915T060959Z
```

Use a new output directory for another run. Ctrl+C stops playback when launched in
your terminal. Processing uses local ROS domain 228; viewing uses local domain 227.
The Thor recording/upload script is separate.

## What is recorded

All generated topics below use the prefix `/odin_nav_graph_node/`.

| Topics | Contents |
|---|---|
| `graph_nodes`, `frontier_cloud`, `graph_edges` | RViz graph displays; graph nodes here are free-space nodes |
| `graph_state` | JSON in `std_msgs/String`: full node positions, stable IDs, types, edge endpoints by ID, edge weights, raw frontiers, score names and per-node scores |
| `elevation_cloud` | Valid height cells in odom, XYZ + intensity=height; 12 m rolling window, 0.10 m cells |
| `matched_odometry`, `/tf` | Pose and LiDAR extrinsic actually used for each cloud |
| `debug/graph_nodes_all` | Every GPU graph node, with uint32 ID/type and floating-point exploration/combined score fields |
| `debug/frontier_cells`, `debug/frontier_mask` | Raw detector cells BEFORE clustering, snapping or frontier filtering |
| `debug/clustered_frontiers` | Frontier centroids before graph-node marking/filtering |
| `debug/local_nodes` | World XY nodes emitted by the local generator before global merging; original Z=0 |
| `debug/elevation_graph_input`, `debug/traversability_graph_input` | Exact arrays entering graph preprocessing |
| `debug/occupancy` | Exact occupancy input to the frontier kernel: -1 unknown, 0 free, 100 occupied |
| `debug/local_cost_grid`, `debug/global_collision_grid` | Exact arrays used by local generation and global edge collision checks; 255 free, 0 other |
| `debug/mapper_*` | Elevation backend layers: elevation, variance, is_valid, traversability, time, upper_bound, is_upper_bound, normals |
| `debug/grid_info` | Geometry for the cropped mapper-layer images |
| `debug/frame_diagnostics` | Frame/stamp association, pose/transform deltas, actual transforms, map geometry, intermediate counts, parse/elevation/graph timings |
| `debug/parameters` | Actual ROS parameters, recorded once as a ParameterEvent |
| All original `/odin1/*` topics | Raw/render/SLAM clouds, raw/compressed images, IMU, both odometry rates, wiwc and Odin TF |

The original bag contains no messages on the undistorted-image topic and has no
CameraInfo topic. These cannot be recovered by recording the downstream pipeline.

## Array conventions

- Mapper debug images are cropped to the same 120 × 120 usable cells as the graph.
  Rows increase in world Y; columns increase in world X. Use `debug/grid_info`.
  Invalid raw mapper elevation values must be masked with `mapper_is_valid`.
- `elevation_graph_input` and `traversability_graph_input` preserve the builder's
  original NE-first array order: row/column 0 are maximum Y/X.
- Occupancy and frontier-mask messages carry the detector's actual origin and
  resolution. Frontier-mask values are 100 for detected cells and 0 elsewhere.
- Raw and clustered frontier point clouds preserve the detector's 2D XY with Z=0.
  They are different from the ground-height frontier graph nodes.
- The local cost image is the detector cost image flipped vertically. The global
  collision image is that local image rotated 180 degrees. They are exact internal
  arrays, not independently georeferenced images.
- The elevation adapter's cell-center convention and the frontier kernel's
  half-cell offset differ by 0.05 m in X/Y at this resolution. Both actual origins
  are recorded in diagnostics; the capture does not hide this difference.

## Code review and remaining concerns

1. **Spot clearance is not configured.** The current safety distance is 0.05 m.
   Measure the Odin-to-Spot body transform and configure footprint/clearance and
   traversability thresholds before testing robot motion.
2. **Disconnected graph nodes occur.** Use connectivity from the robot's nearest
   valid node when selecting goals; do not treat every frontier as reachable.
   The attached quality review measures this on the final recorded graph.
3. **Slope is resolution-dependent.** The pinned builder calls
   `np.gradient(elev_smooth)` without cell spacing. Its slope threshold therefore
   compares height change per cell, not rise/run or an angle.
4. **Mapping/scoring is wired; autonomous exploration is not.** The builder
   computes exploration scores, but this node has no goal-selection or Spot
   motion-control loop.
5. **RGB is not used to generate elevation here.** Optional camera overlays require
   calibration and actual undistorted images. The bag preserves available originals.
6. **Rolling maps and height heuristics:** the map is local, while the graph is
   persistent. A 5×5 minimum filter assigns graph Z; it can pull nodes toward a
   nearby lower surface. Review steps, edges and outliers using the recorded layers.
7. **Reproducibility:** the waypoint generator randomly samples free space.
   The runner seeds Python, NumPy, PyTorch and CuPy and records the seed. GPU atomic
   ordering can still affect tie-breaking; identical graph IDs across runs are
   not guaranteed.
8. **Validation scope:** offline Odin replay on Curiosity. Live ROS synchronization,
   robot-body calibration and closed-loop navigation have not been validated.

## Fixes made for this run

- Handle the actual odom -> imu -> lidar transform chain, including changing
  recorded sensor extrinsics; reject inconsistent frames.
- Pair each cloud with nearest available high- or low-rate recorded odometry.
  High-rate odometry ends before the final cloud; low-rate odometry covers it.
- Remove the 385-cloud cutoff and the coupling between RGB saving and graph updates.
- Correct the installed ROS2 elevation backend's row-Y convention; test asymmetric
  observations, map rolling and graph-grid conversion.
- Add the missing CuPy 14 float16 header for the backend's generated kernels.
- Publish empty frontier/node data and delete empty edge markers to avoid stale views.
- Load YAML parameters, launch RViz, and install its config correctly.
- Record intermediate arrays by observing the actual function results, without
  recomputing them or changing the graph calculation.

## Validation and environment

`verification.json` records read-back topic/count checks, one-MCAP enforcement,
finite published clouds and SHA-256 checks of each original Odin stream's payloads
and receipt timestamps. `SHA256SUMS` covers the final MCAP file.

`run_report.json`, `run_params.yaml`, `environment.json`, `quality_review.json`,
`final_graph.json`, `code_snapshot/` and the preview accompany the bag.

This machine's project venv reuses the existing CUDA-enabled packages in
`/home/rohang73/git/nav_graph_gpu/.venv-ros` and the ROS2 elevation mapper source in
`/home/rohang73/git/gg_autonomy/ros/ros2_ws/src/elevation_mapping_cupy/elevation_mapping_cupy`.
These are local dependencies, not vendored installs. The graph itself uses this
repository's pinned `nav_graph_gpu` submodule. Do not blindly install the generic
CUDA-12 requirement over the working CUDA-13 runtime.
