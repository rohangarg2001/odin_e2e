#!/usr/bin/env bash
# Plays the source rosbag, runs odin_nav_graph nav_graph_node, and records both
# the replayed input topics and the node's published topics into a new bag so
# the visualisation can be replayed later in RViz without re-running the
# pipeline.
#
# Usage:
#   ./record_frontier_bag.sh
#
# Override defaults via env vars:
#   SRC_BAG=/path/to/bag RATE=1.0 RECORD_OUT=my_frontiers ./record_frontier_bag.sh

set -euo pipefail

SRC_BAG="${SRC_BAG:-/home/rohang73/Downloads/rosbag2_2026_04_25-11_52_30/trimmed_bag_output/trimmed_bag_output.db3}"
RATE="${RATE:-0.5}"
RECORD_OUT="${RECORD_OUT:-/home/rohang73/Documents/odin_e2e/recorded_bags/rosbag2_frontiers_$(date +%Y%m%d_%H%M%S)}"

WS_DIR="${WS_DIR:-/home/rohang73/Documents/odin_e2e/ros2_ws}"
CKPT_DIR="${CKPT_DIR:-/home/rohang73/Documents/odin_e2e/nebula2-wildos/ckpts}"
SAVED_OUT="${SAVED_OUT:-/home/rohang73/Documents/odin_e2e/saved_outputs_odin}"

# Topics replayed from the source bag.  Trimmed to just what the node needs +
# the SLAM cloud + RGB image for RViz visualisation context + /tf for transforms.
PLAY_TOPICS=(
  /odin1/cloud_raw
  /odin1/odometry_highfreq
  /odin1/image/undistorted
  /odin1/cloud_slam
  /tf
)

# Topics written into the new bag.  Source-side topics are duplicated so the
# resulting bag is self-contained for RViz playback.
RECORD_TOPICS=(
  # ── from source (replay context) ──
  /odin1/cloud_raw
  /odin1/odometry_highfreq
  /odin1/image/undistorted
  /odin1/cloud_slam
  /tf
  # ── published by nav_graph_node ──
  /odin_nav_graph_node/graph_nodes
  /odin_nav_graph_node/frontier_cloud
  /odin_nav_graph_node/raw_frontiers
  /odin_nav_graph_node/frontier_score_cloud
  /odin_nav_graph_node/trav_score_cloud
  /odin_nav_graph_node/together_markers
  /odin_nav_graph_node/global_occ_grid
)

# Source the workspace so `ros2 run odin_nav_graph ...` resolves.  colcon's
# generated setup.bash references env vars like COLCON_TRACE without first
# defaulting them, so it trips `set -u`.  Temporarily relax it.
if [ -f "$WS_DIR/install/setup.bash" ]; then
  set +u
  # shellcheck disable=SC1091
  source "$WS_DIR/install/setup.bash"
  set -u
fi

mkdir -p "$(dirname "$RECORD_OUT")"

NODE_PID=""
RECORD_PID=""
cleanup() {
  echo "[record_frontier_bag] cleaning up..."
  [ -n "$NODE_PID" ]   && kill -INT "$NODE_PID"   2>/dev/null || true
  [ -n "$RECORD_PID" ] && kill -INT "$RECORD_PID" 2>/dev/null || true
  [ -n "$NODE_PID" ]   && wait "$NODE_PID"   2>/dev/null || true
  [ -n "$RECORD_PID" ] && wait "$RECORD_PID" 2>/dev/null || true
  echo "[record_frontier_bag] saved bag: $RECORD_OUT"
}
trap cleanup EXIT INT TERM

echo "[record_frontier_bag] recording to: $RECORD_OUT"
# --storage sqlite3 forces the classic .db3 rosbag format (default in newer
# distros is mcap).  Drop the flag if you actually want mcap.
ros2 bag record --storage sqlite3 -o "$RECORD_OUT" "${RECORD_TOPICS[@]}" &
RECORD_PID=$!

echo "[record_frontier_bag] starting nav_graph_node..."
ros2 run odin_nav_graph nav_graph_node --ros-args \
  -p out_directory:="$SAVED_OUT" \
  -p cam_fx:=800.0 -p cam_fy:=800.0 -p cam_cx:=800.0 -p cam_cy:=648.0 \
  -p map_length_xy:=30.0 -p cloud_max_range:=15.0 \
  -p enable_explorfm_layers:=true \
  -p explorfm_frontier_ckpt:="$CKPT_DIR/frontier_head.ckpt" \
  -p explorfm_trav_ckpt:="$CKPT_DIR/trav_head.ckpt" \
  -p publish_elevation_cloud:=false &
NODE_PID=$!

# Give the node a beat to come up (RADIO + heads load on first image).
sleep 8

echo "[record_frontier_bag] playing $SRC_BAG at rate $RATE..."
ros2 bag play "$SRC_BAG" -r "$RATE" --topics "${PLAY_TOPICS[@]}"

echo "[record_frontier_bag] playback finished."
