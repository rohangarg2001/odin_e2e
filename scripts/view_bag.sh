#!/usr/bin/env bash
# Open the recorded elevation map and graph on Curiosity.
set -e
ODIN_BAG_PATH="${1:?Usage: bash scripts/view_bag.sh /path/to/derived_bag}"
ODIN_BAG_PATH="$(realpath "$ODIN_BAG_PATH")"
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=227 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
unset ROS_STATIC_PEERS FASTRTPS_DEFAULT_PROFILES_FILE FASTDDS_DEFAULT_PROFILES_FILE
export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-/run/user/1002/gdm/Xauthority}"
rviz2 -d "$ODIN_BAG_PATH/odin_graph.rviz" > "$ODIN_BAG_PATH/rviz.log" 2>&1 &
ODIN_RVIZ_PID=$!
trap 'kill "$ODIN_RVIZ_PID" 2>/dev/null || true' EXIT
ros2 bag play "$ODIN_BAG_PATH" --loop --clock 50 --delay 3 --topics \
  /tf /odin_nav_graph_node/matched_odometry \
  /odin_nav_graph_node/elevation_cloud /odin_nav_graph_node/graph_nodes \
  /odin_nav_graph_node/frontier_cloud /odin_nav_graph_node/graph_edges \
  /odin_nav_graph_node/debug/frontier_cells /odin_nav_graph_node/debug/frontier_mask \
  /odin_nav_graph_node/debug/graph_nodes_all /odin_nav_graph_node/debug/occupancy
