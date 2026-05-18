#!/usr/bin/env bash
# Source this file, don't execute it:  source setup_env.sh

ODIN_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${HOME}/ros2_lyrical/ros2-linux/setup.bash"
source "${ODIN_WS}/.venv/bin/activate"
source "${ODIN_WS}/ros2_ws/install/setup.bash"

export PYTHONPATH="${ODIN_WS}/ros2_ws/src/odin_nav_graph:${ODIN_WS}/.venv/lib/python3.14/site-packages:${ODIN_WS}/nav_graph_gpu:${ODIN_WS}/elevation_mapping_cupy/elevation_mapping_cupy/script:${PYTHONPATH}"

# cupy JIT-compiles CUDA kernels at runtime and needs the CUDA headers
# (cuda_fp16.h, etc.).  There is no system CUDA toolkit; point cupy at the
# headers shipped by the pip ``nvidia-cuda-runtime-cu12`` wheel.
export CUDA_PATH="${ODIN_WS}/.venv/lib/python3.14/site-packages/nvidia/cuda_runtime"

# Use Fast DDS instead of the default rmw_zenoh.  rmw_zenoh matches endpoints
# strictly on the rosidl type hash; this install's sensor_msgs/visualization_msgs
# packages were built without a valid type hash, so rosbag2's generic
# subscriptions never link to the node's PointCloud2/Marker publishers and
# record 0 messages.  Fast DDS matches on type name and records them fine.
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
