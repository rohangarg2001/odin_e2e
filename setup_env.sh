#!/usr/bin/env bash
# Source this file, don't execute it:  source setup_env.sh

ODIN_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /opt/ros/jazzy/setup.bash
source "${ODIN_WS}/.venv/bin/activate"
source "${ODIN_WS}/ros2_ws/install/setup.bash"

export PYTHONPATH="${ODIN_WS}/ros2_ws/src/odin_nav_graph:${ODIN_WS}/.venv/lib/python3.12/site-packages:${ODIN_WS}/nav_graph_gpu:${ODIN_WS}/elevation_mapping_cupy/elevation_mapping_cupy/script:${PYTHONPATH}"
