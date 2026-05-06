#!/usr/bin/env bash
# Run the odin_nav_graph node with ExploRFM layers enabled.
# Source setup_env.sh first, then execute this script.

set -euo pipefail

ODIN_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPTS="${ODIN_WS}/nebula2-wildos/ckpts"

ros2 run odin_nav_graph nav_graph_node --ros-args \
  -p out_directory:="${ODIN_WS}/saved_outputs_odin" \
  -p cam_fx:=800.0 \
  -p cam_fy:=800.0 \
  -p cam_cx:=800.0 \
  -p cam_cy:=648.0 \
  -p map_length_xy:=30.0 \
  -p cloud_max_range:=15.0 \
  -p enable_explorfm_layers:=true \
  -p explorfm_frontier_ckpt:="${CKPTS}/frontier_head.ckpt" \
  -p explorfm_trav_ckpt:="${CKPTS}/trav_head.ckpt" \
  -p explorfm_adaptor_version:=siglip2 \
  -p explorfm_adaptor_ckpt_path:="${CKPTS}/siglip2" \
  -p explorfm_radio_version:=c-radio_v3-b \
  -p explorfm_use_naclip:=true \
  -p explorfm_radio_dim:=768 \
  -p explorfm_static_scale_factor:=0.5 \
  -p explorfm_precision:=FP16 \
  -p explorfm_every_n_images:=1
