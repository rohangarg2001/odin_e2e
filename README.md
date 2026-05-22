# odin_e2e
Code for forming navigation graph from rgb image from Odin camera


(.venv) rohang73@opportunity:~/Documents/odin_e2e$ ros2 run odin_nav_graph nav_graph_node_e2e --ros-args   -p model_checkpoint:=/home/rohang73/ASL/e2e_rgb_nav_graph/four_longrange_size280_eightfoldersval_v4/epoch_999.pth   -p e2e_repo_path:=/home/rohang73/ASL/e2e_rgb_nav_graph   -p cam_fx:=800.0 -p cam_fy:=800.0 -p cam_cx:=800.0 -p cam_cy:=648.0   -p max_node_range:=5.0









DETR model (the default, unchanged):


ros2 run odin_nav_graph nav_graph_node_e2e --ros-args \
  -p model_type:=detr \
  -p model_checkpoint:=/home/rohang73/ASL/e2e_rgb_nav_graph/four_longrange_size280_overfit/epoch_999.pth \
  -p e2e_repo_path:=/home/rohang73/ASL/e2e_rgb_nav_graph \
  -p cam_fx:=800.0 -p cam_fy:=800.0 -p cam_cx:=800.0 -p cam_cy:=648.0 \
  -p max_node_range:=6.0
Heatmap model:


ros2 run odin_nav_graph nav_graph_node_e2e --ros-args \
  -p model_type:=heatmap \
  -p heatmap_checkpoint:=/home/rohang73/ASL/e2e_rgb_nav_graph/heatmap_model_training_v5_wall_neg_more_vitb518/best.pth \
  -p heatmap_repo_path:=/home/rohang73/ASL/e2e_rgb_nav_graph/heatmap_model \
  -p cam_fx:=800.0 -p cam_fy:=800.0 -p cam_cx:=800.0 -p cam_cy:=648.0 \
  -p max_node_range:=6.0

Heatmap-specific knobs (all optional — ckpt defaults are used otherwise)
Param	Meaning	Default
heatmap_sample_threshold	Heatmap probability a pixel must exceed to be eligible for sampling	0.5
heatmap_sample_min_dist	Poisson-disk min spacing between sampled pixels (heatmap-pixel units). Lower = denser nodes.	-1 (from ckpt: 12.0)
heatmap_sample_window	Morphological-erosion window; pixel needs every pixel in a window×window box to clear threshold (strict, deep in traversable region).	-1 (from ckpt: 5)
Tune by passing e.g. -p heatmap_sample_min_dist:=8.0 -p heatmap_sample_threshold:=0.6.






GOOD E2E:

ros2 run odin_nav_graph nav_graph_node_e2e --ros-args   -p model_type:=heatmap   -p heatmap_checkpoint:=/home/rohang73/ASL/e2e_rgb_nav_graph/heatmap_model_training_v5_wall_neg_more_vitb518/best.pth   -p heatmap_repo_path:=/home/rohang73/ASL/e2e_rgb_nav_graph/heatmap_model   -p cam_fx:=800.0 -p cam_fy:=800.0 -p cam_cx:=800.0 -p cam_cy:=648.0   -p max_node_range:=6.0 -p heatmap_sample_min_dist:=8.0 -p heatmap_sample_threshold:=0.6 -p heatmap_sample_window:=9  -p prune_isolated_nodes:=true -p prune_every_n_frames:=300







FOR NORMAL NAV GRAPH

1. Build (once, after code changes)

cd ~/Documents/odin_e2e/ros2_ws
colcon build --packages-select odin_nav_graph
2. Per-terminal setup — source setup_env.sh
Every new terminal needs this first (setup_env.sh). It sources ROS Jazzy, the .venv, the workspace install, and sets PYTHONPATH for nav_graph_gpu + elevation_mapping_cupy:


cd ~/Documents/odin_e2e
source setup_env.sh
3. Run the node — run_nav_graph.sh
run_nav_graph.sh is the main entry point — ros2 run odin_nav_graph nav_graph_node with all the ExploRFM checkpoints/params wired up (camera intrinsics, map_length_xy:=30, enable_explorfm_layers:=true, the frontier/trav/siglip2 checkpoints from nebula2-wildos/ckpts):


./run_nav_graph.sh
4. Play the data — separate terminal
The node listens on /odin1/cloud_raw + /odin1/odometry_highfreq + /odin1/image/undistorted. In another terminal (after source setup_env.sh):


ros2 bag play /path/to/your/rosbag2_...