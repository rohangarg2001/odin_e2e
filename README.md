# odin_e2e
Code for forming navigation graph from rgb image from Odin camera


(.venv) rohang73@opportunity:~/Documents/odin_e2e$ ros2 run odin_nav_graph nav_graph_node_e2e --ros-args   -p model_checkpoint:=/home/rohang73/ASL/e2e_rgb_nav_graph/four_longrange_size280_eightfoldersval_v4/epoch_579.pth   -p e2e_repo_path:=/home/rohang73/ASL/e2e_rgb_nav_graph   -p cam_fx:=800.0 -p cam_fy:=800.0 -p cam_cx:=800.0 -p cam_cy:=648.0   -p max_node_range:=5.0
