# nav-graph comparison report

## Sources

- **baseline** — `/home/rohang73/Documents/odin_e2e/graph_compaRISION/inputs/nav_graph_node.json` (saved 2026-05-23T06:08:40.255109+00:00, 3816 nodes, 17170 edges, frame_count=2363)
- **e2e_vitb_518** — `/home/rohang73/Documents/odin_e2e/graph_compaRISION/inputs/e2e_vitb_518.json` (saved 2026-05-23T06:28:12.458926+00:00, 4337 nodes, 13444 edges, frame_count=2002)
- **e2e_vits** — `/home/rohang73/Documents/odin_e2e/graph_compaRISION/inputs/e2e_vits.json` (saved 2026-05-23T04:49:58.078568+00:00, 4563 nodes, 14593 edges, frame_count=1991)

> **Note:** the first **10.0 s** of each timing CSV (measured against that CSV's own first `frame_timestamp_sec`) were dropped before any plot / mean / median was computed — covers the e2e JIT-warmup spike.  The same filter is propagated to the graph snapshots: every node whose ID was assigned during the warmup window is also excluded from the XY overlay, structural metrics, planning queries, and nearest-neighbour comparisons.

## Summary table

| metric | baseline | e2e_vitb_518 | e2e_vits | δ e2e_vitb_518 | δ e2e_vits |
|---|---:|---:|---:|---:|---:|
| num_nodes | 3816.000 | 4337.000 | 4563.000 | 521.000 | 747.000 |
| num_edges | 17170.000 | 13444.000 | 14593.000 | -3726.000 | -2577.000 |
| num_free | 3693.000 | 4337.000 | 4563.000 | 644.000 | 870.000 |
| num_frontier | 123.000 | 0.000 | 0.000 | -123.000 | -123.000 |
| frame_count | 2363.000 | 2002.000 | 1991.000 | -361.000 | -372.000 |
| bbox_x | 73.100 | 66.479 | 67.406 | -6.620 | -5.694 |
| bbox_y | 146.600 | 149.048 | 148.065 | 2.448 | 1.465 |
| bbox_z | 7.959 | 8.347 | 8.389 | 0.387 | 0.430 |
| xy_area_m2 | 10716.410 | 9908.630 | 9980.470 | -807.780 | -735.940 |
| covered_area_m2 | 1988.500 | 1583.250 | 1763.500 | -405.250 | -225.000 |
| xy_density_per_m2 | 1.919 | 2.739 | 2.587 | 0.820 | 0.668 |
| z_mean | -2.070 | -2.203 | -2.305 | -0.133 | -0.235 |
| z_std | 2.376 | 2.560 | 2.550 | 0.184 | 0.174 |
| mean_nn_distance_m | 0.658 | 0.568 | 0.572 | -0.089 | -0.085 |
| median_nn_distance_m | 0.633 | 0.543 | 0.548 | -0.089 | -0.085 |
| avg_degree | 8.999 | 6.200 | 6.396 | -2.799 | -2.603 |
| mean_edge_len_m | 1.107 | 1.387 | 1.354 | 0.280 | 0.247 |
| max_edge_len_m | 2.997 | 2.976 | 2.979 | -0.021 | -0.018 |
| num_cc | 107.000 | 242.000 | 191.000 | 135.000 | 84.000 |
| largest_cc_size | 3661.000 | 3603.000 | 4357.000 | -58.000 | 696.000 |
| largest_cc_frac | 0.959 | 0.831 | 0.955 | -0.129 | -0.005 |
| nodes_in_free | 2921.000 | 3442.000 | 3388.000 | 521.000 | 467.000 |
| nodes_in_free_pct | 76.546 | 79.364 | 74.249 | 2.817 | -2.297 |
| nodes_in_occ | 685.000 | 568.000 | 649.000 | -117.000 | -36.000 |
| nodes_in_occ_pct | 17.951 | 13.097 | 14.223 | -4.854 | -3.728 |
| edges_mid_in_free | 14778.000 | 11758.000 | 11979.000 | -3020.000 | -2799.000 |
| edges_mid_in_free_pct | 86.069 | 87.459 | 82.087 | 1.390 | -3.981 |
| edges_mid_in_occ | 1724.000 | 1095.000 | 1498.000 | -629.000 | -226.000 |
| edges_mid_in_occ_pct | 10.041 | 8.145 | 10.265 | -1.896 | 0.224 |
| free_area_coverage_pct | 96.732 | 84.097 | 88.369 | -12.635 | -8.363 |
| planning_success_rate | 0.975 | 0.725 | 0.920 | -0.250 | -0.055 |
| planning_mean_path_length_m | 74.373 | 98.798 | 123.371 | 24.425 | 48.998 |
| planning_median_path_length_m | 68.835 | 93.260 | 108.755 | 24.425 | 39.920 |
| planning_mean_snap_distance_m | 0.232 | 0.243 | 0.219 | 0.011 | -0.013 |
| planning_n_pairs | 200.000 | 200.000 | 200.000 | 0.000 | 0.000 |
| mean_t_parse_ms | 2.446 | NaN | NaN | NaN | NaN |
| mean_t_emap_ms | 3.022 | NaN | NaN | NaN | NaN |
| mean_t_local_ms | 6.406 | NaN | NaN | NaN | NaN |
| mean_t_merge_ms | 2.759 | 0.772 | 0.700 | -1.987 | -2.059 |
| mean_t_other_ms | 7.098 | 0.745 | 0.713 | -6.353 | -6.385 |
| mean_t_graph_total_ms | 16.263 | NaN | NaN | NaN | NaN |
| mean_t_frame_total_ms | 22.324 | 25.392 | 14.257 | 3.069 | -8.067 |
| until2500_nodes_in_free | 1790.000 | NaN | NaN | NaN | NaN |
| until2500_nodes_in_free_pct | 77.725 | NaN | NaN | NaN | NaN |
| until2500_nodes_in_occ | 345.000 | NaN | NaN | NaN | NaN |
| until2500_nodes_in_occ_pct | 14.980 | NaN | NaN | NaN | NaN |
| until2500_edges_mid_in_free | 8388.000 | NaN | NaN | NaN | NaN |
| until2500_edges_mid_in_free_pct | 86.725 | NaN | NaN | NaN | NaN |
| until2500_edges_mid_in_occ | 720.000 | NaN | NaN | NaN | NaN |
| until2500_edges_mid_in_occ_pct | 7.444 | NaN | NaN | NaN | NaN |
| until2500_planning_success_rate | 0.955 | NaN | NaN | NaN | NaN |
| until2500_planning_mean_path_length_m | 44.853 | NaN | NaN | NaN | NaN |
| until2500_planning_median_path_length_m | 42.593 | NaN | NaN | NaN | NaN |
| until2500_planning_mean_snap_distance_m | 0.531 | NaN | NaN | NaN | NaN |
| until2500_planning_n_pairs | 200.000 | NaN | NaN | NaN | NaN |
| mean_t_inference_ms | NaN | 22.940 | 12.004 | NaN | NaN |
| mean_t_edges_ms | NaN | 0.935 | 0.839 | NaN | NaN |
| until3396_nodes_in_free | NaN | 2437.000 | NaN | NaN | NaN |
| until3396_nodes_in_free_pct | NaN | 81.233 | NaN | NaN | NaN |
| until3396_nodes_in_occ | NaN | 273.000 | NaN | NaN | NaN |
| until3396_nodes_in_occ_pct | NaN | 9.100 | NaN | NaN | NaN |
| until3396_edges_mid_in_free | NaN | 8729.000 | NaN | NaN | NaN |
| until3396_edges_mid_in_free_pct | NaN | 88.494 | NaN | NaN | NaN |
| until3396_edges_mid_in_occ | NaN | 568.000 | NaN | NaN | NaN |
| until3396_edges_mid_in_occ_pct | NaN | 5.758 | NaN | NaN | NaN |
| until3396_covered_area_iou_2d | NaN | 0.603 | NaN | NaN | NaN |
| until3396_planning_success_rate | NaN | 0.980 | NaN | NaN | NaN |
| until3396_planning_mean_path_length_m | NaN | 73.907 | NaN | NaN | NaN |
| until3396_planning_median_path_length_m | NaN | 65.103 | NaN | NaN | NaN |
| until3396_planning_mean_snap_distance_m | NaN | 0.230 | NaN | NaN | NaN |
| until3396_planning_n_pairs | NaN | 200.000 | NaN | NaN | NaN |
| until3418_nodes_in_free | NaN | NaN | 2209.000 | NaN | NaN |
| until3418_nodes_in_free_pct | NaN | NaN | 73.633 | NaN | NaN |
| until3418_nodes_in_occ | NaN | NaN | 317.000 | NaN | NaN |
| until3418_nodes_in_occ_pct | NaN | NaN | 10.567 | NaN | NaN |
| until3418_edges_mid_in_free | NaN | NaN | 8151.000 | NaN | NaN |
| until3418_edges_mid_in_free_pct | NaN | NaN | 81.355 | NaN | NaN |
| until3418_edges_mid_in_occ | NaN | NaN | 799.000 | NaN | NaN |
| until3418_edges_mid_in_occ_pct | NaN | NaN | 7.975 | NaN | NaN |
| until3418_covered_area_iou_2d | NaN | NaN | 0.659 | NaN | NaN |
| until3418_planning_success_rate | NaN | NaN | 0.975 | NaN | NaN |
| until3418_planning_mean_path_length_m | NaN | NaN | 74.733 | NaN | NaN |
| until3418_planning_median_path_length_m | NaN | NaN | 68.581 | NaN | NaN |
| until3418_planning_mean_snap_distance_m | NaN | NaN | 0.279 | NaN | NaN |
| until3418_planning_n_pairs | NaN | NaN | 200.000 | NaN | NaN |
| chamfer_distance_3d_m [baseline vs e2e_vitb_518] |  | 0.546 |  |  |  |
| hausdorff_distance_3d_m [baseline vs e2e_vitb_518] |  | 8.647 |  |  |  |
| bbox_iou_2d [baseline vs e2e_vitb_518] |  | 0.879 |  |  |  |
| occupancy_iou_2d [baseline vs e2e_vitb_518] |  | 0.285 |  |  |  |
| covered_area_iou_2d [baseline vs e2e_vitb_518] |  | 0.680 |  |  |  |
| node_density_correlation [baseline vs e2e_vitb_518] |  | 0.376 |  |  |  |
| chamfer_distance_3d_m [baseline vs e2e_vits] |  |  | 0.596 |  |  |
| hausdorff_distance_3d_m [baseline vs e2e_vits] |  |  | 7.474 |  |  |
| bbox_iou_2d [baseline vs e2e_vits] |  |  | 0.886 |  |  |
| occupancy_iou_2d [baseline vs e2e_vits] |  |  | 0.291 |  |  |
| covered_area_iou_2d [baseline vs e2e_vits] |  |  | 0.707 |  |  |
| node_density_correlation [baseline vs e2e_vits] |  |  | 0.388 |  |  |

## Plots

Each plot is saved as a PDF (canonical, vector) **and** as a PNG twin so the markdown preview can render it inline.  Click any PDF link to open the print-quality version.

### xy_overlay

![xy_overlay](plots/xy_overlay.png)

_[Open as PDF](plots/xy_overlay.pdf)_

### z_histograms

![z_histograms](plots/z_histograms.png)

_[Open as PDF](plots/z_histograms.pdf)_

### degree_histograms

![degree_histograms](plots/degree_histograms.png)

_[Open as PDF](plots/degree_histograms.pdf)_

### nn_distance_histograms

![nn_distance_histograms](plots/nn_distance_histograms.png)

_[Open as PDF](plots/nn_distance_histograms.pdf)_

### timing_per_frame

![timing_per_frame](plots/timing_per_frame.png)

_[Open as PDF](plots/timing_per_frame.pdf)_

### timing_vs_nodes

![timing_vs_nodes](plots/timing_vs_nodes.png)

_[Open as PDF](plots/timing_vs_nodes.pdf)_

### timing_breakdown

![timing_breakdown](plots/timing_breakdown.png)

_[Open as PDF](plots/timing_breakdown.pdf)_

### timing_histograms

![timing_histograms](plots/timing_histograms.png)

_[Open as PDF](plots/timing_histograms.pdf)_

### graph_growth

![graph_growth](plots/graph_growth.png)

_[Open as PDF](plots/graph_growth.pdf)_

### largest_cc

![largest_cc](plots/largest_cc.png)

_[Open as PDF](plots/largest_cc.pdf)_

### covered_area

![covered_area](plots/covered_area.png)

_[Open as PDF](plots/covered_area.pdf)_

### covered_area_iou

![covered_area_iou](plots/covered_area_iou.png)

_[Open as PDF](plots/covered_area_iou.pdf)_

### graphs_on_occ_map

![graphs_on_occ_map](plots/graphs_on_occ_map.png)

_[Open as PDF](plots/graphs_on_occ_map.pdf)_

### free_area_coverage_map

![free_area_coverage_map](plots/free_area_coverage_map.png)

_[Open as PDF](plots/free_area_coverage_map.pdf)_

### overlay_baseline_vits

![overlay_baseline_vits](plots/overlay_baseline_vits.png)

_[Open as PDF](plots/overlay_baseline_vits.pdf)_

### graphs_on_occ_map_until

![graphs_on_occ_map_until](plots/graphs_on_occ_map_until.png)

_[Open as PDF](plots/graphs_on_occ_map_until.pdf)_

### covered_area_iou_until

![covered_area_iou_until](plots/covered_area_iou.png)

_[Open as PDF](plots/covered_area_iou.pdf)_

### summary_table

![summary_table](plots/summary_table.png)

_[Open as PDF](plots/summary_table.pdf)_


## Interpretation hints

- **mean_nn_distance_m / median_nn_distance_m** — average and median distance from each node to its nearest neighbour in 3-D ("node density" proxy).  Lower ⇒ denser graph.  Independent of the map extent, so the two methods compare fairly even when they cover slightly different regions.
- **Chamfer / Hausdorff** measure how spatially close the two node sets are.  Lower = more similar spatial coverage.  Chamfer is the mean of the two-way nearest-neighbour means; Hausdorff is the worst nearest-neighbour distance — sensitive to outliers.
- **Bbox / occupancy / density correlation** describe extent and distribution agreement.  Occupancy IoU close to 1.0 ⇒ both graphs cover the same XY region at the chosen cell size.
- **Planning success rate** is the apples-to-apples quality proxy: random world-locations are projected to each graph, then Dijkstra checks whether a path exists.  Equal or higher e2e success rate at comparable mean-path-length argues parity; higher mean-path-length is suspicious (detours).
- **Timing breakdown** is mean per-step time over the whole run. Stacks are not directly comparable beyond their totals because the two methods have different sub-steps — but the total height is.  Watch the *vs. node count* scatter for scaling behaviour.
