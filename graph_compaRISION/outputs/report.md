# nav-graph comparison report

## Sources

- **baseline** — `/home/rohang73/Documents/odin_e2e/graph_compaRISION/inputs/nav_graph_node.json` (saved 2026-05-27T10:54:55.494651+00:00, 3107 nodes, 10211 edges, frame_count=1449)
- **e2e_vitb_518** — `/home/rohang73/Documents/odin_e2e/graph_compaRISION/inputs/e2e_vitb_518.json` (saved 2026-05-23T06:28:12.458926+00:00, 4337 nodes, 13444 edges, frame_count=2002)
- **e2e_vits** — `/home/rohang73/Documents/odin_e2e/graph_compaRISION/inputs/e2e_vits.json` (saved 2026-05-23T04:49:58.078568+00:00, 4563 nodes, 14593 edges, frame_count=1991)

> **Note:** the first **10.0 s** of each timing CSV (measured against that CSV's own first `frame_timestamp_sec`) were dropped before any plot / mean / median was computed — covers the e2e JIT-warmup spike.  The same filter is propagated to the graph snapshots: every node whose ID was assigned during the warmup window is also excluded from the XY overlay, structural metrics, planning queries, and nearest-neighbour comparisons.

## Summary table

| metric | baseline | e2e_vitb_518 | e2e_vits | δ e2e_vitb_518 | δ e2e_vits |
|---|---:|---:|---:|---:|---:|
| num_nodes | 3107.000 | 4337.000 | 4563.000 | 1230.000 | 1456.000 |
| num_edges | 10211.000 | 13444.000 | 14593.000 | 3233.000 | 4382.000 |
| num_free | 2821.000 | 4337.000 | 4563.000 | 1516.000 | 1742.000 |
| num_frontier | 286.000 | 0.000 | 0.000 | -286.000 | -286.000 |
| frame_count | 1449.000 | 2002.000 | 1991.000 | 553.000 | 542.000 |
| bbox_x | 76.800 | 66.479 | 67.406 | -10.321 | -9.394 |
| bbox_y | 148.000 | 149.048 | 148.065 | 1.048 | 0.065 |
| bbox_z | 7.414 | 8.347 | 8.389 | 0.933 | 0.976 |
| xy_area_m2 | 11366.388 | 9908.630 | 9980.470 | -1457.758 | -1385.918 |
| covered_area_m2 | 1613.750 | 1583.250 | 1763.500 | -30.500 | 149.750 |
| xy_density_per_m2 | 1.925 | 2.739 | 2.587 | 0.814 | 0.662 |
| z_mean | -1.939 | -2.203 | -2.305 | -0.264 | -0.366 |
| z_std | 2.367 | 2.560 | 2.550 | 0.193 | 0.183 |
| mean_nn_distance_m | 0.685 | 0.568 | 0.572 | -0.116 | -0.113 |
| median_nn_distance_m | 0.634 | 0.543 | 0.548 | -0.090 | -0.086 |
| avg_degree | 6.573 | 6.200 | 6.396 | -0.373 | -0.177 |
| mean_edge_len_m | 0.952 | 1.387 | 1.354 | 0.435 | 0.402 |
| max_edge_len_m | 1.800 | 2.976 | 2.979 | 1.176 | 1.179 |
| num_cc | 174.000 | 242.000 | 191.000 | 68.000 | 17.000 |
| largest_cc_size | 1242.000 | 3603.000 | 4357.000 | 2361.000 | 3115.000 |
| largest_cc_frac | 0.400 | 0.831 | 0.955 | 0.431 | 0.555 |
| nodes_in_free | 2230.000 | 2372.000 | 2415.000 | 142.000 | 185.000 |
| nodes_in_free_pct | 71.773 | 54.692 | 52.926 | -17.081 | -18.848 |
| nodes_in_occ | 783.000 | 1756.000 | 1837.000 | 973.000 | 1054.000 |
| nodes_in_occ_pct | 25.201 | 40.489 | 40.259 | 15.288 | 15.057 |
| edges_mid_in_free | 8045.000 | 8276.000 | 8540.000 | 231.000 | 495.000 |
| edges_mid_in_free_pct | 78.788 | 61.559 | 58.521 | -17.229 | -20.266 |
| edges_mid_in_occ | 1901.000 | 4725.000 | 5287.000 | 2824.000 | 3386.000 |
| edges_mid_in_occ_pct | 18.617 | 35.146 | 36.230 | 16.529 | 17.613 |
| free_area_coverage_pct | 95.685 | 79.974 | 84.621 | -15.712 | -11.064 |
| planning_success_rate | 0.240 | 0.790 | 0.935 | 0.550 | 0.695 |
| planning_mean_path_length_m | 27.695 | 95.559 | 107.013 | 67.864 | 79.318 |
| planning_median_path_length_m | 25.045 | 88.499 | 93.411 | 63.454 | 68.367 |
| planning_mean_snap_distance_m | 0.299 | 0.278 | 0.250 | -0.021 | -0.049 |
| planning_n_pairs | 200.000 | 200.000 | 200.000 | 0.000 | 0.000 |
| planning_mean_path_length_common_m | 27.747 | 46.303 | 52.176 | 18.555 | 24.429 |
| planning_n_common_success_pairs | 43.000 | 43.000 | 43.000 | 0.000 | 0.000 |
| mean_t_parse_ms | 4.908 | NaN | NaN | NaN | NaN |
| mean_t_emap_ms | 3.298 | NaN | NaN | NaN | NaN |
| mean_t_elev_extract_ms | 0.415 | NaN | NaN | NaN | NaN |
| mean_t_local_ms | 6.061 | NaN | NaN | NaN | NaN |
| mean_t_merge_ms | 2.334 | 0.772 | 0.700 | -1.562 | -1.634 |
| mean_t_update_residual_ms | 10.244 | NaN | NaN | NaN | NaN |
| mean_t_visited_time_ms | 0.415 | NaN | NaN | NaN | NaN |
| mean_t_layers_ms | 1.289 | NaN | NaN | NaN | NaN |
| mean_t_graph_total_ms | 20.786 | NaN | NaN | NaN | NaN |
| mean_t_frame_total_ms | 29.571 | 25.392 | 14.257 | -4.178 | -15.314 |
| until2500_nodes_in_free | 1786.000 | NaN | NaN | NaN | NaN |
| until2500_nodes_in_free_pct | 75.774 | NaN | NaN | NaN | NaN |
| until2500_nodes_in_occ | 480.000 | NaN | NaN | NaN | NaN |
| until2500_nodes_in_occ_pct | 20.365 | NaN | NaN | NaN | NaN |
| until2500_edges_mid_in_free | 6462.000 | NaN | NaN | NaN | NaN |
| until2500_edges_mid_in_free_pct | 83.650 | NaN | NaN | NaN | NaN |
| until2500_edges_mid_in_occ | 1011.000 | NaN | NaN | NaN | NaN |
| until2500_edges_mid_in_occ_pct | 13.087 | NaN | NaN | NaN | NaN |
| until2500_planning_success_rate | 0.385 | NaN | NaN | NaN | NaN |
| until2500_planning_mean_path_length_m | 26.851 | NaN | NaN | NaN | NaN |
| until2500_planning_median_path_length_m | 21.672 | NaN | NaN | NaN | NaN |
| until2500_planning_mean_snap_distance_m | 0.284 | NaN | NaN | NaN | NaN |
| until2500_planning_n_pairs | 200.000 | NaN | NaN | NaN | NaN |
| mean_t_inference_ms | NaN | 22.940 | 12.004 | NaN | NaN |
| mean_t_edges_ms | NaN | 0.935 | 0.839 | NaN | NaN |
| mean_t_other_ms | NaN | 0.745 | 0.713 | NaN | NaN |
| until3396_nodes_in_free | NaN | 1876.000 | NaN | NaN | NaN |
| until3396_nodes_in_free_pct | NaN | 62.533 | NaN | NaN | NaN |
| until3396_nodes_in_occ | NaN | 943.000 | NaN | NaN | NaN |
| until3396_nodes_in_occ_pct | NaN | 31.433 | NaN | NaN | NaN |
| until3396_edges_mid_in_free | NaN | 6713.000 | NaN | NaN | NaN |
| until3396_edges_mid_in_free_pct | NaN | 68.056 | NaN | NaN | NaN |
| until3396_edges_mid_in_occ | NaN | 2733.000 | NaN | NaN | NaN |
| until3396_edges_mid_in_occ_pct | NaN | 27.707 | NaN | NaN | NaN |
| until3396_covered_area_iou_2d | NaN | 0.633 | NaN | NaN | NaN |
| until3396_planning_success_rate | NaN | 0.960 | NaN | NaN | NaN |
| until3396_planning_mean_path_length_m | NaN | 78.380 | NaN | NaN | NaN |
| until3396_planning_median_path_length_m | NaN | 70.793 | NaN | NaN | NaN |
| until3396_planning_mean_snap_distance_m | NaN | 0.310 | NaN | NaN | NaN |
| until3396_planning_n_pairs | NaN | 200.000 | NaN | NaN | NaN |
| until3418_nodes_in_free | NaN | NaN | 1695.000 | NaN | NaN |
| until3418_nodes_in_free_pct | NaN | NaN | 56.500 | NaN | NaN |
| until3418_nodes_in_occ | NaN | NaN | 1027.000 | NaN | NaN |
| until3418_nodes_in_occ_pct | NaN | NaN | 34.233 | NaN | NaN |
| until3418_edges_mid_in_free | NaN | NaN | 6177.000 | NaN | NaN |
| until3418_edges_mid_in_free_pct | NaN | NaN | 61.653 | NaN | NaN |
| until3418_edges_mid_in_occ | NaN | NaN | 3121.000 | NaN | NaN |
| until3418_edges_mid_in_occ_pct | NaN | NaN | 31.151 | NaN | NaN |
| until3418_covered_area_iou_2d | NaN | NaN | 0.587 | NaN | NaN |
| until3418_planning_success_rate | NaN | NaN | 0.980 | NaN | NaN |
| until3418_planning_mean_path_length_m | NaN | NaN | 77.209 | NaN | NaN |
| until3418_planning_median_path_length_m | NaN | NaN | 72.520 | NaN | NaN |
| until3418_planning_mean_snap_distance_m | NaN | NaN | 0.411 | NaN | NaN |
| until3418_planning_n_pairs | NaN | NaN | 200.000 | NaN | NaN |
| chamfer_distance_3d_m [baseline vs e2e_vitb_518] |  | 0.601 |  |  |  |
| hausdorff_distance_3d_m [baseline vs e2e_vitb_518] |  | 10.885 |  |  |  |
| bbox_iou_2d [baseline vs e2e_vitb_518] |  | 0.849 |  |  |  |
| occupancy_iou_2d [baseline vs e2e_vitb_518] |  | 0.259 |  |  |  |
| covered_area_iou_2d [baseline vs e2e_vitb_518] |  | 0.626 |  |  |  |
| node_density_correlation [baseline vs e2e_vitb_518] |  | 0.358 |  |  |  |
| chamfer_distance_3d_m [baseline vs e2e_vits] |  |  | 0.629 |  |  |
| hausdorff_distance_3d_m [baseline vs e2e_vits] |  |  | 9.981 |  |  |
| bbox_iou_2d [baseline vs e2e_vits] |  |  | 0.856 |  |  |
| occupancy_iou_2d [baseline vs e2e_vits] |  |  | 0.252 |  |  |
| covered_area_iou_2d [baseline vs e2e_vits] |  |  | 0.628 |  |  |
| node_density_correlation [baseline vs e2e_vits] |  |  | 0.351 |  |  |

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

### timing_breakdown_2

![timing_breakdown_2](plots/timing_breakdown_2.png)

_[Open as PDF](plots/timing_breakdown_2.pdf)_

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
