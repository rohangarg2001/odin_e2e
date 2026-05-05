"""Wrapper around elevation_mapping_cupy that exposes an
nav_graph-compatible elevation grid.

Two convention details that matter and are *easy to get wrong*:

1. ``elevation_mapping_cupy`` stores ``elev[row, col]`` at world position
       (cx + (row - H/2) * res, cy + (col - W/2) * res)
   — i.e. **rows index X, cols index Y**.  Verified empirically: a
   single point at world (X=2, Y=0) lands in cell (row=60, col=40) on
   an 80x80 map centered at (0, 0); a point at (0, 2) lands in
   (40, 60).  (The kernel source uses ``x0`` and ``y0`` as pixel
   coordinates that look like image axes, but in the array layout the
   first axis maps to X, not Y, so a transpose is needed.)

2. ``nav_graph`` (with ``input_type='elevation_map'``) expects the input
   ``grid[r, c]`` at world position
       (cx + (W/2 - 1 - c) * res, cy + (H/2 - 1 - r) * res)
   — i.e. **rows index Y, cols index X** with row 0 = max-y (north)
   and col 0 = max-x (east).  This is the grid_map / image-with-y-up
   orientation.  (Verified by inspecting ``_assign_z_from_elevation``
   which reads through ``elevation_flipped = np.flipud(np.fliplr(grid))``.)

Going from (1) to (2) is ``np.flipud(np.fliplr(elev.T))`` — first
transpose to swap the row/col-axis meanings, then rot180 to put NE
at [0, 0].  We do that inside ``get_elevation_for_navgraph`` so the
consumer doesn't need to think about it.

Equivalently: the example_how_to_use.py script left the elevation
*untransposed* and applied a post-hoc x↔y swap to the resulting graph
(``R_fix = Ry @ Rz``) — that workaround works for one-shot
per-frame graphs but breaks global accumulation, because the global
graph would store everything in swapped coordinates.

Frames:
    * ``move_to(robot_xy_z, R=I3)`` recenters the map at the robot's
      current XY in odom; we pass identity R so the map axes stay
      world-aligned (axis-aligned recentering).
    * ``integrate(points_in_base_frame, t_base_in_odom, R_base_to_odom)``
      hands the library points in the body frame plus the body-pose-
      in-odom; the library transforms internally.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

# elevation_mapping_cupy uses ``cp.bool8`` which falls back to
# ``np.bool8`` — removed in numpy 2.x.  Restore it before the library
# is imported.  (``np.bool_`` is the modern equivalent.)
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_  # type: ignore[attr-defined]


def _resolve_em_cupy_paths() -> tuple[Path, Path]:
    """Locate weights.dat + plugin_config.yaml regardless of install layout.

    The leggedrobotics repo ships them at
    ``<repo>/elevation_mapping_cupy/config/core/`` while the importable
    package lives at ``<repo>/elevation_mapping_cupy/script/elevation_mapping_cupy/``.
    """
    import elevation_mapping_cupy

    pkg_root = Path(elevation_mapping_cupy.__file__).resolve().parent
    candidates = [
        pkg_root / "config" / "core",
        pkg_root.parent.parent / "config" / "core",
        Path(os.environ.get("ELEVATION_MAPPING_CUPY_CONFIG_DIR", "")) / "core",
    ]
    for c in candidates:
        if (c / "weights.dat").exists() and (c / "plugin_config.yaml").exists():
            return c / "weights.dat", c / "plugin_config.yaml"
    raise FileNotFoundError(
        "Could not find elevation_mapping_cupy config (weights.dat + plugin_config.yaml). "
        "Tried: " + ", ".join(str(c) for c in candidates) + ". "
        "Set ELEVATION_MAPPING_CUPY_CONFIG_DIR to the directory containing 'core/'."
    )


class ElevationMapWrapper:
    """Thin facade around ``elevation_mapping_cupy.ElevationMap``.

    Exposes the few methods the ROS node actually needs and hides the
    rot180 fix-up between the two conventions.
    """

    def __init__(
        self,
        map_length: float = 12.0,
        resolution: float = 0.10,
        sensor_noise_factor: float = 0.05,
        max_height_range: float = 1.5,
        recordable_fps: float = 0.0,
    ):
        from elevation_mapping_cupy import ElevationMap, Parameter

        weights_path, plugin_path = _resolve_em_cupy_paths()
        param = Parameter(
            use_chainer=False,
            weight_file=str(weights_path),
            plugin_config_file=str(plugin_path),
        )
        param.enable_drift_compensation = False
        # subscriber is required by Parameter.update() — content is unused
        # because we feed pointclouds via input_pointcloud directly.
        param.subscriber = {
            "odin_cloud": {
                "topic_name": "/odin1/cloud_raw",
                "data_type": "pointcloud",
            }
        }
        param.map_length = float(map_length)
        param.resolution = float(resolution)
        param.sensor_noise_factor = float(sensor_noise_factor)
        param.max_height_range = float(max_height_range)
        param.recordable_fps = float(recordable_fps)
        param.update()

        self._map = ElevationMap(param)
        self._param = param
        self.resolution = float(resolution)
        self.map_length = float(map_length)

    # ──────────────────────────────────────────────────────────────────
    def move_to(self, position_xyz: np.ndarray, rotation_3x3: Optional[np.ndarray] = None) -> None:
        """Recenter the map. Pass robot XY (and ideally z=0) plus identity
        rotation to keep the map axis-aligned with odom."""
        if rotation_3x3 is None:
            rotation_3x3 = np.eye(3, dtype=np.float32)
        self._map.move_to(np.asarray(position_xyz, dtype=np.float32), rotation_3x3)

    def integrate(
        self,
        points_sensor_xyz: np.ndarray,
        t_sensor_in_odom: np.ndarray,
        R_sensor_to_odom: np.ndarray,
        position_noise: float = 0.0,
        orientation_noise: float = 0.0,
    ) -> None:
        """Fuse a pointcloud into the map.

        Args:
            points_sensor_xyz: (N, 3) float32 points in the sensor (or
                body) frame from which the pose ``(R, t)`` is taken.
            t_sensor_in_odom: (3,) sensor-origin position in odom.
            R_sensor_to_odom: (3, 3) rotation that takes sensor-frame
                vectors to odom-frame vectors.
        """
        if points_sensor_xyz.ndim != 2 or points_sensor_xyz.shape[1] < 3:
            raise ValueError(
                f"points must have shape (N, >=3), got {points_sensor_xyz.shape}"
            )
        self._map.input_pointcloud(
            np.ascontiguousarray(points_sensor_xyz, dtype=np.float32),
            ["x", "y", "z"],
            np.asarray(R_sensor_to_odom, dtype=np.float32),
            np.asarray(t_sensor_in_odom, dtype=np.float32),
            float(position_noise),
            float(orientation_noise),
        )

    def tick(self) -> None:
        """Update variance + time layers (call after integrate())."""
        self._map.update_variance()
        self._map.update_time()

    # ──────────────────────────────────────────────────────────────────
    def get_elevation_emcupy(self) -> np.ndarray:
        """Return the elevation in elevation_mapping_cupy's *native* layout:
        ``elev[row, col]`` at world (cx + (row - H/2)*res, cy + (col - W/2)*res).
        Rows index X, cols index Y.  Outer 1-cell border stripped."""
        elevation = self._map.get_layer("elevation").get()
        is_valid = self._map.get_layer("is_valid").get()
        elevation = np.asarray(elevation, dtype=np.float32).copy()
        elevation[is_valid == 0] = np.nan
        return elevation[1:-1, 1:-1]

    def get_elevation_for_navgraph(self) -> np.ndarray:
        """Return the elevation in nav_graph's expected convention
        (rows index Y / col index X, NE at [0, 0]).  Composed of:
        transpose (to swap which axis is X vs Y) followed by rot180
        (to put north/east at the origin)."""
        return np.flipud(np.fliplr(self.get_elevation_emcupy().T)).copy()

    def center_xy(self) -> tuple[float, float]:
        """Current cell-aligned map center in odom, as (cx, cy).

        Note: ``move_to`` rounds the requested center to the nearest cell,
        so this may differ from the requested robot xy by up to half a
        cell.  Always use the value returned here as ``origin_x/origin_y``
        when handing the elevation off to nav_graph."""
        try:
            import cupy as cp

            c = cp.asnumpy(self._map.center)
        except Exception:
            c = np.asarray(self._map.center)
        return float(c[0]), float(c[1])

    def grid_shape(self) -> tuple[int, int]:
        """(H, W) of the elevation grid we hand to nav_graph (post-strip)."""
        n = int(self._map.cell_n) - 2
        return n, n
