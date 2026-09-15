#!/usr/bin/env python3
"""GPU geometry regression: asymmetric observations must stay fixed as the map rolls."""
import numpy as np
import rclpy
from odin_nav_graph.elevation_map import ElevationMapWrapper

def world_cells(m):
    e = m.get_elevation_emcupy()
    rows, cols = np.where(np.isfinite(e))
    cx, cy = m.center_xy()
    return np.c_[cx+(rows-e.shape[0]/2)*m.resolution,
                 cy+(cols-e.shape[1]/2)*m.resolution, e[rows,cols]]

rclpy.init()
try:
    m = ElevationMapWrapper(native_rows_axis='y')
    patches = []
    for x0,y0,z in [(1.5,-0.6,-1.0),(-0.7,2.0,-0.5)]:
        patches.extend((x,y,z) for x in np.arange(x0-.15,x0+.15,.025)
                       for y in np.arange(y0-.15,y0+.15,.025))
    m.integrate(np.asarray(patches,np.float32),np.zeros(3),np.eye(3))
    before = world_cells(m)
    for x0,y0,z in [(1.5,-0.6,-1.0),(-0.7,2.0,-0.5)]:
        chosen=before[np.abs(before[:,2]-z)<.05]
        assert len(chosen)>5
        assert np.linalg.norm(chosen[:,:2].mean(0)-[x0,y0])<.2
    m.move_to(np.array([.7,-.4,0.]))
    after = world_cells(m)
    def sorted_cells(a):
        return a[np.lexsort((np.round(a[:,1],3),np.round(a[:,0],3)))]
    np.testing.assert_allclose(sorted_cells(before),sorted_cells(after),atol=2e-6)
    grid=m.get_elevation_for_navgraph()
    cy,cx=np.where(np.isfinite(grid))
    cx0,cy0=m.center_xy()
    xy=np.c_[cx0+(grid.shape[1]/2-1-cx)*.1,
             cy0+(grid.shape[0]/2-1-cy)*.1,grid[cy,cx]]
    np.testing.assert_allclose(sorted_cells(xy),sorted_cells(after),atol=2e-6)
    print('PASS: distinct X/Y patches, measured heights, rolling-map world position, graph-grid conversion')
finally:
    rclpy.shutdown()
