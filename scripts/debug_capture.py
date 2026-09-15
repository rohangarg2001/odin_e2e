"""Capture the actual intermediate arrays without changing graph calculations."""
import json
import numpy as np
import cupy as cp
from nav_msgs.msg import OccupancyGrid, MapMetaData
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header, String
from rcl_interfaces.msg import ParameterEvent
from rclpy.parameter import Parameter

DEBUG='/odin_nav_graph_node/debug/'
LAYER_NAMES=('elevation','variance','is_valid','traversability','time','upper_bound','is_upper_bound')
DEBUG_TYPES={
    DEBUG+'grid_info':'nav_msgs/msg/MapMetaData',
    DEBUG+'occupancy':'nav_msgs/msg/OccupancyGrid',
    DEBUG+'frontier_mask':'nav_msgs/msg/OccupancyGrid',
    DEBUG+'elevation_graph_input':'sensor_msgs/msg/Image',
    DEBUG+'traversability_graph_input':'sensor_msgs/msg/Image',
    DEBUG+'local_cost_grid':'sensor_msgs/msg/Image',
    DEBUG+'global_collision_grid':'sensor_msgs/msg/Image',
    DEBUG+'frontier_cells':'sensor_msgs/msg/PointCloud2',
    DEBUG+'clustered_frontiers':'sensor_msgs/msg/PointCloud2',
    DEBUG+'local_nodes':'sensor_msgs/msg/PointCloud2',
    DEBUG+'graph_nodes_all':'sensor_msgs/msg/PointCloud2',
    DEBUG+'frame_diagnostics':'std_msgs/msg/String',
    DEBUG+'parameters':'rcl_interfaces/msg/ParameterEvent',
    **{DEBUG+'mapper_'+name:'sensor_msgs/msg/Image' for name in LAYER_NAMES},
    DEBUG+'mapper_normals':'sensor_msgs/msg/Image',
}

class DebugCapture:
    def __init__(self,node):
        self.node=node
        self.values={}
        b=node.builder
        self._wrap(b,'_compute_traversability',self._traversability)
        self._wrap(b.frontier_detector,'detect_from_occupancy_grid',self._detection)
        self._wrap(b.frontier_detector,'cluster_frontiers',self._clusters)
        self._wrap(b.local_generator,'build_graph_from_grid_map',self._local)

    def _wrap(self,obj,name,capture):
        original=getattr(obj,name)
        def call(*args,**kwargs):
            result=original(*args,**kwargs)
            capture(args,kwargs,result)
            return result
        setattr(obj,name,call)

    def reset(self):
        self.values={'clustered':np.empty((0,2))}

    def _traversability(self,args,kwargs,result):
        self.values['elevation_input']=np.array(args[0],copy=True)
        self.values['traversability_input']=np.array(result,copy=True)

    def _detection(self,args,kwargs,result):
        grid,h,w,ox,oy,res=args
        self.values.update(occupancy=np.array(grid,copy=True),
                           occupancy_origin=(ox,oy),raw_frontiers=np.array(result[0],copy=True))

    def _clusters(self,args,kwargs,result):
        self.values['clustered']=np.array(result,copy=True)

    def _local(self,args,kwargs,result):
        self.values['local_cost']=kwargs['image'].detach().cpu().numpy().copy()
        self.values['local_pos']=result[0].detach().cpu().numpy().copy()
        self.values['local_types']=result[1].detach().cpu().numpy().copy()

    def image(self,array,stamp,encoding='32FC1'):
        array=np.ascontiguousarray(array,dtype=np.uint8 if encoding=='mono8' else np.dtype('<f4'))
        m=Image(header=Header(stamp=stamp,frame_id=self.node.frame_id))
        m.height,m.width=array.shape[:2]
        m.encoding=encoding
        m.is_bigendian=0
        m.step=array.strides[0]
        m.data=array.tobytes()
        return m

    def grid_info(self,origin,shape,stamp):
        m=MapMetaData()
        m.map_load_time=stamp
        m.resolution=self.node.emap.resolution
        m.height,m.width=map(int,shape)
        m.origin.position.x=float(origin[0]);m.origin.position.y=float(origin[1])
        m.origin.orientation.w=1.
        return m

    def grid(self,array,origin,stamp):
        m=OccupancyGrid(header=Header(stamp=stamp,frame_id=self.node.frame_id))
        m.info=self.grid_info(origin,array.shape,stamp)
        m.data=np.asarray(array,dtype=np.int8).ravel().tolist()
        return m

    def xy_cloud(self,xy,stamp):
        xyz=np.c_[xy,np.zeros(len(xy),np.float32)].astype(np.float32)
        return self.node._make_xyz_intensity_cloud(xyz,np.ones(len(xy)),stamp)

    def all_nodes_cloud(self,result,stamp):
        ids=result.node_ids.detach().cpu().numpy()
        if len(ids) and (ids.min()<0 or ids.max()>np.iinfo(np.uint32).max):
            raise ValueError('PointCloud2 node_id cannot represent these IDs; graph_state retains int64 IDs')
        names=['x','y','z','node_id','node_type']
        formats=['<f4','<f4','<f4','<u4','<u4']
        scores=result.node_scores
        if scores is not None:
            names+=['score_'+name for name in result.score_layer_names]
            formats+=['<f4']*len(result.score_layer_names)
        packed=np.empty(len(ids),dtype=np.dtype(list(zip(names,formats))))
        pos=result.node_positions.detach().cpu().numpy()
        for i,name in enumerate(('x','y','z')):packed[name]=pos[:,i]
        packed['node_id']=ids;packed['node_type']=result.node_types.detach().cpu().numpy()
        if scores is not None:
            sc=scores.detach().cpu().numpy()
            for i,name in enumerate(result.score_layer_names):packed['score_'+name]=sc[:,i]
        m=PointCloud2(header=Header(stamp=stamp,frame_id=self.node.frame_id))
        m.height=1;m.width=len(ids);m.point_step=packed.dtype.itemsize;m.row_step=m.width*m.point_step
        m.fields=[PointField(name=name,offset=packed.dtype.fields[name][1],
                            datatype=PointField.UINT32 if fmt=='<u4' else PointField.FLOAT32,count=1)
                  for name,fmt in zip(names,formats)]
        m.data=packed.tobytes();m.is_dense=True
        return m

    def record(self,rec,cloud,pose,transform,pose_source,pose_dt,tf_dt):
        n=self.node;v=self.values;stamp=cloud.header.stamp;emap=n.emap
        shape=emap.grid_shape();cx,cy=emap.center_xy();res=emap.resolution
        # Float mapper grids use increasing world Y rows and X columns.
        # Their sample centers follow the elevation adapter exactly.
        origin=(cx-(shape[1]/2+.5)*res,cy-(shape[0]/2+.5)*res)
        rec.write(DEBUG+'grid_info',self.grid_info(origin,shape,stamp))
        layers=cp.asnumpy(emap._map.elevation_map)[:,1:-1,1:-1]
        normals=cp.asnumpy(emap._map.normal_map)[:,1:-1,1:-1]
        if emap.native_rows_axis=='x':
            layers=layers.transpose(0,2,1);normals=normals.transpose(0,2,1)
        for i,name in enumerate(LAYER_NAMES):
            rec.write(DEBUG+'mapper_'+name,self.image(layers[i],stamp))
        rec.write(DEBUG+'mapper_normals',self.image(normals.transpose(1,2,0),stamp,'32FC3'))
        rec.write(DEBUG+'elevation_graph_input',self.image(v['elevation_input'],stamp))
        rec.write(DEBUG+'traversability_graph_input',self.image(v['traversability_input'],stamp))
        rec.write(DEBUG+'occupancy',self.grid(v['occupancy'],v['occupancy_origin'],stamp))
        rec.write(DEBUG+'local_cost_grid',self.image(v['local_cost'],stamp,'mono8'))
        rec.write(DEBUG+'global_collision_grid',self.image(n.builder.global_builder.occ_grid,stamp,'mono8'))
        front=v['raw_frontiers'];mask=np.zeros(shape,np.int8)
        if len(front):
            pixel=np.rint((front-np.asarray(v['occupancy_origin']))/res-.5).astype(int)
            if not ((pixel>=0).all() and (pixel[:,0]<shape[1]).all() and (pixel[:,1]<shape[0]).all()):
                raise ValueError('Frontier cell outside detector grid')
            mask[pixel[:,1],pixel[:,0]]=100
            if (v['occupancy'][pixel[:,1],pixel[:,0]]!=-1).any():
                raise ValueError('Raw frontier kernel emitted a non-unknown cell')
        rec.write(DEBUG+'frontier_mask',self.grid(mask,v['occupancy_origin'],stamp))
        rec.write(DEBUG+'frontier_cells',self.xy_cloud(front,stamp))
        rec.write(DEBUG+'clustered_frontiers',self.xy_cloud(v['clustered'],stamp))
        rec.write(DEBUG+'local_nodes',n._make_xyz_intensity_cloud(v['local_pos'],v['local_types'],stamp))
        rec.write(DEBUG+'graph_nodes_all',self.all_nodes_cloud(n._last_result,stamp))
        p,q=transform.transform.translation,transform.transform.rotation
        pp,pq=pose.pose.pose.position,pose.pose.pose.orientation
        report=dict(frame=n.frame_count,cloud_stamp_ns=stamp.sec*1000000000+stamp.nanosec,
                    receipt_ns=rec.timestamp,pose_source=pose_source,pose_delta_seconds=pose_dt,
                    transform_delta_seconds=tf_dt,map_center=[cx,cy],resolution=res,shape=shape,
                    elevation_grid_origin=origin,occupancy_grid_origin=v['occupancy_origin'],
                    cloud_to_imu=[p.x,p.y,p.z,q.x,q.y,q.z,q.w],
                    matched_pose=[pp.x,pp.y,pp.z,pq.x,pq.y,pq.z,pq.w],
                    raw_frontier_cells=len(front),clustered_frontiers=len(v['clustered']),
                    local_nodes=len(v['local_pos']),timings_ms=getattr(n,'_last_timings',{}),
                    graph_nodes=n._last_result.num_nodes,graph_edges=n._last_result.num_edges)
        rec.write(DEBUG+'frame_diagnostics',String(data=json.dumps(report,allow_nan=False)))
        if n.frame_count==1:
            event=ParameterEvent(stamp=stamp,node=n.get_fully_qualified_name())
            event.new_parameters=[p.to_parameter_msg() for p in n.get_parameters(n.list_parameters([],0).names)]
            rec.write(DEBUG+'parameters',event)
