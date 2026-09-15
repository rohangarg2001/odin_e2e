#!/usr/bin/env python3
"""Read back a derived MCAP, verify its counts, and render elevation/graph previews."""
import argparse
from collections import Counter
import json
import hashlib
import copy
from pathlib import Path
import numpy as np
import yaml
import rosbag2_py as bag
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2, Image
from nav_msgs.msg import OccupancyGrid
from debug_capture import DEBUG, DEBUG_TYPES
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

ap=argparse.ArgumentParser()
ap.add_argument('bag',type=Path)
args=ap.parse_args()
root=args.bag
report=json.loads((root/'run_report.json').read_text())
assert report['status']=='complete', report['status']
r=bag.SequentialReader()
r.open(bag.StorageOptions(uri=str(root),storage_id='mcap'),bag.ConverterOptions('',''))
counts=Counter()
source_hashes={name:hashlib.sha256() for name in report.get('source_topic_sha256',{})}
first,last={},{}
while r.has_next():
    topic,raw,stamp=r.read_next()
    counts[topic]+=1
    if topic in source_hashes:
        h=source_hashes[topic]
        h.update(int(stamp).to_bytes(8,'little',signed=True))
        h.update(len(raw).to_bytes(8,'little'))
        h.update(raw)
    first.setdefault(topic,stamp);last[topic]=stamp
    if topic.endswith(('/elevation_cloud','/graph_nodes','/frontier_cloud')):
        msg=deserialize_message(raw,PointCloud2)
        values=np.frombuffer(msg.data,dtype='<f4').reshape(-1,4)
        assert msg.header.frame_id=='odom'
        assert np.isfinite(values).all()
    if topic in DEBUG_TYPES:
        typ=DEBUG_TYPES[topic]
        if typ=='sensor_msgs/msg/Image':
            msg=deserialize_message(raw,Image)
            assert (msg.height,msg.width)==(120,120)
            assert len(msg.data)==msg.step*msg.height
        elif typ=='nav_msgs/msg/OccupancyGrid':
            msg=deserialize_message(raw,OccupancyGrid)
            assert len(msg.data)==msg.info.width*msg.info.height
            assert set(msg.data)<=({0,100} if topic.endswith('frontier_mask') else {-1,0,100})
        elif typ=='sensor_msgs/msg/PointCloud2':
            msg=deserialize_message(raw,PointCloud2)
            assert len(msg.data)==msg.width*msg.point_step
            assert msg.header.frame_id=='odom'
assert dict(counts)==report['topic_counts'], (counts,report['topic_counts'])
for suffix in ('elevation_cloud','graph_nodes','frontier_cloud','graph_edges','graph_state'):
    assert counts['/odin_nav_graph_node/'+suffix]==report['processed_clouds']
assert len(list(root.glob('*.mcap')))==1
if report.get('debug_enabled'):
    for topic in DEBUG_TYPES:
        assert counts[topic]==(1 if topic.endswith('/parameters') else report['processed_clouds']), topic
    for topic,digest in source_hashes.items():
        assert digest.hexdigest()==report['source_topic_sha256'][topic], topic
        assert counts[topic]==report['source_topic_counts'].get(topic,0), topic
validation=dict(passed=True, topic_counts=dict(counts), first_receipt_ns=first,
                last_receipt_ns=last, one_mcap=True, finite_clouds=True,
                duration_seconds=(max(last.values())-min(first.values()))*1e-9)
validation['original_odin_payloads_and_timestamps_verified']=bool(source_hashes)
(root/'verification.json').write_text(json.dumps(validation,indent=2))

a=np.load(root/'map_preview_data.npz')
elev=a['elevation']; nodes=a['node_positions']; types=a['node_types']
trajectory=a['trajectory']; ids=a['node_ids']; edges=a['edge_index']
id_to_row={int(nid):i for i,nid in enumerate(ids)}
segments=np.asarray([[nodes[id_to_row[int(u)],:2],nodes[id_to_row[int(v)],:2]] for u,v in edges])
vmin,vmax=np.percentile(elev[:,2],[2,98])
fig,axes=plt.subplots(1,2,figsize=(11,12),layout='constrained',sharex=True,sharey=True)
fig.suptitle('Odin elevation map and navigation graph',fontsize=18,fontweight='bold')
for ax in axes:
    ax.set_aspect('equal')
    ax.set_facecolor('#f1f3f5')
    ax.set_xlabel('Odin odom X (m)')
    ax.grid(alpha=.18)
axes[0].set_ylabel('Odin odom Y (m)')
im=axes[0].scatter(elev[:,0],elev[:,1],c=elev[:,2],s=5,marker='s',
                   cmap='terrain',vmin=vmin,vmax=vmax,linewidths=0,rasterized=True)
axes[0].plot(trajectory[:,0],trajectory[:,1],color='#18283a',lw=1.1,label='Recorded trajectory')
axes[0].scatter(*trajectory[0,:2],c='#00b779',s=65,edgecolors='white',zorder=5,label='Start')
axes[0].scatter(*trajectory[-1,:2],c='#ec4660',s=65,marker='X',edgecolors='white',zorder=5,label='End')
axes[0].set_title('Elevation accumulated from rolling maps')
axes[0].legend(loc='upper left',fontsize=8)
fig.colorbar(im,ax=axes[0],shrink=.5,label='Height in Odin odom (m)')
axes[1].scatter(elev[:,0],elev[:,1],c='#cbd1d7',s=5,marker='s',linewidths=0,rasterized=True)
axes[1].add_collection(LineCollection(segments,colors='#4180aa',linewidths=.55,alpha=.7))
axes[1].scatter(nodes[types==1,0],nodes[types==1,1],s=6,c='#185780',label='Graph nodes',zorder=3)
axes[1].scatter(nodes[types==2,0],nodes[types==2,1],s=35,c='#f6a623',
                edgecolors='#8c5d00',linewidths=.5,label='Frontiers',zorder=4)
axes[1].plot(trajectory[:,0],trajectory[:,1],color='#171f2a',lw=.9,alpha=.8,label='Recorded trajectory')
axes[1].set_title(f"Final graph: {len(nodes)} nodes, {len(edges)} edges, {(types==2).sum()} frontiers")
axes[1].legend(loc='upper left',fontsize=8)
fig.supxlabel(f"{report['processed_clouds']:,} LiDAR frames · {validation['duration_seconds']:.1f} s recording · 0.10 m cells\n"
              "Offline mapping and frontier detection; robot-frame calibration and planning validation remain.",
              fontsize=10)
fig.savefig(root/'elevation_and_graph.png',dpi=180)
plt.close(fig)
# Keep the viewing command reusable for every newly processed bag.
config_path=Path(__file__).resolve().parents[1] / 'ros2_ws/src/odin_nav_graph/config/odin_nav_graph.rviz'
cfg=yaml.safe_load(config_path.read_text())
vm=cfg['Visualization Manager']
for display in vm['Displays']:
    topic=display.get('Topic',{}).get('Value')
    if topic=='/odin1/odometry_highfreq':
        display['Topic']['Value']='/odin_nav_graph_node/matched_odometry'
    if topic=='/odin1/cloud_slam':
        display.update(Enabled=False,Value=False)
center=(elev[:,:2].min(0)+elev[:,:2].max(0))/2
view=vm['Views']['Current']
view['Focal Point']={'X':float(center[0]),'Y':float(center[1]),'Z':float(np.median(elev[:,2]))}
view['Distance']=float(max(25.,np.ptp(elev[:,:2],axis=0).max()*1.1))
if report.get('debug_enabled'):
    base=next(d for d in vm['Displays'] if d.get('Topic',{}).get('Value')=='/odin_nav_graph_node/graph_nodes')
    raw=copy.deepcopy(base)
    raw.update(Name='Raw frontier cells - before clustering',Enabled=True,Value=True,
               **{'Color Transformer':'FlatColor','Color':'0; 220; 200','Size (m)':0.07})
    raw['Topic']['Value']=DEBUG+'frontier_cells'
    vm['Displays'].append(raw)
    all_nodes=copy.deepcopy(base)
    all_nodes.update(Name='All GPU nodes - includes IDs and scores',Enabled=False,Value=False,
                     **{'Channel Name':'node_type'})
    all_nodes['Topic']['Value']=DEBUG+'graph_nodes_all'
    vm['Displays'].append(all_nodes)
(root/'odin_graph.rviz').write_text(yaml.safe_dump(cfg,sort_keys=False))
print(json.dumps(dict(verification='passed',messages=sum(counts.values()),
                     duration=validation['duration_seconds'], nodes=len(nodes),edges=len(edges),
                     frontiers=int((types==2).sum()), preview=str(root/'elevation_and_graph.png')),indent=2))
