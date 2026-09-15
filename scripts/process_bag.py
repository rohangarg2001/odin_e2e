#!/usr/bin/env python3
"""Process every Odin cloud synchronously; publish and write one derived MCAP bag."""
import argparse
from bisect import bisect_left
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import rclpy
import rosbag2_py as bag
from rclpy.serialization import deserialize_message, serialize_message
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker
import yaml

from odin_nav_graph.nav_graph_node import OdinNavGraphNode, parse_xyz_points, stamp_to_sec
from debug_capture import DebugCapture, DEBUG_TYPES

CLOUD = '/odin1/cloud_raw'
ODOM = '/odin1/odometry_highfreq'
ODOM_LOW = '/odin1/odometry'
MATCHED_ODOM = '/odin_nav_graph_node/matched_odometry'
TF = '/odin1/tf'
PREFIX = '/odin_nav_graph_node/'
OUTPUT_TYPES = {
    PREFIX+'elevation_cloud': 'sensor_msgs/msg/PointCloud2',
    PREFIX+'graph_nodes': 'sensor_msgs/msg/PointCloud2',
    PREFIX+'frontier_cloud': 'sensor_msgs/msg/PointCloud2',
    PREFIX+'graph_edges': 'visualization_msgs/msg/Marker',
    PREFIX+'graph_state': 'std_msgs/msg/String',
    ODOM: 'nav_msgs/msg/Odometry',
    ODOM_LOW: 'nav_msgs/msg/Odometry',
    MATCHED_ODOM: 'nav_msgs/msg/Odometry',
    '/tf': 'tf2_msgs/msg/TFMessage',
}

def reader(path, topics):
    r = bag.SequentialReader()
    r.open(bag.StorageOptions(uri=str(path), storage_id='mcap'),
           bag.ConverterOptions('', ''))
    r.set_filter(bag.StorageFilter(topics=topics))
    return r

def stamp_ns(stamp):
    return stamp.sec * 1000000000 + stamp.nanosec

def calibration_and_poses(path):
    r = reader(path, [ODOM, ODOM_LOW, TF])
    poses, calibrations = [], []
    tf_checks = 0
    while r.has_next():
        topic, raw, receipt = r.read_next()
        if topic in (ODOM, ODOM_LOW):
            msg = deserialize_message(raw, Odometry)
            if (msg.header.frame_id, msg.child_frame_id) != ('odom', 'imu'):
                raise ValueError('Expected odom -> imu odometry')
            poses.append((stamp_ns(msg.header.stamp), msg, topic))
        else:
            for t in deserialize_message(raw, TFMessage).transforms:
                if (t.header.frame_id, t.child_frame_id) != ('imu', 'lidar'):
                    continue
                p, q = t.transform.translation, t.transform.rotation
                values = np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w])
                calibrations.append((stamp_ns(t.header.stamp), t, values))
                tf_checks += 1
    if not poses or not calibrations:
        raise ValueError('Missing Odin odometry or imu -> lidar calibration')
    poses.sort(key=lambda p: p[0])
    calibrations.sort(key=lambda item: item[0])
    return poses, calibrations, tf_checks

class RecordedPublisher:
    def __init__(self, live, topic, owner):
        self.live, self.topic, self.owner = live, topic, owner
    def publish(self, msg):
        if isinstance(msg, PointCloud2):
            xyz = parse_xyz_points(msg)
            if not np.isfinite(xyz).all():
                raise ValueError('Nonfinite output: '+self.topic)
            if self.topic.endswith('elevation_cloud'):
                self.owner.last_elevation = xyz
        self.owner.write(self.topic, msg)
        self.live.publish(msg)

class Recorder:
    def __init__(self, path, source_topics=(), debug=False):
        self.writer = bag.SequentialWriter()
        self.writer.open(bag.StorageOptions(uri=str(path), storage_id='mcap',
                         max_bagfile_size=0, max_bagfile_duration=0),
                         bag.ConverterOptions('', ''))
        types=dict(OUTPUT_TYPES)
        if debug:
            types.update(DEBUG_TYPES)
        originals={t.name:t for t in source_topics}
        types.update({t.name:t.type for t in source_topics})
        for i, (topic, typ) in enumerate(types.items()):
            extra={}
            if topic in originals:
                extra['offered_qos_profiles']=originals[topic].offered_qos_profiles
            self.writer.create_topic(bag.TopicMetadata(
                id=i, name=topic, type=typ, serialization_format='cdr', **extra))
        self.counts = Counter()
        self.timestamp = 0
        self.last_elevation = None
        self.source_hashes={t.name:hashlib.sha256() for t in source_topics}
        self.source_counts=Counter()
    def write(self, topic, msg):
        self.writer.write(topic, serialize_message(msg), self.timestamp)
        self.counts[topic] += 1

    def write_original(self,topic,raw,receipt):
        self.writer.write(topic,raw,receipt)
        self.counts[topic]+=1
        self.source_counts[topic]+=1
        digest=self.source_hashes[topic]
        digest.update(int(receipt).to_bytes(8,'little',signed=True))
        digest.update(len(raw).to_bytes(8,'little'))
        digest.update(raw)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input', type=Path)
    ap.add_argument('output', type=Path, help='new bag directory, must not exist')
    ap.add_argument('--max-frames', type=int, default=0, help='0 processes every cloud')
    ap.add_argument('--debug', action='store_true', help='Include all original Odin topics and exact intermediate arrays')
    ap.add_argument('--seed', type=int, default=0, help='Seed Python, NumPy, PyTorch and CuPy sampling')
    ap.add_argument('--native-rows-axis', choices=['x','y'], required=True,
                    help='Mapper backend convention; verify with check_geometry.py')
    args = ap.parse_args()
    import random, torch, cupy as cp
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); cp.random.seed(args.seed)
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    poses, calibrations, tf_checks = calibration_and_poses(args.input)
    extrinsic = calibrations[0][2]
    tf_times = [item[0] for item in calibrations]
    times = [p[0] for p in poses]
    print('AUDIT', len(poses), 'poses;', tf_checks, 'time-varying sensor transforms', flush=True)
    params_path = Path(__file__).resolve().parents[1] / 'ros2_ws/src/odin_nav_graph/config/default_params.yaml'
    config = yaml.safe_load(params_path.read_text())
    params = config['odin_nav_graph_node']['ros__parameters']
    params.update(cloud_extrinsic_calibrated=True, cloud_extrinsic_source='tf',
                  cloud_to_robot_translation=extrinsic[:3].tolist(),
                  cloud_to_robot_quaternion=extrinsic[3:].tolist(),
                  elevation_native_rows_axis='row_'+args.native_rows_axis,
                  max_cloud_frames=0, process_every_n=1, viz_z_offset=0.,
                  out_directory='')
    # Place reproducibility artifacts beside the bag until MCAP creates its directory.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_params = args.output.parent / (args.output.name + '_params.yaml')
    run_params.write_text(yaml.safe_dump(config))
    rclpy.init(args=['--ros-args','--params-file',str(run_params)])
    node = OdinNavGraphNode()
    source_topics=[]
    if args.debug:
        inventory=reader(args.input,[CLOUD])
        source_topics=[t for t in inventory.get_all_topics_and_types() if t.name.startswith('/odin1/')]
    rec = Recorder(args.output,source_topics=source_topics,debug=args.debug)
    debug=DebugCapture(node) if args.debug else None
    run_params.replace(args.output / 'run_params.yaml')
    for attr, suffix in [('elev_pub','elevation_cloud'), ('graph_pub','graph_nodes'),
                         ('frontier_pub','frontier_cloud'), ('edges_pub','graph_edges')]:
        setattr(node, attr, RecordedPublisher(getattr(node, attr), PREFIX+suffix, rec))

    rows, trajectory, mosaic = [], [], {}
    r = reader(args.input, [t.name for t in source_topics] if args.debug else [CLOUD])
    max_dt = 0.
    status = 'failed'
    try:
        while r.has_next():
            topic, raw, receipt = r.read_next()
            if args.max_frames and len(rows) >= args.max_frames:
                break
            if args.debug:
                rec.write_original(topic,raw,receipt)
                if topic != CLOUD:
                    continue
            cloud = deserialize_message(raw, PointCloud2)
            if cloud.header.frame_id != 'lidar':
                raise ValueError('Expected lidar-frame raw cloud')
            target = stamp_ns(cloud.header.stamp)
            i = bisect_left(times, target)
            choices = [j for j in (i-1, i) if 0 <= j < len(poses)]
            index = min(choices, key=lambda j: abs(times[j]-target))
            dt = abs(times[index]-target)*1e-9
            max_dt = max(dt, max_dt)
            if dt > node.odom_match_max_dt:
                raise ValueError('No synchronized odometry for cloud')
            pose = poses[index][1]
            i = bisect_left(tf_times, target)
            choices = [j for j in (i-1, i) if 0 <= j < len(calibrations)]
            ti = min(choices, key=lambda j: abs(tf_times[j]-target))
            tf_dt = abs(tf_times[ti]-target)*1e-9
            if tf_dt > node.odom_match_max_dt:
                raise ValueError('No synchronized sensor transform')
            sensor_tf = calibrations[ti][1]
            rec.timestamp = receipt
            if not args.debug:
                rec.write(poses[index][2], pose)
            rec.write(MATCHED_ODOM, pose)
            dynamic = TransformStamped()
            dynamic.header = pose.header
            dynamic.child_frame_id = pose.child_frame_id
            p, q = pose.pose.pose.position, pose.pose.pose.orientation
            dynamic.transform.translation.x = p.x
            dynamic.transform.translation.y = p.y
            dynamic.transform.translation.z = p.z
            dynamic.transform.rotation = q
            rec.write('/tf', TFMessage(transforms=[dynamic, sensor_tf]))
            node.odom_buf.clear()
            node.extrinsic_buf.clear()
            node.sensor_tf_callback(TFMessage(transforms=[sensor_tf]))
            node.odom_callback(pose)
            previous = rec.counts.copy()
            tick = time.perf_counter()
            if debug:
                debug.reset()
            node.cloud_callback(cloud)
            for suffix in ('elevation_cloud','graph_nodes','frontier_cloud','graph_edges'):
                topic = PREFIX + suffix
                if rec.counts[topic] != previous[topic] + 1:
                    raise RuntimeError('Cloud produced no output on '+topic)
            result = node._last_result
            data = {name: getattr(result, name).detach().cpu().tolist()
                    for name in ('node_positions','node_ids','node_types','edge_index','edge_weights')}
            data['node_scores'] = None if result.node_scores is None else result.node_scores.detach().cpu().tolist()
            data['frontiers'] = None if result.frontiers is None else result.frontiers.detach().cpu().tolist()
            data.update(schema_version=1, frame_id=node.frame_id, stamp_ns=target,
                        score_layer_names=result.score_layer_names)
            ids = np.asarray(data['node_ids'])
            if len(np.unique(ids)) != len(ids):
                raise ValueError('Duplicate graph node IDs')
            if not np.isin(np.asarray(data['edge_index']), ids).all():
                raise ValueError('Graph edge endpoint missing from node IDs')
            rec.write(PREFIX+'graph_state', String(data=json.dumps(data, allow_nan=False)))
            if debug:
                debug.record(rec,cloud,pose,sensor_tf,poses[index][2],dt,tf_dt)
            xyz = rec.last_elevation
            cells = np.rint(xyz[:,:2] / node.emap.resolution).astype(np.int64)
            mosaic.update(zip(map(tuple, cells.tolist()), xyz.tolist()))
            trajectory.append([float(pose.pose.pose.position.x),float(pose.pose.pose.position.y),
                               float(pose.pose.pose.position.z)])
            rows.append(dict(frame=len(rows)+1, receipt_ns=receipt, stamp_ns=target,
                             pose_dt_sec=dt, pose_source_topic=poses[index][2],
                             transform_dt_sec=tf_dt, valid_cells=len(xyz), nodes=result.num_nodes,
                             frontiers=int((result.node_types==2).sum().item()),
                             edges=result.num_edges, elapsed_ms=(time.perf_counter()-tick)*1000))
            if len(rows) % 100 == 0:
                print('PROGRESS',json.dumps(rows[-1]),flush=True)
                (args.output / 'progress.json').write_text(json.dumps(rows[-1],indent=2))
        if not rows:
            raise ValueError('No clouds processed')
        if args.debug and not args.max_frames:
            original_metadata=yaml.safe_load((args.input/'metadata.yaml').read_text())['rosbag2_bagfile_information']
            expected={x['topic_metadata']['name']:x['message_count'] for x in original_metadata['topics_with_message_count']
                      if x['topic_metadata']['name'].startswith('/odin1/') and x['message_count']}
            if dict(rec.source_counts)!=expected:
                raise ValueError('Original Odin topic counts were not preserved')
        (args.output / 'final_graph.json').write_text(json.dumps(data,allow_nan=False))
        np.savez_compressed(args.output / 'map_preview_data.npz',
                            elevation=np.asarray(list(mosaic.values()),dtype=np.float32),
                            local_elevation=rec.last_elevation, trajectory=np.asarray(trajectory),
                            node_positions=np.asarray(data['node_positions']),
                            node_ids=np.asarray(data['node_ids']), node_types=np.asarray(data['node_types']),
                            edge_index=np.asarray(data['edge_index']))
        status = 'complete'
    finally:
        # Closing the writer finalizes metadata.yaml and the single MCAP.
        rec.writer.close()
        node.destroy_node()
        rclpy.shutdown()
        report = dict(status=status, input=str(args.input.resolve()),
                      output=str(args.output.resolve()), processed_clouds=len(rows),
                      calibration_checks=tf_checks, cloud_to_imu_first=extrinsic.tolist(),
                      calibration_source='nearest recorded imu->lidar transform per cloud',
                      maximum_pose_delta_seconds=max_dt,
                      elapsed_seconds=time.perf_counter()-started,
                      topic_counts=dict(rec.counts), frames=rows,
                      last_frame=rows[-1] if rows else None)
        report['debug_enabled']=args.debug
        report['random_seed']=args.seed
        report['source_topic_counts']=dict(rec.source_counts)
        report['source_topic_sha256']={name:d.hexdigest() for name,d in rec.source_hashes.items()}
        (args.output / 'run_report.json').write_text(json.dumps(report,indent=2))
    files = list(args.output.glob('*.mcap'))
    if len(files) != 1:
        raise RuntimeError('Expected one MCAP file')
    digest = hashlib.file_digest(files[0].open('rb'), 'sha256').hexdigest()
    (args.output / 'SHA256SUMS').write_text(digest+'  '+files[0].name+'\n')
    print('COMPLETE',str(args.output),'frames',len(rows),'sha256',digest,flush=True)

if __name__ == '__main__':
    main()
