#!/usr/bin/env python3
"""Save connectivity review, runtime provenance and code beside a generated bag."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import torch
import cupy
import nav_graph
import elevation_mapping_cupy

ap=argparse.ArgumentParser();ap.add_argument('bag',type=Path);a=ap.parse_args()
b=a.bag;repo=Path(__file__).resolve().parents[1]
g=json.loads((b/'final_graph.json').read_text());r=json.loads((b/'run_report.json').read_text())
ids=g['node_ids'];parent={i:i for i in ids};degree=Counter()
def find(x):
    while parent[x]!=x:
        parent[x]=parent[parent[x]];x=parent[x]
    return x
for u,v in g['edge_index']:
    parent[find(u)]=find(v);degree[u]+=1;degree[v]+=1
sizes=sorted(Counter(find(i) for i in ids).values(),reverse=True)
pos=np.asarray(g['node_positions']);end=np.load(b/'map_preview_data.npz')['trajectory'][-1,:2]
nearest=int(np.linalg.norm(pos[:,:2]-end,axis=1).argmin());component=find(ids[nearest])
quality=dict(components=len(sizes),component_sizes=sizes,
             isolated_nodes=sum(degree[i]==0 for i in ids),
             robot_component_nodes=sum(find(i)==component for i in ids),
             reachable_frontiers_by_connectivity=sum(find(i)==component and t==2 for i,t in zip(ids,g['node_types'])),
             nearest_node_to_final_pose_m=float(np.linalg.norm(pos[nearest,:2]-end)),
             odometry_sources=dict(Counter(x['pose_source_topic'] for x in r['frames'])),
             max_pose_dt_sec=max(x['pose_dt_sec'] for x in r['frames']),
             max_tf_dt_sec=max(x['transform_dt_sec'] for x in r['frames']),
             median_frame_processing_ms=float(np.median([x['elapsed_ms'] for x in r['frames']])),
             score_layers=g['score_layer_names'])
(b/'quality_review.json').write_text(json.dumps(quality,indent=2))
def git(*args):
    return subprocess.check_output(['git',*args],cwd=repo,text=True).strip()
environment=dict(python=sys.version,torch=torch.__version__,cupy=cupy.__version__,numpy=np.__version__,
                 gpu=torch.cuda.get_device_name(0),graph_module=nav_graph.__file__,
                 elevation_module=elevation_mapping_cupy.__file__,
                 graph_sha=git('-C','nav_graph_gpu','rev-parse','HEAD'),project_head=git('rev-parse','HEAD'),
                 random_seed=r.get('random_seed'),note='Local working changes are included in code_snapshot and code_changes.patch')
(b/'environment.json').write_text(json.dumps(environment,indent=2))
(b/'code_changes.patch').write_text(git('diff')+'\n')
snapshot=b/'code_snapshot'
for rel in ['scripts','ros2_ws/src/odin_nav_graph']:
    shutil.copytree(repo/rel,snapshot/rel,dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.egg-info'))
for rel in ['setup_env.sh','requirements.txt','.gitmodules']:
    shutil.copy2(repo/rel,snapshot/rel)
shutil.copy2(repo/'docs/odin_debug_recording.md',b/'DEBUGGING.md')
shutil.copy2(Path(r['input'])/'metadata.yaml',b/'source_metadata.yaml')
for p in b.glob('*.mcap'):
    print('MCAP',p.name,p.stat().st_size,'bytes')
print(json.dumps({k:v for k,v in quality.items() if k!='component_sizes'},indent=2))
