from pathlib import Path
from rosbags.rosbag2 import Writer
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

# Update paths to match your system
src = Path('/home/rohang73/Downloads/rosbag2_2026_04_25-11_52_30/rosbag2_2026_04_25-11_52_30_0.db3')
dst = Path('/home/rohang73/Downloads/rosbag2_2026_04_25-11_52_30/trimmed_bag_output')

# Load the explicit ROS 2 Jazzy system message profiles
typestore = get_typestore(Stores.ROS2_JAZZY)

# Initialize standard Writer with version=8 and explicitly enforce sqlite3 storage
with AnyReader([src], default_typestore=typestore) as reader, Writer(dst, version=8) as writer:
    # Shift 50s and 90s from the true initial timestamp
    start_ns = reader.start_time + (50 * 10**9)
    end_ns = reader.start_time + (90 * 10**9)

    conn_map = {}
    for conn in reader.connections:
        # Link connections through the validated typestore
        conn_map[conn.id] = writer.add_connection(
            conn.topic, 
            conn.msgtype, 
            msgdef=conn.msgdef, 
            typestore=reader.typestore
        )

    # Filter and write messages within the 50s-90s window
    for conn, timestamp, data in reader.messages():
        if start_ns <= timestamp <= end_ns:
            writer.write(conn_map[conn.id], timestamp, data)

print("Trimming complete!")
