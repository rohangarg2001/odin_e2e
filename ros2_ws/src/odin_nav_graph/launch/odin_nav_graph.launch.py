"""Launch the Odin nav-graph node.

Usage:
    ros2 launch odin_nav_graph odin_nav_graph.launch.py \
        cloud_topic:=/odin1/cloud_raw \
        odom_topic:=/odin1/odometry_highfreq

Then in another terminal:
    ros2 bag play /path/to/rosbag2_2026_04_25-...

RViz: set fixed frame to ``odom`` and add PointCloud2 displays for:
    /odin_nav_graph_node/elevation_cloud
    /odin_nav_graph_node/graph_nodes
    /odin_nav_graph_node/frontier_cloud
plus a Marker display for /odin_nav_graph_node/graph_edges.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory('odin_nav_graph')
    default_rviz = os.path.join(pkg_share, 'config', 'odin_nav_graph.rviz')

    args = [
        DeclareLaunchArgument('cloud_topic', default_value='/odin1/cloud_raw'),
        DeclareLaunchArgument('odom_topic', default_value='/odin1/odometry_highfreq'),
        DeclareLaunchArgument('frame_id', default_value='odom'),
        DeclareLaunchArgument('robot_frame', default_value='odin1_base_link'),
        DeclareLaunchArgument('map_length_xy', default_value='12.0'),
        DeclareLaunchArgument('map_resolution', default_value='0.10'),
        DeclareLaunchArgument('process_every_n', default_value='1'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('rviz', default_value='true',
                              description='Launch RViz2 with a preset config alongside the node.'),
        DeclareLaunchArgument('rviz_config', default_value=default_rviz),
    ]

    node = Node(
        package='odin_nav_graph',
        executable='nav_graph_node',
        name='odin_nav_graph_node',
        output='screen',
        emulate_tty=True,
        parameters=[
            {
                'cloud_topic': LaunchConfiguration('cloud_topic'),
                'odom_topic': LaunchConfiguration('odom_topic'),
                'frame_id': LaunchConfiguration('frame_id'),
                'robot_frame': LaunchConfiguration('robot_frame'),
                'map_length_xy': LaunchConfiguration('map_length_xy'),
                'map_resolution': LaunchConfiguration('map_resolution'),
                'process_every_n': LaunchConfiguration('process_every_n'),
                'use_sim_time': LaunchConfiguration('use_sim_time'),
            }
        ],
    )
    return LaunchDescription(args + [node])
