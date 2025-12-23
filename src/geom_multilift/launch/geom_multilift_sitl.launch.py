#!/usr/bin/env python3
"""
Launch per-drone geometric multilift controller for SITL.
This mirrors px4_offboard/geom_multilift.py inputs:
 - payload odometry on /payload_odom (Odometry, ENU)
 - simulation drone pose on /simulation/position_drone_<i>
 - attitude/bodyrate from PX4 odom, linear velocity/accel from PX4 local_position
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_drone_id = os.environ.get("DRONE_ID", "0")
    default_total = os.environ.get("TOTAL_DRONES", "3")
    default_params = PathJoinSubstitution([
        FindPackageShare("geom_multilift"),
        "config",
        "geom_multilift_sitl_params.yaml",
    ])
    return LaunchDescription([
        DeclareLaunchArgument("drone_id", default_value=default_drone_id,
                              description="Drone ID"),
        DeclareLaunchArgument("total_drones", default_value=default_total,
                              description="Total drones in swarm"),
        DeclareLaunchArgument("params_file", default_value=default_params,
                              description="YAML file with geom_multilift SITL parameters"),
        LogInfo(msg=["Starting SITL geom_multilift C++ for drone ", LaunchConfiguration("drone_id")]),
        Node(
            package="geom_multilift",
            executable="geom_multilift_sitl_node",
            name="geom_multilift_sitl",
            output="screen",
            arguments=[
                LaunchConfiguration("drone_id"),
                LaunchConfiguration("total_drones"),
            ],
            parameters=[LaunchConfiguration("params_file")],
        ),
    ])
