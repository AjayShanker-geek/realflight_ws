#!/usr/bin/env python3
"""
Launch per-drone L2C controller for SITL.
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
        FindPackageShare("l2c_pkg"),
        "config",
        "l2c_sitl_params.yaml",
    ])
    return LaunchDescription([
        DeclareLaunchArgument("drone_id", default_value=default_drone_id,
                              description="Drone ID"),
        DeclareLaunchArgument("total_drones", default_value=default_total,
                              description="Total drones in swarm"),
        DeclareLaunchArgument("params_file", default_value=default_params,
                              description="YAML file with L2C SITL parameters"),
        LogInfo(msg=["Starting L2C SITL for drone ", LaunchConfiguration("drone_id")]),
        Node(
            package="l2c_pkg",
            executable="l2c_node",
            name="l2c_sitl",
            output="screen",
            arguments=[
                LaunchConfiguration("drone_id"),
                LaunchConfiguration("total_drones"),
            ],
            parameters=[LaunchConfiguration("params_file")],
        ),
    ])
