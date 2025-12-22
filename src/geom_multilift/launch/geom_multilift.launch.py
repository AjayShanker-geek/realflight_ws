#!/usr/bin/env python3
"""
Launch per-drone geometric multilift controller (C++).
Reads preprocessed offline data and streams attitude + thrust to PX4.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_drone_id = os.environ.get("DRONE_ID", "0")
    default_total = os.environ.get("TOTAL_DRONES", "3")
    default_data = os.environ.get(
        "MULTILIFT_DATA",
        "data/preprocessed/rg_001_-001_3quad_l1m_100hz"
    )
    return LaunchDescription([
        DeclareLaunchArgument("drone_id", default_value=default_drone_id,
                              description="Drone ID"),
        DeclareLaunchArgument("total_drones", default_value=default_total,
                              description="Total drones in swarm"),
        DeclareLaunchArgument("payload_pose_topic",
                              default_value="/vrpn_mocap/multilift_payload/pose",
                              description="Payload PoseStamped topic (ENU)"),
        DeclareLaunchArgument("data_root", default_value=default_data,
                              description="Preprocessed trajectory root"),
        DeclareLaunchArgument("timer_period", default_value="0.01",
                              description="Controller period (s)"),
        DeclareLaunchArgument("l", default_value="1.0",
                              description="Cable length (m)"),
        DeclareLaunchArgument("kq", default_value="9.5",
                              description="Cable direction gain kq (1/s^2)"),
        DeclareLaunchArgument("kw", default_value="3.4",
                              description="Cable angular rate gain kw (1/s)"),
        DeclareLaunchArgument("z_weight", default_value="0.3",
                              description="Blend weight for body z (0..1)"),
        DeclareLaunchArgument("thrust_bias", default_value="0.0",
                              description="Normalized thrust bias"),
        DeclareLaunchArgument("alpha_gain", default_value="0.0",
                              description="DDP feedback gain (alpha)"),
        DeclareLaunchArgument("log_path", default_value="",
                              description="Optional CSV log file path"),
        LogInfo(msg=["Starting geom_multilift C++ for drone ", LaunchConfiguration("drone_id")]),
        Node(
            package="geom_multilift",
            executable="geom_multilift_node",
            name="geom_multilift",
            output="screen",
            arguments=[
                LaunchConfiguration("drone_id"),
                LaunchConfiguration("total_drones"),
            ],
            parameters=[{
                "data_root": LaunchConfiguration("data_root"),
                "timer_period": LaunchConfiguration("timer_period"),
                "payload_pose_topic": LaunchConfiguration("payload_pose_topic"),
                "l": LaunchConfiguration("l"),
                "kq": LaunchConfiguration("kq"),
                "kw": LaunchConfiguration("kw"),
                "z_weight": LaunchConfiguration("z_weight"),
                "thrust_bias": LaunchConfiguration("thrust_bias"),
                "alpha_gain": LaunchConfiguration("alpha_gain"),
                "log_path": LaunchConfiguration("log_path"),
            }],
        ),
    ])
