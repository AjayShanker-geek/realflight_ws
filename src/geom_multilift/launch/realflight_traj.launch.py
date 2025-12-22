#!/usr/bin/env python3
"""
Launch per-drone realflight trajectory publisher plus the swarm offboard FSM.
Relies on DRONE_ID env var to decide which CSV to load so each drone runs its
own node instead of looping over all vehicles (geom_multilift style).
"""

import csv
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def read_first_trajectory_point(base_path: str, drone_id: str):
    csv_path = f"{base_path}/drone_{drone_id}_traj.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Trajectory file not found: {csv_path}")

    with open(csv_path, "r") as csvfile:
        csv_reader = csv.reader(csvfile)
        _ = next(csv_reader)  # header
        first_row = next(csv_reader)

    x = float(first_row[1])
    y = float(first_row[2])
    z = float(first_row[3])
    return x, y, z


def launch_setup(context, *args, **kwargs):
    drone_id = LaunchConfiguration("drone_id").perform(context)
    total_drones = int(LaunchConfiguration("total_drones").perform(context))
    csv_base_path = LaunchConfiguration("csv_base_path").perform(context)
    timer_period = float(LaunchConfiguration("timer_period").perform(context))
    yaw_setpoint = float(LaunchConfiguration("yaw_setpoint").perform(context))

    goto_x, goto_y, goto_z = read_first_trajectory_point(csv_base_path, drone_id)
    takeoff_altitude = abs(goto_z)

    fsm_node = Node(
        package="offboard_state_machine",
        executable="offboard_fsm_node",
        name=f"offboard_fsm_node_{drone_id}",
        output="screen",
        parameters=[{
            "drone_id": int(drone_id),
            "takeoff_alt": takeoff_altitude,
            "takeoff_time": 10.0,
            "climb_rate": 1.0,
            "landing_time": 2.0,
            "goto_x": goto_x,
            "goto_y": goto_y,
            "goto_z": goto_z,
            "goto_tol": 0.05,
            "goto_max_vel": 1.0,
            "goto_accel_time": 4.0,
            "inward_offset": 0.0,
            "payload_offset_x": goto_x,
            "payload_offset_y": goto_y,
            "num_drones": total_drones,
            "timer_period": 0.02,
            "alt_tol": 0.01,
        }],
    )

    traj_node = Node(
        package="geom_multilift",
        executable="realflight_traj_node",
        name=f"realflight_traj_node_{drone_id}",
        output="screen",
        arguments=[drone_id, str(total_drones)],
        parameters=[{
            "csv_base_path": csv_base_path,
            "timer_period": timer_period,
            "yaw_setpoint": yaw_setpoint,
        }],
    )

    return [
        LogInfo(msg=f"Launching FSM + trajectory node for drone {drone_id}"),
        fsm_node,
        traj_node,
    ]


def generate_launch_description():
    default_drone_id = os.environ.get("DRONE_ID", "0")
    return LaunchDescription([
        DeclareLaunchArgument(
            "drone_id",
            default_value=default_drone_id,
            description="Drone ID (also driven by DRONE_ID env var)"
        ),
        DeclareLaunchArgument(
            "total_drones",
            default_value=os.environ.get("TOTAL_DRONES", "1"),
            description="Total number of drones in the swarm"
        ),
        DeclareLaunchArgument(
            "csv_base_path",
            default_value="data/3drone_trajectories_001_-001",
            description="Base directory containing drone_X_traj.csv files"
        ),
        DeclareLaunchArgument(
            "timer_period",
            default_value="0.01",
            description="Streaming period (seconds) for trajectory setpoints"
        ),
        DeclareLaunchArgument(
            "yaw_setpoint",
            default_value="3.1415926",
            description="Yaw (rad) applied to every setpoint"
        ),
        OpaqueFunction(function=launch_setup),
    ])
