#!/usr/bin/env python3
"""
Launch all SITL geometric multilift nodes in one shot.
Spins one node per drone_id in [0, total_drones-1] using the SITL inputs:
 - /payload_odom
 - /simulation/position_drone_<i+1>
 - PX4 odom/local_position topics per drone namespace
"""

import csv
import os
import sys
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def read_first_trajectory_point(base_dir: str, drone_id: int):
  csv_path = Path(base_dir) / f"drone_{drone_id}_traj_smoothed_100hz.csv"
  if not csv_path.exists():
    print(f"ERROR: Trajectory file not found: {csv_path}", file=sys.stderr)
    raise FileNotFoundError(f"Trajectory file not found: {csv_path}")

  with csv_path.open("r") as csvfile:
    csv_reader = csv.reader(csvfile)
    try:
      _ = next(csv_reader)
    except StopIteration:
      raise ValueError(f"Trajectory file is empty: {csv_path}")
    try:
      first_row = next(csv_reader)
    except StopIteration:
      raise ValueError(f"Trajectory file has no data rows: {csv_path}")

  x = float(first_row[1])
  y = float(first_row[2])
  z = float(first_row[3])
  return x, y, z


def make_nodes(context):
  total = int(LaunchConfiguration("total_drones").perform(context))
  params_file = LaunchConfiguration("params_file")
  traj_base_dir = LaunchConfiguration("traj_base_dir").perform(context)
  log_dir = LaunchConfiguration("log_dir").perform(context)
  launch_fsm = LaunchConfiguration("launch_fsm").perform(context).lower() in ("1", "true", "yes")
  takeoff_time = float(LaunchConfiguration("takeoff_time").perform(context))
  climb_rate = float(LaunchConfiguration("climb_rate").perform(context))
  landing_time = float(LaunchConfiguration("landing_time").perform(context))
  goto_tol = float(LaunchConfiguration("goto_tol").perform(context))
  goto_max_vel = float(LaunchConfiguration("goto_max_vel").perform(context))
  goto_accel_time = float(LaunchConfiguration("goto_accel_time").perform(context))
  fsm_timer_period = float(LaunchConfiguration("fsm_timer_period").perform(context))
  alt_tol = float(LaunchConfiguration("alt_tol").perform(context))
  inward_offset = float(LaunchConfiguration("inward_offset").perform(context))

  nodes = [
    LogInfo(msg=f"Starting SITL geom_multilift swarm with {total} drones (launch_fsm={launch_fsm})")
  ]
  for drone_id in range(total):
    log_path = ""
    if log_dir:
      log_path = os.path.join(log_dir, f"geom_multilift_sitl_drone_{drone_id}.csv")
    params = [params_file]
    if log_path:
      params.append({"log_path": log_path})
    nodes.append(
      Node(
        package="geom_multilift",
        executable="geom_multilift_sitl_node",
        name=f"geom_multilift_sitl_{drone_id}",
        output="screen",
        arguments=[str(drone_id), str(total)],
        parameters=params,
      )
    )
    if launch_fsm:
      goto_x, goto_y, goto_z = read_first_trajectory_point(traj_base_dir, drone_id)
      takeoff_altitude = abs(goto_z)
      nodes.append(
        Node(
          package="offboard_state_machine",
          executable="offboard_fsm_node",
          name=f"offboard_fsm_node_{drone_id}",
          output="screen",
          parameters=[{
            "drone_id": drone_id,
            "takeoff_alt": takeoff_altitude,
            "takeoff_time": takeoff_time,
            "climb_rate": climb_rate,
            "landing_time": landing_time,
            "goto_x": goto_x,
            "goto_y": goto_y,
            "goto_z": goto_z,
            "goto_tol": goto_tol,
            "goto_max_vel": goto_max_vel,
            "goto_accel_time": goto_accel_time,
            "inward_offset": inward_offset,
            "payload_offset_x": goto_x,
            "payload_offset_y": goto_y,
            "num_drones": total,
            "timer_period": fsm_timer_period,
            "alt_tol": alt_tol,
          }],
        )
      )
  return nodes


def generate_launch_description():
  default_total = os.environ.get("TOTAL_DRONES", "3")
  default_traj = os.environ.get("MULTILIFT_DATA",
                                "data/realflight_traj_new")
  default_params = PathJoinSubstitution([
    FindPackageShare("geom_multilift"),
    "config",
    "geom_multilift_sitl_params.yaml",
  ])
  return LaunchDescription([
    DeclareLaunchArgument("total_drones", default_value=default_total,
                          description="Total number of drones to launch"),
    DeclareLaunchArgument("params_file", default_value=default_params,
                          description="YAML file with geom_multilift SITL parameters"),
    DeclareLaunchArgument("traj_base_dir", default_value=default_traj,
                          description="Directory containing per-drone trajectory CSVs"),
    DeclareLaunchArgument("log_dir", default_value="",
                          description="Optional directory for per-drone CSV logs"),
    DeclareLaunchArgument("launch_fsm", default_value="true",
                          description="Launch offboard FSM nodes per drone"),
    DeclareLaunchArgument("takeoff_time", default_value="10.0",
                          description="FSM takeoff time (s)"),
    DeclareLaunchArgument("climb_rate", default_value="1.0",
                          description="FSM climb rate (m/s)"),
    DeclareLaunchArgument("landing_time", default_value="2.0",
                          description="FSM landing time (s)"),
    DeclareLaunchArgument("goto_tol", default_value="0.05",
                          description="FSM goto tolerance (m)"),
    DeclareLaunchArgument("goto_max_vel", default_value="1.0",
                          description="FSM goto max velocity (m/s)"),
    DeclareLaunchArgument("goto_accel_time", default_value="4.0",
                          description="FSM goto accel time (s)"),
    DeclareLaunchArgument("fsm_timer_period", default_value="0.02",
                          description="FSM timer period (s)"),
    DeclareLaunchArgument("alt_tol", default_value="0.01",
                          description="FSM altitude tolerance (m)"),
    DeclareLaunchArgument("inward_offset", default_value="0.0",
                          description="FSM inward offset (m)"),
    OpaqueFunction(function=make_nodes),
  ])
