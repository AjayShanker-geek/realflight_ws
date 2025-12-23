#!/usr/bin/env python3
"""
Launch all SITL geometric multilift nodes in one shot.
Spins one node per drone_id in [0, total_drones-1] using the SITL inputs:
 - /payload_odom
 - /simulation/position_drone_<i+1>
 - attitude/bodyrate from PX4 odom, linear velocity/accel from PX4 local_position
Also launches the sync_goto state machine and a single swarm coordinator.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def make_nodes(context):
  total = int(LaunchConfiguration("total_drones").perform(context))
  drone_ids_raw = LaunchConfiguration("drone_ids").perform(context).strip()
  params_file = LaunchConfiguration("params_file")
  traj_base_dir = LaunchConfiguration("traj_base_dir").perform(context)
  log_dir = LaunchConfiguration("log_dir").perform(context)
  launch_sync_goto = LaunchConfiguration("launch_sync_goto").perform(context).lower() in ("1", "true", "yes")
  launch_coordinator = LaunchConfiguration("launch_coordinator").perform(context).lower() in ("1", "true", "yes")
  use_raw_traj = LaunchConfiguration("use_raw_traj").perform(context).lower() in ("1", "true", "yes")
  takeoff_time = float(LaunchConfiguration("takeoff_time").perform(context))
  goto_time = float(LaunchConfiguration("goto_time").perform(context))
  sync_timer_period = float(LaunchConfiguration("sync_timer_period").perform(context))
  takeoff_alt = float(LaunchConfiguration("takeoff_alt").perform(context))
  alt_tol = float(LaunchConfiguration("alt_tol").perform(context))
  hover_wait_time = float(LaunchConfiguration("hover_wait_time").perform(context))

  if drone_ids_raw:
    drone_ids = [int(x.strip()) for x in drone_ids_raw.split(",") if x.strip()]
  else:
    drone_ids = list(range(total))
  if len(drone_ids) != total:
    raise ValueError(f"Mismatch: drone_ids has {len(drone_ids)} IDs but total_drones={total}")

  nodes = [
    LogInfo(msg=f"Starting SITL geom_multilift swarm with {total} drones (sync_goto={launch_sync_goto})")
  ]

  if launch_coordinator:
    nodes.append(
      Node(
        package="traj_test",
        executable="swarm_goto_coordinator_node",
        name="swarm_goto_coordinator",
        output="screen",
        parameters=[{
          "total_drones": total,
          "drone_ids": drone_ids,
          "takeoff_alt": takeoff_alt,
          "alt_tol": alt_tol,
          "hover_wait_time": hover_wait_time,
        }],
      )
    )

  for drone_id in drone_ids:
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
    if launch_sync_goto:
      nodes.append(
        Node(
          package="offboard_state_machine",
          executable="offboard_sync_goto_node",
          name=f"offboard_sync_goto_{drone_id}",
          output="screen",
          parameters=[{
            "drone_id": drone_id,
            "takeoff_alt": takeoff_alt,
            "takeoff_time": takeoff_time,
            "goto_time": goto_time,
            "traj_base_dir": traj_base_dir,
            "use_raw_traj": use_raw_traj,
            "timer_period": sync_timer_period,
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
    DeclareLaunchArgument("drone_ids", default_value="",
                          description="Comma-separated drone IDs (default 0..total-1)"),
    DeclareLaunchArgument("params_file", default_value=default_params,
                          description="YAML file with geom_multilift SITL parameters"),
    DeclareLaunchArgument("traj_base_dir", default_value=default_traj,
                          description="Directory containing per-drone trajectory CSVs"),
    DeclareLaunchArgument("use_raw_traj", default_value="false",
                          description="Use *_traj_raw_20hz.csv instead of smoothed CSVs"),
    DeclareLaunchArgument("log_dir", default_value="",
                          description="Optional directory for per-drone CSV logs"),
    DeclareLaunchArgument("launch_sync_goto", default_value="true",
                          description="Launch sync_goto state machine per drone"),
    DeclareLaunchArgument("launch_coordinator", default_value="true",
                          description="Launch a single swarm_goto_coordinator_node"),
    DeclareLaunchArgument("takeoff_alt", default_value="0.4",
                          description="Takeoff altitude for sync_goto/coordinator (m, positive up)"),
    DeclareLaunchArgument("takeoff_time", default_value="10.0",
                          description="sync_goto takeoff time (s)"),
    DeclareLaunchArgument("goto_time", default_value="10.0",
                          description="sync_goto goto time (s)"),
    DeclareLaunchArgument("sync_timer_period", default_value="0.02",
                          description="sync_goto timer period (s)"),
    DeclareLaunchArgument("alt_tol", default_value="0.01",
                          description="Coordinator altitude tolerance (m)"),
    DeclareLaunchArgument("hover_wait_time", default_value="5.0",
                          description="Coordinator hover wait before TRAJ (s)"),
    OpaqueFunction(function=make_nodes),
  ])
