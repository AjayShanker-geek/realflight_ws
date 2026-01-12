#!/usr/bin/env python3
"""
Launch smooth-feedback PVA control nodes for realflight swarm.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def read_data_root(params_path: str) -> str:
  if not params_path:
    return ""
  try:
    import yaml  # type: ignore
  except Exception:
    yaml = None

  if yaml is not None:
    try:
      with open(params_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
      if isinstance(data, dict):
        for scope in data.values():
          if not isinstance(scope, dict):
            continue
          params = scope.get("ros__parameters")
          if isinstance(params, dict) and "data_root" in params:
            return str(params["data_root"])
    except Exception:
      return ""

  try:
    with open(params_path, "r", encoding="utf-8") as f:
      in_params = False
      for line in f:
        stripped = line.strip()
        if stripped.startswith("ros__parameters:"):
          in_params = True
          continue
        if in_params and stripped.startswith("data_root:"):
          value = stripped.split("data_root:", 1)[1].strip()
          return value.strip("\"' ")
  except Exception:
    return ""
  return ""


def make_nodes(context):
  total = int(LaunchConfiguration("total_drones").perform(context))
  drone_ids_raw = LaunchConfiguration("drone_ids").perform(context).strip()
  params_file = LaunchConfiguration("params_file")
  params_file_path = LaunchConfiguration("params_file").perform(context)
  traj_base_dir = LaunchConfiguration("traj_base_dir").perform(context)
  use_params_data_root = LaunchConfiguration("use_params_data_root").perform(context).lower() in ("1", "true", "yes")

  if drone_ids_raw:
    drone_ids = [int(x.strip()) for x in drone_ids_raw.split(",") if x.strip()]
  else:
    drone_ids = list(range(total))
  if len(drone_ids) != total:
    raise ValueError(f"Mismatch: drone_ids has {len(drone_ids)} IDs but total_drones={total}")

  if use_params_data_root:
    data_root = read_data_root(params_file_path)
    if data_root:
      traj_base_dir = data_root

  nodes = [
    LogInfo(msg=f"Starting smooth PVA REALFLIGHT swarm with {total} drones")
  ]

  for drone_id in drone_ids:
    override = {
      "use_raw_traj": LaunchConfiguration("use_raw_traj").perform(context).lower() in ("1", "true", "yes"),
    }
    if traj_base_dir:
      override["data_root"] = traj_base_dir
    params = [params_file, override]
    nodes.append(
      Node(
        package="pva_control",
        executable="pva_smooth_feedback_control_node",
        name=f"pva_smooth_realflight_{drone_id}",
        output="screen",
        arguments=[str(drone_id), str(total)],
        parameters=params,
      )
    )
  return nodes


def generate_launch_description():
  default_total = os.environ.get("TOTAL_DRONES", "3")
  default_traj = os.environ.get("MULTILIFT_DATA",
                                "data/realflight_traj_new")
  default_params = PathJoinSubstitution([
    FindPackageShare("pva_control"),
    "config",
    "pva_smooth_feedback_realflight_params.yaml",
  ])
  return LaunchDescription([
    DeclareLaunchArgument("total_drones", default_value=default_total,
                          description="Total number of drones to launch"),
    DeclareLaunchArgument("drone_ids", default_value="",
                          description="Comma-separated drone IDs (default 0..total-1)"),
    DeclareLaunchArgument("params_file", default_value=default_params,
                          description="YAML file with smooth PVA realflight parameters"),
    DeclareLaunchArgument("traj_base_dir", default_value=default_traj,
                          description="Directory containing per-drone trajectory CSVs"),
    DeclareLaunchArgument("use_params_data_root", default_value="true",
                          description="Override traj_base_dir from params_file data_root"),
    DeclareLaunchArgument("use_raw_traj", default_value="false",
                          description="Use *_traj_raw_20hz.csv instead of smoothed CSVs"),
    OpaqueFunction(function=make_nodes),
  ])
