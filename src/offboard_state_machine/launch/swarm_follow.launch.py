#!/usr/bin/env python3
"""
Single drone launch file for offboard FSM
Launches one drone that takes off and goes to the first point from a CSV trajectory file
"""

import csv
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import EnvironmentVariable
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def read_first_trajectory_point(drone_id: str):
    """
    Read the first point from the drone's trajectory CSV file.
    
    Args:
        drone_id: The drone ID (e.g., '0', '1', '2')
    
    Returns:
        tuple: (x, y, z) position of the first trajectory point
    """
    csv_path = f"data/3drone_trajectories_001_-001/drone_{drone_id}_traj.csv"
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Trajectory file not found: {csv_path}")
    
    with open(csv_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        
        # Skip header if exists
        header = next(csv_reader)
        
        # Read first data row
        first_row = next(csv_reader)
        
        # Assuming CSV format: x, y, z
        x = float(first_row[0])
        y = float(first_row[1])
        z = float(first_row[2])
        
        return x, y, z

def generate_launch_description() -> LaunchDescription:
    """
    Launch a single drone with FSM node.
    The drone will:
    1. Take off to the first point in the trajectory CSV (using minimum jerk trajectory)
    2. Hover at that position
    """
    
    # Get drone ID
    drone_id_default = os.environ.get('DRONE_ID', '0')
    
    drone_id_arg = DeclareLaunchArgument(
        'drone_id',
        default_value=drone_id_default,
        description='Drone ID'
    )
    
    # Read the first trajectory point
    goto_x, goto_y, goto_z = read_first_trajectory_point(drone_id_default)
    
    # Calculate takeoff altitude from z position (NED frame: negative = up)
    takeoff_altitude = abs(goto_z)
    
    drone_id = LaunchConfiguration('drone_id')
    
    fsm_node = Node(
        package="offboard_state_machine",
        executable="offboard_fsm_node",
        name=["offboard_fsm_node_", drone_id],
        output="screen",
        parameters=[{
            "drone_id": drone_id,
            "takeoff_alt": takeoff_altitude,
            "takeoff_time": 10.0,
            "climb_rate": 1.0,
            "landing_time": 2.0,
            
            # GOTO target (first CSV point)
            'goto_x': goto_x,
            'goto_y': goto_y,
            'goto_z': goto_z,
            'goto_tol': 0.05,
            'goto_max_vel': 1.0,
            'goto_accel_time': 4.0,
            
            # Takeoff position (use payload_offset with inward_offset=0)
            "inward_offset": 0.0,           # Must be 0 for direct control
            "payload_offset_x": goto_x,     # Takeoff x = first CSV point x
            "payload_offset_y": goto_y,     # Takeoff y = first CSV point y
            
            "num_drones": 1,
            "timer_period": 0.02,
            "alt_tol": 0.01,
        }],
    )

    return LaunchDescription([
        drone_id_arg,
        fsm_node,
    ])