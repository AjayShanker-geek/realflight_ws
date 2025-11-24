#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import LogInfo


def generate_launch_description():
    # Declare launch arguments
    total_drones_arg = DeclareLaunchArgument(
        'total_drones',
        default_value='3',
        description='Total number of drones in the swarm'
    )
    
    csv_base_path_arg = DeclareLaunchArgument(
        'csv_base_path',
        default_value='data/3drone_trajectories_001_-001',
        description='Base path for CSV trajectory files'
    )
    
    timer_period_arg = DeclareLaunchArgument(
        'timer_period',
        default_value='0.01',
        description='Timer period in seconds (default 100Hz)'
    )
    
    yaw_setpoint_arg = DeclareLaunchArgument(
        'yaw_setpoint',
        default_value='3.1415926',
        description='Yaw setpoint for all drones (radians)'
    )
    
    return LaunchDescription([
        total_drones_arg,
        csv_base_path_arg,
        timer_period_arg,
        yaw_setpoint_arg,
        OpaqueFunction(function=launch_setup)
    ])


def launch_setup(context, *args, **kwargs):
    total_drones = int(LaunchConfiguration('total_drones').perform(context))
    csv_base_path = LaunchConfiguration('csv_base_path').perform(context)
    timer_period = LaunchConfiguration('timer_period').perform(context)
    yaw_setpoint = LaunchConfiguration('yaw_setpoint').perform(context)
    
    nodes = []
    
    # Create follow_traj node for each drone
    for drone_id in range(total_drones):
        csv_path = f"{csv_base_path}/drone_{drone_id}_traj.csv"
        
        node = Node(
            package='traj_test',
            executable='follow_traj_node',
            name=f'follow_traj_node_{drone_id}',
            namespace='',
            arguments=[str(drone_id), str(total_drones)],
            parameters=[{
                'timer_period': float(timer_period),
                'csv_path': csv_path,
                'yaw_setpoint': float(yaw_setpoint),
            }],
            output='screen',
            emulate_tty=True,
        )
        nodes.append(node)
    
    # Add swarm coordinator node
    coordinator_node = Node(
        package='traj_test',
        executable='swarm_coordinator_node',
        name='swarm_coordinator',
        namespace='',
        arguments=[str(total_drones)],
        output='screen',
        emulate_tty=True,
    )
    nodes.append(coordinator_node)
    
    # Add log messages
    nodes.append(
        LogInfo(msg=f'Launching {total_drones} follow_traj nodes + coordinator')
    )
    
    return nodes