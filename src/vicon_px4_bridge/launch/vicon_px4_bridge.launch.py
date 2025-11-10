from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    """
    Launch file for Vicon PX4 Bridge with configurable frame transformations.
    
    This launch file demonstrates how to use the corrected bridge with
    proper world and body frame transformations.
    """
    
    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('vicon_px4_bridge'),
            'config',
            'vicon_px4_bridge_params.yaml'
        ]),
        description='Path to the configuration YAML file'
    )
    
    # You can also override individual parameters here
    vicon_topic_arg = DeclareLaunchArgument(
        'vicon_topic',
        default_value='/vicon/drone/pose',
        description='Vicon topic name'
    )
    
    input_world_frame_arg = DeclareLaunchArgument(
        'input_world_frame',
        default_value='ENU',
        description='Input world coordinate frame (ENU, NED, etc.)'
    )
    
    output_world_frame_arg = DeclareLaunchArgument(
        'output_world_frame',
        default_value='NED',
        description='Output world coordinate frame (ENU, NED, etc.)'
    )
    
    input_body_frame_arg = DeclareLaunchArgument(
        'input_body_frame',
        default_value='FLU',
        description='Input body coordinate frame (FLU, FRD, etc.)'
    )
    
    output_body_frame_arg = DeclareLaunchArgument(
        'output_body_frame',
        default_value='FRD',
        description='Output body coordinate frame (FLU, FRD, etc.)'
    )
    
    # Create the bridge node
    vicon_px4_bridge_node = Node(
        package='vicon_px4_bridge',
        executable='vicon_px4_bridge_node',
        name='vicon_px4_bridge',
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
            {
                'vicon_topic_name': LaunchConfiguration('vicon_topic'),
                'input_world_frame': LaunchConfiguration('input_world_frame'),
                'output_world_frame': LaunchConfiguration('output_world_frame'),
                'input_body_frame': LaunchConfiguration('input_body_frame'),
                'output_body_frame': LaunchConfiguration('output_body_frame'),
            }
        ],
        # Remap if needed
        remappings=[
            # Example remapping if your Vicon topic has a different name
            # ('/vicon/drone/pose', '/your_actual_vicon_topic'),
        ]
    )
    
    return LaunchDescription([
        config_file_arg,
        vicon_topic_arg,
        input_world_frame_arg,
        output_world_frame_arg,
        input_body_frame_arg,
        output_body_frame_arg,
        vicon_px4_bridge_node,
    ])


# Alternative: Simplified launch without arguments
def generate_simple_launch_description():
    """
    Simplified launch file that uses only the config file.
    Use this if you prefer to configure everything in the YAML file.
    """
    return LaunchDescription([
        Node(
            package='vicon_px4_bridge',
            executable='vicon_px4_bridge_node',
            name='vicon_px4_bridge',
            output='screen',
            parameters=[
                PathJoinSubstitution([
                    FindPackageShare('vicon_px4_bridge'),
                    'config',
                    'vicon_px4_bridge_params.yaml'
                ])
            ],
        )
    ])