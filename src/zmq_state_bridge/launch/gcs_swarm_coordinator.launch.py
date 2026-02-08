from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    drone_ids_arg = DeclareLaunchArgument(
        "drone_ids_csv",
        default_value="0,1,2",
        description="comma-separated drone ids"
    )
    vicon_names_arg = DeclareLaunchArgument(
        "vicon_names_csv",
        default_value="",
        description="comma-separated vicon object names (optional)"
    )
    vicon_prefix_arg = DeclareLaunchArgument(
        "vicon_topic_prefix",
        default_value="/vrpn_mocap/",
        description="vicon topic prefix"
    )
    vicon_suffix_arg = DeclareLaunchArgument(
        "vicon_topic_suffix",
        default_value="/pose",
        description="vicon topic suffix"
    )
    takeoff_alt_arg = DeclareLaunchArgument(
        "takeoff_alt",
        default_value="0.4",
        description="takeoff altitude in meters (positive up)"
    )
    alt_tol_arg = DeclareLaunchArgument(
        "alt_tol",
        default_value="0.05",
        description="altitude tolerance in meters"
    )
    hover_wait_arg = DeclareLaunchArgument(
        "hover_wait_time",
        default_value="5.0",
        description="hover wait time before TRAJ"
    )
    vicon_is_enu_arg = DeclareLaunchArgument(
        "vicon_is_enu",
        default_value="true",
        description="vicon frame is ENU (z up)"
    )
    use_vicon_alt_arg = DeclareLaunchArgument(
        "use_vicon_altitude",
        default_value="true",
        description="gate GOTO based on vicon altitude"
    )

    node = Node(
        package="zmq_state_bridge",
        executable="gcs_swarm_coordinator_node",
        name="gcs_swarm_coordinator",
        output="screen",
        parameters=[{
            "drone_ids_csv": LaunchConfiguration("drone_ids_csv"),
            "vicon_names_csv": LaunchConfiguration("vicon_names_csv"),
            "vicon_topic_prefix": LaunchConfiguration("vicon_topic_prefix"),
            "vicon_topic_suffix": LaunchConfiguration("vicon_topic_suffix"),
            "takeoff_alt": LaunchConfiguration("takeoff_alt"),
            "alt_tol": LaunchConfiguration("alt_tol"),
            "hover_wait_time": LaunchConfiguration("hover_wait_time"),
            "vicon_is_enu": LaunchConfiguration("vicon_is_enu"),
            "use_vicon_altitude": LaunchConfiguration("use_vicon_altitude"),
        }],
    )

    return LaunchDescription([
        drone_ids_arg,
        vicon_names_arg,
        vicon_prefix_arg,
        vicon_suffix_arg,
        takeoff_alt_arg,
        alt_tol_arg,
        hover_wait_arg,
        vicon_is_enu_arg,
        use_vicon_alt_arg,
        node,
    ])
