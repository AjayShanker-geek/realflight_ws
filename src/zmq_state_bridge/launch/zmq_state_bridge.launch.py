from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    mode_arg = DeclareLaunchArgument(
        "mode",
        default_value="drone",
        description="bridge mode: 'drone' or 'gcs'"
    )
    drone_ids_arg = DeclareLaunchArgument(
        "drone_ids_csv",
        default_value="0,1,2",
        description="comma-separated drone ids (used in gcs mode)"
    )
    state_push_endpoint_arg = DeclareLaunchArgument(
        "state_push_endpoint",
        default_value="tcp://127.0.0.1:5555",
        description="ZMQ PUSH connect endpoint for state (drone mode)"
    )
    cmd_sub_endpoint_arg = DeclareLaunchArgument(
        "cmd_sub_endpoint",
        default_value="tcp://127.0.0.1:5560",
        description="ZMQ SUB connect endpoint for command (drone mode)"
    )
    state_pull_bind_arg = DeclareLaunchArgument(
        "state_pull_bind",
        default_value="tcp://*:5555",
        description="ZMQ PULL bind endpoint for state (gcs mode)"
    )
    cmd_pub_bind_arg = DeclareLaunchArgument(
        "cmd_pub_bind",
        default_value="tcp://*:5560",
        description="ZMQ PUB bind endpoint for command (gcs mode)"
    )
    poll_period_ms_arg = DeclareLaunchArgument(
        "poll_period_ms",
        default_value="10",
        description="poll period in milliseconds"
    )

    bridge_node = Node(
        package="zmq_state_bridge",
        executable="zmq_state_bridge_node",
        name="zmq_state_bridge",
        output="screen",
        parameters=[{
            "mode": LaunchConfiguration("mode"),
            "state_push_endpoint": LaunchConfiguration("state_push_endpoint"),
            "cmd_sub_endpoint": LaunchConfiguration("cmd_sub_endpoint"),
            "state_pull_bind": LaunchConfiguration("state_pull_bind"),
            "cmd_pub_bind": LaunchConfiguration("cmd_pub_bind"),
            "poll_period_ms": LaunchConfiguration("poll_period_ms"),
            "drone_ids_csv": LaunchConfiguration("drone_ids_csv"),
        }],
    )

    return LaunchDescription([
        mode_arg,
        drone_ids_arg,
        state_push_endpoint_arg,
        cmd_sub_endpoint_arg,
        state_pull_bind_arg,
        cmd_pub_bind_arg,
        poll_period_ms_arg,
        bridge_node,
    ])
