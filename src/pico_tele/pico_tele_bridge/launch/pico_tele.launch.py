from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    default_params = PathJoinSubstitution(
        [FindPackageShare("pico_tele_bridge"), "config", "pico_tele.yaml"]
    )
    params_file = LaunchConfiguration("params_file")
    device_id = LaunchConfiguration("device_id")

    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", default_value=default_params),
            DeclareLaunchArgument(
                "device_id",
                default_value="",
                description="Optional PICO device ID; empty locks to the first device.",
            ),
            Node(
                package="pico_tele_bridge",
                executable="pico_sdk_bridge",
                name="pico_sdk_bridge",
                output="screen",
                parameters=[params_file, {"device_id": device_id}],
            ),
            Node(
                package="pico_tele_bridge",
                executable="pico_gesture_mapper",
                name="pico_gesture_mapper",
                output="screen",
                parameters=[params_file],
            ),
            Node(
                package="pico_tele_bridge",
                executable="pico_command_router",
                name="pico_command_router",
                output="screen",
                parameters=[params_file],
            ),
        ]
    )
