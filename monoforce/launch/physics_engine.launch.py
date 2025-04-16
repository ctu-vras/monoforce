from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='monoforce',
            executable='physics_engine',
            name='physics_engine_node',
            output='screen',
            parameters=[
                {
                    'gridmap_topic': '/terrain/grid_map',
                    'gridmap_layer': 'elevation',
                    'robot_frame': 'base_link',
                }
            ]
        )
    ])
