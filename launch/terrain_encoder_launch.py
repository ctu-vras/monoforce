from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='monoforce',
            executable='terrain_encoder',
            name='terrain_encoder_node',
            output='screen',
            parameters=[
                {'img_topics': ['/image_raw'], 'camera_info_topics': ['/camera_info'],
                 'robot_frame': 'base_link', 'fixed_frame': 'odom',}
            ]),
        # optional: web-cam input data generation
        Node(
            package='monoforce',
            executable='camera_publisher',
            name='camera_publisher_node',
            output='screen'
        ),
    ])
