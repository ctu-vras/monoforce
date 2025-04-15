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
                {'img_topics': ['/camera_front/image_color/compressed'],
                 'camera_info_topics': ['/camera_front/camera_info'],
                 'robot_frame': 'base_link',
                 'fixed_frame': 'odom',}
            ])
    ])
