from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='monoforce',
            executable='monoforce_node',
            name='monoforce_node',
            output='screen',
            parameters=[
                {
                    'img_topics': ['/camera_left/image_color/compressed',
                                   '/camera_front/image_color/compressed',
                                   '/camera_right/image_color/compressed',
                                   '/camera_rear/image_color/compressed'],
                    'camera_info_topics': ['/camera_left/image_color/camera_info',
                                           '/camera_front/image_color/camera_info',
                                           '/camera_right/image_color/camera_info',
                                           '/camera_rear/image_color/camera_info'],
                    'robot_frame': 'base_link',
                    'fixed_frame': 'odom',
                    'use_sim_time': True
                }
            ])
    ])
