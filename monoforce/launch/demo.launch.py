from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess


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
            ]),
        # optional: web-cam input data generation
        # Node(
        #     package='monoforce',
        #     executable='camera_publisher',
        #     name='camera_publisher_node',
        #     output='screen'
        # ),

        # node to play bag file
        ExecuteProcess(
            cmd=['ros2', 'bag', 'play',
                 '/media/ruslan/VRAS-DATA 4TB 2/outdoor_dataset/25-03-19-petrin/marv_2025-03-19-15-35-24_mcap'],
            output='screen'
        )
    ])
