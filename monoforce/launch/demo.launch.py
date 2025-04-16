from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, TimerAction
from ament_index_python.packages import get_package_share_directory
import os


monoforce_pkg_path = get_package_share_directory('monoforce')
rviz_cfg_path = os.path.join(monoforce_pkg_path, 'config', 'rviz', 'monoforce.rviz')
bag_path = '/media/ruslan/VRAS-DATA 4TB 2/outdoor_dataset/25-03-19-petrin/marv_2025-03-19-15-35-24'

def generate_launch_description():
    return LaunchDescription([
        # Start playing the bag file first
        ExecuteProcess(
            cmd=['ros2', 'bag', 'play', bag_path,
                 '--clock',
                 '--loop',
                 '--remap',
                     '/camera_left/camera_info:=/camera_left/image_color/camera_info',
                     '/camera_front/camera_info:=/camera_front/image_color/camera_info',
                     '/camera_right/camera_info:=/camera_right/image_color/camera_info',
                     '/camera_rear/camera_info:=/camera_rear/image_color/camera_info'
                 ],
            output='screen'
        ),

        # Delay terrain_encoder_node to ensure TF is being published
        TimerAction(
            period=10.0,  # seconds delay: for bag file to start
            actions=[
                Node(
                    package='monoforce',
                    executable='terrain_encoder',
                    name='terrain_encoder_node',
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
                    ]
                ),
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
                            'max_age': 1.0,
                            'use_sim_time': True
                        }
                    ]
                )
            ]
        ),

        # RVIZ2 node
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            parameters=[
                {'use_sim_time': True}
            ],
            arguments=['-d', rviz_cfg_path]
        )
    ])
