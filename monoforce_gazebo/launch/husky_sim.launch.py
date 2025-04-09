from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

import os

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    world_path = os.path.join(
        get_package_share_directory('monoforce_gazebo'),
        'worlds',
        'clearpath_playpen.world'
    )

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(
                    get_package_share_directory('husky_gazebo'),
                    'launch',
                    'gazebo.launch.py'
                )
            ]),
            launch_arguments={'world_path': world_path}.items()
        ),
    ])
