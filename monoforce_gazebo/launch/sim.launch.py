from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
import os
import subprocess
from glob import glob


world_name = 'Harmonic Terrain'
# world_name = 'Rubicon'
world_path = os.path.expanduser(f'~/.gz/fuel/fuel.gazebosim.org/openrobotics/models/{world_name.lower()}/*/model.sdf')
world_path = glob(world_path.replace(' ', '%20'))[0]

def download_world_if_missing(context, *args, **kwargs):
    if not os.path.exists(world_path):
        print(f'[INFO] {world_name} world not found locally. Downloading from Gazebo Fuel...')
        subprocess.run([
            'gz', 'fuel', 'download',
            '-u', f'https://fuel.gazebosim.org/1.0/OpenRobotics/models/{world_name}'.replace(' ', '%20'),
        ], check=True)
    else:
        print(f'[INFO] {world_name} world already downloaded.')
    return []


def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=download_world_if_missing),
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', "$HOME/.gz/fuel/fuel.gazebosim.org/openrobotics/models"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    FindPackageShare('ros_gz_sim').find('ros_gz_sim'),
                    'launch', 'gz_sim.launch.py'
                )
            ),
            launch_arguments={'gz_args': world_path}.items(),
        )
    ])
