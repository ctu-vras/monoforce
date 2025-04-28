from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.substitutions import PythonExpression


def generate_launch_description():
    # Define launch arguments
    DeclareLaunchArgument(
        'img_topics',
        default_value='/camera/image_raw/compressed',
        description='Comma-separated list of image topics'
    )
    DeclareLaunchArgument(
        'camera_info_topics',
        default_value='/camera/image_raw/camera_info',
        description='Comma-separated list of camera info topics'
    )
    DeclareLaunchArgument('robot_frame', default_value='base_link', description='Robot base frame id')
    DeclareLaunchArgument('fixed_frame', default_value='odom', description='Fixed world frame id')
    DeclareLaunchArgument('use_sim_time', default_value='true', description='Use simulation time')

    # Node definition with PythonExpression to parse arguments into lists
    return LaunchDescription([
        TimerAction(
            period=0.0,
            actions=[
                Node(
                    package='monoforce',
                    executable='monoforce_node',
                    name='monoforce_node',
                    output='screen',
                    parameters=[{
                        'img_topics': PythonExpression([
                            '["" + topic.strip() for topic in "', LaunchConfiguration('img_topics'), '" .split(",")]'
                        ]),
                        'camera_info_topics': PythonExpression([
                            '["" + topic.strip() for topic in "', LaunchConfiguration('camera_info_topics'), '" .split(",")]'
                        ]),
                        'robot_frame': LaunchConfiguration('robot_frame'),
                        'fixed_frame': LaunchConfiguration('fixed_frame'),
                        'use_sim_time': LaunchConfiguration('use_sim_time')
                    }]
                )
            ]
        )
    ])
