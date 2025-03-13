from setuptools import find_packages, setup
from glob import glob


package_name = 'monoforce'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(include=['monoforce']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ruslan',
    maintainer_email='agishrus@fel.cvut.cz',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'terrain_encoder = monoforce.nodes.terrain_encoder:main',
            'camera_publisher = monoforce.nodes.camera_publisher:main',
            'terrain_encoder_launch = monoforce.launch.terrain_encoder_launch:generate_launch_description',
        ],
    },
)
