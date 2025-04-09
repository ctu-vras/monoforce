from setuptools import setup, find_packages
from glob import glob
import os


package_name = 'monoforce_gazebo'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(include=[package_name]),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
        (os.path.join('share', package_name, 'worlds', 'meshes'), glob('worlds/meshes/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ruslan',
    maintainer_email='agishrus@fel.cvut.cz',
    description='Gazebo simulation for MonoForce model deployment experiments',
    license='BSD-3-Clause',
    entry_points={
        'console_scripts': [
        ],
    },
)
