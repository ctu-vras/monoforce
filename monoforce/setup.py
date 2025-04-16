from setuptools import find_packages, setup
from glob import glob
import os


package_name = 'monoforce'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(include=[package_name]),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config', 'rviz'), glob('config/rviz/*')),
        (os.path.join('share', package_name, 'config', 'robots'), glob('config/robots/*')),
        (os.path.join('share', package_name, 'config', 'meshes'), glob('config/meshes/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ruslan Agishev',
    maintainer_email='agishrus@fel.cvut.cz',
    description='MonoForce: Learnable Image-conditioned Physics Engine',
    license='BSD-3-Clause',
    entry_points={
        'console_scripts': [
            'terrain_encoder = monoforce.nodes.terrain_encoder:main',
            'physics_engine = monoforce.nodes.physics_engine:main',
        ],
    },
)
