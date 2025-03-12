from setuptools import find_packages, setup

package_name = 'monoforce'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=['monoforce']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ruslan',
    maintainer_email='agishrus@fel.cvut.cz',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'publisher = monoforce.nodes.publisher:main',
            'subscriber = monoforce.nodes.subscriber:main',
            'terrain_encoder = monoforce.nodes.terrain_encoder:main',
        ],
    },
)
