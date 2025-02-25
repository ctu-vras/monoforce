cmake_minimum_required(VERSION 3.9)
project(monoforce)

find_package(PkgConfig REQUIRED)

# Catkin
find_package(catkin REQUIRED)
catkin_python_setup()
catkin_package()

find_package(catkin REQUIRED COMPONENTS ${CATKIN_DEPS})
include_directories(${catkin_INCLUDE_DIRS})
link_directories(${catkin_LIBRARY_DIRS})

catkin_package(
    CATKIN_DEPENDS ${CATKIN_DEPS}
)

install(
    PROGRAMS
        nodes/diff_physics.py
        nodes/terrain_encoder.py
        nodes/monoforce_node.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
    DIRECTORY
        launch/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
