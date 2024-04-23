cmake_minimum_required(VERSION 3.9)
project(monoforce)

find_package(PkgConfig REQUIRED)

pkg_check_modules(lz4 REQUIRED liblz4)
include_directories(${lz4_INCLUDE_DIRS})
link_directories(${lz4_LIBRARY_DIRS})

# Catkin
set(CATKIN_DEPS
    geometry_msgs
    nav_msgs
    sensor_msgs
    grid_map_msgs
    std_msgs
    tf2_ros
)
find_package(catkin REQUIRED COMPONENTS ${CATKIN_DEPS})
include_directories(${catkin_INCLUDE_DIRS})
link_directories(${catkin_LIBRARY_DIRS})

catkin_package(
    CATKIN_DEPENDS ${CATKIN_DEPS}
)

install(
    PROGRAMS
        nodes/diff_physics
        nodes/terrain_encoder
        nodes/play_tf_static
        nodes/monoforce
        nodes/geom_hm_estimator
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
    DIRECTORY
        launch/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
