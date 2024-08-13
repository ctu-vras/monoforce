cmake_minimum_required(VERSION 3.0.2)
project(monoforce_navigation)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(
    PROGRAMS
        path_follower
        path_to_map
        path_selector
        publish_gt_gridmap
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}/nodes
)