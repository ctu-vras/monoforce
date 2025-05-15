cmake_minimum_required(VERSION 3.0.2)
project(monoforce_demos)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(
    PROGRAMS
        nodes/monoforce_demo
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}/nodes
)