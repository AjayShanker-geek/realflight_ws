cmake_minimum_required(VERSION 3.8)
project(zmq_state_bridge)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED libzmq)

add_executable(zmq_state_bridge_node
  src/zmq_state_bridge_node.cpp
)

target_compile_features(zmq_state_bridge_node PUBLIC cxx_std_17)
target_include_directories(zmq_state_bridge_node PRIVATE ${ZMQ_INCLUDE_DIRS})
target_link_directories(zmq_state_bridge_node PRIVATE ${ZMQ_LIBRARY_DIRS})
target_link_libraries(zmq_state_bridge_node ${ZMQ_LIBRARIES})
ament_target_dependencies(zmq_state_bridge_node rclcpp std_msgs)

add_executable(gcs_swarm_coordinator_node
  src/gcs_swarm_coordinator_node.cpp
)
target_compile_features(gcs_swarm_coordinator_node PUBLIC cxx_std_17)
ament_target_dependencies(gcs_swarm_coordinator_node rclcpp std_msgs geometry_msgs)

install(TARGETS zmq_state_bridge_node
  DESTINATION lib/${PROJECT_NAME}
)
install(TARGETS gcs_swarm_coordinator_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

ament_package()
