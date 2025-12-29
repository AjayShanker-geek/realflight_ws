cmake_minimum_required(VERSION 3.8)
project(l2c_pkg)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(include)

add_executable(l2c_node
  src/data_loader_new.cpp
  src/l2c_node.cpp
  src/l2c_main.cpp
)
ament_target_dependencies(l2c_node
  rclcpp
  geometry_msgs
  nav_msgs
  px4_msgs
  std_msgs
)
target_link_libraries(l2c_node Eigen3::Eigen)

install(TARGETS
  l2c_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

install(FILES README.md
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
