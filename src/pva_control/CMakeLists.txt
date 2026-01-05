cmake_minimum_required(VERSION 3.8)
project(pva_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(offboard_state_machine REQUIRED)

include_directories(
  include
)

add_executable(pva_control_node
  src/pva_control_node.cpp
  src/pva_control_main.cpp
)

ament_target_dependencies(pva_control_node
  rclcpp
  std_msgs
  px4_msgs
  offboard_state_machine
)

add_executable(pva_feedback_control_node
  src/pva_feedback_control_node.cpp
  src/pva_feedback_control_main.cpp
)

ament_target_dependencies(pva_feedback_control_node
  rclcpp
  std_msgs
  px4_msgs
  geometry_msgs
  nav_msgs
  offboard_state_machine
)
target_link_libraries(pva_feedback_control_node Eigen3::Eigen)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

install(TARGETS
  pva_control_node
  pva_feedback_control_node
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
