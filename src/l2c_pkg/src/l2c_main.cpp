#include "l2c_pkg/l2c_node.hpp"

#include <rclcpp/rclcpp.hpp>
#include <cstdlib>

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  int drone_id = 0;
  int total = 3;
  if (const char *env = std::getenv("DRONE_ID")) drone_id = std::atoi(env);
  if (const char *env = std::getenv("TOTAL_DRONES")) total = std::atoi(env);
  if (argc > 1) drone_id = std::atoi(argv[1]);
  if (argc > 2) total = std::atoi(argv[2]);
  auto node = std::make_shared<L2CNode>(drone_id, total);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
