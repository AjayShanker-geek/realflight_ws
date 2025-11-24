#include "traj_test/swarm_coordinator_node.hpp"
#include <rclcpp/rclcpp.hpp>

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  
  int total_drones = 1;
  if (argc > 1) {
    total_drones = std::atoi(argv[1]);
  }
  
  auto node = std::make_shared<SwarmCoordinator>(total_drones);
  
  RCLCPP_INFO(node->get_logger(), "Swarm Coordinator Node Started");
  
  rclcpp::spin(node);
  rclcpp::shutdown();
  
  return 0;
}