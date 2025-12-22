#include "geom_multilift/realflight_traj_node.hpp"

#include <cstdlib>
#include <iostream>

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);

  int drone_id = 0;
  int total_drones = 1;

  if (const char *env_id = std::getenv("DRONE_ID")) {
    try {
      drone_id = std::stoi(env_id);
    } catch (...) {
      std::cerr << "Invalid DRONE_ID env var, defaulting to 0\n";
    }
  }

  if (const char *env_total = std::getenv("TOTAL_DRONES")) {
    try {
      total_drones = std::stoi(env_total);
    } catch (...) {
      std::cerr << "Invalid TOTAL_DRONES env var, defaulting to 1\n";
    }
  }

  if (argc > 1) {
    drone_id = std::stoi(argv[1]);
  }
  if (argc > 2) {
    total_drones = std::stoi(argv[2]);
  }

  auto node = std::make_shared<RealflightTrajNode>(drone_id, total_drones);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
