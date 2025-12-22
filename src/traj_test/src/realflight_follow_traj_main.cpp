#include "follow_traj/realflight_follow_traj_node.hpp"
#include <rclcpp/rclcpp.hpp>
#include <cstdlib>

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  if (argc < 3) {
    RCLCPP_ERROR(rclcpp::get_logger("rclcpp"),
                 "Usage: realflight_follow_traj_node <drone_id> <total_drones>");
    return 1;
  }

  int drone_id = std::atoi(argv[1]);
  int total_drones = std::atoi(argv[2]);

  auto node = std::make_shared<RealflightFollowTrajNode>(drone_id, total_drones);

  RCLCPP_INFO(node->get_logger(),
              "Realflight follow trajectory node started for drone %d", drone_id);

  rclcpp::spin(node);
  rclcpp::shutdown();

  return 0;
}
