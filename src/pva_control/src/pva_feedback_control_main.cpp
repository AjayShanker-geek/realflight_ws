#include "pva_control/pva_feedback_control_node.hpp"

#include <cstdlib>
#include <rclcpp/rclcpp.hpp>

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);

  if (argc < 3) {
    RCLCPP_ERROR(rclcpp::get_logger("rclcpp"),
                 "Usage: pva_feedback_control_node <drone_id> <total_drones>");
    return 1;
  }

  int drone_id = std::atoi(argv[1]);
  int total_drones = std::atoi(argv[2]);

  auto node = std::make_shared<PvaFeedbackControlNode>(drone_id, total_drones);
  RCLCPP_INFO(node->get_logger(), "PVA feedback control node started for drone %d", drone_id);

  rclcpp::spin(node);
  rclcpp::shutdown();

  return 0;
}
