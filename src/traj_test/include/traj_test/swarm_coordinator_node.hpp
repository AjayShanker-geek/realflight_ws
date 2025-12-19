#ifndef TRAJ_TEST_SWARM_COORDINATOR_HPP
#define TRAJ_TEST_SWARM_COORDINATOR_HPP

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int32.hpp>
#include <map>
#include <vector>

class SwarmCoordinator : public rclcpp::Node
{
public:
  SwarmCoordinator(int total_drones);

private:
  void timer_callback();
  void state_callback(const std_msgs::msg::Int32::SharedPtr msg, int drone_id);
  bool all_drones_in_hover();
  void send_traj_command_to_all();

  int total_drones_;
  std::vector<int> drone_ids_;
  bool traj_sent_;
  
  std::map<int, int> drone_states_;  // drone_id -> state
  std::vector<rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr> state_subs_;
  std::vector<rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr> cmd_pubs_;
  rclcpp::TimerBase::SharedPtr timer_;
  
  static constexpr int HOVER_STATE = 4;
  static constexpr int TRAJ_STATE = 5;
};

#endif // TRAJ_TEST_SWARM_COORDINATOR_HPP