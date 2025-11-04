// offboard_fsm_node.hpp
// Header file for PX4 offboard control FSM
// Author: Yichao Gao

#ifndef OFFBOARD_FSM_NODE_HPP_
#define OFFBOARD_FSM_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int32.hpp>

// Include full headers instead of forward declarations
#include <px4_msgs/msg/vehicle_status.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>

// FSM states
enum class FsmState {
  INIT = 0,      // Initialize offboard stream
  ARMING = 1,    // Switch to offboard & arm
  TAKEOFF = 2,   // Take off to target altitude
  GOTO = 3,      // Go to specified position
  HOVER = 4,     // Hover at current position
  TRAJ = 5,      // Follow trajectory
  END_TRAJ = 6,  // End trajectory mode
  LAND = 7,      // Landing
  DONE = 8       // Landed & disarmed
};

class OffboardFSM : public rclcpp::Node
{
public:
  explicit OffboardFSM(int drone_id);

private:
  // Callbacks
  void timer_cb();
  void status_cb(const px4_msgs::msg::VehicleStatus::SharedPtr msg);
  void odom_cb(const px4_msgs::msg::VehicleOdometry::SharedPtr msg);
  void state_cmd_cb(const std_msgs::msg::Int32::SharedPtr msg);

  // Helper functions
  void publish_offboard_mode();
  void publish_current_setpoint();
  void send_vehicle_cmd(uint16_t cmd, float p1, float p2);
  void try_set_offboard_and_arm();
  void generate_trajectory();
  bool has_goto_target() const;
  uint64_t get_timestamp_us();
  // Publishers
  rclcpp::Publisher<px4_msgs::msg::OffboardControlMode>::SharedPtr pub_offb_mode_;
  rclcpp::Publisher<px4_msgs::msg::TrajectorySetpoint>::SharedPtr pub_traj_sp_;
  rclcpp::Publisher<px4_msgs::msg::VehicleCommand>::SharedPtr pub_cmd_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr pub_state_;

  // Subscribers
  rclcpp::Subscription<px4_msgs::msg::VehicleStatus>::SharedPtr sub_status_;
  rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr sub_odom_;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr sub_state_cmd_;

  // Timer
  rclcpp::TimerBase::SharedPtr timer_;

  // Parameters
  int drone_id_;
  double takeoff_alt_;
  double takeoff_time_s_;
  double climb_rate_;
  double landing_time_s_;
  double circle_radius_;
  double inward_offset_;
  int num_drones_;
  double timer_period_s_;
  double alt_tol_;
  double radius_;
  double period_s_;
  
  // GOTO parameters
  double goto_x_;
  double goto_y_;
  double goto_z_;
  double goto_tol_;
  
  // Payload offset
  double payload_offset_x_;
  double payload_offset_y_;

  // State variables
  FsmState current_state_;
  int offb_counter_;
  int takeoff_start_count_;
  int takeoff_complete_count_;
  int landing_start_count_;
  
  // Command tracking
  int offboard_cmd_count_;
  int arm_cmd_count_;
  int64_t last_cmd_time_;
  
  // Position tracking
  double takeoff_pos_x_;
  double takeoff_pos_y_;
  double hover_x_;
  double hover_y_;
  double hover_z_;
  double landing_x_;
  double landing_y_;
  double landing_start_z_;
  
  // Current position from odometry
  double current_x_;
  double current_y_;
  double current_z_;
  double last_z_;
  bool odom_ready_;
  
  // Control mode
  bool use_attitude_control_;
  
  // PX4 status
  uint8_t nav_state_;
  uint8_t arming_state_;
  
  // Time tracking
  rclcpp::Time state_start_time_;
  std::chrono::steady_clock::time_point start_time_;
  // Namespace
  std::string px4_ns_;
};

#endif  // OFFBOARD_FSM_NODE_HPP_