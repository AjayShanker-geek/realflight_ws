#ifndef PVA_CONTROL_NODE_HPP_
#define PVA_CONTROL_NODE_HPP_

#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int32.hpp>

#include <fstream>
#include <map>
#include <string>
#include <vector>

// Keep this enum aligned with offboard_state_machine FsmState values.
enum class FsmState {
  INIT = 0,
  ARMING = 1,
  TAKEOFF = 2,
  GOTO = 3,
  HOVER = 4,
  TRAJ = 5,
  END_TRAJ = 6,
  LAND = 7,
  DONE = 8
};

struct TrajectoryPoint {
  double time;
  double x;
  double y;
  double z;
  double vx;
  double vy;
  double vz;
  double ax;
  double ay;
  double az;
};

struct CablePoint {
  double time;
  double dir_x;
  double dir_y;
  double dir_z;
  double mu;
};

class PvaControlNode : public rclcpp::Node
{
public:
  explicit PvaControlNode(int drone_id, int total_drones);

private:
  void timer_callback();
  void state_callback(const std_msgs::msg::Int32::SharedPtr msg);
  void swarm_state_callback(const std_msgs::msg::Int32::SharedPtr msg, int other_drone_id);
  void odom_callback(const px4_msgs::msg::VehicleOdometry::SharedPtr msg);

  bool load_trajectory_from_csv(const std::string &filepath);
  bool load_cable_from_csv(const std::string &filepath);
  TrajectoryPoint interpolate_trajectory(double t) const;
  CablePoint interpolate_cable(double t) const;
  void publish_trajectory_setpoint(const TrajectoryPoint &traj,
                                   const CablePoint &cable,
                                   double yaw);
  void log_sample(double now_s,
                  const TrajectoryPoint &traj,
                  const CablePoint &cable,
                  double acc_x,
                  double acc_y,
                  double acc_z);
  void debug_log_sample(double now_s);
  void send_state_command(int state);

  std::string get_px4_namespace(int drone_id);
  bool all_drones_in_traj_state() const;

  // Publishers
  rclcpp::Publisher<px4_msgs::msg::TrajectorySetpoint>::SharedPtr traj_pub_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr state_cmd_pub_;

  // Subscribers
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr state_sub_;
  rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr odom_sub_;
  std::vector<rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr> swarm_state_subs_;

  // Timer
  rclcpp::TimerBase::SharedPtr timer_;

  // Drone identification
  int drone_id_;
  int total_drones_;
  std::string px4_namespace_;

  // FSM state tracking
  FsmState current_state_;
  std::map<int, FsmState> swarm_states_;

  // Trajectory data
  std::vector<TrajectoryPoint> trajectory_;
  std::vector<CablePoint> cable_;
  bool trajectory_loaded_;
  bool cable_loaded_;

  // Trajectory execution state
  bool waiting_for_swarm_;
  bool traj_started_;
  bool traj_completed_;
  bool traj_time_initialized_;
  rclcpp::Time traj_start_time_;

  // Odometry
  double current_x_;
  double current_y_;
  double current_z_;
  bool odom_ready_;

  // Parameters
  double timer_period_;
  std::string trajectory_csv_;
  std::string cable_csv_;
  double yaw_setpoint_;
  double drone_mass_;
  double feedforward_weight_;
  bool use_wall_timer_;

  // Logging
  bool log_enabled_;
  bool debug_log_enabled_;
  double debug_log_period_s_;
  double last_debug_log_time_s_;
  std::ofstream log_file_;
  std::ofstream debug_log_file_;
};

#endif  // PVA_CONTROL_NODE_HPP_
