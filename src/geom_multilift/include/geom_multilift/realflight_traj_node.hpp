// Realflight per-drone trajectory publisher using swarm FSM gates.
// Keeps the realflight assumption: mocap publishes world-frame local_position,
// so we send trajectory points directly without re-centering.
#ifndef GEOM_MULTILIFT_REALFLIGHT_TRAJ_NODE_HPP_
#define GEOM_MULTILIFT_REALFLIGHT_TRAJ_NODE_HPP_

#include <map>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <std_msgs/msg/int32.hpp>

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
};

class RealflightTrajNode : public rclcpp::Node {
public:
  RealflightTrajNode(int drone_id, int total_drones);

private:
  void timer_callback();
  void state_callback(const std_msgs::msg::Int32::SharedPtr msg);
  void swarm_state_callback(const std_msgs::msg::Int32::SharedPtr msg, int other_drone_id);
  void odom_callback(const px4_msgs::msg::VehicleOdometry::SharedPtr msg);

  bool load_trajectory_from_csv(const std::string &filepath);
  TrajectoryPoint interpolate_trajectory(double t);
  void publish_trajectory_setpoint(double x, double y, double z,
                                   double vx, double vy, double vz,
                                   double yaw);
  void send_state_command(int state);

  std::string get_px4_namespace(int drone_id) const;
  bool all_drones_in_traj_state() const;

  // ROS interfaces
  rclcpp::Publisher<px4_msgs::msg::TrajectorySetpoint>::SharedPtr traj_pub_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr state_cmd_pub_;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr state_sub_;
  rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr odom_sub_;
  std::vector<rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr> swarm_state_subs_;
  rclcpp::TimerBase::SharedPtr timer_;

  // Parameters and identifiers
  int drone_id_;
  int total_drones_;
  std::string px4_namespace_;
  double timer_period_;
  std::string csv_base_path_;
  std::string csv_path_;
  double yaw_setpoint_;

  // Trajectory data/state
  std::vector<TrajectoryPoint> trajectory_;
  bool trajectory_loaded_;
  bool waiting_for_swarm_;
  bool traj_started_;
  bool traj_completed_;
  rclcpp::Time traj_start_time_;
  bool traj_time_initialized_;

  // Current state estimation (no origin shift in realflight)
  double current_x_;
  double current_y_;
  double current_z_;
  bool odom_ready_;

  FsmState current_state_;
  std::map<int, FsmState> swarm_states_;
};

#endif  // GEOM_MULTILIFT_REALFLIGHT_TRAJ_NODE_HPP_
