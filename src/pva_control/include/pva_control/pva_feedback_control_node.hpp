#ifndef PVA_FEEDBACK_CONTROL_NODE_HPP_
#define PVA_FEEDBACK_CONTROL_NODE_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
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

struct PayloadPoint {
  double time;
  Eigen::Vector3d pos;
  Eigen::Vector3d vel;
  Eigen::Vector3d acc;
  Eigen::Quaterniond q;
  Eigen::Vector3d omega;
};

struct KfbPoint {
  double time;
  Eigen::Matrix<double, 6, 13> K;
};

class AccKalmanFilter
{
public:
  AccKalmanFilter(double dt_init, double q_var, double r_var);
  void step(const Eigen::Vector3d &z, double dt);
  const Eigen::Vector3d &pos() const { return pos_; }
  const Eigen::Vector3d &vel() const { return vel_; }
  const Eigen::Vector3d &acc() const { return acc_; }

private:
  void build(double dt);

  bool init_;
  double q_;
  double dt_;
  Eigen::Matrix<double, 9, 9> F_;
  Eigen::Matrix<double, 9, 9> Q_;
  Eigen::Matrix<double, 3, 9> H_;
  Eigen::Matrix3d R_;
  Eigen::Matrix<double, 9, 9> P_;
  Eigen::Matrix<double, 9, 1> x_;
  Eigen::Vector3d pos_;
  Eigen::Vector3d vel_;
  Eigen::Vector3d acc_;
};

class PvaFeedbackControlNode : public rclcpp::Node
{
public:
  explicit PvaFeedbackControlNode(int drone_id, int total_drones);

private:
  void timer_callback();
  void state_callback(const std_msgs::msg::Int32::SharedPtr msg);
  void swarm_state_callback(const std_msgs::msg::Int32::SharedPtr msg, int other_drone_id);
  void odom_callback(const px4_msgs::msg::VehicleOdometry::SharedPtr msg);
  void payload_pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void payload_odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg);
  void update_payload(const Eigen::Vector3d &pos,
                      const Eigen::Quaterniond &q_raw,
                      const rclcpp::Time &stamp);

  bool load_trajectory_from_csv(const std::string &filepath);
  bool load_cable_from_csv(const std::string &filepath);
  bool load_payload_from_csv(const std::string &filepath);
  bool load_kfb_from_csv(const std::string &filepath);

  TrajectoryPoint interpolate_trajectory(double t) const;
  CablePoint interpolate_cable(double t) const;
  PayloadPoint interpolate_payload(double t) const;
  KfbPoint interpolate_kfb(double t) const;

  void publish_trajectory_setpoint(const TrajectoryPoint &traj,
                                   const CablePoint &cable,
                                   const PayloadPoint &payload_des,
                                   const KfbPoint &kfb,
                                   double yaw);
  void log_sample(double now_s,
                  const TrajectoryPoint &traj,
                  const PayloadPoint &payload_des,
                  const Eigen::Vector3d &acc_cmd,
                  const Eigen::Vector3d &mu_ff,
                  const Eigen::Vector3d &mu_fb,
                  const Eigen::Vector3d &payload_pos_enu,
                  const Eigen::Vector3d &payload_vel_enu,
                  const Eigen::Vector3d &e_x_enu,
                  const Eigen::Vector3d &e_v_enu);
  void debug_log_sample(double now_s);
  void send_state_command(int state);

  void build_allocation_matrix();
  Eigen::Matrix3d hat(const Eigen::Vector3d &v) const;
  std::string get_px4_namespace(int drone_id);
  bool all_drones_in_traj_state() const;

  // Publishers
  rclcpp::Publisher<px4_msgs::msg::TrajectorySetpoint>::SharedPtr traj_pub_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr state_cmd_pub_;

  // Subscribers
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr state_sub_;
  rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr payload_pose_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr payload_odom_sub_;
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
  std::vector<PayloadPoint> payload_;
  std::vector<KfbPoint> kfb_;
  bool trajectory_loaded_;
  bool cable_loaded_;
  bool payload_loaded_;
  bool kfb_loaded_;

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

  // Payload state (NED)
  AccKalmanFilter acc_filter_;
  bool payload_ready_;
  bool payload_omega_init_;
  Eigen::Vector3d payload_pos_;
  Eigen::Vector3d payload_vel_;
  Eigen::Vector3d payload_acc_;
  Eigen::Quaterniond payload_q_;
  Eigen::Quaterniond payload_q_enu_;
  Eigen::Matrix3d payload_R_;
  Eigen::Vector3d payload_omega_;
  Eigen::Vector3d payload_omega_prev_;
  Eigen::Vector3d payload_omega_dot_;
  rclcpp::Time last_payload_stamp_;
  bool use_payload_stamp_time_;
  double sim_time_;
  double last_sim_time_;
  double traj_start_time_sim_;

  // Payload geometry for pseudo inverse
  Eigen::Vector3d payload_rp_;
  Eigen::Vector3d payload_rg_;
  std::vector<Eigen::Vector3d> rho_;
  Eigen::MatrixXd P_;
  Eigen::MatrixXd P_pinv_;
  Eigen::Matrix3d T_enu2ned_;
  Eigen::Matrix3d T_body_;

  // Parameters
  double timer_period_;
  std::string trajectory_csv_;
  std::string cable_csv_;
  std::string payload_csv_;
  std::string kfb_csv_;
  std::string payload_input_;
  std::string payload_pose_topic_;
  std::string payload_odom_topic_;
  double yaw_setpoint_;
  double drone_mass_;
  double payload_mass_;
  double payload_added_mass_;
  double payload_radius_;
  double alpha_gain_;
  double feedforward_weight_;
  double feedback_weight_;
  bool payload_is_enu_;
  bool use_wall_timer_;

  // Logging
  bool log_enabled_;
  bool debug_log_enabled_;
  double debug_log_period_s_;
  double last_debug_log_time_s_;
  std::ofstream log_file_;
  std::ofstream debug_log_file_;
};

#endif  // PVA_FEEDBACK_CONTROL_NODE_HPP_
