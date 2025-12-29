#pragma once

#include "l2c_pkg/data_loader_new.hpp"

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <px4_msgs/msg/vehicle_attitude_setpoint.hpp>
#include <px4_msgs/msg/vehicle_local_position.hpp>
#include <px4_msgs/msg/vehicle_local_position_setpoint.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <std_msgs/msg/int32.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

class AccKalmanFilter {
public:
  explicit AccKalmanFilter(double dt_init = 0.01, double q_var = 1e-5, double r_var = 2e-5);
  void step(const Eigen::Vector3d &z, double dt);
  const Eigen::Vector3d &pos() const { return pos_; }
  const Eigen::Vector3d &vel() const { return vel_; }
  const Eigen::Vector3d &acc() const { return acc_; }

private:
  void build(double dt);

  Eigen::Matrix<double, 9, 9> F_;
  Eigen::Matrix<double, 9, 9> Q_;
  Eigen::Matrix<double, 3, 9> H_;
  Eigen::Matrix3d R_;
  Eigen::Matrix<double, 9, 1> x_;
  Eigen::Matrix<double, 9, 9> P_;
  bool init_;
  double q_;
  double dt_;
  Eigen::Vector3d pos_;
  Eigen::Vector3d vel_;
  Eigen::Vector3d acc_;
};

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

class L2CNode : public rclcpp::Node {
public:
  L2CNode(int drone_id, int total_drones);

private:
  // callbacks
  void payload_pose_cb(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void payload_odom_cb(const nav_msgs::msg::Odometry::SharedPtr msg);
  void odom_cb(const px4_msgs::msg::VehicleOdometry::SharedPtr msg);
  void local_pos_cb(const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg);
  void lps_setpoint_cb(const px4_msgs::msg::VehicleLocalPositionSetpoint::SharedPtr msg);
  void init_pose_cb(const geometry_msgs::msg::TransformStamped::SharedPtr msg);
  void state_cb(const std_msgs::msg::Int32::SharedPtr msg);
  void swarm_state_cb(const std_msgs::msg::Int32::SharedPtr msg, int other);
  void timer_cb();

  // helpers
  Eigen::Matrix3d hat(const Eigen::Vector3d &v) const;
  Eigen::Vector3d vee(const Eigen::Matrix3d &M) const;
  Eigen::Quaterniond quat_from_msg(const geometry_msgs::msg::PoseStamped &msg) const;
  Eigen::Quaterniond quat_from_px4(const px4_msgs::msg::VehicleOdometry &msg) const;
  Eigen::Vector3d lowpass(const Eigen::Vector3d &x, Eigen::Vector3d &y_prev,
                          double cutoff_hz, double dt, bool &init) const;
  bool all_ready() const;

  void update_payload(const Eigen::Vector3d &pos, const Eigen::Quaterniond &q_raw,
                      const rclcpp::Time &stamp);
  void update_sim_time(const rclcpp::Time &stamp);
  void check_state_internal(double t);
  void reset_trajectory_state();
  void publish_command(FsmState state, bool all_drones);

  void compute_desired(double t);
  void run_control(double sim_t);
  void log_sample(double sim_t,
                  const Eigen::Vector3d &q_i,
                  const Eigen::Vector3d &omega_i,
                  const Eigen::Vector3d &e_qi,
                  const Eigen::Vector3d &e_omega_i);
  void log_debug(double sim_t);

  // parameters
  int drone_id_;
  int total_drones_;
  double dt_nom_;
  double l_;
  double payload_m_;
  double payload_added_mass_;
  double payload_radius_;
  double m_drones_;
  double kq_;
  double kw_;
  double alpha_gain_;
  double z_weight_;
  double thrust_bias_;
  double thrust_to_weight_ratio_;
  double slowdown_;
  bool payload_enu_;
  bool apply_payload_offset_;
  bool apply_init_pos_offset_;
  bool use_internal_state_machine_;
  bool use_payload_stamp_time_;
  bool command_all_drones_;
  bool lps_transient_local_;
  std::string payload_input_;
  std::string payload_pose_topic_;
  std::string payload_odom_topic_;
  std::string init_pose_topic_;
  std::string attitude_setpoint_topic_;
  double acc_sp_timeout_s_;
  bool acc_sp_valid_;
  bool acc_sp_from_setpoint_;
  rclcpp::Time last_acc_sp_stamp_;
  Eigen::Vector3d local_pos_offset_ned_;
  bool init_pos_received_;
  Eigen::Vector3d init_pos_offset_ned_;
  double initial_ready_delay_;
  Eigen::Vector3d payload_rp_;
  Eigen::Vector3d payload_rg_;

  Eigen::Matrix3d T_enu2ned_;
  Eigen::Matrix3d T_body_;
  Eigen::Matrix3d T_flu2frd_;
  std::vector<Eigen::Vector3d> rho_;
  std::vector<Eigen::Vector3d> offset_pos_;
  Eigen::Matrix<double, 6, Eigen::Dynamic> P_;
  std::ofstream log_file_;
  bool log_enabled_{false};
  std::ofstream debug_log_file_;
  bool debug_log_enabled_{false};
  double debug_log_period_s_{0.5};
  double last_debug_log_time_s_{std::numeric_limits<double>::quiet_NaN()};
  Eigen::Quaternionf att_sp_prev_;
  bool att_sp_prev_valid_{false};

  // offline data
  std::shared_ptr<DataLoaderNew> data_;
  double traj_duration_;

  // filters
  AccKalmanFilter acc_filter_;

  // ROS handles
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr payload_pose_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr payload_odom_sub_;
  rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<px4_msgs::msg::VehicleLocalPosition>::SharedPtr local_pos_sub_;
  rclcpp::Subscription<px4_msgs::msg::VehicleLocalPositionSetpoint>::SharedPtr lps_sub_;
  rclcpp::Subscription<geometry_msgs::msg::TransformStamped>::SharedPtr init_pos_sub_;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr state_sub_;
  std::vector<rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr> swarm_subs_;
  rclcpp::Publisher<px4_msgs::msg::VehicleAttitudeSetpoint>::SharedPtr att_pub_;
  rclcpp::Publisher<px4_msgs::msg::TrajectorySetpoint>::SharedPtr traj_pub_;
  std::vector<rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr> cmd_pubs_;
  rclcpp::TimerBase::SharedPtr timer_;

  // current states
  bool payload_ready_;
  bool odom_ready_;
  bool local_pos_ready_;
  rclcpp::Time last_payload_stamp_;
  Eigen::Vector3d payload_pos_;
  Eigen::Vector3d payload_vel_;
  Eigen::Vector3d payload_acc_;
  Eigen::Quaterniond payload_q_;
  Eigen::Matrix3d payload_R_;
  Eigen::Quaterniond payload_q_enu_;
  Eigen::Vector3d payload_omega_;
  Eigen::Vector3d payload_omega_dot_;
  Eigen::Vector3d payload_omega_prev_;
  bool payload_omega_init_;

  Eigen::Vector3d drone_pos_;
  Eigen::Vector3d drone_vel_;
  Eigen::Vector3d drone_omega_;
  Eigen::Matrix3d drone_R_;
  Eigen::Vector3d drone_acc_sp_;
  Eigen::Vector3d drone_acc_est_;
  bool last_local_xy_valid_{false};
  bool last_local_z_valid_{false};
  bool last_local_vxy_valid_{false};
  bool last_local_vz_valid_{false};
  double last_vec_norm_{std::numeric_limits<double>::quiet_NaN()};
  double last_u_total_norm_{std::numeric_limits<double>::quiet_NaN()};

  // desired states
  Eigen::Vector3d x_d_;
  Eigen::Vector3d v_d_;
  Eigen::Matrix3d R_d_;
  Eigen::Vector3d a_d_;
  Eigen::Vector3d e_x_;
  Eigen::Vector3d e_v_;
  Eigen::Vector3d e_R_;
  Eigen::Vector3d e_Omega_;
  Eigen::Matrix<double, 6, 13> k_ddp_;
  std::vector<Eigen::Vector3d> q_id_;
  std::vector<Eigen::Vector3d> mu_id_;
  std::vector<Eigen::Vector3d> omega_id_;
  std::vector<Eigen::Vector3d> omega_id_dot_;
  std::vector<Eigen::Vector3d> mu_id_prev_;
  std::vector<Eigen::Vector3d> mu_id_dot_;
  std::vector<Eigen::Vector3d> mu_id_ddot_;
  std::vector<Eigen::Vector3d> mu_id_dot_prev_;
  std::vector<Eigen::Vector3d> mu_id_ddot_prev_;
  std::vector<Eigen::Vector3d> q_id_dot_;
  std::vector<Eigen::Vector3d> q_id_ddot_;
  std::vector<bool> mu_dot_filter_init_;
  std::vector<bool> mu_ddot_filter_init_;
  std::vector<bool> mu_history_init_;

  // timers / trajectory state
  rclcpp::Time traj_start_;
  bool traj_started_;
  bool traj_completed_;
  bool traj_time_init_;
  rclcpp::Time last_control_stamp_;
  bool control_dt_init_{false};
  double control_dt_{0.0};

  bool traj_ready_;
  bool traj_done_;
  std::vector<bool> traj_done_bit_;
  double sim_time_;
  double last_sim_time_;
  double sim_dt_;
  double sim_t0_;
  double traj_t0_;
  double t_wait_traj_;
  double ready_since_;
  bool ready_since_valid_;

  // fsm
  FsmState current_state_;
  std::map<int, FsmState> swarm_states_;
};
