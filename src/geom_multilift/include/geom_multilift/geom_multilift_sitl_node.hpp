#pragma once

#include "geom_multilift/geom_multilift_node.hpp"
#include "geom_multilift/data_loader_new.hpp"

#include <nav_msgs/msg/odometry.hpp>
#include <px4_msgs/msg/vehicle_local_position.hpp>
#include <fstream>

// SITL variant of the geometric multilift controller.
// Mirrors the Python controller used in px4_offboard/geom_multilift.py:
// - payload pose from /payload_odom (Odometry, ENU -> NED conversion)
// - drone pose from /simulation/position_drone_<i>
// - velocities/omegas from PX4 odom/local_position topics
class GeomMultiliftSitlNode : public rclcpp::Node {
public:
  GeomMultiliftSitlNode(int drone_id, int total_drones);

private:
  // callbacks
  void payload_odom_cb(const nav_msgs::msg::Odometry::SharedPtr msg);
  void sim_pose_cb(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void odom_cb(const px4_msgs::msg::VehicleOdometry::SharedPtr msg);
  void local_pos_cb(const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg);
  void lps_setpoint_cb(const px4_msgs::msg::VehicleLocalPositionSetpoint::SharedPtr msg);
  void state_cb(const std_msgs::msg::Int32::SharedPtr msg);
  void swarm_state_cb(const std_msgs::msg::Int32::SharedPtr msg, int other);
  void timer_cb();
  void check_state(double t);

  // helpers
  Eigen::Matrix3d hat(const Eigen::Vector3d &v) const;
  Eigen::Quaterniond quat_from_px4(const px4_msgs::msg::VehicleOdometry &msg) const;
  bool all_ready() const;
  void compute_desired(double t);
  void run_control(double sim_t);
  Eigen::Vector3d lowpass(const Eigen::Vector3d &x, Eigen::Vector3d &y_prev,
                          double cutoff_hz, double dt, bool &init) const;
  Eigen::Vector3d vee(const Eigen::Matrix3d &M) const;
  void log_sample(double sim_t,
                  const Eigen::Vector3d &q_i,
                  const Eigen::Vector3d &omega_i,
                  const Eigen::Vector3d &e_qi,
                  const Eigen::Vector3d &e_omega_i);

  // parameters
  int drone_id_;
  int total_drones_;
  double dt_nom_;
  double l_;
  double payload_m_;
  double m_drones_;
  double kq_;
  double kw_;
  double alpha_gain_;
  double z_weight_;
  double thrust_bias_;
  double slowdown_;
  bool payload_enu_;

  Eigen::Matrix3d T_enu2ned_;
  Eigen::Matrix3d T_body_;
  Eigen::Matrix3d T_flu2frd_;
  std::vector<Eigen::Vector3d> rho_;
  std::vector<Eigen::Vector3d> offset_pos_;
  Eigen::Matrix<double, 6, Eigen::Dynamic> P_;  // 6 x 3n
  std::ofstream log_file_;
  bool log_enabled_{false};
  Eigen::Quaternionf att_sp_prev_;
  bool att_sp_prev_valid_{false};

  // offline data
  std::shared_ptr<DataLoaderNew> data_;
  double traj_duration_;

  // filters
  AccKalmanFilter acc_filter_;

  // ROS handles
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr payload_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sim_pose_sub_;
  rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<px4_msgs::msg::VehicleLocalPosition>::SharedPtr local_pos_sub_;
  rclcpp::Subscription<px4_msgs::msg::VehicleLocalPositionSetpoint>::SharedPtr lps_sub_;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr state_sub_;
  std::vector<rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr> swarm_subs_;
  rclcpp::Publisher<px4_msgs::msg::VehicleAttitudeSetpoint>::SharedPtr att_pub_;
  rclcpp::Publisher<px4_msgs::msg::TrajectorySetpoint>::SharedPtr traj_pub_;
  std::vector<rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr> cmd_pubs_;
  rclcpp::TimerBase::SharedPtr timer_;

  // current states
  bool payload_ready_;
  bool odom_ready_;
  bool sim_pose_ready_;
  geometry_msgs::msg::PoseStamped last_sim_pose_;
  nav_msgs::msg::Odometry last_payload_odom_;
  rclcpp::Time last_payload_stamp_;
  double last_sim_time_;
  double sim_dt_;
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
  Eigen::Vector3d sim_offset_;
  bool sim_offset_init_;

  // desired states
  Eigen::Vector3d x_d_;
  Eigen::Vector3d v_d_;
  Eigen::Matrix3d R_d_;
  Eigen::Vector3d a_d_;
  Eigen::Quaterniond q_d_enu_;
  Eigen::Vector3d omega_d_enu_;
  Eigen::Vector3d e_x_;
  Eigen::Vector3d e_v_;
  Eigen::Vector3d e_R_;
  Eigen::Vector3d e_Omega_;
  std::vector<Eigen::Vector3d> x_id_;
  std::vector<Eigen::Vector3d> v_id_;
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

  // timers
  rclcpp::Time traj_start_;
  bool traj_ready_;
  bool traj_done_;
  std::vector<bool> traj_done_bit_;
  double sim_time_;
  double sim_t0_;
  double traj_t0_;
  double t_wait_traj_;
  double ready_since_;
  bool ready_since_valid_;
  double initial_ready_delay_;

  // fsm
  FsmState current_state_;
  std::map<int, FsmState> swarm_states_;
};
