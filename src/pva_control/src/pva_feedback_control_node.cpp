#include "pva_control/pva_feedback_control_node.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <unordered_map>

using std::placeholders::_1;

namespace {
bool getline_safe(std::ifstream &f, std::string &line)
{
  if (!std::getline(f, line)) {
    return false;
  }
  if (!line.empty() && line.back() == '\r') {
    line.pop_back();
  }
  return true;
}

std::vector<std::string> split_csv(const std::string &line)
{
  std::vector<std::string> out;
  std::stringstream ss(line);
  std::string item;
  while (std::getline(ss, item, ',')) {
    out.push_back(item);
  }
  return out;
}

std::unordered_map<std::string, size_t> header_index(const std::string &line)
{
  std::unordered_map<std::string, size_t> idx;
  auto cols = split_csv(line);
  for (size_t i = 0; i < cols.size(); ++i) {
    idx[cols[i]] = i;
  }
  return idx;
}

double read_value(const std::vector<std::string> &cols,
                  const std::unordered_map<std::string, size_t> &idx,
                  const std::string &key,
                  double default_val)
{
  auto it = idx.find(key);
  if (it == idx.end() || it->second >= cols.size()) {
    return default_val;
  }
  const std::string &cell = cols[it->second];
  if (cell.empty()) {
    return default_val;
  }
  return std::stod(cell);
}

void require_columns(const std::unordered_map<std::string, size_t> &idx,
                     const std::vector<std::string> &cols,
                     const std::string &path)
{
  for (const auto &key : cols) {
    if (idx.find(key) == idx.end()) {
      throw std::runtime_error("Missing required column '" + key + "' in " + path);
    }
  }
}

bool line_is_header(const std::vector<std::string> &cols)
{
  if (cols.empty()) {
    return false;
  }
  try {
    static_cast<void>(std::stod(cols[0]));
    return false;
  } catch (const std::exception &) {
    return true;
  }
}

std::string timestamp_string()
{
  auto now = std::chrono::system_clock::now();
  std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  std::tm tm {};
#ifdef _WIN32
  localtime_s(&tm, &now_time);
#else
  localtime_r(&now_time, &tm);
#endif
  std::ostringstream oss;
  oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
  return oss.str();
}

bool looks_like_dir(const std::string &path)
{
  if (path.empty()) {
    return false;
  }
  const char last = path.back();
  if (last == '/' || last == '\\') {
    return true;
  }
  std::filesystem::path p(path);
  if (p.has_extension()) {
    return false;
  }
  std::error_code ec;
  if (std::filesystem::is_directory(p, ec)) {
    return true;
  }
  return !p.has_extension();
}

std::string create_run_dir(const std::string &base_dir)
{
  std::filesystem::path base(base_dir);
  std::error_code ec;
  std::filesystem::create_directories(base, ec);
  if (ec) {
    return base.string();
  }
  const std::string stamp = timestamp_string();
  for (int idx = 1; idx <= 999; ++idx) {
    std::ostringstream name;
    name << "exp_" << stamp << "_run" << std::setfill('0') << std::setw(3) << idx;
    std::filesystem::path candidate = base / name.str();
    std::error_code ec_try;
    bool created = std::filesystem::create_directory(candidate, ec_try);
    if (created) {
      return candidate.string();
    }
    if (ec_try && ec_try != std::errc::file_exists) {
      break;
    }
  }
  return base.string();
}

std::string create_sitl_run_dir(const std::string &base_dir)
{
  std::filesystem::path base(base_dir);
  std::error_code ec;
  std::filesystem::create_directories(base, ec);
  if (ec) {
    return base.string();
  }
  const std::string stamp = timestamp_string();
  std::ostringstream name;
  name << "exp_" << stamp << "_run001";
  std::filesystem::path candidate = base / name.str();
  std::filesystem::create_directory(candidate, ec);
  if (ec && ec != std::errc::file_exists) {
    return base.string();
  }
  return candidate.string();
}
}  // namespace

AccKalmanFilter::AccKalmanFilter(double dt_init, double q_var, double r_var)
: init_(false)
, q_(q_var)
, dt_(dt_init)
{
  F_.setIdentity();
  Q_.setZero();
  H_.setZero();
  H_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
  R_ = r_var * Eigen::Matrix3d::Identity();
  x_.setZero();
  P_.setIdentity();
  P_ *= 1e3;
  pos_.setZero();
  vel_.setZero();
  acc_.setZero();
}

void AccKalmanFilter::build(double dt)
{
  Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
  double dt2 = dt * dt / 2.0;
  F_.setIdentity();
  F_.block<3, 3>(0, 3) = dt * I3;
  F_.block<3, 3>(0, 6) = dt2 * I3;
  F_.block<3, 3>(3, 6) = dt * I3;

  Q_.setZero();
  Q_.block<3, 3>(0, 0) = (std::pow(dt, 5) / 20.0) * q_ * I3;
  Q_.block<3, 3>(0, 3) = (std::pow(dt, 4) / 8.0) * q_ * I3;
  Q_.block<3, 3>(0, 6) = (std::pow(dt, 3) / 6.0) * q_ * I3;
  Q_.block<3, 3>(3, 0) = Q_.block<3, 3>(0, 3).transpose();
  Q_.block<3, 3>(3, 3) = (std::pow(dt, 3) / 3.0) * q_ * I3;
  Q_.block<3, 3>(3, 6) = (std::pow(dt, 2) / 2.0) * q_ * I3;
  Q_.block<3, 3>(6, 0) = Q_.block<3, 3>(0, 6).transpose();
  Q_.block<3, 3>(6, 3) = Q_.block<3, 3>(3, 6).transpose();
  Q_.block<3, 3>(6, 6) = dt * q_ * I3;
}

void AccKalmanFilter::step(const Eigen::Vector3d &z, double dt)
{
  if (!init_) {
    x_.setZero();
    x_.segment<3>(0) = z;
    pos_ = z;
    init_ = true;
    return;
  }
  dt_ = dt;
  build(dt_);
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;

  Eigen::Matrix<double, 3, 1> y = z - H_ * x_;
  Eigen::Matrix3d S = H_ * P_ * H_.transpose() + R_;
  Eigen::Matrix<double, 9, 3> K = P_ * H_.transpose() * S.inverse();
  x_ = x_ + K * y;
  Eigen::Matrix<double, 9, 9> I9 = Eigen::Matrix<double, 9, 9>::Identity();
  P_ = (I9 - K * H_) * P_;

  pos_ = x_.segment<3>(0);
  vel_ = x_.segment<3>(3);
  acc_ = x_.segment<3>(6);
}

PvaFeedbackControlNode::PvaFeedbackControlNode(int drone_id, int total_drones)
: Node("pva_feedback_control_node_" + std::to_string(drone_id))
, drone_id_(drone_id)
, total_drones_(total_drones)
, current_state_(FsmState::INIT)
, trajectory_loaded_(false)
, cable_loaded_(false)
, payload_loaded_(false)
, kfb_loaded_(false)
, waiting_for_swarm_(false)
, traj_started_(false)
, traj_completed_(false)
, traj_time_initialized_(false)
, current_x_(0.0)
, current_y_(0.0)
, current_z_(0.0)
, odom_ready_(false)
, acc_filter_(0.01, 1e-5, 2e-5)
, payload_ready_(false)
, payload_omega_init_(false)
, use_payload_stamp_time_(false)
, sim_time_(0.0)
, last_sim_time_(0.0)
, traj_start_time_sim_(0.0)
, payload_input_("pose")
, payload_pose_topic_("")
, payload_odom_topic_("")
, log_enabled_(false)
, debug_log_enabled_(false)
, debug_log_period_s_(0.2)
, last_debug_log_time_s_(-1.0)
{
  timer_period_ = this->declare_parameter("timer_period", 0.01);
  std::string data_root = this->declare_parameter("data_root", std::string(""));
  bool use_raw_traj = this->declare_parameter("use_raw_traj", false);
  std::string trajectory_csv_param = this->declare_parameter("trajectory_csv", std::string(""));
  std::string cable_csv_param = this->declare_parameter("cable_csv", std::string(""));
  std::string payload_csv_param = this->declare_parameter("payload_csv", std::string(""));
  std::string kfb_csv_param = this->declare_parameter("kfb_csv", std::string(""));
  payload_input_ = this->declare_parameter("payload_input", payload_input_);
  payload_pose_topic_ = this->declare_parameter(
    "payload_pose_topic", std::string("/vrpn_mocap/multilift_payload/pose"));
  payload_odom_topic_ = this->declare_parameter(
    "payload_odom_topic", payload_pose_topic_);
  if (payload_input_ == "odom" && payload_odom_topic_.empty()) {
    payload_odom_topic_ = payload_pose_topic_;
  }
  if (payload_input_ == "pose" && payload_pose_topic_.empty()) {
    payload_pose_topic_ = payload_odom_topic_;
  }
  payload_is_enu_ = this->declare_parameter("payload_is_enu", true);
  yaw_setpoint_ = this->declare_parameter("yaw_setpoint", 3.1415926);
  drone_mass_ = this->declare_parameter("drone_mass", 0.25);
  payload_mass_ = this->declare_parameter("payload_mass", 0.15);
  payload_added_mass_ = this->declare_parameter("payload_added_mass", 0.1);
  payload_radius_ = this->declare_parameter("payload_radius", 0.13);
  alpha_gain_ = this->declare_parameter("alpha_gain", 0.10);
  feedforward_weight_ = this->declare_parameter("feedforward_weight", 1.0);
  feedback_weight_ = this->declare_parameter("feedback_weight", 1.0);
  use_wall_timer_ = this->declare_parameter("use_wall_timer", false);
  use_payload_stamp_time_ = this->declare_parameter("use_payload_stamp_time", false);
  std::string mode = this->declare_parameter("mode", std::string("sitl"));
  if (!this->has_parameter("use_sim_time")) {
    this->declare_parameter("use_sim_time", false);
  }
  bool use_sim_time = false;
  this->get_parameter("use_sim_time", use_sim_time);
  if (use_sim_time) {
    RCLCPP_INFO(this->get_logger(), "Using simulated time");
  }
  if (use_payload_stamp_time_ && !use_wall_timer_) {
    RCLCPP_WARN(this->get_logger(),
                "use_payload_stamp_time=true: forcing use_wall_timer=true");
    use_wall_timer_ = true;
  }
  this->declare_parameter("use_internal_state_machine", false);
  this->declare_parameter("command_all_drones", false);
  this->declare_parameter("lps_transient_local", false);
  this->declare_parameter<std::string>("attitude_setpoint_topic", "in/vehicle_attitude_setpoint_v1");
  this->declare_parameter("l", 1.0);
  this->declare_parameter("acc_sp_timeout", 0.5);
  this->declare_parameter("thrust_to_weight_ratio", 7.90);
  this->declare_parameter("initial_ready_delay", 5.0);
  bool enable_log = this->declare_parameter("enable_log", false);
  bool enable_debug_log = this->declare_parameter("enable_debug_log", false);
  std::string log_path = this->declare_parameter("log_path", std::string(""));
  std::string default_debug_log =
    "log/pva_feedback_debug_" + mode + "_drone_" + std::to_string(drone_id_) + ".csv";
  std::string debug_log_path = this->declare_parameter("debug_log_path", default_debug_log);
  debug_log_period_s_ = this->declare_parameter("debug_log_period", 0.2);
  if (!enable_log) {
    log_path.clear();
  }
  if (!enable_debug_log) {
    debug_log_path.clear();
  }
  std::unordered_map<std::string, std::string> run_dirs;
  auto resolve_log_path = [this, &mode, &run_dirs](const std::string &path,
                                                   const std::string &prefix) -> std::string {
    if (path.empty()) {
      return std::string();
    }
    if (!looks_like_dir(path)) {
      return path;
    }
    auto it = run_dirs.find(path);
    if (it == run_dirs.end()) {
      if (mode == "sitl") {
        run_dirs[path] = create_sitl_run_dir(path);
      } else {
        run_dirs[path] = create_run_dir(path);
      }
      it = run_dirs.find(path);
    }
    std::filesystem::path p(it->second);
    std::string filename = prefix + "_" + mode + "_drone_" + std::to_string(drone_id_) + ".csv";
    return (p / filename).string();
  };
  log_path = resolve_log_path(log_path, "pva_feedback_log");
  debug_log_path = resolve_log_path(debug_log_path, "pva_feedback_debug");

  std::vector<double> rp_param = this->declare_parameter(
    "payload_rp", std::vector<double>{0.0, 0.0, 0.0});
  if (rp_param.size() == 3) {
    payload_rp_ = Eigen::Vector3d(rp_param[0], rp_param[1], rp_param[2]);
  } else if (rp_param.size() == 2) {
    payload_rp_ = Eigen::Vector3d(rp_param[0], rp_param[1], 0.0);
  } else {
    payload_rp_.setZero();
    RCLCPP_WARN(this->get_logger(),
                "payload_rp must have 2 or 3 elements; using zeros");
  }

  std::filesystem::path base_dir = data_root.empty()
    ? std::filesystem::path("data/realflight_traj_new")
    : std::filesystem::path(data_root);
  std::string traj_suffix = use_raw_traj ? "_traj_raw_20hz.csv" : "_traj_smoothed_100hz.csv";
  if (trajectory_csv_param.empty()) {
    trajectory_csv_ = (base_dir / ("drone_" + std::to_string(drone_id_) + traj_suffix)).string();
  } else {
    trajectory_csv_ = trajectory_csv_param;
  }
  if (cable_csv_param.empty()) {
    cable_csv_ = (base_dir / ("cable_" + std::to_string(drone_id_) + ".csv")).string();
  } else {
    cable_csv_ = cable_csv_param;
  }
  if (payload_csv_param.empty()) {
    payload_csv_ = (base_dir / "payload.csv").string();
  } else {
    payload_csv_ = payload_csv_param;
  }
  if (kfb_csv_param.empty()) {
    kfb_csv_ = (base_dir / "kfb.csv").string();
  } else {
    kfb_csv_ = kfb_csv_param;
  }

  for (int i = 0; i < total_drones_; ++i) {
    swarm_states_[i] = FsmState::INIT;
  }

  px4_namespace_ = get_px4_namespace(drone_id_);

  T_enu2ned_ << 0, 1, 0,
                1, 0, 0,
                0, 0,-1;
  T_body_ = T_enu2ned_;

  double payload_total_mass = payload_mass_ + payload_added_mass_;
  if (payload_total_mass > 1e-9) {
    payload_rg_ = (payload_added_mass_ / payload_total_mass) * payload_rp_;
  } else {
    payload_rg_.setZero();
  }
  build_allocation_matrix();

  payload_pos_.setZero();
  payload_vel_.setZero();
  payload_acc_.setZero();
  payload_q_.setIdentity();
  payload_q_enu_.setIdentity();
  payload_R_.setIdentity();
  payload_omega_.setZero();
  payload_omega_prev_.setZero();
  payload_omega_dot_.setZero();

  RCLCPP_INFO(this->get_logger(),
              "=== PVA Feedback Control Node for Drone %d ===", drone_id_);
  RCLCPP_INFO(this->get_logger(), "Total drones in swarm: %d", total_drones_);
  if (!data_root.empty()) {
    RCLCPP_INFO(this->get_logger(), "Data root: %s", data_root.c_str());
  }
  RCLCPP_INFO(this->get_logger(), "Trajectory CSV: %s", trajectory_csv_.c_str());
  RCLCPP_INFO(this->get_logger(), "Cable CSV: %s", cable_csv_.c_str());
  RCLCPP_INFO(this->get_logger(), "Payload CSV: %s", payload_csv_.c_str());
  RCLCPP_INFO(this->get_logger(), "Kfb CSV: %s", kfb_csv_.c_str());
  RCLCPP_INFO(this->get_logger(), "Drone mass: %.3f kg", drone_mass_);
  if (payload_added_mass_ > 0.0 || payload_rp_.norm() > 0.0) {
    RCLCPP_INFO(this->get_logger(),
                "payload_rp=[%.4f %.4f %.4f], payload_rg=[%.4f %.4f %.4f]",
                payload_rp_.x(), payload_rp_.y(), payload_rp_.z(),
                payload_rg_.x(), payload_rg_.y(), payload_rg_.z());
  }

  if (!log_path.empty()) {
    std::filesystem::path log_file_path(log_path);
    std::error_code ec;
    if (log_file_path.has_parent_path()) {
      std::filesystem::create_directories(log_file_path.parent_path(), ec);
    }
    log_file_.open(log_path, std::ios::out | std::ios::trunc);
    if (log_file_.is_open()) {
      log_file_ << "t,traj_t,state,sp_x,sp_y,sp_z,sp_vx,sp_vy,sp_vz,"
                << "acc_x,acc_y,acc_z,acc_dir_x,acc_dir_y,acc_dir_z,"
                << "ff_acc_x,ff_acc_y,ff_acc_z,"
                << "mu_ff_x,mu_ff_y,mu_ff_z,mu_fb_x,mu_fb_y,mu_fb_z,"
                << "payload_x_enu,payload_y_enu,payload_z_enu,payload_vx_enu,payload_vy_enu,payload_vz_enu,"
                << "payload_des_x,payload_des_y,payload_des_z,ex_enu_x,ex_enu_y,ex_enu_z,"
                << "ev_enu_x,ev_enu_y,ev_enu_z\n";
      log_enabled_ = true;
      RCLCPP_INFO(this->get_logger(), "Logging enabled: %s", log_path.c_str());
    } else {
      RCLCPP_WARN(this->get_logger(), "Failed to open log file: %s", log_path.c_str());
    }
  }

  if (!debug_log_path.empty()) {
    std::filesystem::path dbg_path(debug_log_path);
    std::error_code ec;
    if (dbg_path.has_parent_path()) {
      std::filesystem::create_directories(dbg_path.parent_path(), ec);
    }
    debug_log_file_.open(debug_log_path, std::ios::out | std::ios::trunc);
    if (debug_log_file_.is_open()) {
      debug_log_file_ << "t,sim_t,state,traj_started,waiting_swarm,traj_completed,"
                      << "odom_ready,payload_ready\n";
      debug_log_enabled_ = true;
      RCLCPP_INFO(this->get_logger(), "Debug logging enabled: %s", debug_log_path.c_str());
    } else {
      RCLCPP_WARN(this->get_logger(), "Failed to open debug log file: %s", debug_log_path.c_str());
    }
  }

  if (!load_trajectory_from_csv(trajectory_csv_)) {
    RCLCPP_ERROR(this->get_logger(), "Failed to load trajectory from CSV!");
    return;
  }
  if (!load_cable_from_csv(cable_csv_)) {
    RCLCPP_ERROR(this->get_logger(), "Failed to load cable data from CSV!");
    return;
  }
  if (!load_payload_from_csv(payload_csv_)) {
    RCLCPP_ERROR(this->get_logger(), "Failed to load payload data from CSV!");
    return;
  }
  if (!load_kfb_from_csv(kfb_csv_)) {
    RCLCPP_ERROR(this->get_logger(), "Failed to load Kfb from CSV!");
    return;
  }

  traj_pub_ = this->create_publisher<px4_msgs::msg::TrajectorySetpoint>(
    px4_namespace_ + "in/trajectory_setpoint",
    rclcpp::QoS(10).reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE));

  state_cmd_pub_ = this->create_publisher<std_msgs::msg::Int32>(
    "/state/command_drone_" + std::to_string(drone_id_),
    rclcpp::QoS(10).reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE));

  state_sub_ = this->create_subscription<std_msgs::msg::Int32>(
    "/state/state_drone_" + std::to_string(drone_id_),
    rclcpp::QoS(10).reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE),
    std::bind(&PvaFeedbackControlNode::state_callback, this, _1));

  for (int i = 0; i < total_drones_; ++i) {
    if (i == drone_id_) {
      continue;
    }
    auto sub = this->create_subscription<std_msgs::msg::Int32>(
      "/state/state_drone_" + std::to_string(i),
      rclcpp::QoS(10).reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE),
      [this, i](const std_msgs::msg::Int32::SharedPtr msg) {
        this->swarm_state_callback(msg, i);
      });
    swarm_state_subs_.push_back(sub);
  }

  odom_sub_ = this->create_subscription<px4_msgs::msg::VehicleOdometry>(
    px4_namespace_ + "out/vehicle_odometry",
    rclcpp::SensorDataQoS(),
    std::bind(&PvaFeedbackControlNode::odom_callback, this, _1));

  if (payload_input_ == "pose") {
    payload_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      payload_pose_topic_, rclcpp::SensorDataQoS(),
      std::bind(&PvaFeedbackControlNode::payload_pose_callback, this, _1));
  } else if (payload_input_ == "odom") {
    payload_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      payload_odom_topic_, rclcpp::SensorDataQoS(),
      std::bind(&PvaFeedbackControlNode::payload_odom_callback, this, _1));
  } else {
    RCLCPP_WARN(this->get_logger(), "Unknown payload_input '%s'; defaulting to odom",
                payload_input_.c_str());
    payload_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      payload_odom_topic_, rclcpp::SensorDataQoS(),
      std::bind(&PvaFeedbackControlNode::payload_odom_callback, this, _1));
  }

  if (use_wall_timer_) {
    timer_ = this->create_wall_timer(
      std::chrono::duration<double>(timer_period_),
      std::bind(&PvaFeedbackControlNode::timer_callback, this));
  } else {
    timer_ = rclcpp::create_timer(
      this,
      this->get_clock(),
      rclcpp::Duration(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(timer_period_))),
      std::bind(&PvaFeedbackControlNode::timer_callback, this));
  }

  RCLCPP_INFO(this->get_logger(), "Timer initialized at %.0f Hz", 1.0 / timer_period_);
}

std::string PvaFeedbackControlNode::get_px4_namespace(int drone_id)
{
  if (drone_id == 0) {
    return "/fmu/";
  }
  return "/px4_" + std::to_string(drone_id) + "/fmu/";
}

Eigen::Matrix3d PvaFeedbackControlNode::hat(const Eigen::Vector3d &v) const
{
  Eigen::Matrix3d m;
  m << 0, -v.z(), v.y(),
       v.z(), 0, -v.x(),
      -v.y(), v.x(), 0;
  return m;
}

void PvaFeedbackControlNode::build_allocation_matrix()
{
  rho_.clear();
  rho_.reserve(static_cast<size_t>(total_drones_));

  for (int i = 0; i < total_drones_; ++i) {
    double angle = 2.0 * M_PI * i / static_cast<double>(total_drones_);
    double y = payload_radius_ * std::cos(angle);
    double x = payload_radius_ * std::sin(angle);
    Eigen::Vector3d attach(x, y, 0.0);
    rho_.push_back(attach - payload_rg_);
  }

  P_.resize(6, 3 * total_drones_);
  P_.setZero();
  for (int i = 0; i < total_drones_; ++i) {
    P_.block<3, 3>(0, 3 * i) = Eigen::Matrix3d::Identity();
    P_.block<3, 3>(3, 3 * i) = hat(rho_[i]);
  }

  if (total_drones_ >= 2) {
    Eigen::MatrixXd gram = P_ * P_.transpose();
    if (gram.fullPivLu().isInvertible()) {
      P_pinv_ = P_.transpose() * gram.inverse();
    } else {
      RCLCPP_ERROR(this->get_logger(), "Allocation matrix is singular; feedback disabled");
      P_pinv_.setZero(3 * total_drones_, 6);
    }
  } else {
    RCLCPP_WARN(this->get_logger(),
                "Allocation matrix requires at least 2 drones; feedback disabled");
    P_pinv_.setZero(3 * total_drones_, 6);
  }
}

bool PvaFeedbackControlNode::load_trajectory_from_csv(const std::string &filepath)
{
  std::ifstream file(filepath);
  if (!file.is_open()) {
    RCLCPP_ERROR(this->get_logger(), "Cannot open trajectory CSV: %s", filepath.c_str());
    return false;
  }

  std::string line;
  if (!getline_safe(file, line)) {
    RCLCPP_ERROR(this->get_logger(), "Trajectory CSV is empty: %s", filepath.c_str());
    return false;
  }

  auto idx = header_index(line);
  try {
    require_columns(idx, {"time", "x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az"}, filepath);
  } catch (const std::exception &e) {
    RCLCPP_ERROR(this->get_logger(), "%s", e.what());
    return false;
  }

  int point_count = 0;
  while (getline_safe(file, line)) {
    if (line.empty()) {
      continue;
    }
    auto cols = split_csv(line);
    TrajectoryPoint point{};
    try {
      point.time = read_value(cols, idx, "time", 0.0);
      point.x = read_value(cols, idx, "x", 0.0);
      point.y = read_value(cols, idx, "y", 0.0);
      point.z = read_value(cols, idx, "z", 0.0);
      point.vx = read_value(cols, idx, "vx", 0.0);
      point.vy = read_value(cols, idx, "vy", 0.0);
      point.vz = read_value(cols, idx, "vz", 0.0);
      point.ax = read_value(cols, idx, "ax", 0.0);
      point.ay = read_value(cols, idx, "ay", 0.0);
      point.az = read_value(cols, idx, "az", 0.0);
    } catch (const std::exception &e) {
      RCLCPP_WARN(this->get_logger(), "Skipping trajectory line: %s", line.c_str());
      continue;
    }
    trajectory_.push_back(point);
    point_count++;
  }

  if (trajectory_.empty()) {
    RCLCPP_ERROR(this->get_logger(), "No valid trajectory points loaded!");
    return false;
  }

  trajectory_loaded_ = true;
  RCLCPP_INFO(this->get_logger(), "Loaded %d trajectory points", point_count);
  RCLCPP_INFO(this->get_logger(), "Trajectory duration: %.2f seconds",
              trajectory_.back().time);
  return true;
}

bool PvaFeedbackControlNode::load_cable_from_csv(const std::string &filepath)
{
  std::ifstream file(filepath);
  if (!file.is_open()) {
    RCLCPP_ERROR(this->get_logger(), "Cannot open cable CSV: %s", filepath.c_str());
    return false;
  }

  std::string line;
  if (!getline_safe(file, line)) {
    RCLCPP_ERROR(this->get_logger(), "Cable CSV is empty: %s", filepath.c_str());
    return false;
  }

  auto idx = header_index(line);
  try {
    require_columns(idx, {"time", "dir_x", "dir_y", "dir_z", "mu"}, filepath);
  } catch (const std::exception &e) {
    RCLCPP_ERROR(this->get_logger(), "%s", e.what());
    return false;
  }

  int point_count = 0;
  while (getline_safe(file, line)) {
    if (line.empty()) {
      continue;
    }
    auto cols = split_csv(line);
    CablePoint point{};
    try {
      point.time = read_value(cols, idx, "time", 0.0);
      point.dir_x = read_value(cols, idx, "dir_x", 0.0);
      point.dir_y = read_value(cols, idx, "dir_y", 0.0);
      point.dir_z = read_value(cols, idx, "dir_z", 0.0);
      point.mu = read_value(cols, idx, "mu", 0.0);
    } catch (const std::exception &e) {
      RCLCPP_WARN(this->get_logger(), "Skipping cable line: %s", line.c_str());
      continue;
    }
    cable_.push_back(point);
    point_count++;
  }

  if (cable_.empty()) {
    RCLCPP_ERROR(this->get_logger(), "No valid cable points loaded!");
    return false;
  }

  cable_loaded_ = true;
  RCLCPP_INFO(this->get_logger(), "Loaded %d cable points", point_count);
  return true;
}

bool PvaFeedbackControlNode::load_payload_from_csv(const std::string &filepath)
{
  std::ifstream file(filepath);
  if (!file.is_open()) {
    RCLCPP_ERROR(this->get_logger(), "Cannot open payload CSV: %s", filepath.c_str());
    return false;
  }

  std::string line;
  if (!getline_safe(file, line)) {
    RCLCPP_ERROR(this->get_logger(), "Payload CSV is empty: %s", filepath.c_str());
    return false;
  }

  auto idx = header_index(line);
  try {
    require_columns(idx, {"time", "x", "y", "z", "vx", "vy", "vz",
                          "qw", "qx", "qy", "qz", "wx", "wy", "wz"}, filepath);
  } catch (const std::exception &e) {
    RCLCPP_ERROR(this->get_logger(), "%s", e.what());
    return false;
  }

  int point_count = 0;
  while (getline_safe(file, line)) {
    if (line.empty()) {
      continue;
    }
    auto cols = split_csv(line);
    PayloadPoint point{};
    try {
      point.time = read_value(cols, idx, "time", 0.0);
      point.pos = Eigen::Vector3d(
        read_value(cols, idx, "x", 0.0),
        read_value(cols, idx, "y", 0.0),
        read_value(cols, idx, "z", 0.0));
      point.vel = Eigen::Vector3d(
        read_value(cols, idx, "vx", 0.0),
        read_value(cols, idx, "vy", 0.0),
        read_value(cols, idx, "vz", 0.0));
      point.acc = Eigen::Vector3d(
        read_value(cols, idx, "ax", 0.0),
        read_value(cols, idx, "ay", 0.0),
        read_value(cols, idx, "az", 0.0));
      double qw = read_value(cols, idx, "qw", 1.0);
      double qx = read_value(cols, idx, "qx", 0.0);
      double qy = read_value(cols, idx, "qy", 0.0);
      double qz = read_value(cols, idx, "qz", 0.0);
      point.q = Eigen::Quaterniond(qw, qx, qy, qz);
      point.omega = Eigen::Vector3d(
        read_value(cols, idx, "wx", 0.0),
        read_value(cols, idx, "wy", 0.0),
        read_value(cols, idx, "wz", 0.0));
    } catch (const std::exception &e) {
      RCLCPP_WARN(this->get_logger(), "Skipping payload line: %s", line.c_str());
      continue;
    }
    payload_.push_back(point);
    point_count++;
  }

  if (payload_.empty()) {
    RCLCPP_ERROR(this->get_logger(), "No valid payload points loaded!");
    return false;
  }

  payload_loaded_ = true;
  RCLCPP_INFO(this->get_logger(), "Loaded %d payload points", point_count);
  return true;
}

bool PvaFeedbackControlNode::load_kfb_from_csv(const std::string &filepath)
{
  std::ifstream file(filepath);
  if (!file.is_open()) {
    RCLCPP_ERROR(this->get_logger(), "Cannot open Kfb CSV: %s", filepath.c_str());
    return false;
  }

  std::string line;
  if (!getline_safe(file, line)) {
    RCLCPP_ERROR(this->get_logger(), "Kfb CSV is empty: %s", filepath.c_str());
    return false;
  }

  auto first_cols = split_csv(line);
  bool has_header = line_is_header(first_cols);
  std::unordered_map<std::string, size_t> idx;
  if (has_header) {
    idx = header_index(line);
    std::vector<std::string> required = {"time"};
    for (int r = 0; r < 6; ++r) {
      for (int c = 0; c < 13; ++c) {
        required.push_back("k" + std::to_string(r) + "_" + std::to_string(c));
      }
    }
    try {
      require_columns(idx, required, filepath);
    } catch (const std::exception &e) {
      RCLCPP_ERROR(this->get_logger(), "%s", e.what());
      return false;
    }
  } else {
    file.clear();
    file.seekg(0);
  }

  int point_count = 0;
  while (getline_safe(file, line)) {
    if (line.empty()) {
      continue;
    }
    auto cols = split_csv(line);
    KfbPoint point{};
    if (has_header) {
      point.time = read_value(cols, idx, "time", 0.0);
      for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 13; ++c) {
          std::string key = "k" + std::to_string(r) + "_" + std::to_string(c);
          point.K(r, c) = read_value(cols, idx, key, 0.0);
        }
      }
    } else {
      if (cols.size() < 1 + 6 * 13) {
        continue;
      }
      size_t col_idx = 0;
      point.time = std::stod(cols[col_idx++]);
      for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 13; ++c) {
          point.K(r, c) = std::stod(cols[col_idx++]);
        }
      }
    }
    kfb_.push_back(point);
    point_count++;
  }

  if (kfb_.empty()) {
    RCLCPP_ERROR(this->get_logger(), "No valid Kfb points loaded!");
    return false;
  }

  kfb_loaded_ = true;
  RCLCPP_INFO(this->get_logger(), "Loaded %d Kfb points", point_count);
  return true;
}

TrajectoryPoint PvaFeedbackControlNode::interpolate_trajectory(double t) const
{
  if (t <= trajectory_.front().time) {
    return trajectory_.front();
  }
  if (t >= trajectory_.back().time) {
    return trajectory_.back();
  }

  auto it = std::lower_bound(
    trajectory_.begin(), trajectory_.end(), t,
    [](const TrajectoryPoint &p, double value) { return p.time < value; });
  size_t idx = static_cast<size_t>(std::distance(trajectory_.begin(), it));
  const auto &p2 = trajectory_[idx];
  const auto &p1 = trajectory_[idx - 1];

  double dt = p2.time - p1.time;
  if (dt <= 1e-6) {
    return p1;
  }
  double alpha = (t - p1.time) / dt;

  TrajectoryPoint result{};
  result.time = t;
  result.x = p1.x + alpha * (p2.x - p1.x);
  result.y = p1.y + alpha * (p2.y - p1.y);
  result.z = p1.z + alpha * (p2.z - p1.z);
  result.vx = p1.vx + alpha * (p2.vx - p1.vx);
  result.vy = p1.vy + alpha * (p2.vy - p1.vy);
  result.vz = p1.vz + alpha * (p2.vz - p1.vz);
  result.ax = p1.ax + alpha * (p2.ax - p1.ax);
  result.ay = p1.ay + alpha * (p2.ay - p1.ay);
  result.az = p1.az + alpha * (p2.az - p1.az);
  return result;
}

CablePoint PvaFeedbackControlNode::interpolate_cable(double t) const
{
  if (cable_.empty()) {
    return CablePoint{};
  }
  if (t <= cable_.front().time) {
    return cable_.front();
  }
  if (t >= cable_.back().time) {
    return cable_.back();
  }

  auto it = std::lower_bound(
    cable_.begin(), cable_.end(), t,
    [](const CablePoint &p, double value) { return p.time < value; });
  size_t idx = static_cast<size_t>(std::distance(cable_.begin(), it));
  const auto &p2 = cable_[idx];
  const auto &p1 = cable_[idx - 1];

  double dt = p2.time - p1.time;
  if (dt <= 1e-6) {
    return p1;
  }
  double alpha = (t - p1.time) / dt;

  CablePoint result{};
  result.time = t;
  result.dir_x = p1.dir_x + alpha * (p2.dir_x - p1.dir_x);
  result.dir_y = p1.dir_y + alpha * (p2.dir_y - p1.dir_y);
  result.dir_z = p1.dir_z + alpha * (p2.dir_z - p1.dir_z);
  result.mu = p1.mu + alpha * (p2.mu - p1.mu);
  return result;
}

PayloadPoint PvaFeedbackControlNode::interpolate_payload(double t) const
{
  if (payload_.empty()) {
    return PayloadPoint{};
  }
  if (t <= payload_.front().time) {
    return payload_.front();
  }
  if (t >= payload_.back().time) {
    return payload_.back();
  }

  auto it = std::lower_bound(
    payload_.begin(), payload_.end(), t,
    [](const PayloadPoint &p, double value) { return p.time < value; });
  size_t idx = static_cast<size_t>(std::distance(payload_.begin(), it));
  const auto &p2 = payload_[idx];
  const auto &p1 = payload_[idx - 1];

  double dt = p2.time - p1.time;
  if (dt <= 1e-6) {
    return p1;
  }
  double alpha = (t - p1.time) / dt;

  PayloadPoint result{};
  result.time = t;
  result.pos = p1.pos + alpha * (p2.pos - p1.pos);
  result.vel = p1.vel + alpha * (p2.vel - p1.vel);
  result.acc = p1.acc + alpha * (p2.acc - p1.acc);
  result.omega = p1.omega + alpha * (p2.omega - p1.omega);
  Eigen::Quaterniond q1 = p1.q.normalized();
  Eigen::Quaterniond q2 = p2.q.normalized();
  if (q1.dot(q2) < 0.0) {
    q2.coeffs() *= -1.0;
  }
  result.q = q1.slerp(alpha, q2);
  return result;
}

KfbPoint PvaFeedbackControlNode::interpolate_kfb(double t) const
{
  if (kfb_.empty()) {
    return KfbPoint{};
  }
  if (t <= kfb_.front().time) {
    return kfb_.front();
  }
  if (t >= kfb_.back().time) {
    return kfb_.back();
  }

  auto it = std::lower_bound(
    kfb_.begin(), kfb_.end(), t,
    [](const KfbPoint &p, double value) { return p.time < value; });
  size_t idx = static_cast<size_t>(std::distance(kfb_.begin(), it));
  const auto &p2 = kfb_[idx];
  const auto &p1 = kfb_[idx - 1];

  double dt = p2.time - p1.time;
  if (dt <= 1e-6) {
    return p1;
  }
  double alpha = (t - p1.time) / dt;

  KfbPoint result{};
  result.time = t;
  result.K = (1.0 - alpha) * p1.K + alpha * p2.K;
  return result;
}

bool PvaFeedbackControlNode::all_drones_in_traj_state() const
{
  for (const auto &entry : swarm_states_) {
    if (entry.second != FsmState::TRAJ) {
      return false;
    }
  }
  return true;
}

void PvaFeedbackControlNode::state_callback(const std_msgs::msg::Int32::SharedPtr msg)
{
  auto old_state = current_state_;
  auto state = static_cast<FsmState>(msg->data);
  current_state_ = state;
  swarm_states_[drone_id_] = state;

  if (old_state != state) {
    RCLCPP_WARN(this->get_logger(),
                "STATE CHANGE: %d -> %d",
                static_cast<int>(old_state), static_cast<int>(state));
  }

  if (state == FsmState::TRAJ && !waiting_for_swarm_ && !traj_started_ && !traj_completed_) {
    waiting_for_swarm_ = true;
    RCLCPP_WARN(this->get_logger(),
                ">>> Entered TRAJ state, waiting for swarm sync...");

    if (total_drones_ == 1) {
      traj_started_ = true;
      waiting_for_swarm_ = false;
      RCLCPP_WARN(this->get_logger(),
                  "SINGLE DRONE - READY TO START TRAJECTORY");
    }
  }

  if (waiting_for_swarm_ && !traj_started_ && all_drones_in_traj_state()) {
    traj_started_ = true;
    waiting_for_swarm_ = false;
    RCLCPP_WARN(this->get_logger(),
                "ALL DRONES IN TRAJ - READY TO START TRAJECTORY");
  }

  if (traj_started_ && state != FsmState::TRAJ && !traj_completed_) {
    traj_started_ = false;
    waiting_for_swarm_ = false;
    traj_time_initialized_ = false;
    RCLCPP_ERROR(this->get_logger(),
                 "!!! Left TRAJ state early (to state %d), resetting !!!",
                 static_cast<int>(state));
  }
}

void PvaFeedbackControlNode::swarm_state_callback(
  const std_msgs::msg::Int32::SharedPtr msg, int other_drone_id)
{
  auto state = static_cast<FsmState>(msg->data);
  swarm_states_[other_drone_id] = state;

  if (state == FsmState::TRAJ && waiting_for_swarm_) {
    int traj_count = 0;
    for (const auto &entry : swarm_states_) {
      if (entry.second == FsmState::TRAJ) {
        traj_count++;
      }
    }
    RCLCPP_INFO(this->get_logger(),
                "Drone %d entered TRAJ (%d/%d drones ready)",
                other_drone_id, traj_count, total_drones_);

    if (all_drones_in_traj_state() && !traj_started_) {
      traj_started_ = true;
      waiting_for_swarm_ = false;
      RCLCPP_WARN(this->get_logger(),
                  "ALL DRONES IN TRAJ - READY TO START TRAJECTORY");
    }
  }
}

void PvaFeedbackControlNode::odom_callback(
  const px4_msgs::msg::VehicleOdometry::SharedPtr msg)
{
  current_x_ = msg->position[0];
  current_y_ = msg->position[1];
  current_z_ = msg->position[2];
  odom_ready_ = true;
}

void PvaFeedbackControlNode::payload_pose_callback(
  const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
  Eigen::Quaterniond q_raw(msg->pose.orientation.w,
                           msg->pose.orientation.x,
                           msg->pose.orientation.y,
                           msg->pose.orientation.z);
  Eigen::Vector3d pos(msg->pose.position.x,
                      msg->pose.position.y,
                      msg->pose.position.z);
  update_payload(pos, q_raw, rclcpp::Time(msg->header.stamp));
}

void PvaFeedbackControlNode::payload_odom_callback(
  const nav_msgs::msg::Odometry::SharedPtr msg)
{
  Eigen::Quaterniond q_raw(msg->pose.pose.orientation.w,
                           msg->pose.pose.orientation.x,
                           msg->pose.pose.orientation.y,
                           msg->pose.pose.orientation.z);
  Eigen::Vector3d pos(msg->pose.pose.position.x,
                      msg->pose.pose.position.y,
                      msg->pose.pose.position.z);
  update_payload(pos, q_raw, rclcpp::Time(msg->header.stamp));
}

void PvaFeedbackControlNode::update_payload(
  const Eigen::Vector3d &pos,
  const Eigen::Quaterniond &q_raw_in,
  const rclcpp::Time &stamp)
{
  double dt = payload_ready_
    ? (stamp - last_payload_stamp_).seconds()
    : timer_period_;
  if (dt <= 0.0) {
    dt = timer_period_;
  }

  Eigen::Quaterniond q_raw = q_raw_in;
  if (q_raw.w() < 0.0) {
    q_raw.coeffs() *= -1.0;
  }
  if (payload_ready_) {
    double dot = q_raw.w() * payload_q_enu_.w() + q_raw.vec().dot(payload_q_enu_.vec());
    if (dot < 0.0) {
      q_raw.coeffs() *= -1.0;
    }
  }

  Eigen::Vector3d meas = pos;
  Eigen::Quaterniond q_meas = q_raw;
  if (payload_is_enu_) {
    payload_q_enu_ = q_raw;
    meas = T_enu2ned_ * meas;
    Eigen::Matrix3d R_enu = q_raw.toRotationMatrix();
    Eigen::Matrix3d R_ned = T_enu2ned_ * R_enu * T_body_;
    q_meas = Eigen::Quaterniond(R_ned);
  } else {
    Eigen::Matrix3d R_ned = q_raw.toRotationMatrix();
    Eigen::Matrix3d R_enu = T_enu2ned_ * R_ned * T_body_;
    payload_q_enu_ = Eigen::Quaterniond(R_enu);
  }

  acc_filter_.step(meas, dt);
  payload_pos_ = acc_filter_.pos();
  payload_vel_ = acc_filter_.vel();
  payload_acc_ = acc_filter_.acc();

  Eigen::Matrix3d R_prev = payload_R_;
  Eigen::Matrix3d R_curr = q_meas.toRotationMatrix();
  Eigen::Matrix3d delta = R_prev.transpose() * R_curr;
  Eigen::AngleAxisd aa(delta);
  Eigen::Vector3d omega = aa.axis() * aa.angle() / dt;

  if (!payload_omega_init_) {
    payload_omega_dot_.setZero();
    payload_omega_init_ = true;
  } else {
    payload_omega_dot_ = (omega - payload_omega_prev_) / dt;
  }
  payload_omega_prev_ = omega;
  payload_omega_ = omega;
  payload_q_ = q_meas;
  payload_R_ = R_curr;

  last_payload_stamp_ = stamp;
  payload_ready_ = true;

  if (use_payload_stamp_time_) {
    sim_time_ = stamp.seconds();
    last_sim_time_ = sim_time_;
  }
}

void PvaFeedbackControlNode::timer_callback()
{
  static int debug_counter = 0;
  if (debug_counter++ % 100 == 0) {
    RCLCPP_INFO(this->get_logger(),
                "State: %d, traj_started: %d, time_init: %d, waiting_swarm: %d, completed: %d, odom: %d, payload: %d",
                static_cast<int>(current_state_), traj_started_, traj_time_initialized_,
                waiting_for_swarm_, traj_completed_, odom_ready_, payload_ready_);
  }
  double now_s = use_payload_stamp_time_ ? sim_time_ : this->now().seconds();
  debug_log_sample(now_s);

  if (!odom_ready_) {
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                         "Waiting for odometry");
    return;
  }
  if (!payload_ready_) {
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                         "Waiting for payload pose");
    return;
  }

  if (!trajectory_loaded_) {
    RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                          "Trajectory not loaded!");
    return;
  }
  if (!cable_loaded_) {
    RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                          "Cable data not loaded!");
    return;
  }
  if (!payload_loaded_) {
    RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                          "Payload data not loaded!");
    return;
  }
  if (!kfb_loaded_) {
    RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                          "Kfb data not loaded!");
    return;
  }

  if (current_state_ == FsmState::TRAJ && traj_started_ && !traj_completed_) {
    if (!traj_time_initialized_) {
      traj_start_time_ = this->now();
      traj_start_time_sim_ = sim_time_;
      traj_time_initialized_ = true;
      RCLCPP_WARN(this->get_logger(),
                  "TRAJECTORY EXECUTION STARTED (t=0.00s)");
    }

    double elapsed = 0.0;
    if (use_payload_stamp_time_) {
      elapsed = sim_time_ - traj_start_time_sim_;
    } else {
      elapsed = (this->now() - traj_start_time_).seconds();
    }

    if (elapsed >= trajectory_.back().time) {
      if (!traj_completed_) {
        RCLCPP_WARN(this->get_logger(),
                    "========================================");
        RCLCPP_WARN(this->get_logger(),
                    "TRAJECTORY COMPLETE - SENDING END_TRAJ");
        RCLCPP_WARN(this->get_logger(),
                    "========================================");

        for (int i = 0; i < 5; i++) {
          send_state_command(static_cast<int>(FsmState::END_TRAJ));
        }

        traj_completed_ = true;
      }
      return;
    }

    TrajectoryPoint traj = interpolate_trajectory(elapsed);
    CablePoint cable = interpolate_cable(elapsed);
    PayloadPoint payload_des = interpolate_payload(elapsed);
    KfbPoint kfb = interpolate_kfb(elapsed);
    publish_trajectory_setpoint(traj, cable, payload_des, kfb, yaw_setpoint_);

    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "EXECUTING TRAJ: t=%.2fs | pos=(%.2f,%.2f,%.2f) | vel=(%.2f,%.2f,%.2f)",
                         elapsed, traj.x, traj.y, traj.z,
                         traj.vx, traj.vy, traj.vz);
  } else if (current_state_ == FsmState::TRAJ && waiting_for_swarm_) {
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "In TRAJ state, waiting for swarm synchronization...");
  }
}

void PvaFeedbackControlNode::publish_trajectory_setpoint(
  const TrajectoryPoint &traj,
  const CablePoint &cable,
  const PayloadPoint &payload_des,
  const KfbPoint &kfb,
  double yaw)
{
  Eigen::Vector3d acc_cmd(traj.ax, traj.ay, traj.az);

  Eigen::Vector3d dir_enu(cable.dir_x, cable.dir_y, cable.dir_z);
  Eigen::Vector3d dir_ned = T_enu2ned_ * dir_enu;
  Eigen::Vector3d mu_ff = feedforward_weight_ * cable.mu * dir_ned;

  Eigen::Vector3d payload_pos_enu = T_enu2ned_ * payload_pos_;
  Eigen::Vector3d payload_vel_enu = T_enu2ned_ * payload_vel_;
  Eigen::Vector3d payload_omega_enu = T_enu2ned_ * payload_omega_;
  Eigen::Vector3d e_x_enu = payload_pos_enu - payload_des.pos;
  Eigen::Vector3d e_v_enu = payload_vel_enu - payload_des.vel;

  Eigen::Vector3d mu_fb = Eigen::Vector3d::Zero();
  if (P_pinv_.cols() == 6 && payload_ready_) {
    Eigen::Quaterniond q_meas_enu = payload_q_enu_;
    Eigen::Quaterniond q_des_enu = payload_des.q.normalized();
    if (q_des_enu.w() < 0.0) {
      q_des_enu.coeffs() *= -1.0;
    }
    if (q_meas_enu.dot(q_des_enu) < 0.0) {
      q_des_enu.coeffs() *= -1.0;
    }

    Eigen::Vector4d e_q_enu(q_meas_enu.w() - q_des_enu.w(),
                            q_meas_enu.x() - q_des_enu.x(),
                            q_meas_enu.y() - q_des_enu.y(),
                            q_meas_enu.z() - q_des_enu.z());

    Eigen::Vector3d e_omega_enu = payload_omega_enu - payload_des.omega;

    Eigen::Matrix<double, 13, 1> e_ddp;
    e_ddp << e_x_enu, e_v_enu, e_q_enu, e_omega_enu;
    Eigen::Matrix<double, 6, 1> FM_enu = alpha_gain_ * kfb.K * e_ddp;

    Eigen::Matrix<double, 6, 1> FM_ned;
    FM_ned.segment<3>(0) = T_enu2ned_ * FM_enu.segment<3>(0);
    FM_ned.segment<3>(3) = T_enu2ned_ * FM_enu.segment<3>(3);

    Eigen::VectorXd delta_mu = P_pinv_ * FM_ned;
    if (static_cast<int>(delta_mu.size()) >= 3 * (drone_id_ + 1)) {
      Eigen::Vector3d delta_mu_i = delta_mu.segment<3>(3 * drone_id_);
      mu_fb = feedback_weight_ * (payload_R_ * delta_mu_i);
    }
  }

  if (drone_mass_ > 0.0) {
    acc_cmd += (mu_ff + mu_fb) / drone_mass_;
  } else {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                         "drone_mass <= 0, skipping cable feedback");
  }

  px4_msgs::msg::TrajectorySetpoint msg;
  msg.position[0] = static_cast<float>(traj.x);
  msg.position[1] = static_cast<float>(traj.y);
  msg.position[2] = static_cast<float>(traj.z);

  msg.velocity[0] = static_cast<float>(traj.vx);
  msg.velocity[1] = static_cast<float>(traj.vy);
  msg.velocity[2] = static_cast<float>(traj.vz);

  msg.acceleration[0] = static_cast<float>(acc_cmd.x());
  msg.acceleration[1] = static_cast<float>(acc_cmd.y());
  msg.acceleration[2] = static_cast<float>(acc_cmd.z());

  msg.yaw = static_cast<float>(yaw);
  msg.timestamp = 0;

  traj_pub_->publish(msg);
  double now_s = use_payload_stamp_time_ ? sim_time_ : this->now().seconds();
  log_sample(now_s, traj, payload_des, acc_cmd, mu_ff, mu_fb,
             payload_pos_enu, payload_vel_enu, e_x_enu, e_v_enu);
}

void PvaFeedbackControlNode::log_sample(double now_s,
                                        const TrajectoryPoint &traj,
                                        const PayloadPoint &payload_des,
                                        const Eigen::Vector3d &acc_cmd,
                                        const Eigen::Vector3d &mu_ff,
                                        const Eigen::Vector3d &mu_fb,
                                        const Eigen::Vector3d &payload_pos_enu,
                                        const Eigen::Vector3d &payload_vel_enu,
                                        const Eigen::Vector3d &e_x_enu,
                                        const Eigen::Vector3d &e_v_enu)
{
  if (!log_enabled_ || !log_file_.is_open()) {
    return;
  }
  double acc_norm = acc_cmd.norm();
  double acc_dir_x = 0.0;
  double acc_dir_y = 0.0;
  double acc_dir_z = 0.0;
  if (acc_norm > 1e-6) {
    acc_dir_x = acc_cmd.x() / acc_norm;
    acc_dir_y = acc_cmd.y() / acc_norm;
    acc_dir_z = acc_cmd.z() / acc_norm;
  }
  double ff_acc_x = 0.0;
  double ff_acc_y = 0.0;
  double ff_acc_z = 0.0;
  if (drone_mass_ > 1e-9) {
    ff_acc_x = mu_ff.x() / drone_mass_;
    ff_acc_y = mu_ff.y() / drone_mass_;
    ff_acc_z = mu_ff.z() / drone_mass_;
  }
  log_file_ << std::fixed << std::setprecision(6)
            << now_s << ","
            << traj.time << ","
            << static_cast<int>(current_state_) << ","
            << traj.x << "," << traj.y << "," << traj.z << ","
            << traj.vx << "," << traj.vy << "," << traj.vz << ","
            << acc_cmd.x() << "," << acc_cmd.y() << "," << acc_cmd.z() << ","
            << acc_dir_x << "," << acc_dir_y << "," << acc_dir_z << ","
            << ff_acc_x << "," << ff_acc_y << "," << ff_acc_z << ","
            << mu_ff.x() << "," << mu_ff.y() << "," << mu_ff.z() << ","
            << mu_fb.x() << "," << mu_fb.y() << "," << mu_fb.z() << ","
            << payload_pos_enu.x() << "," << payload_pos_enu.y() << "," << payload_pos_enu.z() << ","
            << payload_vel_enu.x() << "," << payload_vel_enu.y() << "," << payload_vel_enu.z() << ","
            << payload_des.pos.x() << "," << payload_des.pos.y() << "," << payload_des.pos.z() << ","
            << e_x_enu.x() << "," << e_x_enu.y() << "," << e_x_enu.z() << ","
            << e_v_enu.x() << "," << e_v_enu.y() << "," << e_v_enu.z() << "\n";
  log_file_.flush();
}

void PvaFeedbackControlNode::debug_log_sample(double now_s)
{
  if (!debug_log_enabled_ || !debug_log_file_.is_open()) {
    return;
  }
  if (last_debug_log_time_s_ >= 0.0 &&
      (now_s - last_debug_log_time_s_) < debug_log_period_s_) {
    return;
  }
  last_debug_log_time_s_ = now_s;
  debug_log_file_ << std::fixed << std::setprecision(6)
                  << now_s << ","
                  << sim_time_ << ","
                  << static_cast<int>(current_state_) << ","
                  << traj_started_ << ","
                  << waiting_for_swarm_ << ","
                  << traj_completed_ << ","
                  << odom_ready_ << ","
                  << payload_ready_ << "\n";
  debug_log_file_.flush();
}

void PvaFeedbackControlNode::send_state_command(int state)
{
  std_msgs::msg::Int32 msg;
  msg.data = state;
  state_cmd_pub_->publish(msg);
  RCLCPP_INFO(this->get_logger(), "Sent state command: %d", state);
}
