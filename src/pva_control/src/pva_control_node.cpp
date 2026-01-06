#include "pva_control/pva_control_node.hpp"

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

PvaControlNode::PvaControlNode(int drone_id, int total_drones)
: Node("pva_control_node_" + std::to_string(drone_id))
, drone_id_(drone_id)
, total_drones_(total_drones)
, current_state_(FsmState::INIT)
, trajectory_loaded_(false)
, cable_loaded_(false)
, waiting_for_swarm_(false)
, traj_started_(false)
, traj_completed_(false)
, traj_time_initialized_(false)
, current_x_(0.0)
, current_y_(0.0)
, current_z_(0.0)
, odom_ready_(false)
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
  yaw_setpoint_ = this->declare_parameter("yaw_setpoint", 3.1415926);
  drone_mass_ = this->declare_parameter("drone_mass", 0.25);
  feedforward_weight_ = this->declare_parameter("feedforward_weight", 1.0);
  use_wall_timer_ = this->declare_parameter("use_wall_timer", false);
  std::string mode = this->declare_parameter("mode", std::string("sitl"));
  bool enable_log = this->declare_parameter("enable_log", false);
  bool enable_debug_log = this->declare_parameter("enable_debug_log", false);
  std::string log_path = this->declare_parameter("log_path", std::string(""));
  std::string default_debug_log =
    "log/pva_debug_" + mode + "_drone_" + std::to_string(drone_id_) + ".csv";
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
  log_path = resolve_log_path(log_path, "pva_log");
  debug_log_path = resolve_log_path(debug_log_path, "pva_debug");

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

  for (int i = 0; i < total_drones_; ++i) {
    swarm_states_[i] = FsmState::INIT;
  }

  px4_namespace_ = get_px4_namespace(drone_id_);

  RCLCPP_INFO(this->get_logger(),
              "=== PVA Control Node for Drone %d ===", drone_id_);
  RCLCPP_INFO(this->get_logger(), "Total drones in swarm: %d", total_drones_);
  if (!data_root.empty()) {
    RCLCPP_INFO(this->get_logger(), "Data root: %s", data_root.c_str());
  }
  RCLCPP_INFO(this->get_logger(), "Use raw traj: %s", use_raw_traj ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "Trajectory CSV: %s", trajectory_csv_.c_str());
  RCLCPP_INFO(this->get_logger(), "Cable CSV: %s", cable_csv_.c_str());
  RCLCPP_INFO(this->get_logger(), "Drone mass: %.3f kg", drone_mass_);

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
                << "cable_dir_x,cable_dir_y,cable_dir_z,"
                << "cable_mu,odom_x,odom_y,odom_z\n";
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
      debug_log_file_ << "t,state,traj_started,waiting_swarm,traj_completed,odom_ready,"
                      << "traj_loaded,cable_loaded\n";
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

  traj_pub_ = this->create_publisher<px4_msgs::msg::TrajectorySetpoint>(
    px4_namespace_ + "in/trajectory_setpoint",
    rclcpp::QoS(10).reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE));

  state_cmd_pub_ = this->create_publisher<std_msgs::msg::Int32>(
    "/state/command_drone_" + std::to_string(drone_id_),
    rclcpp::QoS(10).reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE));

  state_sub_ = this->create_subscription<std_msgs::msg::Int32>(
    "/state/state_drone_" + std::to_string(drone_id_),
    rclcpp::QoS(10).reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE),
    std::bind(&PvaControlNode::state_callback, this, std::placeholders::_1));

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
    std::bind(&PvaControlNode::odom_callback, this, std::placeholders::_1));

  if (use_wall_timer_) {
    timer_ = this->create_wall_timer(
      std::chrono::duration<double>(timer_period_),
      std::bind(&PvaControlNode::timer_callback, this));
  } else {
    timer_ = rclcpp::create_timer(
      this,
      this->get_clock(),
      rclcpp::Duration(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(timer_period_))),
      std::bind(&PvaControlNode::timer_callback, this));
  }

  RCLCPP_INFO(this->get_logger(), "Timer initialized at %.0f Hz", 1.0 / timer_period_);
}

std::string PvaControlNode::get_px4_namespace(int drone_id)
{
  if (drone_id == 0) {
    return "/fmu/";
  }
  return "/px4_" + std::to_string(drone_id) + "/fmu/";
}

bool PvaControlNode::load_trajectory_from_csv(const std::string &filepath)
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

bool PvaControlNode::load_cable_from_csv(const std::string &filepath)
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
      double dir_x_enu = read_value(cols, idx, "dir_x", 0.0);
      double dir_y_enu = read_value(cols, idx, "dir_y", 0.0);
      double dir_z_enu = read_value(cols, idx, "dir_z", 0.0);
      // Cable directions are stored in ENU; convert to NED to match trajectory.
      point.dir_x = dir_y_enu;
      point.dir_y = dir_x_enu;
      point.dir_z = -dir_z_enu;
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

  if (trajectory_loaded_ && cable_.size() != trajectory_.size()) {
    RCLCPP_WARN(this->get_logger(),
                "Cable length (%zu) differs from trajectory length (%zu); using time interpolation",
                cable_.size(), trajectory_.size());
  }

  return true;
}

TrajectoryPoint PvaControlNode::interpolate_trajectory(double t) const
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

CablePoint PvaControlNode::interpolate_cable(double t) const
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

bool PvaControlNode::all_drones_in_traj_state() const
{
  for (const auto &entry : swarm_states_) {
    if (entry.second != FsmState::TRAJ) {
      return false;
    }
  }
  return true;
}

void PvaControlNode::state_callback(const std_msgs::msg::Int32::SharedPtr msg)
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

void PvaControlNode::swarm_state_callback(
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

void PvaControlNode::odom_callback(
  const px4_msgs::msg::VehicleOdometry::SharedPtr msg)
{
  current_x_ = msg->position[0];
  current_y_ = msg->position[1];
  current_z_ = msg->position[2];
  odom_ready_ = true;
}

void PvaControlNode::timer_callback()
{
  static int debug_counter = 0;
  if (debug_counter++ % 100 == 0) {
    RCLCPP_INFO(this->get_logger(),
                "State: %d, traj_started: %d, time_init: %d, waiting_swarm: %d, completed: %d, odom: %d",
                static_cast<int>(current_state_), traj_started_, traj_time_initialized_,
                waiting_for_swarm_, traj_completed_, odom_ready_);
  }
  debug_log_sample(this->now().seconds());

  if (!odom_ready_) {
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                         "Waiting for odometry");
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

  if (current_state_ == FsmState::TRAJ && traj_started_ && !traj_completed_) {
    if (!traj_time_initialized_) {
      traj_start_time_ = this->now();
      traj_time_initialized_ = true;
      RCLCPP_WARN(this->get_logger(),
                  "TRAJECTORY EXECUTION STARTED (t=0.00s)");
    }

    double elapsed = (this->now() - traj_start_time_).seconds();

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

    TrajectoryPoint point = interpolate_trajectory(elapsed);
    CablePoint cable = interpolate_cable(elapsed);
    publish_trajectory_setpoint(point, cable, yaw_setpoint_);

    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "EXECUTING TRAJ: t=%.2fs | pos=(%.2f,%.2f,%.2f) | vel=(%.2f,%.2f,%.2f)",
                         elapsed, point.x, point.y, point.z,
                         point.vx, point.vy, point.vz);
  } else if (current_state_ == FsmState::TRAJ && waiting_for_swarm_) {
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "In TRAJ state, waiting for swarm synchronization...");
  }
}

void PvaControlNode::publish_trajectory_setpoint(
  const TrajectoryPoint &traj,
  const CablePoint &cable,
  double yaw)
{
  double ax = traj.ax;
  double ay = traj.ay;
  double az = traj.az;
  double ff_ax = 0.0;
  double ff_ay = 0.0;
  double ff_az = 0.0;

  if (drone_mass_ > 0.0) {
    double scale = feedforward_weight_ * cable.mu / drone_mass_;
    ff_ax = scale * cable.dir_x;
    ff_ay = scale * cable.dir_y;
    ff_az = scale * cable.dir_z;
    ax += ff_ax;
    ay += ff_ay;
    az += ff_az;
  } else {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                         "drone_mass <= 0, skipping feedforward");
  }
  RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                       "FF acc [%.3f %.3f %.3f] mu=%.3f dir=[%.3f %.3f %.3f]",
                       ff_ax, ff_ay, ff_az, cable.mu,
                       cable.dir_x, cable.dir_y, cable.dir_z);

  px4_msgs::msg::TrajectorySetpoint msg;
  msg.position[0] = static_cast<float>(traj.x);
  msg.position[1] = static_cast<float>(traj.y);
  msg.position[2] = static_cast<float>(traj.z);

  msg.velocity[0] = static_cast<float>(traj.vx);
  msg.velocity[1] = static_cast<float>(traj.vy);
  msg.velocity[2] = static_cast<float>(traj.vz);

  msg.acceleration[0] = static_cast<float>(ax);
  msg.acceleration[1] = static_cast<float>(ay);
  msg.acceleration[2] = static_cast<float>(az);

  msg.yaw = static_cast<float>(yaw);
  msg.timestamp = 0;

  traj_pub_->publish(msg);
  log_sample(this->now().seconds(), traj, cable, ax, ay, az, ff_ax, ff_ay, ff_az);
}

void PvaControlNode::log_sample(double now_s,
                                const TrajectoryPoint &traj,
                                const CablePoint &cable,
                                double acc_x,
                                double acc_y,
                                double acc_z,
                                double ff_acc_x,
                                double ff_acc_y,
                                double ff_acc_z)
{
  if (!log_enabled_ || !log_file_.is_open()) {
    return;
  }
  double acc_norm = std::sqrt(acc_x * acc_x + acc_y * acc_y + acc_z * acc_z);
  double acc_dir_x = 0.0;
  double acc_dir_y = 0.0;
  double acc_dir_z = 0.0;
  if (acc_norm > 1e-6) {
    acc_dir_x = acc_x / acc_norm;
    acc_dir_y = acc_y / acc_norm;
    acc_dir_z = acc_z / acc_norm;
  }
  log_file_ << std::fixed << std::setprecision(6)
            << now_s << ","
            << traj.time << ","
            << static_cast<int>(current_state_) << ","
            << traj.x << "," << traj.y << "," << traj.z << ","
            << traj.vx << "," << traj.vy << "," << traj.vz << ","
            << acc_x << "," << acc_y << "," << acc_z << ","
            << acc_dir_x << "," << acc_dir_y << "," << acc_dir_z << ","
            << ff_acc_x << "," << ff_acc_y << "," << ff_acc_z << ","
            << cable.dir_x << "," << cable.dir_y << "," << cable.dir_z << ","
            << cable.mu << ","
            << current_x_ << "," << current_y_ << "," << current_z_ << "\n";
  log_file_.flush();
}

void PvaControlNode::debug_log_sample(double now_s)
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
                  << static_cast<int>(current_state_) << ","
                  << traj_started_ << ","
                  << waiting_for_swarm_ << ","
                  << traj_completed_ << ","
                  << odom_ready_ << ","
                  << trajectory_loaded_ << ","
                  << cable_loaded_ << "\n";
  debug_log_file_.flush();
}

void PvaControlNode::send_state_command(int state)
{
  std_msgs::msg::Int32 msg;
  msg.data = state;
  state_cmd_pub_->publish(msg);
  RCLCPP_INFO(this->get_logger(), "Sent state command: %d", state);
}
