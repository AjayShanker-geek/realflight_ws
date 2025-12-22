#include "geom_multilift/realflight_traj_node.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <sstream>

RealflightTrajNode::RealflightTrajNode(int drone_id, int total_drones)
: Node("realflight_traj_node_" + std::to_string(drone_id))
, drone_id_(drone_id)
, total_drones_(total_drones)
, timer_period_(0.02)
, yaw_setpoint_(3.1415926)
, trajectory_loaded_(false)
, waiting_for_swarm_(false)
, traj_started_(false)
, traj_completed_(false)
, traj_time_initialized_(false)
, current_x_(0.0)
, current_y_(0.0)
, current_z_(0.0)
, odom_ready_(false)
, current_state_(FsmState::INIT)
{
  // Parameters
  timer_period_ = this->declare_parameter("timer_period", timer_period_);
  csv_base_path_ = this->declare_parameter("csv_base_path",
    std::string("data/3drone_trajectories_001_-001"));
  std::string csv_override = this->declare_parameter("csv_path", std::string(""));
  yaw_setpoint_ = this->declare_parameter("yaw_setpoint", yaw_setpoint_);

  if (csv_override.empty()) {
    csv_path_ = csv_base_path_ + "/drone_" + std::to_string(drone_id_) + "_traj.csv";
  } else {
    csv_path_ = csv_override;
  }

  // Initialize swarm state tracking
  for (int i = 0; i < total_drones_; ++i) {
    swarm_states_[i] = FsmState::INIT;
  }

  px4_namespace_ = get_px4_namespace(drone_id_);

  RCLCPP_INFO(this->get_logger(),
              "Realflight trajectory node for drone %d (total %d)", drone_id_, total_drones_);
  RCLCPP_INFO(this->get_logger(), "CSV path: %s", csv_path_.c_str());

  if (!load_trajectory_from_csv(csv_path_)) {
    RCLCPP_ERROR(this->get_logger(), "Failed to load trajectory file");
    return;
  }

  // Publishers: only talk to this drone's PX4 namespace
  traj_pub_ = this->create_publisher<px4_msgs::msg::TrajectorySetpoint>(
    px4_namespace_ + "in/trajectory_setpoint",
    rclcpp::QoS(10).reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE));

  state_cmd_pub_ = this->create_publisher<std_msgs::msg::Int32>(
    "/state/command_drone_" + std::to_string(drone_id_),
    rclcpp::QoS(10).reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE));

  // Subscriptions for FSM and odometry
  state_sub_ = this->create_subscription<std_msgs::msg::Int32>(
    "/state/state_drone_" + std::to_string(drone_id_),
    rclcpp::QoS(10).reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE),
    std::bind(&RealflightTrajNode::state_callback, this, std::placeholders::_1));

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
    std::bind(&RealflightTrajNode::odom_callback, this, std::placeholders::_1));

  // Use wall timer to stream setpoints
  timer_ = this->create_wall_timer(
    std::chrono::duration<double>(timer_period_),
    std::bind(&RealflightTrajNode::timer_callback, this));

  RCLCPP_INFO(this->get_logger(),
              "Initialized timer at %.1f Hz", 1.0 / timer_period_);
}

std::string RealflightTrajNode::get_px4_namespace(int drone_id) const
{
  if (drone_id == 0) {
    return "/fmu/";
  }
  return "/px4_" + std::to_string(drone_id) + "/fmu/";
}

bool RealflightTrajNode::load_trajectory_from_csv(const std::string &filepath)
{
  std::ifstream file(filepath);
  if (!file.is_open()) {
    RCLCPP_ERROR(this->get_logger(), "Cannot open CSV file: %s", filepath.c_str());
    return false;
  }

  std::string line;
  bool first_line = true;
  int point_count = 0;
  trajectory_.clear();

  while (std::getline(file, line)) {
    if (first_line) {
      first_line = false;
      continue;
    }
    if (line.empty()) {
      continue;
    }

    std::stringstream ss(line);
    std::string value;
    TrajectoryPoint point{};
    try {
      std::getline(ss, value, ',');
      point.time = std::stod(value);
      std::getline(ss, value, ',');
      point.x = std::stod(value);
      std::getline(ss, value, ',');
      point.y = std::stod(value);
      std::getline(ss, value, ',');
      point.z = std::stod(value);
      std::getline(ss, value, ',');
      point.vx = std::stod(value);
      std::getline(ss, value, ',');
      point.vy = std::stod(value);
      std::getline(ss, value, ',');
      point.vz = std::stod(value);
    } catch (const std::exception &e) {
      RCLCPP_WARN(this->get_logger(), "Skipping malformed line: %s", line.c_str());
      continue;
    }

    trajectory_.push_back(point);
    ++point_count;
  }

  if (trajectory_.empty()) {
    RCLCPP_ERROR(this->get_logger(), "Trajectory file contained no usable points");
    return false;
  }

  trajectory_loaded_ = true;
  RCLCPP_INFO(this->get_logger(), "Loaded %d trajectory samples (duration %.2fs)",
              point_count, trajectory_.back().time);
  return true;
}

TrajectoryPoint RealflightTrajNode::interpolate_trajectory(double t)
{
  if (t <= trajectory_.front().time) {
    return trajectory_.front();
  }
  if (t >= trajectory_.back().time) {
    return trajectory_.back();
  }

  size_t idx = 0;
  for (size_t i = 0; i < trajectory_.size() - 1; ++i) {
    if (trajectory_[i].time <= t && t < trajectory_[i + 1].time) {
      idx = i;
      break;
    }
  }

  const auto &p1 = trajectory_[idx];
  const auto &p2 = trajectory_[idx + 1];
  double dt = p2.time - p1.time;
  double alpha = (t - p1.time) / dt;

  TrajectoryPoint out;
  out.time = t;
  out.x = p1.x + alpha * (p2.x - p1.x);
  out.y = p1.y + alpha * (p2.y - p1.y);
  out.z = p1.z + alpha * (p2.z - p1.z);
  out.vx = p1.vx + alpha * (p2.vx - p1.vx);
  out.vy = p1.vy + alpha * (p2.vy - p1.vy);
  out.vz = p1.vz + alpha * (p2.vz - p1.vz);
  return out;
}

bool RealflightTrajNode::all_drones_in_traj_state() const
{
  for (const auto &entry : swarm_states_) {
    if (entry.second != FsmState::TRAJ) {
      return false;
    }
  }
  return true;
}

void RealflightTrajNode::state_callback(const std_msgs::msg::Int32::SharedPtr msg)
{
  auto new_state = static_cast<FsmState>(msg->data);
  auto old_state = current_state_;
  current_state_ = new_state;
  swarm_states_[drone_id_] = new_state;

  if (old_state != new_state) {
    RCLCPP_INFO(this->get_logger(), "State change: %d -> %d",
                static_cast<int>(old_state), static_cast<int>(new_state));
  }

  if (new_state == FsmState::TRAJ && !waiting_for_swarm_ && !traj_started_ && !traj_completed_) {
    waiting_for_swarm_ = true;
    if (total_drones_ == 1) {
      traj_started_ = true;
      waiting_for_swarm_ = false;
    }
  }

  if (waiting_for_swarm_ && !traj_started_ && all_drones_in_traj_state()) {
    traj_started_ = true;
    waiting_for_swarm_ = false;
  }

  if (traj_started_ && new_state != FsmState::TRAJ && !traj_completed_) {
    traj_started_ = false;
    waiting_for_swarm_ = false;
    traj_time_initialized_ = false;
    RCLCPP_ERROR(this->get_logger(), "Exited TRAJ early, stopping trajectory");
  }
}

void RealflightTrajNode::swarm_state_callback(
  const std_msgs::msg::Int32::SharedPtr msg, int other_drone_id)
{
  swarm_states_[other_drone_id] = static_cast<FsmState>(msg->data);
}

void RealflightTrajNode::odom_callback(const px4_msgs::msg::VehicleOdometry::SharedPtr msg)
{
  // Realflight: odometry already referenced to mocap/world frame
  current_x_ = msg->position[0];
  current_y_ = msg->position[1];
  current_z_ = msg->position[2];
  odom_ready_ = true;
}

void RealflightTrajNode::timer_callback()
{
  static int tick = 0;
  if ((tick++ % 100) == 0) {
    RCLCPP_INFO(this->get_logger(),
                "State %d | started %d | swarm_wait %d | completed %d | odom %d",
                static_cast<int>(current_state_), traj_started_,
                waiting_for_swarm_, traj_completed_, odom_ready_);
  }

  if (!odom_ready_) {
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "Waiting for odometry (realflight expects mocap local_position)");
    return;
  }

  if (!trajectory_loaded_) {
    RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                          "Trajectory not loaded");
    return;
  }

  if (current_state_ == FsmState::TRAJ && traj_started_ && !traj_completed_) {
    if (!traj_time_initialized_) {
      traj_start_time_ = this->now();
      traj_time_initialized_ = true;
      RCLCPP_INFO(this->get_logger(), "Trajectory execution started");
    }

    double elapsed = (this->now() - traj_start_time_).seconds();
    if (elapsed >= trajectory_.back().time) {
      if (!traj_completed_) {
        traj_completed_ = true;
        for (int i = 0; i < 5; ++i) {
          send_state_command(static_cast<int>(FsmState::END_TRAJ));
        }
        RCLCPP_INFO(this->get_logger(), "Trajectory complete -> END_TRAJ sent");
      }
      return;
    }

    auto point = interpolate_trajectory(elapsed);
    publish_trajectory_setpoint(point.x, point.y, point.z,
                                point.vx, point.vy, point.vz,
                                yaw_setpoint_);
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "t=%.2f pos=(%.2f, %.2f, %.2f) vel=(%.2f, %.2f, %.2f)",
                         elapsed, point.x, point.y, point.z,
                         point.vx, point.vy, point.vz);
  } else if (current_state_ == FsmState::TRAJ && waiting_for_swarm_) {
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "In TRAJ, waiting for swarm to be ready");
  }
}

void RealflightTrajNode::publish_trajectory_setpoint(
  double x, double y, double z,
  double vx, double vy, double vz,
  double yaw)
{
  px4_msgs::msg::TrajectorySetpoint msg;
  msg.position[0] = static_cast<float>(x);
  msg.position[1] = static_cast<float>(y);
  msg.position[2] = static_cast<float>(z);
  msg.velocity[0] = static_cast<float>(vx);
  msg.velocity[1] = static_cast<float>(vy);
  msg.velocity[2] = static_cast<float>(vz);
  msg.yaw = static_cast<float>(yaw);
  msg.timestamp = 0;
  traj_pub_->publish(msg);
}

void RealflightTrajNode::send_state_command(int state)
{
  std_msgs::msg::Int32 msg;
  msg.data = state;
  state_cmd_pub_->publish(msg);
}
