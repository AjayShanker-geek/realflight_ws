// offboard_fsm_node.cpp – fixed version
// Finite-state machine for PX4 offboard control (ROS 2)
// Author: Yichao Gao 

#include "offboard_fsm_node.hpp"
#include <Eigen/Core>
#include <algorithm>
#include <chrono>
#include <cmath>

using namespace px4_msgs::msg;

/* ------------------------------------------------------------------ */
/*  Helpers                                                           */
/* ------------------------------------------------------------------ */
static float wrap_pi(float x)
{
  while (x >  M_PI) x -= 2.f * M_PI;
  while (x < -M_PI) x += 2.f * M_PI;
  return x;
}

// ENU → NED yaw: +90 deg rotation
static float enu_to_ned_yaw(float yaw_enu)
{
  float y = yaw_enu + static_cast<float>(M_PI_2);
  return wrap_pi(y);
}

/* ------------------------------------------------------------------ */
/*  Constructor                                                       */
/* ------------------------------------------------------------------ */
OffboardFSM::OffboardFSM(int drone_id)
: Node("offboard_fsm_node_" + std::to_string(drone_id))
, drone_id_(drone_id)
, takeoff_alt_(declare_parameter("takeoff_alt", 1.51))
, takeoff_time_s_ (declare_parameter("takeoff_time",    1.0))
, climb_rate_     (declare_parameter("climb_rate",      1.0))  
, landing_time_s_ (declare_parameter("landing_time",    5.0))
, circle_radius_  (declare_parameter("circle_radius",   1.4))
, inward_offset_  (declare_parameter("inward_offset",   0.80))
, num_drones_     (declare_parameter("num_drones",      6))
, timer_period_s_ (declare_parameter("timer_period",    0.02))
, alt_tol_        (declare_parameter("alt_tol",         0.03))
, radius_         (declare_parameter("circle_radius_traj", 3.0))
, period_s_       (declare_parameter("circle_period",   20.0))
// goto parameters
, goto_x_       (declare_parameter<double>("goto_x", std::numeric_limits<double>::quiet_NaN()))
, goto_y_       (declare_parameter<double>("goto_y", std::numeric_limits<double>::quiet_NaN()))
, goto_z_       (declare_parameter<double>("goto_z", std::numeric_limits<double>::quiet_NaN()))
, goto_tol_     (declare_parameter("goto_tol", 0.1))
// payload offset
, payload_offset_x_(declare_parameter("payload_offset_x", 0.0))
, payload_offset_y_(declare_parameter("payload_offset_y", 0.0))
// initial state
, current_state_(FsmState::INIT)
, offb_counter_(0)
, takeoff_start_count_(0)
, takeoff_complete_count_(-1)
, landing_start_count_(0)
, use_attitude_control_(false)         
, odom_ready_(false)
// hover position
, hover_x_(0.0)
, hover_y_(0.0)
, hover_z_(-1.2)
// command tracking
, offboard_cmd_count_(0)
, arm_cmd_count_(0)
, last_cmd_time_(0)
// Initialize other position variables
, takeoff_pos_x_(0.0)
, takeoff_pos_y_(0.0)
, landing_x_(0.0)
, landing_y_(0.0)
, landing_start_z_(0.0)
, current_x_(0.0)
, current_y_(0.0)
, current_z_(0.0)
, last_z_(0.0)
, nav_state_(0)
, arming_state_(0)
, start_time_(std::chrono::steady_clock::now())
{
  // Calculate takeoff position
  double theta    = 2.0 * M_PI * drone_id_ / static_cast<double>(num_drones_);
  double r_target = - inward_offset_;
  takeoff_pos_x_  =  r_target * std::sin(theta) + payload_offset_x_;
  takeoff_pos_y_  =  r_target * std::cos(theta) + payload_offset_y_;
  
  climb_rate_ = takeoff_alt_ / takeoff_time_s_;

  RCLCPP_INFO(get_logger(),
              "Init FSM for drone %d: take-off to %.2f m",
              drone_id_, takeoff_alt_);

  // PX4 namespace
  px4_ns_ = (drone_id_ == 0) ? "/fmu/" :
            "/px4_" + std::to_string(drone_id_) + "/fmu/";

  // Publishers with increased QoS
  pub_offb_mode_ = create_publisher<OffboardControlMode>(
      px4_ns_ + "in/offboard_control_mode", rclcpp::QoS{10});
  pub_cmd_       = create_publisher<VehicleCommand>(
      px4_ns_ + "in/vehicle_command",       rclcpp::QoS{10});
  pub_state_     = create_publisher<std_msgs::msg::Int32>(
      "/state/state_drone_" + std::to_string(drone_id_), rclcpp::QoS{10});
  pub_traj_sp_   = create_publisher<TrajectorySetpoint>(
      px4_ns_ + "in/trajectory_setpoint",   rclcpp::QoS{10});

  // Subscriptions
  sub_status_ = create_subscription<VehicleStatus>(
      px4_ns_ + "out/vehicle_status_v1", rclcpp::SensorDataQoS(),
      std::bind(&OffboardFSM::status_cb, this, std::placeholders::_1));

  sub_state_cmd_ = create_subscription<std_msgs::msg::Int32>(
      "/state/command_drone_" + std::to_string(drone_id_), rclcpp::QoS{10},
      std::bind(&OffboardFSM::state_cmd_cb, this, std::placeholders::_1));

  sub_odom_ = create_subscription<VehicleOdometry>(
      px4_ns_ + "out/vehicle_odometry", rclcpp::SensorDataQoS(),
      std::bind(&OffboardFSM::odom_cb, this, std::placeholders::_1));

  // Timer
  timer_ = create_wall_timer(std::chrono::duration<double>(timer_period_s_),
                             std::bind(&OffboardFSM::timer_cb, this));
}

uint64_t OffboardFSM::get_timestamp_us()
{
  // use chrono to get time since start in microseconds
  auto now = std::chrono::steady_clock::now();
  auto duration = now - start_time_;
  return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
}

/* ------------------------------------------------------------------ */
/*  Callbacks                                                         */
/* ------------------------------------------------------------------ */
void OffboardFSM::status_cb(const VehicleStatus::SharedPtr msg)
{
  nav_state_    = msg->nav_state;
  arming_state_ = msg->arming_state;
}

void OffboardFSM::odom_cb(const VehicleOdometry::SharedPtr msg)
{
  current_x_ = msg->position[0];
  current_y_ = msg->position[1];
  current_z_ = msg->position[2];
  odom_ready_ = true;
}

void OffboardFSM::state_cmd_cb(const std_msgs::msg::Int32::SharedPtr msg)
{
  int s = msg->data;
  if (s < static_cast<int>(FsmState::INIT) ||
      s > static_cast<int>(FsmState::DONE))
    return;

  // External state override
  current_state_ = static_cast<FsmState>(s);

  // Handle LAND state
  if (current_state_ == FsmState::LAND) {
    landing_start_count_ = offb_counter_;
    landing_start_z_     = -current_z_;  // Convert to +up
    landing_x_           = current_x_;
    landing_y_           = current_y_;
  }

  // Handle TRAJ state
  if (current_state_ == FsmState::TRAJ) {
    state_start_time_     = now();
    use_attitude_control_ = true;
  }

  RCLCPP_WARN(get_logger(), "External state override → %d", s);
}

bool OffboardFSM::has_goto_target() const
{
  return std::isfinite(goto_x_) &&
         std::isfinite(goto_y_) &&
         std::isfinite(goto_z_);
}

/* ------------------------------------------------------------------ */
/*  Publish trajectory setpoint                                      */
/* ------------------------------------------------------------------ */
void OffboardFSM::publish_current_setpoint()
{
  TrajectorySetpoint sp;
  
  // Initialize NaN for unused fields
  sp.position[0] = 0.0f;
  sp.position[1] = 0.0f;
  sp.position[2] = 0.0f;
  sp.velocity[0] = std::nanf("");
  sp.velocity[1] = std::nanf("");
  sp.velocity[2] = std::nanf("");
  sp.acceleration[0] = std::nanf("");
  sp.acceleration[1] = std::nanf("");
  sp.acceleration[2] = std::nanf("");
  sp.yaw = std::nanf("");
  sp.yawspeed = std::nanf("");
  
  // Set position based on state
  switch (current_state_) {
    case FsmState::INIT:
    case FsmState::ARMING:
      // Send takeoff position before armed
      sp.position[0] = takeoff_pos_x_;
      sp.position[1] = takeoff_pos_y_;
      sp.position[2] = -0.05f;  // Slightly above ground
      break;
      
    case FsmState::TAKEOFF: {
      // Smooth takeoff trajectory
      double elapsed = (offb_counter_ - takeoff_start_count_) * timer_period_s_;
      double alt_sp = std::min(takeoff_alt_, climb_rate_ * elapsed);
      sp.position[0] = takeoff_pos_x_;
      sp.position[1] = takeoff_pos_y_;
      sp.position[2] = -alt_sp;  // NED: negative is up
      
      // Add climb velocity for better tracking
      if (alt_sp < takeoff_alt_) {
        sp.velocity[2] = -climb_rate_;  // Climbing (negative in NED)
      } else {
        sp.velocity[2] = 0.0f;  // Stop at target altitude
      }
      break;
    }
    
    case FsmState::GOTO:
      if (has_goto_target()) {
        sp.position[0] = goto_x_;
        sp.position[1] = goto_y_;
        sp.position[2] = goto_z_;
      } else {
        // Use current position as fallback
        sp.position[0] = current_x_;
        sp.position[1] = current_y_;
        sp.position[2] = current_z_;
      }
      break;
      
    case FsmState::HOVER:
      sp.position[0] = hover_x_;
      sp.position[1] = hover_y_;
      sp.position[2] = hover_z_;
      // Set zero velocity for stable hover
      sp.velocity[0] = 0.0f;
      sp.velocity[1] = 0.0f;
      sp.velocity[2] = 0.0f;
      break;
      
    case FsmState::TRAJ:
    case FsmState::END_TRAJ:
      // Maintain hover position
      sp.position[0] = hover_x_;
      sp.position[1] = hover_y_;
      sp.position[2] = hover_z_;
      break;
      
    case FsmState::LAND: {
      // Smooth landing trajectory
      double elapsed = (offb_counter_ - landing_start_count_) * timer_period_s_;
      double descent_rate = landing_start_z_ / landing_time_s_;
      double alt = std::max(0.0, landing_start_z_ - descent_rate * elapsed);
      sp.position[0] = landing_x_;
      sp.position[1] = landing_y_;
      sp.position[2] = -alt;  // NED
      
      // Add descent velocity
      if (alt > 0.1) {
        sp.velocity[2] = descent_rate;  // Descending (positive in NED)
      } else {
        sp.velocity[2] = 0.0f;
      }
      break;
    }
    
    case FsmState::DONE:
      // Ground position
      sp.position[0] = landing_x_;
      sp.position[1] = landing_y_;
      sp.position[2] = 0.0f;
      sp.velocity[0] = 0.0f;
      sp.velocity[1] = 0.0f;
      sp.velocity[2] = 0.0f;
      break;
  }
  
  // Validate setpoint
  if (!std::isfinite(sp.position[0]) || 
      !std::isfinite(sp.position[1]) || 
      !std::isfinite(sp.position[2])) {
    RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000,
                          "Invalid setpoint! Using current position.");
    if (odom_ready_) {
      sp.position[0] = current_x_;
      sp.position[1] = current_y_;
      sp.position[2] = current_z_;
    } else {
      sp.position[0] = 0.0f;
      sp.position[1] = 0.0f;
      sp.position[2] = 0.0f;
    }
  }
  
  sp.timestamp = get_timestamp_us();  // Convert to microseconds
  pub_traj_sp_->publish(sp);
}

/* ------------------------------------------------------------------ */
/*  Main timer callback                                               */
/* ------------------------------------------------------------------ */
void OffboardFSM::timer_cb()
{
  // CRITICAL: Always publish heartbeat signals first
  publish_offboard_mode();
  publish_current_setpoint();
  
  // State machine logic
  switch (current_state_) {
  case FsmState::INIT:
    // Wait for stable control stream before attempting mode switch
    if (offb_counter_ >= 20) {  // 400ms of stable signals
      RCLCPP_INFO_ONCE(get_logger(), "Starting arming sequence");
      current_state_ = FsmState::ARMING;
      offb_counter_ = 0;
    }
    break;

  case FsmState::ARMING: {
    bool is_offboard = (nav_state_ == VehicleStatus::NAVIGATION_STATE_OFFBOARD);
    bool is_armed = (arming_state_ == VehicleStatus::ARMING_STATE_ARMED);
    
    if (is_offboard && is_armed) {
      // Success
      RCLCPP_INFO(get_logger(), "Drone %d: Armed & Offboard", drone_id_);
      current_state_       = FsmState::TAKEOFF;
      takeoff_start_count_ = offb_counter_;
      last_z_              = current_z_;
      offb_counter_        = 0;
      offboard_cmd_count_  = 0;
      arm_cmd_count_       = 0;
    } else {
      // Send commands with rate limiting
      if (!is_offboard && offboard_cmd_count_ % 50 == 0) {  // Every 1s
        send_vehicle_cmd(VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1.f, 6.f);
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                            "Requesting offboard mode...");
      }
      offboard_cmd_count_++;
      
      if (is_offboard && !is_armed && arm_cmd_count_ % 50 == 0) {  // Every 1s
        send_vehicle_cmd(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.f, 0.f);
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                            "Requesting arm...");
      }
      if (is_offboard) arm_cmd_count_++;
    }
    break;
  }

  case FsmState::TAKEOFF: {
    double actual_alt = -current_z_;  // Convert to +up
    
    // Check altitude reached
    if (takeoff_complete_count_ < 0) {
      bool alt_ok = actual_alt >= (takeoff_alt_ - alt_tol_);
      bool vel_ok = std::abs(current_z_ - last_z_) / timer_period_s_ < 0.1;  // <0.1 m/s
      
      if (alt_ok && vel_ok) {
        takeoff_complete_count_ = offb_counter_;
        RCLCPP_INFO(get_logger(), "Altitude reached: %.2f m", actual_alt);
      }
    }
    
    // Hover for 5s then transition
    if (takeoff_complete_count_ >= 0) {
      double hover_time = (offb_counter_ - takeoff_complete_count_) * timer_period_s_;
      if (hover_time >= 5.0) {
        if (has_goto_target()) {
          current_state_ = FsmState::GOTO;
          RCLCPP_INFO(get_logger(), "Moving to GOTO target");
        } else {
          hover_x_ = takeoff_pos_x_;
          hover_y_ = takeoff_pos_y_;
          hover_z_ = -takeoff_alt_;
          current_state_ = FsmState::HOVER;
          RCLCPP_INFO(get_logger(), "Entering HOVER mode");
        }
        state_start_time_ = now();
        offb_counter_ = 0;
      }
    }
    
    last_z_ = current_z_;
    break;
  }

  case FsmState::GOTO: {
    if (!has_goto_target()) {
      // No target, switch to hover
      hover_x_ = current_x_;
      hover_y_ = current_y_;
      hover_z_ = current_z_;
      current_state_ = FsmState::HOVER;
      RCLCPP_WARN(get_logger(), "No GOTO target, switching to HOVER");
      break;
    }

    // Check if reached target
    double dx = current_x_ - goto_x_;
    double dy = current_y_ - goto_y_;
    double dz = current_z_ - goto_z_;
    double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
    
    if (dist < goto_tol_) {
      RCLCPP_INFO(get_logger(), "GOTO target reached (error: %.3f m)", dist);
      hover_x_ = goto_x_;
      hover_y_ = goto_y_;
      hover_z_ = goto_z_;
      current_state_ = FsmState::HOVER;
      state_start_time_ = now();
      offb_counter_ = 0;
    }
    break;
  }

  case FsmState::HOVER:
    // Just hover at position
    if (use_attitude_control_) {
      use_attitude_control_ = false;
      RCLCPP_INFO(get_logger(), "Switched to position control");
    }
    break;

  case FsmState::TRAJ:
    // Trajectory mode (future implementation)
    if (!use_attitude_control_) {
      use_attitude_control_ = true;
      state_start_time_ = now();
      RCLCPP_INFO(get_logger(), "Switched to attitude control");
    }
    // generate_trajectory(); // Uncomment when implemented
    break;

  case FsmState::END_TRAJ:
    // End trajectory
    if (use_attitude_control_) {
      use_attitude_control_ = false;
      state_start_time_ = now();
      RCLCPP_INFO(get_logger(), "Trajectory complete");
    }
    break;

  case FsmState::LAND: {
    double elapsed = (offb_counter_ - landing_start_count_) * timer_period_s_;
    double alt = -current_z_;  // Convert to +up
    
    if (alt <= 0.15 || elapsed >= landing_time_s_) {
      RCLCPP_INFO(get_logger(), "Drone %d landed", drone_id_);
      current_state_ = FsmState::DONE;
      // Disarm
      send_vehicle_cmd(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.f, 0.f);
    }
    break;
  }

  case FsmState::DONE:
    // Stay on ground
    break;
  }

  // Publish state
  std_msgs::msg::Int32 st; 
  st.data = static_cast<int>(current_state_);
  pub_state_->publish(st);
  
  ++offb_counter_;
}

/* ------------------------------------------------------------------ */
/*  Publish offboard control mode                                    */
/* ------------------------------------------------------------------ */
void OffboardFSM::publish_offboard_mode()
{
  OffboardControlMode m;
  m.position     = true;   // Position control
  m.velocity     = false;  
  m.acceleration = false;  
  m.attitude     = false;  
  m.body_rate    = false;
  m.timestamp    = get_timestamp_us();  // Microseconds
  pub_offb_mode_->publish(m);
}

/* ------------------------------------------------------------------ */
/*  Send vehicle command                                             */
/* ------------------------------------------------------------------ */
void OffboardFSM::send_vehicle_cmd(uint16_t cmd, float p1, float p2)
{
  VehicleCommand m;
  m.command       = cmd;
  m.param1        = p1;
  m.param2        = p2;
  m.target_system = drone_id_ + 1;  // PX4 system ID
  m.target_component = 1;  // Autopilot component
  m.source_system = 1;
  m.source_component = 1;
  m.from_external = true;
  m.timestamp     = get_timestamp_us();
  pub_cmd_->publish(m);
}

/* ------------------------------------------------------------------ */
/*  Try to set offboard and arm (deprecated - use separate calls)   */
/* ------------------------------------------------------------------ */
void OffboardFSM::try_set_offboard_and_arm()
{
  // This function is deprecated - use rate-limited calls in ARMING state
  send_vehicle_cmd(VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1.f, 6.f);
  send_vehicle_cmd(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.f, 0.f);
}

/* ------------------------------------------------------------------ */
/*  Trajectory generator placeholder                                 */
/* ------------------------------------------------------------------ */
void OffboardFSM::generate_trajectory()
{
  // Future implementation for trajectory generation
}