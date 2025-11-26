#include "traj_test/swarm_coordinator_node.hpp"

SwarmCoordinator::SwarmCoordinator(int total_drones)
  : Node("swarm_coordinator")
  , total_drones_(total_drones)
  , traj_sent_(false)
{
  RCLCPP_INFO(this->get_logger(), "=== Swarm Coordinator initialized for %d drones ===", total_drones_);
  
  // NEW: Declare and get drone_ids parameter
  this->declare_parameter<std::vector<int64_t>>("drone_ids", std::vector<int64_t>{});
  std::vector<int64_t> drone_ids_param = this->get_parameter("drone_ids").as_integer_array();
  
  // Convert int64_t to int and store
  if (drone_ids_param.empty()) {
    // Fallback: use sequential IDs 0, 1, 2, ...
    RCLCPP_WARN(this->get_logger(), "No drone_ids parameter found, using default sequential IDs");
    for (int i = 0; i < total_drones_; i++) {
      drone_ids_.push_back(i);
    }
  } else {
    for (auto id : drone_ids_param) {
      drone_ids_.push_back(static_cast<int>(id));
    }
  }
  
  // Validate
  if (drone_ids_.size() != static_cast<size_t>(total_drones_)) {
    RCLCPP_ERROR(this->get_logger(), 
                 "Mismatch: total_drones=%d but got %zu drone IDs", 
                 total_drones_, drone_ids_.size());
    throw std::runtime_error("Drone ID count mismatch");
  }
  
  // Log the drone IDs being used
  std::stringstream ss;
  ss << "Using drone IDs: [";
  for (size_t i = 0; i < drone_ids_.size(); i++) {
    ss << drone_ids_[i];
    if (i < drone_ids_.size() - 1) ss << ", ";
  }
  ss << "]";
  RCLCPP_INFO(this->get_logger(), "%s", ss.str().c_str());
  
  // Initialize state tracking with custom IDs
  for (int drone_id : drone_ids_) {
    drone_states_[drone_id] = -1;  // Unknown state initially
  }
  
  // Subscribe to all drone states with RELIABLE QoS using custom IDs
  for (int drone_id : drone_ids_) {
    auto qos = rclcpp::QoS(10).reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
    
    auto sub = this->create_subscription<std_msgs::msg::Int32>(
      "/state/state_drone_" + std::to_string(drone_id),
      qos,
      [this, drone_id](const std_msgs::msg::Int32::SharedPtr msg) {
        this->state_callback(msg, drone_id);
      });
    state_subs_.push_back(sub);
    
    RCLCPP_INFO(this->get_logger(), "Subscribed to /state/state_drone_%d", drone_id);
  }
  
  // Create command publishers with RELIABLE QoS using custom IDs
  for (int drone_id : drone_ids_) {
    auto qos = rclcpp::QoS(10).reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
    auto pub = this->create_publisher<std_msgs::msg::Int32>(
      "/state/command_drone_" + std::to_string(drone_id),
      qos);
    cmd_pubs_.push_back(pub);
    RCLCPP_INFO(this->get_logger(), "Created command publisher for drone %d", drone_id);
  }

  // Timer at 10 Hz
  timer_ = rclcpp::create_timer(
      this,
      this->get_clock(),
      rclcpp::Duration(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(0.1)
      )),
      std::bind(&SwarmCoordinator::timer_callback, this));
    
  RCLCPP_INFO(this->get_logger(), "Swarm Coordinator ready - Timer started at 10 Hz");
}

void SwarmCoordinator::state_callback(const std_msgs::msg::Int32::SharedPtr msg, int drone_id)
{
  int old_state = drone_states_[drone_id];
  drone_states_[drone_id] = msg->data;
  
  // Log state changes
  if (old_state != msg->data) {
    RCLCPP_INFO(this->get_logger(), 
                "Drone %d: state %d -> %d%s", 
                drone_id, old_state, msg->data,
                (msg->data == HOVER_STATE) ? " (HOVER)" : "");
  }
}

bool SwarmCoordinator::all_drones_in_hover()
{
  for (const auto& [drone_id, state] : drone_states_) {
    if (state != HOVER_STATE) {
      return false;
    }
  }
  return true;
}

void SwarmCoordinator::send_traj_command_to_all()
{
  std_msgs::msg::Int32 cmd;
  cmd.data = TRAJ_STATE;
  
  RCLCPP_WARN(this->get_logger(),
              "SENDING TRAJ COMMAND TO ALL DRONES");
  
  // Send MULTIPLE times to ensure delivery
  for (int repeat = 0; repeat < 5; repeat++) {
    for (size_t i = 0; i < drone_ids_.size(); i++) {
      int drone_id = drone_ids_[i];
      cmd_pubs_[i]->publish(cmd);
      RCLCPP_INFO(this->get_logger(), 
                  "  -> Sent TRAJ (state=%d) to drone %d (attempt %d/5)", 
                  TRAJ_STATE, drone_id, repeat + 1);
    }
    // Small delay between retries for message delivery
    rclcpp::sleep_for(std::chrono::milliseconds(50));
  }
  
  traj_sent_ = true;
  RCLCPP_WARN(this->get_logger(), "TRAJ commands sent successfully");
}

void SwarmCoordinator::timer_callback()
{
  static int tick = 0;
  tick++;
  
  // Periodic status logging
  if (tick % 50 == 0) {
    RCLCPP_INFO(this->get_logger(), "--- Status Check (tick=%d) ---", tick);
    for (const auto& [drone_id, state] : drone_states_) {
      RCLCPP_INFO(this->get_logger(), "  Drone %d: state=%d", drone_id, state);
    }
  }
  
  // If already sent, do nothing
  if (traj_sent_) {
    return;
  }
  
  // Check if all drones are in HOVER
  if (all_drones_in_hover()) {
    RCLCPP_INFO(this->get_logger(), "All drones in HOVER state, sending TRAJ command...");
    send_traj_command_to_all();
  } else {
    // Log waiting status every 5 ticks (0.5 seconds)
    if (tick % 5 == 0) {
      int hover_count = 0;
      int unknown_count = 0;
      for (const auto& [drone_id, state] : drone_states_) {
        if (state == HOVER_STATE) hover_count++;
        if (state == -1) unknown_count++;
      }
      
      RCLCPP_INFO(this->get_logger(), 
                  "Waiting for HOVER: %d/%d ready, %d unknown", 
                  hover_count, total_drones_, unknown_count);
    }
  }
}