#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int32.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <chrono>
#include <cmath>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

class GcsSwarmCoordinator : public rclcpp::Node
{
public:
  GcsSwarmCoordinator()
  : Node("gcs_swarm_coordinator")
  {
    takeoff_alt_ = this->declare_parameter<double>("takeoff_alt", 0.4);
    alt_tol_ = this->declare_parameter<double>("alt_tol", 0.05);
    hover_wait_time_ = this->declare_parameter<double>("hover_wait_time", 5.0);
    vicon_is_enu_ = this->declare_parameter<bool>("vicon_is_enu", true);
    use_vicon_altitude_ = this->declare_parameter<bool>("use_vicon_altitude", true);

    std::string drone_ids_csv = this->declare_parameter<std::string>("drone_ids_csv", "0");
    parse_csv_ids(drone_ids_csv, drone_ids_);
    if (drone_ids_.empty()) {
      throw std::runtime_error("drone_ids_csv is empty");
    }

    std::string vicon_names_csv = this->declare_parameter<std::string>("vicon_names_csv", "");
    std::string vicon_prefix = this->declare_parameter<std::string>("vicon_topic_prefix", "/vrpn_mocap/");
    std::string vicon_suffix = this->declare_parameter<std::string>("vicon_topic_suffix", "/pose");

    if (!vicon_names_csv.empty()) {
      parse_csv_strings(vicon_names_csv, vicon_names_);
      if (vicon_names_.size() != drone_ids_.size()) {
        throw std::runtime_error("vicon_names_csv count must match drone_ids_csv count");
      }
    } else {
      for (int id : drone_ids_) {
        vicon_names_.push_back("multilift_" + std::to_string(id));
      }
    }

    for (size_t i = 0; i < drone_ids_.size(); ++i) {
      int drone_id = drone_ids_[i];
      drone_states_[drone_id] = -1;
      alt_ready_[drone_id] = false;
      drone_alt_[drone_id] = 0.0;

      std::string state_topic = "/state/state_drone_" + std::to_string(drone_id);
      auto state_sub = this->create_subscription<std_msgs::msg::Int32>(
        state_topic, rclcpp::QoS(10),
        [this, drone_id](const std_msgs::msg::Int32::SharedPtr msg) {
          this->state_callback(msg, drone_id);
        });
      state_subs_.push_back(state_sub);

      std::string cmd_topic = "/state/command_drone_" + std::to_string(drone_id);
      auto cmd_pub = this->create_publisher<std_msgs::msg::Int32>(cmd_topic, rclcpp::QoS(10));
      cmd_pubs_.push_back(cmd_pub);

      if (use_vicon_altitude_) {
        std::string vicon_topic = vicon_prefix + vicon_names_[i] + vicon_suffix;
        auto vicon_sub = this->create_subscription<geometry_msgs::msg::PoseStamped>(
          vicon_topic, rclcpp::SensorDataQoS(),
          [this, drone_id](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
            this->vicon_callback(msg, drone_id);
          });
        vicon_subs_.push_back(vicon_sub);
        RCLCPP_INFO(this->get_logger(),
                    "Subscribed Vicon pose: %s (drone %d)", vicon_topic.c_str(), drone_id);
      }
    }

    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(100),
      std::bind(&GcsSwarmCoordinator::timer_callback, this));

    RCLCPP_INFO(this->get_logger(),
                "GCS coordinator: drones=%zu takeoff_alt=%.2f alt_tol=%.2f vicon=%s",
                drone_ids_.size(), takeoff_alt_, alt_tol_, use_vicon_altitude_ ? "on" : "off");
  }

private:
  void state_callback(const std_msgs::msg::Int32::SharedPtr msg, int drone_id)
  {
    int old_state = drone_states_[drone_id];
    drone_states_[drone_id] = msg->data;
    if (old_state != msg->data) {
      RCLCPP_INFO(this->get_logger(),
                  "Drone %d: state %d -> %d%s",
                  drone_id, old_state, msg->data,
                  (msg->data == HOVER_STATE) ? " (HOVER)" : "");
    }
  }

  void vicon_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg, int drone_id)
  {
    double z = msg->pose.position.z;
    // ENU: z up; NED: z down.
    double alt = vicon_is_enu_ ? z : -z;
    drone_alt_[drone_id] = alt;
    alt_ready_[drone_id] = true;
  }

  void timer_callback()
  {
    tick_++;

    // Phase 1: wait for takeoff altitude, then send GOTO once.
    if (!goto_sent_) {
      if (all_drones_ready_for_goto()) {
        RCLCPP_INFO(this->get_logger(), "All drones at takeoff altitude, sending GOTO");
        send_goto_command_to_all();
      } else if (tick_ % 20 == 0) {
        int ready_count = 0;
        int alt_count = 0;
        for (int id : drone_ids_) {
          if (alt_ready_.at(id)) {
            alt_count++;
          }
          if (is_altitude_ready(id)) {
            ready_count++;
          }
        }
        RCLCPP_INFO(this->get_logger(),
                    "Waiting takeoff altitude: %d/%zu ready (vicon %d/%zu)",
                    ready_count, drone_ids_.size(), alt_count, drone_ids_.size());
      }
      return;
    }

    // Phase 2: wait for all HOVER, then delay and send TRAJ once.
    if (!traj_sent_) {
      if (all_drones_ready_for_traj()) {
        if (!hover_timer_started_) {
          hover_ready_time_ = this->now();
          hover_timer_started_ = true;
          RCLCPP_INFO(this->get_logger(),
                      "All drones in HOVER after GOTO, starting %.1fs wait before TRAJ",
                      hover_wait_time_);
        } else {
          double hover_wait = (this->now() - hover_ready_time_).seconds();
          if (hover_wait >= hover_wait_time_) {
            RCLCPP_INFO(this->get_logger(),
                        "Hover wait complete (%.1fs >= %.1fs), sending TRAJ",
                        hover_wait, hover_wait_time_);
            send_traj_command_to_all();
          } else if (tick_ % 20 == 0) {
            RCLCPP_INFO(this->get_logger(),
                        "Hover wait: %.1fs / %.1fs", hover_wait, hover_wait_time_);
          }
        }
      } else {
        hover_timer_started_ = false;
        if (tick_ % 20 == 0) {
          int hover_count = 0;
          int unknown_count = 0;
          for (const auto &entry : drone_states_) {
            if (entry.second == HOVER_STATE) hover_count++;
            if (entry.second == -1) unknown_count++;
          }
          RCLCPP_INFO(this->get_logger(),
                      "Waiting post-GOTO: %d/%zu in HOVER, unknown=%d",
                      hover_count, drone_ids_.size(), unknown_count);
        }
      }
    }
  }

  bool is_altitude_ready(int drone_id) const
  {
    if (!use_vicon_altitude_) {
      return true;
    }
    auto it_ready = alt_ready_.find(drone_id);
    if (it_ready == alt_ready_.end() || !it_ready->second) {
      return false;
    }
    auto it_alt = drone_alt_.find(drone_id);
    if (it_alt == drone_alt_.end()) {
      return false;
    }
    return it_alt->second >= (takeoff_alt_ - alt_tol_);
  }

  bool all_drones_ready_for_goto() const
  {
    for (int id : drone_ids_) {
      if (!is_altitude_ready(id)) {
        return false;
      }
    }
    return true;
  }

  bool all_drones_ready_for_traj() const
  {
    for (const auto &entry : drone_states_) {
      if (entry.second != HOVER_STATE) {
        return false;
      }
    }
    return true;
  }

  void send_goto_command_to_all()
  {
    std_msgs::msg::Int32 cmd;
    cmd.data = GOTO_STATE;
    for (int repeat = 0; repeat < 5; ++repeat) {
      for (size_t i = 0; i < drone_ids_.size(); ++i) {
        cmd_pubs_[i]->publish(cmd);
      }
      rclcpp::sleep_for(std::chrono::milliseconds(50));
    }
    goto_sent_ = true;
  }

  void send_traj_command_to_all()
  {
    std_msgs::msg::Int32 cmd;
    cmd.data = TRAJ_STATE;
    for (int repeat = 0; repeat < 5; ++repeat) {
      for (size_t i = 0; i < drone_ids_.size(); ++i) {
        cmd_pubs_[i]->publish(cmd);
      }
      rclcpp::sleep_for(std::chrono::milliseconds(50));
    }
    traj_sent_ = true;
  }

  void parse_csv_ids(const std::string &csv, std::vector<int> &out_ids)
  {
    std::istringstream iss(csv);
    std::string token;
    while (std::getline(iss, token, ',')) {
      if (token.empty()) {
        continue;
      }
      try {
        out_ids.push_back(std::stoi(token));
      } catch (const std::exception &) {
        RCLCPP_WARN(this->get_logger(), "Invalid drone id token '%s'", token.c_str());
      }
    }
  }

  void parse_csv_strings(const std::string &csv, std::vector<std::string> &out)
  {
    std::istringstream iss(csv);
    std::string token;
    while (std::getline(iss, token, ',')) {
      if (!token.empty()) {
        out.push_back(token);
      }
    }
  }

  double takeoff_alt_{0.4};
  double alt_tol_{0.05};
  double hover_wait_time_{5.0};
  bool vicon_is_enu_{true};
  bool use_vicon_altitude_{true};

  std::vector<int> drone_ids_;
  std::vector<std::string> vicon_names_;

  std::unordered_map<int, int> drone_states_;
  std::unordered_map<int, double> drone_alt_;
  std::unordered_map<int, bool> alt_ready_;

  std::vector<rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr> state_subs_;
  std::vector<rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr> cmd_pubs_;
  std::vector<rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr> vicon_subs_;
  rclcpp::TimerBase::SharedPtr timer_;

  bool goto_sent_{false};
  bool traj_sent_{false};
  bool hover_timer_started_{false};
  rclcpp::Time hover_ready_time_;
  int tick_{0};

  static constexpr int HOVER_STATE = 4;
  static constexpr int GOTO_STATE = 3;
  static constexpr int TRAJ_STATE = 5;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<GcsSwarmCoordinator>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
