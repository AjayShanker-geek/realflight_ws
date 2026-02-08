#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int32.hpp>

#include <zmq.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <cstdint>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

class ZmqStateBridge : public rclcpp::Node
{
public:
  ZmqStateBridge()
  : Node("zmq_state_bridge")
  {
    mode_ = this->declare_parameter<std::string>("mode", "drone");
    poll_period_ms_ = this->declare_parameter<int>("poll_period_ms", 10);
    hwm_ = this->declare_parameter<int>("hwm", 10);
    recv_max_per_poll_ = this->declare_parameter<int>("recv_max_per_poll", 50);

    const char *env_id = std::getenv("DRONE_ID");
    if (!env_id) {
      throw std::runtime_error("DRONE_ID environment variable is not set");
    }
    drone_id_ = std::atoi(env_id);

    // Two roles: drone (ROS->ZMQ state, ZMQ->ROS command) or GCS (reverse).
    if (mode_ == "drone") {
      setup_drone_mode();
    } else if (mode_ == "gcs") {
      setup_gcs_mode();
    } else {
      throw std::runtime_error("Invalid mode parameter: " + mode_);
    }

    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(poll_period_ms_),
      std::bind(&ZmqStateBridge::poll_zmq, this));
  }

  ~ZmqStateBridge() override
  {
    close_sockets();
    close_udp_sockets();
    if (ctx_) {
      zmq_ctx_shutdown(ctx_);
      zmq_ctx_term(ctx_);
      ctx_ = nullptr;
    }
  }

private:
  void setup_drone_mode()
  {
    state_topic_ = "/state/state_drone_" + std::to_string(drone_id_);
    command_topic_ = "/state/command_drone_" + std::to_string(drone_id_);

    state_push_endpoint_ = this->declare_parameter<std::string>(
      "state_push_endpoint", "tcp://127.0.0.1:5555");
    cmd_sub_endpoint_ = this->declare_parameter<std::string>(
      "cmd_sub_endpoint", "tcp://127.0.0.1:5560");

    state_sub_ = this->create_subscription<std_msgs::msg::Int32>(
      state_topic_, rclcpp::QoS(10),
      std::bind(&ZmqStateBridge::on_state_local, this, std::placeholders::_1));

    cmd_pub_ = this->create_publisher<std_msgs::msg::Int32>(
      command_topic_, rclcpp::QoS(10));

    init_context();
    state_push_sock_ = create_socket(ZMQ_PUSH);
    cmd_sub_sock_ = create_socket(ZMQ_SUB);

    connect_or_throw(state_push_sock_, state_push_endpoint_, "state_push_endpoint");
    connect_or_throw(cmd_sub_sock_, cmd_sub_endpoint_, "cmd_sub_endpoint");

    // Filter commands by drone id; also receive state fanout (if enabled).
    std::string cmd_filter = "cmd " + std::to_string(drone_id_) + " ";
    std::string state_filter = "state ";
    zmq_setsockopt(cmd_sub_sock_, ZMQ_SUBSCRIBE, cmd_filter.c_str(), cmd_filter.size());
    zmq_setsockopt(cmd_sub_sock_, ZMQ_SUBSCRIBE, state_filter.c_str(), state_filter.size());

    RCLCPP_INFO(this->get_logger(),
                "DRONE mode: id=%d state_topic=%s command_topic=%s",
                drone_id_, state_topic_.c_str(), command_topic_.c_str());
    RCLCPP_INFO(this->get_logger(),
                "ZMQ state PUSH -> %s, cmd SUB <- %s",
                state_push_endpoint_.c_str(), cmd_sub_endpoint_.c_str());

    // Optional UDP listener for state fanout.
    udp_state_listen_ = this->declare_parameter<bool>("udp_state_listen", false);
    udp_state_port_ = this->declare_parameter<int>("udp_state_port", 5570);
    udp_state_bind_ = this->declare_parameter<std::string>("udp_state_bind", "0.0.0.0");
    if (udp_state_listen_) {
      setup_udp_state_listener();
    }
  }

  void setup_gcs_mode()
  {
    state_pull_bind_ = this->declare_parameter<std::string>(
      "state_pull_bind", "tcp://*:5555");
    cmd_pub_bind_ = this->declare_parameter<std::string>(
      "cmd_pub_bind", "tcp://*:5560");
    fanout_states_ = this->declare_parameter<bool>("fanout_states", true);
    udp_state_fanout_ = this->declare_parameter<bool>("udp_state_fanout", false);
    udp_state_port_ = this->declare_parameter<int>("udp_state_port", 5570);
    std::string udp_ips_csv = this->declare_parameter<std::string>("udp_state_drone_ips_csv", "");

    std::string ids_csv = this->declare_parameter<std::string>(
      "drone_ids_csv", "0");
    parse_csv_ids(ids_csv, drone_ids_);
    if (drone_ids_.empty()) {
      throw std::runtime_error("drone_ids is empty in gcs mode");
    }

    for (int id : drone_ids_) {
      std::string cmd_topic = "/state/command_drone_" + std::to_string(id);
      auto sub = this->create_subscription<std_msgs::msg::Int32>(
        cmd_topic, rclcpp::QoS(10),
        [this, id](const std_msgs::msg::Int32::SharedPtr msg) {
          this->on_command_local(id, msg);
        });
      cmd_subs_.push_back(sub);

      std::string state_topic = "/state/state_drone_" + std::to_string(id);
      auto pub = this->create_publisher<std_msgs::msg::Int32>(
        state_topic, rclcpp::QoS(10));
      state_pubs_.emplace(id, pub);
    }

    init_context();
    state_pull_sock_ = create_socket(ZMQ_PULL);
    cmd_pub_sock_ = create_socket(ZMQ_PUB);

    bind_or_throw(state_pull_sock_, state_pull_bind_, "state_pull_bind");
    bind_or_throw(cmd_pub_sock_, cmd_pub_bind_, "cmd_pub_bind");

    if (udp_state_fanout_) {
      parse_csv_strings(udp_ips_csv, udp_state_drone_ips_);
      if (udp_state_drone_ips_.size() != drone_ids_.size()) {
        throw std::runtime_error("udp_state_drone_ips_csv count must match drone_ids_csv");
      }
      setup_udp_state_fanout();
    }

    RCLCPP_INFO(this->get_logger(), "GCS mode: drone_ids=%zu", drone_ids_.size());
    RCLCPP_INFO(this->get_logger(), "ZMQ state PULL <- %s, cmd PUB -> %s (fanout=%s, udp=%s)",
                state_pull_bind_.c_str(), cmd_pub_bind_.c_str(),
                fanout_states_ ? "true" : "false",
                udp_state_fanout_ ? "true" : "false");
  }

  void init_context()
  {
    ctx_ = zmq_ctx_new();
    if (!ctx_) {
      throw std::runtime_error("Failed to create ZMQ context");
    }
  }

  void *create_socket(int type)
  {
    void *sock = zmq_socket(ctx_, type);
    if (!sock) {
      throw std::runtime_error("Failed to create ZMQ socket");
    }
    int hwm = hwm_;
    int linger = 0;
    zmq_setsockopt(sock, ZMQ_SNDHWM, &hwm, sizeof(hwm));
    zmq_setsockopt(sock, ZMQ_RCVHWM, &hwm, sizeof(hwm));
    zmq_setsockopt(sock, ZMQ_LINGER, &linger, sizeof(linger));
    return sock;
  }

  void connect_or_throw(void *sock, const std::string &endpoint, const std::string &label)
  {
    if (zmq_connect(sock, endpoint.c_str()) != 0) {
      throw std::runtime_error("ZMQ connect failed for " + label + ": " + endpoint);
    }
  }

  void bind_or_throw(void *sock, const std::string &endpoint, const std::string &label)
  {
    if (zmq_bind(sock, endpoint.c_str()) != 0) {
      throw std::runtime_error("ZMQ bind failed for " + label + ": " + endpoint);
    }
  }

  void close_sockets()
  {
    if (state_push_sock_) {
      zmq_close(state_push_sock_);
      state_push_sock_ = nullptr;
    }
    if (state_pull_sock_) {
      zmq_close(state_pull_sock_);
      state_pull_sock_ = nullptr;
    }
    if (cmd_pub_sock_) {
      zmq_close(cmd_pub_sock_);
      cmd_pub_sock_ = nullptr;
    }
    if (cmd_sub_sock_) {
      zmq_close(cmd_sub_sock_);
      cmd_sub_sock_ = nullptr;
    }
  }

  void close_udp_sockets()
  {
    if (udp_recv_sock_ >= 0) {
      close(udp_recv_sock_);
      udp_recv_sock_ = -1;
    }
    if (udp_send_sock_ >= 0) {
      close(udp_send_sock_);
      udp_send_sock_ = -1;
    }
  }

  void on_state_local(const std_msgs::msg::Int32::SharedPtr msg)
  {
    // Message format: "state <id> <value>"
    std::ostringstream oss;
    oss << "state " << drone_id_ << " " << msg->data;
    send_zmq(state_push_sock_, oss.str());
  }

  void on_command_local(int drone_id, const std_msgs::msg::Int32::SharedPtr msg)
  {
    // Message format: "cmd <id> <value>"
    std::ostringstream oss;
    oss << "cmd " << drone_id << " " << msg->data;
    send_zmq(cmd_pub_sock_, oss.str());
  }

  void poll_zmq()
  {
    if (mode_ == "drone") {
      poll_commands_from_zmq();
      poll_udp_states();
    } else if (mode_ == "gcs") {
      poll_states_from_zmq();
    }
  }

  void poll_commands_from_zmq()
  {
    if (!cmd_sub_sock_) {
      return;
    }

    // Drain at most recv_max_per_poll_ to avoid starving ROS callbacks.
    int processed = 0;
    while (processed < recv_max_per_poll_) {
      std::string msg;
      if (!recv_zmq(cmd_sub_sock_, msg)) {
        break;
      }
      int id = -1;
      int value = 0;
      if (parse_triplet(msg, "cmd", id, value)) {
        if (id == drone_id_) {
          std_msgs::msg::Int32 out;
          out.data = value;
          cmd_pub_->publish(out);
        }
      } else if (parse_triplet(msg, "state", id, value)) {
        if (id != drone_id_) {
          auto pub = get_or_create_state_pub(id);
          if (pub) {
            std_msgs::msg::Int32 out;
            out.data = value;
            pub->publish(out);
          }
        }
      } else {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                             "Malformed ZMQ message: '%s'", msg.c_str());
      }
      processed++;
    }
  }

  void poll_states_from_zmq()
  {
    if (!state_pull_sock_) {
      return;
    }

    // Drain at most recv_max_per_poll_ to avoid starving ROS callbacks.
    int processed = 0;
    while (processed < recv_max_per_poll_) {
      std::string msg;
      if (!recv_zmq(state_pull_sock_, msg)) {
        break;
      }
      int id = -1;
      int value = 0;
      if (!parse_triplet(msg, "state", id, value)) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                             "Malformed state message: '%s'", msg.c_str());
      } else {
        auto it = state_pubs_.find(id);
        if (it != state_pubs_.end()) {
          std_msgs::msg::Int32 out;
          out.data = value;
          it->second->publish(out);
        }
        if (fanout_states_) {
          std::ostringstream oss;
          oss << "state " << id << " " << value;
          send_zmq(cmd_pub_sock_, oss.str());
        }
        if (udp_state_fanout_) {
          send_udp_state(id, value);
        }
      }
      processed++;
    }
  }

  void poll_udp_states()
  {
    if (!udp_state_listen_ || udp_recv_sock_ < 0) {
      return;
    }
    int processed = 0;
    while (processed < recv_max_per_poll_) {
      char buf[256];
      sockaddr_in src {};
      socklen_t slen = sizeof(src);
      int rc = recvfrom(udp_recv_sock_, buf, sizeof(buf) - 1, MSG_DONTWAIT,
                        reinterpret_cast<sockaddr *>(&src), &slen);
      if (rc < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
          break;
        }
        break;
      }
      buf[rc] = '\0';
      std::string msg(buf);
      int id = -1;
      int value = 0;
      if (parse_triplet(msg, "state", id, value)) {
        if (id != drone_id_) {
          auto pub = get_or_create_state_pub(id);
          if (pub) {
            std_msgs::msg::Int32 out;
            out.data = value;
            pub->publish(out);
          }
        }
      }
      processed++;
    }
  }

  bool send_zmq(void *sock, const std::string &msg)
  {
    if (!sock) {
      return false;
    }
    // Non-blocking send to avoid backpressure in control loop.
    std::lock_guard<std::mutex> lock(zmq_mutex_);
    int rc = zmq_send(sock, msg.data(), msg.size(), ZMQ_DONTWAIT);
    return rc >= 0;
  }

  bool recv_zmq(void *sock, std::string &out)
  {
    if (!sock) {
      return false;
    }
    // Fixed-size buffer; messages are short (cmd/state + two ints).
    char buf[256];
    std::lock_guard<std::mutex> lock(zmq_mutex_);
    int rc = zmq_recv(sock, buf, sizeof(buf) - 1, ZMQ_DONTWAIT);
    if (rc < 0) {
      return false;
    }
    buf[rc] = '\0';
    out.assign(buf);
    return true;
  }

  bool parse_triplet(const std::string &msg, const std::string &tag, int &id, int &value)
  {
    std::istringstream iss(msg);
    std::string t;
    if (!(iss >> t >> id >> value)) {
      return false;
    }
    return t == tag;
  }

  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr get_or_create_state_pub(int id)
  {
    auto it = remote_state_pubs_.find(id);
    if (it != remote_state_pubs_.end()) {
      return it->second;
    }
    std::string topic = "/state/state_drone_" + std::to_string(id);
    auto pub = this->create_publisher<std_msgs::msg::Int32>(topic, rclcpp::QoS(10));
    remote_state_pubs_.emplace(id, pub);
    return pub;
  }

  void setup_udp_state_listener()
  {
    udp_recv_sock_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (udp_recv_sock_ < 0) {
      throw std::runtime_error("Failed to create UDP recv socket");
    }
    int reuse = 1;
    setsockopt(udp_recv_sock_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    sockaddr_in addr {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(udp_state_port_));
    if (inet_pton(AF_INET, udp_state_bind_.c_str(), &addr.sin_addr) != 1) {
      throw std::runtime_error("Invalid udp_state_bind: " + udp_state_bind_);
    }
    if (bind(udp_recv_sock_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
      throw std::runtime_error("UDP bind failed on port " + std::to_string(udp_state_port_));
    }
    int flags = fcntl(udp_recv_sock_, F_GETFL, 0);
    fcntl(udp_recv_sock_, F_SETFL, flags | O_NONBLOCK);
  }

  void setup_udp_state_fanout()
  {
    udp_send_sock_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (udp_send_sock_ < 0) {
      throw std::runtime_error("Failed to create UDP send socket");
    }
    udp_targets_.clear();
    for (size_t i = 0; i < drone_ids_.size(); ++i) {
      sockaddr_in addr {};
      addr.sin_family = AF_INET;
      addr.sin_port = htons(static_cast<uint16_t>(udp_state_port_));
      if (inet_pton(AF_INET, udp_state_drone_ips_[i].c_str(), &addr.sin_addr) != 1) {
        throw std::runtime_error("Invalid UDP IP: " + udp_state_drone_ips_[i]);
      }
      udp_targets_.emplace(drone_ids_[i], addr);
    }
  }

  void send_udp_state(int id, int value)
  {
    if (udp_targets_.empty() || udp_send_sock_ < 0) {
      return;
    }
    char buf[64];
    int n = std::snprintf(buf, sizeof(buf), "state %d %d", id, value);
    for (const auto &entry : udp_targets_) {
      sendto(udp_send_sock_, buf, static_cast<size_t>(n), 0,
             reinterpret_cast<const sockaddr *>(&entry.second), sizeof(sockaddr_in));
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

  void parse_csv_ids(const std::string &csv, std::vector<int> &out_ids)
  {
    std::istringstream iss(csv);
    std::string token;
    while (std::getline(iss, token, ',')) {
      if (token.empty()) {
        continue;
      }
      try {
        int id = std::stoi(token);
        out_ids.push_back(id);
      } catch (const std::exception &) {
        RCLCPP_WARN(this->get_logger(), "Invalid drone id token '%s' in drone_ids_csv", token.c_str());
      }
    }
  }

  std::string mode_;
  int poll_period_ms_{10};
  int hwm_{10};
  int recv_max_per_poll_{50};
  int drone_id_{0};

  std::string state_topic_;
  std::string command_topic_;
  std::string state_push_endpoint_;
  std::string cmd_sub_endpoint_;
  std::string state_pull_bind_;
  std::string cmd_pub_bind_;
  bool fanout_states_{true};
  bool udp_state_fanout_{false};
  bool udp_state_listen_{false};
  int udp_state_port_{5570};
  std::string udp_state_bind_{"0.0.0.0"};
  std::vector<std::string> udp_state_drone_ips_;

  std::vector<int> drone_ids_;
  std::unordered_map<int, rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr> state_pubs_;
  std::unordered_map<int, rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr> remote_state_pubs_;
  std::vector<rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr> cmd_subs_;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr state_sub_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr cmd_pub_;

  rclcpp::TimerBase::SharedPtr timer_;

  void *ctx_{nullptr};
  void *state_push_sock_{nullptr};
  void *state_pull_sock_{nullptr};
  void *cmd_pub_sock_{nullptr};
  void *cmd_sub_sock_{nullptr};
  int udp_recv_sock_{-1};
  int udp_send_sock_{-1};
  std::unordered_map<int, sockaddr_in> udp_targets_;

  std::mutex zmq_mutex_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ZmqStateBridge>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
