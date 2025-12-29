#include "l2c_pkg/data_loader_new.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace {
bool getline_safe(std::ifstream &f, std::string &line) {
  if (!std::getline(f, line)) return false;
  if (!line.empty() && line.back() == '\r') {
    line.pop_back();
  }
  return true;
}

std::vector<std::string> split_csv(const std::string &line) {
  std::vector<std::string> out;
  std::stringstream ss(line);
  std::string item;
  while (std::getline(ss, item, ',')) {
    out.push_back(item);
  }
  return out;
}

std::unordered_map<std::string, size_t> header_index(const std::string &line) {
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
                  double default_val) {
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
                     const std::string &path) {
  for (const auto &key : cols) {
    if (idx.find(key) == idx.end()) {
      throw std::runtime_error("Missing required column '" + key + "' in " + path);
    }
  }
}
}  // namespace

DataLoaderNew::DataLoaderNew(const std::string &root_dir, int num_cables) : dt_(0.01) {
  load_payload(root_dir + "/payload.csv");
  cables_.resize(num_cables);
  for (int i = 0; i < num_cables; ++i) {
    load_cable(root_dir + "/cable_" + std::to_string(i) + ".csv", cables_[i]);
    if (cables_[i].size() != payload_.size()) {
      throw std::runtime_error("Cable csv length mismatch with payload csv for cable " + std::to_string(i));
    }
  }
  load_kfb(root_dir + "/kfb.csv");
}

void DataLoaderNew::load_payload(const std::string &path) {
  std::ifstream f(path);
  if (!f.is_open()) {
    std::cerr << "ERROR: Cannot open payload csv: " << path << std::endl;
    throw std::runtime_error("Cannot open payload csv: " + path);
  }
  std::string line;
  if (!getline_safe(f, line)) {
    throw std::runtime_error("Payload csv is empty: " + path);
  }
  auto idx = header_index(line);
  require_columns(idx, {"time", "x", "y", "z", "vx", "vy", "vz"}, path);

  while (getline_safe(f, line)) {
    if (line.empty()) continue;
    auto cols = split_csv(line);
    PayloadSampleNew s;
    s.time = read_value(cols, idx, "time", 0.0);
    s.pos = Eigen::Vector3d(
      read_value(cols, idx, "x", 0.0),
      read_value(cols, idx, "y", 0.0),
      read_value(cols, idx, "z", 0.0));
    s.vel = Eigen::Vector3d(
      read_value(cols, idx, "vx", 0.0),
      read_value(cols, idx, "vy", 0.0),
      read_value(cols, idx, "vz", 0.0));
    s.acc = Eigen::Vector3d(
      read_value(cols, idx, "ax", 0.0),
      read_value(cols, idx, "ay", 0.0),
      read_value(cols, idx, "az", 0.0));
    s.jerk = Eigen::Vector3d(
      read_value(cols, idx, "jx", 0.0),
      read_value(cols, idx, "jy", 0.0),
      read_value(cols, idx, "jz", 0.0));
    s.snap = Eigen::Vector3d(
      read_value(cols, idx, "sx", 0.0),
      read_value(cols, idx, "sy", 0.0),
      read_value(cols, idx, "sz", 0.0));
    double qw = read_value(cols, idx, "qw", 1.0);
    double qx = read_value(cols, idx, "qx", 0.0);
    double qy = read_value(cols, idx, "qy", 0.0);
    double qz = read_value(cols, idx, "qz", 0.0);
    s.q = Eigen::Quaterniond(qw, qx, qy, qz);
    s.omega = Eigen::Vector3d(
      read_value(cols, idx, "wx", 0.0),
      read_value(cols, idx, "wy", 0.0),
      read_value(cols, idx, "wz", 0.0));
    payload_.push_back(s);
  }
  if (payload_.empty()) {
    throw std::runtime_error("payload.csv is empty or missing data");
  }

  if (payload_.size() >= 2) {
    dt_ = std::max(1e-6, payload_[1].time - payload_[0].time);
  }
}

void DataLoaderNew::load_cable(const std::string &path, std::vector<CableSampleNew> &out) {
  std::ifstream f(path);
  if (!f.is_open()) {
    std::cerr << "ERROR: Cannot open cable csv: " << path << std::endl;
    throw std::runtime_error("Cannot open cable csv: " + path);
  }
  std::string line;
  if (!getline_safe(f, line)) {
    throw std::runtime_error("Cable csv is empty: " + path);
  }
  auto idx = header_index(line);
  require_columns(idx, {"time", "dir_x", "dir_y", "dir_z", "mu"}, path);

  while (getline_safe(f, line)) {
    if (line.empty()) continue;
    auto cols = split_csv(line);
    CableSampleNew s;
    s.time = read_value(cols, idx, "time", 0.0);
    s.dir = Eigen::Vector3d(
      read_value(cols, idx, "dir_x", 0.0),
      read_value(cols, idx, "dir_y", 0.0),
      read_value(cols, idx, "dir_z", 0.0));
    s.omega = Eigen::Vector3d(
      read_value(cols, idx, "omega_x", 0.0),
      read_value(cols, idx, "omega_y", 0.0),
      read_value(cols, idx, "omega_z", 0.0));
    s.omega_dot = Eigen::Vector3d(
      read_value(cols, idx, "omega_dot_x", 0.0),
      read_value(cols, idx, "omega_dot_y", 0.0),
      read_value(cols, idx, "omega_dot_z", 0.0));
    s.mu = read_value(cols, idx, "mu", 0.0);
    out.push_back(s);
  }
}

void DataLoaderNew::load_kfb(const std::string &path) {
  std::ifstream f(path);
  if (!f.is_open()) {
    std::cerr << "ERROR: Cannot open kfb csv: " << path << std::endl;
    throw std::runtime_error("Cannot open kfb csv: " + path);
  }
  std::string line;
  if (!getline_safe(f, line)) {
    throw std::runtime_error("kfb.csv is empty: " + path);
  }
  while (getline_safe(f, line)) {
    if (line.empty()) continue;
    auto cols = split_csv(line);
    if (cols.size() < 1 + 6 * 13) continue;
    KfbSampleNew s;
    size_t idx = 0;
    s.time = std::stod(cols[idx++]);
    for (int r = 0; r < 6; ++r) {
      for (int c = 0; c < 13; ++c) {
        s.K(r, c) = std::stod(cols[idx++]);
      }
    }
    kfb_.push_back(s);
  }
  if (kfb_.empty()) {
    throw std::runtime_error("kfb.csv is empty or missing data");
  }
}
