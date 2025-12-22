#include "geom_multilift/data_loader.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

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
}  // namespace

DataLoader::DataLoader(const std::string &root_dir, int num_cables) : dt_(0.01) {
  load_payload(root_dir + "/payload.csv");
  // load as many cables as requested (or until a file is missing)
  cables_.resize(num_cables);
  for (int i = 0; i < num_cables; ++i) {
    load_cable(root_dir + "/cable_" + std::to_string(i) + ".csv", cables_[i]);
    if (cables_[i].size() != payload_.size()) {
      throw std::runtime_error("Cable csv length mismatch with payload csv for cable " + std::to_string(i));
    }
  }
  load_kfb(root_dir + "/kfb.csv");
}

void DataLoader::load_payload(const std::string &path) {
  std::ifstream f(path);
  if (!f.is_open()) {
    throw std::runtime_error("Cannot open payload csv: " + path);
  }
  std::string line;
  getline_safe(f, line);  // header
  while (getline_safe(f, line)) {
    auto cols = split_csv(line);
    if (cols.size() < 14) continue;
    PayloadSample s;
    size_t idx = 0;
    s.time = std::stod(cols[idx++]);
    s.pos = Eigen::Vector3d(std::stod(cols[idx++]), std::stod(cols[idx++]), std::stod(cols[idx++]));
    s.vel = Eigen::Vector3d(std::stod(cols[idx++]), std::stod(cols[idx++]), std::stod(cols[idx++]));
    s.q = Eigen::Quaterniond(std::stod(cols[idx++]), std::stod(cols[idx++]), std::stod(cols[idx++]), std::stod(cols[idx++]));
    s.omega = Eigen::Vector3d(std::stod(cols[idx++]), std::stod(cols[idx++]), std::stod(cols[idx++]));
    payload_.push_back(s);
  }
  if (payload_.size() >= 2) {
    dt_ = std::max(1e-6, payload_[1].time - payload_[0].time);
  }
}

void DataLoader::load_cable(const std::string &path, std::vector<CableSample> &out) {
  std::ifstream f(path);
  if (!f.is_open()) {
    throw std::runtime_error("Cannot open cable csv: " + path);
  }
  std::string line;
  getline_safe(f, line);  // header
  while (getline_safe(f, line)) {
    auto cols = split_csv(line);
    if (cols.size() < 11) continue;
    CableSample s;
    size_t idx = 0;
    s.time = std::stod(cols[idx++]);
    s.dir = Eigen::Vector3d(std::stod(cols[idx++]), std::stod(cols[idx++]), std::stod(cols[idx++]));
    s.omega = Eigen::Vector3d(std::stod(cols[idx++]), std::stod(cols[idx++]), std::stod(cols[idx++]));
    s.mu = std::stod(cols[idx++]);
    s.omega_dot = Eigen::Vector3d(std::stod(cols[idx++]), std::stod(cols[idx++]), std::stod(cols[idx++]));
    out.push_back(s);
  }
}

void DataLoader::load_kfb(const std::string &path) {
  std::ifstream f(path);
  if (!f.is_open()) {
    throw std::runtime_error("Cannot open kfb csv: " + path);
  }
  std::string line;
  getline_safe(f, line);  // header
  while (getline_safe(f, line)) {
    auto cols = split_csv(line);
    if (cols.size() < 1 + 6 * 13) continue;
    KfbSample s;
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
