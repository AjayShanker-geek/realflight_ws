#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <string>
#include <vector>

struct PayloadSample {
  double time;
  Eigen::Vector3d pos;
  Eigen::Vector3d vel;
  Eigen::Quaterniond q;
  Eigen::Vector3d omega;
};

struct CableSample {
  double time;
  Eigen::Vector3d dir;
  Eigen::Vector3d omega;
  Eigen::Vector3d omega_dot;
  double mu;
};

struct KfbSample {
  double time;
  Eigen::Matrix<double, 6, 13> K;
};

class DataLoader {
public:
  explicit DataLoader(const std::string &root_dir, int num_cables = 3);

  const std::vector<PayloadSample> &payload() const { return payload_; }
  const std::vector<std::vector<CableSample>> &cables() const { return cables_; }
  const std::vector<KfbSample> &kfb() const { return kfb_; }
  double dt() const { return dt_; }

private:
  double dt_;
  std::vector<PayloadSample> payload_;
  std::vector<std::vector<CableSample>> cables_;
  std::vector<KfbSample> kfb_;

  void load_payload(const std::string &path);
  void load_cable(const std::string &path, std::vector<CableSample> &out);
  void load_kfb(const std::string &path);
};
