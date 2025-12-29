#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <string>
#include <vector>

struct PayloadSampleNew {
  double time;
  Eigen::Vector3d pos;
  Eigen::Vector3d vel;
  Eigen::Vector3d acc;
  Eigen::Vector3d jerk;
  Eigen::Vector3d snap;
  Eigen::Quaterniond q;
  Eigen::Vector3d omega;
};

struct CableSampleNew {
  double time;
  Eigen::Vector3d dir;
  Eigen::Vector3d omega;
  Eigen::Vector3d omega_dot;
  double mu;
};

struct KfbSampleNew {
  double time;
  Eigen::Matrix<double, 6, 13> K;
};

class DataLoaderNew {
public:
  explicit DataLoaderNew(const std::string &root_dir, int num_cables = 3);

  const std::vector<PayloadSampleNew> &payload() const { return payload_; }
  const std::vector<std::vector<CableSampleNew>> &cables() const { return cables_; }
  const std::vector<KfbSampleNew> &kfb() const { return kfb_; }
  double dt() const { return dt_; }

private:
  double dt_;
  std::vector<PayloadSampleNew> payload_;
  std::vector<std::vector<CableSampleNew>> cables_;
  std::vector<KfbSampleNew> kfb_;

  void load_payload(const std::string &path);
  void load_cable(const std::string &path, std::vector<CableSampleNew> &out);
  void load_kfb(const std::string &path);
};
