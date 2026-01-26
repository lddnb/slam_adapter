#pragma once
#include "slam_core/rms/histogram.h"
#include "slam_core/odometry/odom_common.hpp"
#include "slam_core/utils/eigen_nanoflann_kdtree.hpp"

namespace ms_slam::slam_core
{

/* //{ class RMS */

using t_points  = PointCloudType::Ptr;
using t_voxel   = Eigen::Vector3i;

class RMS {
  // | ----------------------- Public API ----------------------- |
  struct VoxelHash
  {
    size_t operator()(const Eigen::Vector3i& voxel) const {
      const std::uint32_t* vec = reinterpret_cast<const uint32_t*>(voxel.data());
      return ((1 << 20) - 1) & (vec[0] * 73856093 ^ vec[1] * 19349663 ^ vec[2] * 83492791);
    }
  };
  
public:
  RMS(size_t K, float lambda, float voxel_input);
  t_points sample(t_points& msg_inout);

  // | ---------------- RMS variables and methods --------------- |
private:
  size_t _K            = 10;
  float  _lambda       = 1.0f;
  float  _voxel_input  = -1.0f;
  float  _voxel_output = -1.0f;

  EigenNanoFlannKdTree3f kdtree_;

  std::vector<t_gfh> computeGFH(const t_points points, const t_indices& indices);
  t_indices          sampleByGFH(const t_points points, const std::vector<t_gfh>& gfh, const size_t K, const float lambda);
};

//}

}  // namespace ms_slam::slam_core
