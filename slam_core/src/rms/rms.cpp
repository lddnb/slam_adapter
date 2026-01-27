#include "slam_core/rms/rms.h"

#include <algorithm>
#include <execution>
#include <numeric>

#include <tsl/robin_map.h>  // Include here since it's a header-only library

namespace ms_slam::slam_core
{

/*//{ RMS() constructor */
RMS::RMS(size_t K, float lambda, float voxel_input) {
  _K = K;
  _lambda = lambda;
  _voxel_input = voxel_input;
}
/*//}*/

/*//{ sample() */
t_points RMS::sample(t_points& pts) {
  if (pts->empty()) {
    return pts;
  }

  // Setup a vector of point indices 0->N
  t_indices indices = t_indices(pts->size());
  std::iota(std::begin(indices), std::end(indices), 0);

  // L12-L14: compute GFH
  const auto &gfh = computeGFH(pts, indices);

  // L15-L28: GFH entropy minimization
  indices = sampleByGFH(pts, gfh, size_t(_K), _lambda);

  // L29: sample the input message by indices
  return pts->extract_ptr(indices);
}
/*//}*/

/*//{ sampleByGFH() */
t_indices RMS::sampleByGFH(const t_points points, const std::vector<t_gfh> &gfh, const size_t K, const float lambda) {

  // Construct histogram of the 1D values
  auto hist = Histogram1D(K, 0.0f, 1.0f, gfh);

  size_t N        = 0;
  float  max_rate = 0.0;

  while (N < points->size()) {

    // Sample one point
    float entropy;
    hist.selectByUniformnessMaximization(entropy);
    N++;

    // Eq. (28): compute mean entropy
    const float rate = entropy / N;

    // L20-L23: Add first K samples to initialize
    if (N <= K) {

      // Eq. (31): Compute maximum entropy rate
      max_rate = std::fmax(rate, max_rate);

    } else {

      // L26: Break if rate of entropy change has slowed down under a threshold
      if ((rate / max_rate) < lambda) {
        break;
      }
    }
  }

  // Retrieve and return the selected indices and return them
  return hist.getSelectedIndices();
}
/*//}*/

/*//{ computeGFH() */
std::vector<t_gfh> RMS::computeGFH(const t_points points_in, const t_indices &indices_in) {

  // Setup output vector
  std::vector<t_gfh> gfh(indices_in.size());

  const auto xyz = points_in->positions();
  kdtree_.SetPoints(xyz);
  kdtree_.Build();

  const float nn_radius    = 2.0 * _voxel_input;  // Nearest-neighbor search radius
  /* const float nn_radius    = 2.236f * _voxel_input;  // Nearest-neighbor search radius (linear scale: sqrt(5)) */

  // 构造 [0, N) 的下标序列，用于并行循环中稳定写入 gfh[i]
  std::vector<size_t> order(indices_in.size());
  std::iota(order.begin(), order.end(), 0);

  // 使用 PSTL 并行计算每个点的 GFH 范数（邻域查询与累加逻辑保持原样）
  std::for_each(std::execution::par_unseq, order.begin(), order.end(), [&](const size_t i) {
    const auto pt_idx = indices_in[i];

    // Get point at index
    const auto &pt = points_in->position(pt_idx);

    // Perform radius search
    std::vector<size_t>   radius_indices;
    std::vector<float> radius_sq_dist;
    /* kdtree.nearestKSearch(pt, 5, radius_indices, radius_sq_dist); // optional search */
    const auto rnn = kdtree_.RadiusSearchSquared(pt, nn_radius, radius_indices, radius_sq_dist);

    // Compute GFH vector
    size_t          N      = 0;
    Eigen::Vector3f pt_gfh = Eigen::Vector3f::Zero();
    for (size_t j = 0; j < radius_indices.size(); j++) {

      const auto ind = radius_indices[j];

      if (ind == pt_idx) {
        continue;
      }

      const auto &pt_neigh = points_in->position(ind);
      pt_gfh += Eigen::Vector3f(pt_neigh.x() - pt.x(), pt_neigh.y() - pt.y(), pt_neigh.z() - pt.z());
      N++;
    }

    if (N > 1) {
      pt_gfh /= float(N);
    }

    // Compute the GFH norm
    gfh[i] = {pt_idx, pt_gfh.norm()};
  });

  // <0, 1> normalizing factor for GFH norm
  float max_gfh_norm = 0.0f;
  for (const auto& item : gfh) {
    max_gfh_norm = std::max(max_gfh_norm, item.second);
  }

  // Normalize to <0, 1>
  if (max_gfh_norm > 0.0f) {
    std::for_each(gfh.begin(), gfh.end(), [max_gfh_norm](auto& item) { item.second /= max_gfh_norm; });
  }

  return gfh;
}
/*//}*/

}  // namespace ms_slam::slam_core
