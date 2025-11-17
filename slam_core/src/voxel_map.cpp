#include "slam_core/voxel_map.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <execution>
#include <mutex>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Eigenvalues>
#include <spdlog/spdlog.h>

namespace ms_slam::slam_core
{

namespace
{
/**
 * @brief 全局平面ID计数器
 */
std::atomic<int> g_plane_id{0};

/**
 * @brief 圆周率常量
 */
constexpr double kPi = 3.14159265358979323846;
}  // namespace

/**
 * @brief 协方差对比函数
 * @param x 点A
 * @param y 点B
 * @return 协方差是否更小
 */
bool VarContrast(const PointWithCov& x, const PointWithCov& y)
{
    return (x.cov.diagonal().norm() < y.cov.diagonal().norm());
}

/**
 * @brief 构造函数
 * @param config 地图配置
 */
VoxelMap::VoxelMap(const VoxelMapConfig& config) : config_(config)
{
    spdlog::info("VoxelMap created with voxel size {:.3f}", config_.voxel_size);
}

/**
 * @brief 初始构建地图
 * @param input_points 输入点云
 */
void VoxelMap::Build(const std::vector<PointWithCov>& input_points)
{
    for (const auto& point : input_points) {
        const VoxelLoc loc = ComputeVoxelLoc(point.world_point);
        auto iter = voxel_map_.find(loc);
        if (iter != voxel_map_.end()) {
            iter->second->AddInitialPoint(point);
        } else {
            auto tree = CreateOctoTree(loc);
            tree->AddInitialPoint(point);
            voxel_map_[loc] = std::move(tree);
        }
    }

    for (auto& [_, tree] : voxel_map_) {
        tree->Initialize();
    }
    spdlog::info("VoxelMap build finished with {} voxels", voxel_map_.size());
}

/**
 * @brief 单线程更新
 * @param input_points 新增点
 */
void VoxelMap::Update(const std::vector<PointWithCov>& input_points)
{
    for (const auto& point : input_points) {
        const VoxelLoc loc = ComputeVoxelLoc(point.world_point);
        auto iter = voxel_map_.find(loc);
        if (iter != voxel_map_.end()) {
            iter->second->Update(point);
        } else {
            auto tree = CreateOctoTree(loc);
            tree->Update(point);
            voxel_map_[loc] = std::move(tree);
        }
    }
}

/**
 * @brief 分桶并更新
 * @param input_points 新增点
 */
void VoxelMap::UpdateParallel(const std::vector<PointWithCov>& input_points)
{
    std::unordered_map<VoxelLoc, std::vector<PointWithCov>, VoxelLocHash> buckets;
    for (const auto& point : input_points) {
        const VoxelLoc loc = ComputeVoxelLoc(point.world_point);
        buckets[loc].push_back(point);
    }

    for (auto& [loc, points] : buckets) {
        auto iter = voxel_map_.find(loc);
        if (iter == voxel_map_.end()) {
            auto tree = CreateOctoTree(loc);
            for (const auto& pt : points) {
                tree->Update(pt);
            }
            voxel_map_[loc] = std::move(tree);
        } else {
            for (const auto& pt : points) {
                iter->second->Update(pt);
            }
        }
    }
}

/**
 * @brief 并行构建残差
 * @param pv_list 输入点
 * @param ptpl_list 匹配结果
 * @param non_match 未匹配点
 */
void VoxelMap::BuildResidualList(
    const std::vector<PointWithCov>& pv_list,
    std::vector<PointToPlaneResidual>& ptpl_list,
    std::vector<Eigen::Vector3d>& non_match) const
{
    ptpl_list.clear();
    std::vector<PointToPlaneResidual> all_ptpl_list(pv_list.size());
    std::vector<bool> useful_ptpl(pv_list.size(), false);

    std::mutex result_mutex;
    std::vector<std::size_t> indices(pv_list.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [&](std::size_t i) {
        const PointWithCov& pv = pv_list[i];
        const VoxelLoc position = ComputeVoxelLoc(pv.world_point);
        auto iter = voxel_map_.find(position);
        if (iter == voxel_map_.end()) {
            return;
        }

        const OctoTree* current_octo = iter->second.get();
        PointToPlaneResidual single_ptpl;
        bool is_success = false;
        double prob = 0.0;
        BuildSingleResidual(pv, current_octo, 0, is_success, prob, single_ptpl);

        if (!is_success) {
            VoxelLoc near_position = position;
            // 判断点是否超出当前voxel中心边界
            if (pv.world_point.x() > (current_octo->Center().x() + current_octo->QuarterLength())) {
                near_position.x += 1;
            } else if (pv.world_point.x() < (current_octo->Center().x() - current_octo->QuarterLength())) {
                near_position.x -= 1;
            }
            if (pv.world_point.y() > (current_octo->Center().y() + current_octo->QuarterLength())) {
                near_position.y += 1;
            } else if (pv.world_point.y() < (current_octo->Center().y() - current_octo->QuarterLength())) {
                near_position.y -= 1;
            }
            if (pv.world_point.z() > (current_octo->Center().z() + current_octo->QuarterLength())) {
                near_position.z += 1;
            } else if (pv.world_point.z() < (current_octo->Center().z() - current_octo->QuarterLength())) {
                near_position.z -= 1;
            }
            auto iter_near = voxel_map_.find(near_position);
            if (iter_near != voxel_map_.end()) {
                BuildSingleResidual(pv, iter_near->second.get(), 0, is_success, prob, single_ptpl);
            }
        }

        if (is_success) {
            std::scoped_lock lock(result_mutex);
            useful_ptpl[i] = true;
            all_ptpl_list[i] = single_ptpl;
        }
    });

    for (std::size_t i = 0; i < useful_ptpl.size(); ++i) {
        if (useful_ptpl[i]) {
            ptpl_list.push_back(all_ptpl_list[i]);
        }
    }
}

/**
 * @brief 常规匹配
 * @param pv_list 输入点
 * @param ptpl_list 匹配结果
 * @param non_match 未匹配点
 */
void VoxelMap::BuildResidualListNormal(
    const std::vector<PointWithCov>& pv_list,
    std::vector<PointToPlaneResidual>& ptpl_list,
    std::vector<Eigen::Vector3d>& non_match) const
{
    ptpl_list.clear();
    for (const PointWithCov& pv : pv_list) {
        const VoxelLoc position = ComputeVoxelLoc(pv.world_point);
        auto iter = voxel_map_.find(position);
        if (iter == voxel_map_.end()) {
            continue;
        }
        const OctoTree* current_octo = iter->second.get();
        PointToPlaneResidual single_ptpl;
        bool is_success = false;
        double prob = 0.0;
        BuildSingleResidual(pv, current_octo, 0, is_success, prob, single_ptpl);
        if (!is_success) {
            non_match.push_back(pv.world_point.cast<double>());
        } else {
            ptpl_list.push_back(single_ptpl);
        }
    }
}

/**
 * @brief 获取更新平面
 * @param pub_max_voxel_layer 发布层数
 * @param plane_list 输出平面
 */
void VoxelMap::GetUpdatedPlanes(int pub_max_voxel_layer, std::vector<Plane>& plane_list) const
{
    plane_list.clear();
    for (const auto& [_, tree] : voxel_map_) {
        CollectUpdatedPlanes(tree.get(), pub_max_voxel_layer, plane_list);
    }
}

/**
 * @brief 计算体素坐标
 * @param point 输入点
 * @return 体素索引
 */
VoxelLoc VoxelMap::ComputeVoxelLoc(const Eigen::Vector3f& point) const
{
    Eigen::Vector3f ratio = point / config_.voxel_size;
    for (int i = 0; i < 3; ++i) {
        if (ratio[i] < 0.0F) {
            ratio[i] -= 1.0F;
        }
    }
    VoxelLoc loc;
    loc.x = static_cast<int64_t>(ratio.x());
    loc.y = static_cast<int64_t>(ratio.y());
    loc.z = static_cast<int64_t>(ratio.z());
    return loc;
}

/**
 * @brief 构建单个残差
 * @param pv 输入点
 * @param current_octo 当前节点
 * @param current_layer 当前层级
 * @param is_success 是否成功
 * @param prob 匹配概率
 * @param single_ptpl 输出残差
 */
void VoxelMap::BuildSingleResidual(
    const PointWithCov& pv,
    const OctoTree* current_octo,
    int current_layer,
    bool& is_success,
    double& prob,
    PointToPlaneResidual& single_ptpl) const
{
    if (current_octo->HasPlane()) {
        const Plane& plane = current_octo->PlaneData();
        const Eigen::Vector3d center = plane.center.cast<double>();
        const Eigen::Vector3d normal = plane.normal.cast<double>();
        const Eigen::Vector3d p = pv.world_point.cast<double>();
        const double dis_to_plane = std::abs(normal.dot(p) + plane.d);
        const double dis_to_center = (center - p).squaredNorm();
        const double range_dis = std::sqrt(std::max(dis_to_center - dis_to_plane * dis_to_plane, 0.0));

        if (range_dis <= 3.0 * static_cast<double>(plane.radius)) {
            Eigen::Matrix<double, 1, 6> J_nq;
            J_nq.block<1, 3>(0, 0) = p - center;
            J_nq.block<1, 3>(0, 3) = -normal;
            double sigma_l = J_nq * plane.plane_cov * J_nq.transpose();
            sigma_l += normal.transpose() * pv.cov * normal;
            const double sigma_safe = std::max(sigma_l, 1e-8);
            if (dis_to_plane < config_.sigma_num * std::sqrt(sigma_safe)) {
                is_success = true;
                // 使用概率密度评估匹配质量
                const double this_prob = 1.0 / std::sqrt(sigma_safe) * std::exp(-0.5 * dis_to_plane * dis_to_plane / sigma_safe);
                if (this_prob > prob) {
                    prob = this_prob;
                    single_ptpl.body_point = pv.body_point;
                    single_ptpl.world_point = pv.world_point;
                    single_ptpl.plane_cov = plane.plane_cov;
                    single_ptpl.normal = plane.normal;
                    single_ptpl.center = plane.center;
                    single_ptpl.d = plane.d;
                    single_ptpl.layer = current_layer;
                    single_ptpl.cov_lidar = pv.cov_lidar;
                }
                return;
            }
        }
        return;
    }

    if (current_layer < current_octo->MaxLayer()) {
        for (std::size_t i = 0; i < 8; ++i) {
            const OctoTree* child = current_octo->GetChild(i);
            if (child != nullptr) {
                BuildSingleResidual(pv, child, current_layer + 1, is_success, prob, single_ptpl);
            }
        }
    }
}

/**
 * @brief 收集更新平面
 * @param current_octo 当前节点
 * @param pub_max_voxel_layer 最大层级
 * @param plane_list 平面列表
 */
void VoxelMap::CollectUpdatedPlanes(const OctoTree* current_octo, int pub_max_voxel_layer, std::vector<Plane>& plane_list) const
{
    if (current_octo->Layer() > pub_max_voxel_layer) {
        return;
    }
    if (current_octo->PlaneData().is_update) {
        plane_list.push_back(current_octo->PlaneData());
    }
    if (current_octo->Layer() < current_octo->MaxLayer() && !current_octo->HasPlane()) {
        for (std::size_t i = 0; i < 8; ++i) {
            const OctoTree* child = current_octo->GetChild(i);
            if (child != nullptr) {
                CollectUpdatedPlanes(child, pub_max_voxel_layer, plane_list);
            }
        }
    }
}

/**
 * @brief 创建八叉树节点
 * @param position 体素索引
 * @return 新节点
 */
std::unique_ptr<OctoTree> VoxelMap::CreateOctoTree(const VoxelLoc& position)
{
    auto octo_tree = std::make_unique<OctoTree>(
        config_.max_layer,
        0,
        config_.layer_point_size,
        config_.max_points_size,
        config_.max_cov_points_size,
        config_.planer_threshold);
    const Eigen::Vector3f center =
        Eigen::Vector3f(static_cast<float>(position.x), static_cast<float>(position.y), static_cast<float>(position.z)) * config_.voxel_size +
        Eigen::Vector3f::Constant(config_.voxel_size * 0.5F);
    octo_tree->ConfigureCenter(center, config_.voxel_size * 0.25F);
    return octo_tree;
}

/**
 * @brief 构造函数
 * @param max_layer 最大层数
 * @param layer 当前层
 * @param layer_point_size 每层阈值
 * @param max_point_size 最大点数
 * @param max_cov_points_size 最大协方差点
 * @param planer_threshold 平面阈值
 */
OctoTree::OctoTree(int max_layer, int layer, const std::vector<int>& layer_point_size, int max_point_size, int max_cov_points_size, float planer_threshold)
: max_layer_(max_layer),
  layer_(layer),
  layer_point_size_(layer_point_size),
  planer_threshold_(planer_threshold),
  max_plane_update_threshold_(layer_point_size_[layer_]),
  update_size_threshold_(5),
  all_points_num_(0),
  new_points_num_(0),
  max_points_size_(max_point_size),
  max_cov_points_size_(max_cov_points_size),
  init_octo_(false),
  update_cov_enable_(true),
  update_enable_(true)
{
    plane_.is_plane = false;
    plane_.is_init = false;
    for (auto& leaf : leaves_) {
        leaf.reset();
    }
}

/**
 * @brief 设置体素中心
 * @param center 体素中心
 * @param quarter_length 四分之一直径
 */
void OctoTree::ConfigureCenter(const Eigen::Vector3f& center, float quarter_length)
{
    voxel_center_ = center;
    quater_length_ = quarter_length;
}

/**
 * @brief 添加构建点
 * @param point 点
 */
void OctoTree::AddInitialPoint(const PointWithCov& point)
{
    temp_points_.push_back(point);
    new_points_num_++;
    if (static_cast<int>(temp_points_.size()) > max_plane_update_threshold_) {
        Initialize();
    }
}

/**
 * @brief 初始化节点
 */
void OctoTree::Initialize()
{
    if (static_cast<int>(temp_points_.size()) <= max_plane_update_threshold_) {
        return;
    }
    InitPlane(temp_points_);
    if (plane_.is_plane) {
        octo_state_ = 0;
        if (static_cast<int>(temp_points_.size()) > max_cov_points_size_) {
            update_cov_enable_ = false;
        }
        if (static_cast<int>(temp_points_.size()) > max_points_size_) {
            update_enable_ = false;
        }
    } else {
        octo_state_ = 1;
        CutTree();
    }
    init_octo_ = true;
    new_points_num_ = 0;
}

/**
 * @brief 更新节点
 * @param point 新点
 */
void OctoTree::Update(const PointWithCov& point)
{
    if (!init_octo_) {
        AddInitialPoint(point);
        return;
    }

    if (plane_.is_plane) {
        if (!update_enable_) {
            return;
        }
        new_points_num_++;
        all_points_num_++;
        if (update_cov_enable_) {
            temp_points_.push_back(point);
        } else {
            new_points_.push_back(point);
        }
        if (new_points_num_ > update_size_threshold_) {
            if (update_cov_enable_) {
                InitPlane(temp_points_);
            } else {
                UpdatePlane(new_points_);
                new_points_.clear();
            }
            new_points_num_ = 0;
        }
        if (all_points_num_ >= max_cov_points_size_) {
            update_cov_enable_ = false;
            std::vector<PointWithCov>().swap(temp_points_);
        }
        if (all_points_num_ >= max_points_size_) {
            update_enable_ = false;
            plane_.update_enable = false;
            std::vector<PointWithCov>().swap(new_points_);
        }
        return;
    }

    if (layer_ < max_layer_) {
        temp_points_.clear();
        new_points_.clear();
        int xyz[3] = {0, 0, 0};
        xyz[0] = static_cast<int>(point.world_point.x() > voxel_center_.x());
        xyz[1] = static_cast<int>(point.world_point.y() > voxel_center_.y());
        xyz[2] = static_cast<int>(point.world_point.z() > voxel_center_.z());
        const int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
        if (leaves_[leafnum]) {
            leaves_[leafnum]->Update(point);
        } else {
            auto child = std::make_unique<OctoTree>(max_layer_, layer_ + 1, layer_point_size_, max_points_size_, max_cov_points_size_, planer_threshold_);
            child->layer_point_size_ = layer_point_size_;
            Eigen::Vector3f child_center = voxel_center_;
            child_center[0] += (2 * xyz[0] - 1) * quater_length_;
            child_center[1] += (2 * xyz[1] - 1) * quater_length_;
            child_center[2] += (2 * xyz[2] - 1) * quater_length_;
            child->ConfigureCenter(child_center, quater_length_ / 2.0F);
            child->Update(point);
            leaves_[leafnum] = std::move(child);
        }
    } else {
        if (update_enable_) {
            new_points_num_++;
            all_points_num_++;
            if (update_cov_enable_) {
                temp_points_.push_back(point);
            } else {
                new_points_.push_back(point);
            }
            if (new_points_num_ > update_size_threshold_) {
                if (update_cov_enable_) {
                    InitPlane(temp_points_);
                } else {
                    UpdatePlane(new_points_);
                    new_points_.clear();
                }
                new_points_num_ = 0;
            }
            if (all_points_num_ >= max_cov_points_size_) {
                update_cov_enable_ = false;
                std::vector<PointWithCov>().swap(temp_points_);
            }
            if (all_points_num_ >= max_points_size_) {
                update_enable_ = false;
                plane_.update_enable = false;
                std::vector<PointWithCov>().swap(new_points_);
            }
        }
    }
}

/**
 * @brief 初始化平面
 * @param points 点集合
 */
void OctoTree::InitPlane(const std::vector<PointWithCov>& points)
{
    plane_.plane_cov = Eigen::Matrix<double, 6, 6>::Zero();
    plane_.covariance = Eigen::Matrix3d::Zero();
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    plane_.points_size = static_cast<int>(points.size());
    plane_.radius = 0.0F;

    for (const auto& pv : points) {
        const Eigen::Vector3d pt = pv.world_point.cast<double>();
        plane_.covariance += pt * pt.transpose();
        center += pt;
    }
    center /= static_cast<double>(plane_.points_size);
    plane_.covariance = plane_.covariance / static_cast<double>(plane_.points_size) - center * center.transpose();

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(plane_.covariance);
    const Eigen::Vector3d evals = es.eigenvalues();
    const Eigen::Matrix3d evecs = es.eigenvectors();

    Eigen::Index evals_min, evals_max;
    evals.minCoeff(&evals_min);
    evals.maxCoeff(&evals_max);
    const int evals_mid = 3 - static_cast<int>(evals_min) - static_cast<int>(evals_max);

    const Eigen::Vector3d evec_min = evecs.col(evals_min);
    const Eigen::Vector3d evec_mid = evecs.col(evals_mid);
    const Eigen::Vector3d evec_max = evecs.col(evals_max);

    Eigen::Matrix3d J_Q = Eigen::Matrix3d::Identity() / static_cast<double>(plane_.points_size);
    if (evals(evals_min) < static_cast<double>(planer_threshold_)) {
        for (const auto& pv : points) {
            Eigen::Matrix<double, 6, 3> J = Eigen::Matrix<double, 6, 3>::Zero();
            Eigen::Matrix3d F = Eigen::Matrix3d::Zero();
            const Eigen::Vector3d pt = pv.world_point.cast<double>();
            for (int m = 0; m < 3; ++m) {
                if (m != evals_min) {
                    const Eigen::Matrix<double, 1, 3> F_m = (pt - center).transpose() /
                                                            (static_cast<double>(plane_.points_size) * (evals(evals_min) - evals(m))) *
                                                            (evecs.col(m) * evec_min.transpose() + evec_min * evecs.col(m).transpose());
                    F.row(m) = F_m;
                }
            }
            J.block<3, 3>(0, 0) = evecs * F;
            J.block<3, 3>(3, 0) = J_Q;
            plane_.plane_cov += J * pv.cov * J.transpose();
        }
        plane_.is_plane = true;
    } else {
        plane_.is_plane = false;
    }

    plane_.center = center.cast<float>();
    plane_.normal = evec_min.cast<float>();
    plane_.y_normal = evec_mid.cast<float>();
    plane_.x_normal = evec_max.cast<float>();
    plane_.min_eigen_value = static_cast<float>(evals(evals_min));
    plane_.mid_eigen_value = static_cast<float>(evals(evals_mid));
    plane_.max_eigen_value = static_cast<float>(evals(evals_max));
    plane_.radius = std::sqrt(static_cast<float>(std::max(evals(evals_max), 0.0)));
    plane_.d = -(plane_.normal.cast<double>().dot(center));
    plane_.is_update = true;
    if (!plane_.is_init) {
        plane_.id = g_plane_id.fetch_add(1);
        plane_.is_init = true;
    }
    plane_.last_update_points_size = plane_.points_size;
}

/**
 * @brief 更新平面
 * @param points 新点
 */
void OctoTree::UpdatePlane(const std::vector<PointWithCov>& points)
{
    Eigen::Matrix3d sum_ppt = (plane_.covariance + plane_.center.cast<double>() * plane_.center.cast<double>().transpose()) * plane_.points_size;
    Eigen::Vector3d sum_p = plane_.center.cast<double>() * static_cast<double>(plane_.points_size);

    for (const auto& pv : points) {
        const Eigen::Vector3d pt = pv.world_point.cast<double>();
        sum_ppt += pt * pt.transpose();
        sum_p += pt;
    }
    plane_.points_size += static_cast<int>(points.size());
    const Eigen::Vector3d center = sum_p / static_cast<double>(plane_.points_size);
    plane_.covariance = sum_ppt / static_cast<double>(plane_.points_size) - center * center.transpose();

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(plane_.covariance);
    const Eigen::Vector3d evals = es.eigenvalues();
    const Eigen::Matrix3d evecs = es.eigenvectors();

    Eigen::Index evals_min, evals_max;
    evals.minCoeff(&evals_min);
    evals.maxCoeff(&evals_max);
    const int evals_mid = 3 - static_cast<int>(evals_min) - static_cast<int>(evals_max);

    plane_.center = center.cast<float>();
    plane_.normal = evecs.col(evals_min).cast<float>();
    plane_.y_normal = evecs.col(evals_mid).cast<float>();
    plane_.x_normal = evecs.col(evals_max).cast<float>();
    plane_.min_eigen_value = static_cast<float>(evals(evals_min));
    plane_.mid_eigen_value = static_cast<float>(evals(evals_mid));
    plane_.max_eigen_value = static_cast<float>(evals(evals_max));
    plane_.radius = std::sqrt(static_cast<float>(std::max(evals(evals_max), 0.0)));
    plane_.d = -(plane_.normal.cast<double>().dot(center));
    plane_.is_plane = evals(evals_min) < static_cast<double>(planer_threshold_);
    plane_.is_update = true;
}

/**
 * @brief 递归切分八叉树
 */
void OctoTree::CutTree()
{
    if (layer_ >= max_layer_) {
        octo_state_ = 0;
        return;
    }
    for (const auto& pv : temp_points_) {
        int xyz[3] = {0, 0, 0};
        xyz[0] = static_cast<int>(pv.world_point.x() > voxel_center_.x());
        xyz[1] = static_cast<int>(pv.world_point.y() > voxel_center_.y());
        xyz[2] = static_cast<int>(pv.world_point.z() > voxel_center_.z());
        const int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
        if (!leaves_[leafnum]) {
            auto child = std::make_unique<OctoTree>(max_layer_, layer_ + 1, layer_point_size_, max_points_size_, max_cov_points_size_, planer_threshold_);
            Eigen::Vector3f child_center = voxel_center_;
            child_center[0] += (2 * xyz[0] - 1) * quater_length_;
            child_center[1] += (2 * xyz[1] - 1) * quater_length_;
            child_center[2] += (2 * xyz[2] - 1) * quater_length_;
            child->ConfigureCenter(child_center, quater_length_ / 2.0F);
            leaves_[leafnum] = std::move(child);
        }
        leaves_[leafnum]->AddInitialPoint(pv);
    }

    for (auto& leaf : leaves_) {
        if (leaf) {
            if (static_cast<int>(leaf->temp_points_.size()) > leaf->max_plane_update_threshold_) {
                leaf->Initialize();
                if (!leaf->plane_.is_plane) {
                    leaf->CutTree();
                }
                leaf->init_octo_ = true;
                leaf->new_points_num_ = 0;
            }
        }
    }
    std::vector<PointWithCov>().swap(temp_points_);
}

/**
 * @brief jet映射
 * @param v 输入值
 * @param vmin 最小
 * @param vmax 最大
 * @param r 红
 * @param g 绿
 * @param b 蓝
 */
void MapJet(double v, double vmin, double vmax, uint8_t& r, uint8_t& g, uint8_t& b)
{
    v = std::clamp(v, vmin, vmax);
    double dr = 1.0;
    double dg = 1.0;
    double db = 1.0;

    if (v < 0.1242) {
        db = 0.504 + ((1.0 - 0.504) / 0.1242) * v;
        dg = dr = 0.0;
    } else if (v < 0.3747) {
        db = 1.0;
        dr = 0.0;
        dg = (v - 0.1242) * (1.0 / (0.3747 - 0.1242));
    } else if (v < 0.6253) {
        db = (0.6253 - v) * (1.0 / (0.6253 - 0.3747));
        dg = 1.0;
        dr = (v - 0.3747) * (1.0 / (0.6253 - 0.3747));
    } else if (v < 0.8758) {
        db = 0.0;
        dr = 1.0;
        dg = (0.8758 - v) * (1.0 / (0.8758 - 0.6253));
    } else {
        db = 0.0;
        dg = 0.0;
        dr = 1.0 - (v - 0.8758) * ((1.0 - 0.504) / (1.0 - 0.8758));
    }

    r = static_cast<uint8_t>(255.0 * dr);
    g = static_cast<uint8_t>(255.0 * dg);
    b = static_cast<uint8_t>(255.0 * db);
}

/**
 * @brief 计算LiDAR噪声
 * @param pb 点
 * @param range_inc 距离噪声
 * @param degree_inc 角度噪声
 * @return 协方差
 */
Eigen::Matrix3d CalcBodyCov(const Eigen::Vector3f& pb, float range_inc, float degree_inc)
{
    const double range = std::sqrt(pb.x() * pb.x() + pb.y() * pb.y() + pb.z() * pb.z());
    const double range_var = static_cast<double>(range_inc) * static_cast<double>(range_inc);
    Eigen::Matrix2d direction_var = Eigen::Matrix2d::Zero();
    const double angle = std::sin(static_cast<double>(degree_inc) * kPi / 180.0);
    direction_var.diagonal().setConstant(angle * angle);

    Eigen::Vector3d direction = pb.cast<double>();
    direction.normalize();
    Eigen::Matrix3d direction_hat;
    direction_hat << 0, -direction(2), direction(1), direction(2), 0, -direction(0), -direction(1), direction(0), 0;
    Eigen::Vector3d base_vector1(1.0, 1.0, -(direction(0) + direction(1)) / std::max(direction(2), 1e-6));
    base_vector1.normalize();
    Eigen::Vector3d base_vector2 = base_vector1.cross(direction);
    base_vector2.normalize();
    Eigen::Matrix<double, 3, 2> N;
    N << base_vector1(0), base_vector2(0), base_vector1(1), base_vector2(1), base_vector1(2), base_vector2(2);
    Eigen::Matrix<double, 3, 2> A = range * direction_hat * N;
    return direction * range_var * direction.transpose() + A * direction_var * A.transpose();
}

}  // namespace ms_slam::slam_core
