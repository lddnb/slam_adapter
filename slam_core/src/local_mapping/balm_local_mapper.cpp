#include "slam_core/local_mapping/balm_local_mapper.hpp"

#include <cmath>
#include <thread>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <spdlog/spdlog.h>

#include "slam_core/config.hpp"

namespace ms_slam::slam_core::local_mapping
{
BalmLocalMapper::BalmLocalMapper(const LocalMapperConfig& config) : config_(config), lidar_factor_(config.window_size), mgsize_(1)
{
    // 配置 VoxelSLAM 全局参数，与原实现保持一致
    voxelslam::voxel_size = config_.voxel_size;
    voxelslam::min_eigen_value = config_.min_eigen_value;
    voxelslam::plane_eigen_value_thre = config_.plane_eigen_value_thresholds;
    voxelslam::max_layer = config_.max_layer;
    voxelslam::max_points = config_.max_points;
    voxelslam::min_ba_point = config_.min_ba_point;
    voxelslam::min_point << 5, 5, 5, 5;

    voxelslam::mp = new int[config_.window_size];
    for (int i = 0; i < config_.window_size; ++i) {
        voxelslam::mp[i] = i;
    }

    const auto& cfg = Config::GetInstance();
    T_i_l_.setIdentity();
    T_i_l_.linear() = cfg.mapping_params.extrinR;
    T_i_l_.translation() = cfg.mapping_params.extrinT;

    dept_err_ = 0.02;
    beam_err_ = 0.05;
    win_count_ = 0;
    win_base_ = 0;
    win_size_ = config_.window_size;
    jour_ = 0.0;

    // 初始化预积分噪声，避免未赋值导致 NaN
    voxelslam::noiseMeas.setZero();
    voxelslam::noiseMeas.diagonal() << 1, 1, 1, 200, 200, 200;
    voxelslam::noiseWalk.setIdentity();
    voxelslam::noiseWalk *= 0.0001;

    last_pos_ = Eigen::Vector3d::Zero();

    window_pool_.resize(thread_num_);

    spdlog::info("BalmLocalMapper initialized, window_size={}", config_.window_size);
}

BalmLocalMapper::~BalmLocalMapper()
{
    if (voxelslam::mp != nullptr) {
        delete[] voxelslam::mp;
        voxelslam::mp = nullptr;
    }
    Reset();
    for (auto* pre : imu_factors_) {
        delete pre;
    }
    imu_factors_.clear();
}

void BalmLocalMapper::PushOdometryOutput(const OdometryOutput& input)
{
    std::lock_guard<std::mutex> lock(mutex_);
    input_queue_.push_back(input);
}

void BalmLocalMapper::ExportMapCloud(std::vector<PointCloudType::Ptr>& out)
{
    std::lock_guard<std::mutex> lock(mutex_);
    out.clear();
    if (!map_cloud_buffer_.empty()) {
        out.swap(map_cloud_buffer_);
    }
}

void BalmLocalMapper::ExportStates(std::unordered_map<int, CommonState>& out)
{
    std::lock_guard<std::mutex> lock(mutex_);
    out.clear();
    if (!output_state_buffer_.empty()) {
        out.swap(output_state_buffer_);
    }
}

std::optional<LocalMappingResult> BalmLocalMapper::TryProcess()
{
    mutex_.lock();
    if (input_queue_.empty()) {
        return std::nullopt;
    }

    OdometryOutput input = std::move(input_queue_.front());
    input_queue_.pop_front();
    mutex_.unlock();

    if (!input.cloud) {
        spdlog::error("BalmLocalMapper frame has null cloud");
        return std::nullopt;
    }

    if (!AppendFrame(input)) {
        spdlog::warn("BalmLocalMapper is initializing ...");
        return std::nullopt;
    }

    if (win_count_ >= win_size_) {
        voxelslam::LI_BA_Optimizer opt_lsv;
        opt_lsv.damping_iter(window_states_, lidar_factor_, imu_factors_, &hess_);

        x_curr_.R(window_states_[win_count_ - 1].R());
        x_curr_.p(window_states_[win_count_ - 1].p());

        auto cloud = std::make_shared<PointCloudType>();
        cloud->resize(window_points_[0]->size());
        auto positions = cloud->positions_vec3();
        for (std::size_t idx = 0; idx < window_points_[0]->size(); ++idx) {
            const Eigen::Vector3d pw = window_states_[0].R() * window_points_[0]->at(idx).pnt + window_states_[0].p();
            positions[idx] = pw.cast<float>();
        }
        
        mutex_.lock();
        map_cloud_buffer_.push_back(std::move(cloud));
        output_state_buffer_.emplace(win_base_, window_states_[0]);
        mutex_.unlock();
        opted_state_ = window_states_[0];

        multi_margi(slide_map_, jour_, win_count_, window_states_, lidar_factor_, window_pool_[0]);

        if ((win_base_ + win_count_) % 10 == 0) {
            double spat = (x_curr_.p() - last_pos_).norm();
            if (spat > 0.5) {
                jour_ += spat;
                last_pos_ = x_curr_.p();
                // release_flag = true;
            }
        }

        for (int i = 0; i < win_size_; i++) {
            voxelslam::mp[i] += mgsize_;
            if (voxelslam::mp[i] >= win_size_) voxelslam::mp[i] -= win_size_;
        }

        // 把第一帧移到队尾
        for (int i = mgsize_; i < win_count_; i++) {
            window_states_[i - mgsize_] = std::move(window_states_[i]);
            window_points_[i - mgsize_] = std::move(window_points_[i]);
        }

        // 删除滑窗第一帧
        for (int i = win_count_ - mgsize_; i < win_count_; i++) {
            window_states_.pop_back();
            window_points_.pop_back();

            delete imu_factors_.front();
            imu_factors_.pop_front();
        }

        win_base_ += mgsize_;
        win_count_ -= mgsize_;
    }

    return BuildResult(true);
}

void BalmLocalMapper::Reset()
{
    if (window_pool_.empty()) {
        window_pool_.resize(thread_num_);
    }

    if (voxelslam::mp != nullptr) {
        for (int i = 0; i < win_size_; ++i) {
            voxelslam::mp[i] = i;
        }
    }

    for (auto& kv : voxel_nodes_) {
        if (kv.second != nullptr) {
            kv.second->clear_slwd(window_pool_[0]);
            delete kv.second;
        }
    }
    voxel_nodes_.clear();
    slide_map_.clear();
    for (auto& pool : window_pool_) {
        for (auto* sw : pool) {
            delete sw;
        }
        pool.clear();
    }
    window_pool_.clear();
    window_states_.clear();
    window_points_.clear();
    input_queue_.clear();
    for (auto* pre : imu_factors_) {
        delete pre;
    }
    imu_factors_.clear();
}

bool BalmLocalMapper::AppendFrame(const OdometryOutput& input)
{
    // 构造点集，保持与 VoxelSLAM 一致的点簇和 world 坐标
    voxelslam::PVecPtr pvec = std::make_shared<voxelslam::PVec>();
    std::vector<Eigen::Vector3d> pwld;
    var_init(input.cloud, pvec);
    pvec_update(pvec, input.state, pwld);

    win_count_++;
    window_states_.push_back(input.state);
    window_points_.push_back(pvec);

    // 构建预积分：使用倒数第二帧的 bias
    if (win_count_ > 1) {
        const std::size_t prev_idx = win_count_ - 2;
        imu_factors_.push_back(new voxelslam::IMU_PRE(window_states_[prev_idx].b_g(), window_states_[prev_idx].b_a()));
        std::deque<IMU> imu_copy = input.imu_buffer;
        if (imu_copy.back().index() - imu_copy.front().index() + 1 != imu_copy.size()) {
            spdlog::warn("Not continuous IMU data for pre-integration");
        }
        // spdlog::info("odom_res imu size : {}, front ts: {}, back ts: {}", imu_copy.size(), imu_copy.front().timestamp(), imu_copy.back().timestamp());
        imu_factors_.back()->push_imu(imu_copy);
    }

    cut_voxel_multi(voxel_nodes_, window_points_[win_count_ - 1], win_count_ - 1, slide_map_, win_size_, pwld, window_pool_);

    if (!initialized_) {
        if (win_count_ < win_size_) return false;
        initialized_ = true;
        spdlog::info("BalmLocalMapper initialization finished at frame timestamp={}", input.cloud->timestamp(0));
    }

    lidar_factor_.clear();
    lidar_factor_.win_size = win_size_;

    multi_recut(slide_map_, win_count_, window_states_, lidar_factor_, window_pool_);
    return true;
}

void BalmLocalMapper::calcBodyVar(Eigen::Vector3d &pb, const float range_inc, const float degree_inc, Eigen::Matrix3d &var) 
{
  if (pb[2] == 0)
    pb[2] = 0.0001;
  float range = sqrt(pb[0] * pb[0] + pb[1] * pb[1] + pb[2] * pb[2]);
  float range_var = range_inc * range_inc;
  Eigen::Matrix2d direction_var;
  direction_var << pow(sin(degree_inc * 0.017453293), 2), 0, 0, pow(sin(degree_inc * 0.017453293), 2);
  Eigen::Vector3d direction(pb);
  direction.normalize();
  Eigen::Matrix3d direction_hat;
  direction_hat << 0, -direction(2), direction(1), direction(2), 0, -direction(0), -direction(1), direction(0), 0;
  Eigen::Vector3d base_vector1(1, 1, -(direction(0) + direction(1)) / direction(2));
  base_vector1.normalize();
  Eigen::Vector3d base_vector2 = base_vector1.cross(direction);
  base_vector2.normalize();
  Eigen::Matrix<double, 3, 2> N;
  N << base_vector1(0), base_vector2(0), base_vector1(1), base_vector2(1), base_vector1(2), base_vector2(2);
  Eigen::Matrix<double, 3, 2> A = range * direction_hat * N;
  var = direction * range_var * direction.transpose() + A * direction_var * A.transpose();
};

void BalmLocalMapper::var_init(const PointCloudType::ConstPtr& pl_cur, voxelslam::PVecPtr& pptr)
{
    auto positions = pl_cur->positions_vec3();
    const int plsize = static_cast<int>(positions.size());
    pptr->clear();
    pptr->resize(plsize);
    for (int i = 0; i < plsize; i++) {
        const Eigen::Vector3f& ap = positions[i];
        voxelslam::pointVar& pv = pptr->at(i);
        pv.pnt << ap[0], ap[1], ap[2];
        calcBodyVar(pv.pnt, dept_err_, beam_err_, pv.var);
        pv.pnt = T_i_l_.rotation() * pv.pnt + T_i_l_.translation();
        pv.var = T_i_l_.rotation() * pv.var * T_i_l_.rotation().transpose();
    }
}

void BalmLocalMapper::pvec_update(voxelslam::PVecPtr pptr, const CommonState& x_curr, std::vector<Eigen::Vector3d> &pwld)
{
  Eigen::Matrix3d rot_var = x_curr.cov().block<3, 3>(0, 0);
  Eigen::Matrix3d tsl_var = x_curr.cov().block<3, 3>(3, 3);

  for(voxelslam::pointVar &pv: *pptr)
  {
    Eigen::Matrix3d phat = manif::skew(pv.pnt);
    pv.var = x_curr.R() * pv.var * x_curr.R().transpose() + phat * rot_var * phat.transpose() + tsl_var;
    pwld.push_back(x_curr.R() * pv.pnt + x_curr.p());
  }
}

void BalmLocalMapper::multi_recut(
    std::unordered_map<voxelslam::VOXEL_LOC, voxelslam::OctoTree*>& feat_map,
    int win_count,
    std::vector<CommonState>& xs,
    voxelslam::LidarFactor& voxopt,
    std::vector<std::vector<voxelslam::SlideWindow*>>& sws)
{
    // for(auto iter=feat_map.begin(); iter!=feat_map.end(); iter++)
    // {
    //   iter->second->recut(win_count, xs, sws[0]);
    //   iter->second->tras_opt(voxopt);
    // }

    int thd_num = thread_num_;
    std::vector<std::vector<voxelslam::OctoTree*>> octss(thd_num);
    int g_size = feat_map.size();
    if (g_size < thd_num) return;
    std::vector<std::thread*> mthreads(thd_num);
    double part = 1.0 * g_size / thd_num;
    int cnt = 0;
    // 把所有voxel五等分
    for (auto iter = feat_map.begin(); iter != feat_map.end(); iter++) {
        octss[cnt].push_back(iter->second);
        if (octss[cnt].size() >= part && cnt < thd_num - 1) cnt++;
    }

    // 线程函数，用于对单个线程中的所有voxel进行初始化或者更新
    auto recut_func =
        [](int win_count, std::vector<voxelslam::OctoTree*>& oct, std::vector<CommonState> xxs, std::vector<voxelslam::SlideWindow*>& sw) {
            for (voxelslam::OctoTree* oc : oct) oc->recut(win_count, xxs, sw);
        };

    for (int i = 1; i < thd_num; i++) {
        mthreads[i] = new std::thread(recut_func, win_count, ref(octss[i]), xs, ref(sws[i]));
    }

    for (int i = 0; i < thd_num; i++) {
        if (i == 0) {
            recut_func(win_count, octss[i], xs, sws[i]);
        } else {
            mthreads[i]->join();
            delete mthreads[i];
        }
    }

    // 把所有线程的滑窗合并到第一个线程中
    for (int i = 1; i < sws.size(); i++) {
        sws[0].insert(sws[0].end(), sws[i].begin(), sws[i].end());
        sws[i].clear();
    }

    // 遍历所有voxel，提取点簇和voxel的平面相关信息
    for (auto iter = feat_map.begin(); iter != feat_map.end(); iter++) iter->second->tras_opt(voxopt);
}

void BalmLocalMapper::multi_margi(
    std::unordered_map<voxelslam::VOXEL_LOC, voxelslam::OctoTree*>& feat_map,
    double jour,
    int win_count,
    std::vector<CommonState>& xs,
    voxelslam::LidarFactor& voxopt,
    std::vector<voxelslam::SlideWindow*>& sw)
{
    // for(auto iter=feat_map.begin(); iter!=feat_map.end();)
    // {
    //   iter->second->jour = jour;
    //   iter->second->margi(win_count, 1, xs, voxopt);
    //   if(iter->second->isexist)
    //     iter++;
    //   else
    //   {
    //     iter->second->clear_slwd(sw);
    //     feat_map.erase(iter++);
    //   }
    // }
    // return;

    int thd_num = thread_num_;
    std::vector<std::vector<voxelslam::OctoTree*>*> octs;
    for (int i = 0; i < thd_num; i++) octs.push_back(new std::vector<voxelslam::OctoTree*>());

    int g_size = feat_map.size();
    if (g_size < thd_num) return;
    std::vector<std::thread*> mthreads(thd_num);
    double part = 1.0 * g_size / thd_num;
    int cnt = 0;
    for (auto iter = feat_map.begin(); iter != feat_map.end(); iter++) {
        iter->second->jour = jour;
        octs[cnt]->push_back(iter->second);
        if (octs[cnt]->size() >= part && cnt < thd_num - 1) cnt++;
    }

    auto margi_func = [](int win_cnt, std::vector<voxelslam::OctoTree*>* oct, std::vector<CommonState> xxs, voxelslam::LidarFactor& voxhess) {
        for (voxelslam::OctoTree* oc : *oct) {
            oc->margi(win_cnt, 1, xxs, voxhess);
        }
    };

    for (int i = 1; i < thd_num; i++) {
        mthreads[i] = new std::thread(margi_func, win_count, octs[i], xs, std::ref(voxopt));
    }

    for (int i = 0; i < thd_num; i++) {
        if (i == 0) {
            margi_func(win_count, octs[i], xs, voxopt);
        } else {
            mthreads[i]->join();
            delete mthreads[i];
        }
    }

    // 如果voxel中点较少，清空voxel并归还滑窗
    for (auto iter = feat_map.begin(); iter != feat_map.end();) {
        if (iter->second->isexist)
            iter++;
        else {
            iter->second->clear_slwd(sw);
            feat_map.erase(iter++);
        }
    }

    for (int i = 0; i < thd_num; i++) delete octs[i];
}

std::optional<LocalMappingResult> BalmLocalMapper::BuildResult(bool optimized)
{
    LocalMappingResult result;
    result.optimized_state = opted_state_;
    result.window_states = window_states_;
    result.is_keyframe = optimized;
    result.index = win_base_ - 1;
    return result;
}

std::unique_ptr<LocalMapper> CreateVoxelLocalMapper(const LocalMapperConfig& config)
{
    return std::make_unique<BalmLocalMapper>(config);
}

}  // namespace ms_slam::slam_core::local_mapping
