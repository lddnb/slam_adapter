#pragma once

#include <span>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <spdlog/spdlog.h>

namespace ms_slam::slam_core
{

struct Match {
    Eigen::Vector3d p;
    Eigen::Vector4d n;  // global normal vector

    Match() = default;
    Match(const Eigen::Vector3d& p_, const Eigen::Vector4d& n_) : p(p_), n(n_){};

    inline static double Dist2Plane(const Eigen::Vector4d& normal, const Eigen::Vector3d& point) { return normal.head<3>().dot(point) + normal(3); }
};

typedef std::vector<Match> Matches;

inline bool
EstimatePlane(Eigen::Vector4d& pabcd, const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& pts, const double& thresh)
{
    int N = pts.size();
    if (N < 3) return false;

    Eigen::Matrix<double, Eigen::Dynamic, 3> neighbors(N, 3);
    for (size_t i = 0; i < N; i++) {
        neighbors.row(i) = pts[i].cast<double>();
    }

    Eigen::Vector3d centroid = neighbors.colwise().mean();
    neighbors.rowwise() -= centroid.transpose();

    Eigen::Matrix3d cov = (neighbors.transpose() * neighbors) / N;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(cov);
    if (eigensolver.info() != Eigen::Success) return false;

    Eigen::Vector3d normal = eigensolver.eigenvectors().col(0);
    double d = -normal.dot(centroid);

    pabcd.head<3>() = normal;
    pabcd(3) = d;

    for (auto& p : pts) {
        double distance = normal.dot(p.cast<double>()) + d;
        if (std::abs(distance) > thresh) return false;
    }

    return true;
}

/**
 * @brief 将按 xyz 排列的浮点 span 重解释为 Eigen::Vector3f 视图
 * @param data 连续 xyz 浮点数据
 * @return 若满足对齐和长度约束则返回有效 span，否则返回空 span
 */
inline std::span<Eigen::Vector3f> MakeVec3Span(std::span<float> data)
{
    if (data.size() % 3 != 0) {
        spdlog::error("MakeVec3Span expects data.size() % 3 == 0, got {}", data.size());
        return {};
    }
    const auto addr = reinterpret_cast<std::uintptr_t>(data.data());
    if (addr % alignof(Eigen::Vector3f) != 0U) {
        spdlog::error("MakeVec3Span expects {}-byte alignment, got address {:#x}", alignof(Eigen::Vector3f), addr);
        return {};
    }
    return {reinterpret_cast<Eigen::Vector3f*>(data.data()), data.size() / 3};
}

}  // namespace ms_slam::slam_core