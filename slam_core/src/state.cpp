#include "slam_core/state.hpp"

#include <spdlog/spdlog.h>
#include "slam_core/config.hpp"

namespace ms_slam::slam_core
{
State::State() : stamp(-1.0)
{
    auto& cfg = Config::GetInstance();
    Eigen::Vector3d zero_vec = Eigen::Vector3d(0., 0., 0.);
    X = BundleState(                                  // X                    Tanget
        manif::R3d(zero_vec),                         // p                     0
        manif::SO3d(Eigen::Quaterniond::Identity()),  // R                     3
        manif::R3d(zero_vec),                         // v                     6
        manif::R3d(zero_vec),                         // b_g                   9
        manif::R3d(zero_vec),                         // b_a                   12
        manif::R3d(cfg.mapping_params.gravity));      // g                     15

    P.setIdentity();
    P *= 1e-3f;

    gyro.setZero();
    acc.setZero();

    // Control signal noise (never changes)
    Q.setZero();

    Q.block<3, 3>(0, 0) = cfg.mapping_params.gyr_cov * Eigen::Matrix3d::Identity();    // n_w
    Q.block<3, 3>(3, 3) = cfg.mapping_params.acc_cov * Eigen::Matrix3d::Identity();    // n_a
    Q.block<3, 3>(6, 6) = cfg.mapping_params.b_gyr_cov * Eigen::Matrix3d::Identity();  // n_{b_g}
    Q.block<3, 3>(9, 9) = cfg.mapping_params.b_acc_cov * Eigen::Matrix3d::Identity();  // n_{b_a}
}

void State::Predict(const BundleInput& imu, double dt, double timestamp)
{
    const Eigen::Vector3d& in_gyro = imu.element<0>().coeffs();
    const Eigen::Vector3d& in_acc = imu.element<1>().coeffs();

    ProcessMatrix Gx, Gf;  // Adjoint_X(u)^{-1}, J_r(u)  Sola-18, [https://arxiv.org/abs/1812.01537]
    BundleState X_tmp = X.plus(f(in_gyro, in_acc) * dt, Gx, Gf);

    // Update covariance
    ProcessMatrix Fx = Gx + Gf * df_dx(imu) * dt;  // He-2021, [https://arxiv.org/abs/2102.03804] Eq. (26)
    MappingMatrix Fw = Gf * df_dw(imu) * dt;       // He-2021, [https://arxiv.org/abs/2102.03804] Eq. (27)

    P = Fx * P * Fx.transpose() + Fw * Q * Fw.transpose();

    X = X_tmp;

    // Save info
    gyro = in_gyro;
    acc = in_acc;

    stamp = timestamp;
}

std::optional<Eigen::Isometry3d> State::Predict(double timestamp) const
{
    double dt = timestamp - stamp;
    if (dt < 0.0) {
        spdlog::critical("State::Predict: dt is negative: {} vs. {}", timestamp, stamp);
        return std::nullopt;
    }

    BundleState X_tmp = X.plus(f(gyro, acc) * dt);
    Eigen::Isometry3d res = Eigen::Isometry3d::Identity();
    res.translation() = X_tmp.element<0>().coeffs();
    res.linear() = X_tmp.element<1>().quat().toRotationMatrix();

    return res;
}

State::Tangent State::f(const Eigen::Vector3d& ang_vel, const Eigen::Vector3d& lin_acc) const
{
    Tangent u = Tangent::Zero();
    u.element<0>().coeffs() = v();
    u.element<1>().coeffs() = ang_vel - b_g();
    u.element<2>().coeffs() = R() * (lin_acc - b_a()) + g();
    // u.element<3>().coeffs() = n_{b_w}
    // u.element<4>().coeffs() = n_{b_a}

    return u;
}

State::ProcessMatrix State::df_dx(const BundleInput& imu) const
{
    ProcessMatrix out = ProcessMatrix::Zero();
    const Eigen::Vector3d& in_gyro = imu.element<0>().coeffs();
    const Eigen::Vector3d& in_acc = imu.element<1>().coeffs();

    // position
    out.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity();  // w.r.t v

    // rotation
    out.block<3, 3>(3, 9) = -Eigen::Matrix3d::Identity();  // w.r.t b_g

    // velocity
    out.block<3, 3>(6, 3) = -R() * manif::skew(acc - b_a());  // w.r.t R
    out.block<3, 3>(6, 12) = -R();                            // w.r.t b_a
    out.block<3, 3>(6, 15) = Eigen::Matrix3d::Identity();     // w.r.t g

    return out;
}

State::MappingMatrix State::df_dw(const BundleInput& imu) const
{
    // w = (n_g, n_a, n_{b_g}, n_{b_a})
    MappingMatrix out = MappingMatrix::Zero();

    out.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();  // R w.r.t n_g
    out.block<3, 3>(6, 3) = -R();                          // v w.r.t n_a
    out.block<3, 3>(9, 6) = Eigen::Matrix3d::Identity();   // b_g w.r.t n_{b_g}
    out.block<3, 3>(12, 9) = Eigen::Matrix3d::Identity();  // b_a w.r.t n_{b_a}

    return out;
}

}  // namespace ms_slam::slam_core
