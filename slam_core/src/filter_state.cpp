#include "slam_core/filter_state.hpp"

#include <algorithm>

#include "slam_core/logging_utils.hpp"
#include "slam_core/config.hpp"

namespace ms_slam::slam_core
{
template<int kObsDim, int kResDim>
FilterStateTemplate<kObsDim, kResDim>::FilterStateTemplate() : stamp(-1.0)
{
    const auto& cfg = Config::GetInstance();
    Eigen::Vector3d zero_vec = Eigen::Vector3d(0., 0., 0.);
    X = BundleState(                                  // X                    Tanget
        manif::R3d(zero_vec),                         // p                     0
        manif::SO3d(Eigen::Quaterniond::Identity()),  // R                     3
        manif::R3d(zero_vec),                         // v                     6
        manif::R3d(zero_vec),                         // b_g                   9
        manif::R3d(zero_vec),                         // b_a                   12
        manif::R3d(cfg.mapping_params.gravity));      // g                     15

    P.setIdentity();
    // bg
    P.block<3, 3>(9, 9).diagonal() << 0.0001, 0.0001, 0.0001;
    // ba
    P.block<3, 3>(12, 12).diagonal() << 0.001, 0.001, 0.001;
    // g
    P.block<3, 3>(15, 15).diagonal() << 0.00001, 0.00001, 0.00001;
    // P *= 1e-3f;

    gyro.setZero();
    acc.setZero();

    // Control signal noise (never changes)
    Q.setZero();

    Q.block<3, 3>(0, 0) = cfg.mapping_params.gyr_cov * Eigen::Matrix3d::Identity();    // n_w
    Q.block<3, 3>(3, 3) = cfg.mapping_params.acc_cov * Eigen::Matrix3d::Identity();    // n_a
    Q.block<3, 3>(6, 6) = cfg.mapping_params.b_gyr_cov * Eigen::Matrix3d::Identity();  // n_{b_g}
    Q.block<3, 3>(9, 9) = cfg.mapping_params.b_acc_cov * Eigen::Matrix3d::Identity();  // n_{b_a}
}

template<int kObsDim, int kResDim>
void FilterStateTemplate<kObsDim, kResDim>::Predict(const BundleInput& imu, double dt, double timestamp)
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

template<int kObsDim, int kResDim>
std::optional<Eigen::Isometry3d> FilterStateTemplate<kObsDim, kResDim>::Predict(double timestamp) const
{
    double dt = timestamp - stamp;
    if (dt < 0.0) {
        spdlog::critical("State::Predict: dt is negative: {} vs. {}", timestamp, stamp);
        return std::nullopt;
    }

    BundleState X_tmp = X.plus(f(gyro, acc) * dt);
    Eigen::Isometry3d res = Eigen::Isometry3d::Identity();
    res.translation() = X_tmp.element<0>().coeffs();
    res.linear() = X_tmp.element<1>().rotation();

    return res;
}

template<int kObsDim, int kResDim>
void FilterStateTemplate<kObsDim, kResDim>::Update()
{
    if (h_models_.empty()) {
        spdlog::warn("State::Update: no observation models available");
        return;
    }

    for (const auto& entry : h_models_) {
        ApplyObservationModel(entry);
    }
}

template<int kObsDim, int kResDim>
void FilterStateTemplate<kObsDim, kResDim>::UpdateWithModel(std::string_view name)
{
    if (h_models_.empty()) {
        spdlog::warn("State::UpdateWithModel: no observation models available");
        return;
    }

    // 零散两个观测模型来说，std::find_if 和手写 for 循环在效率上没有本质差别：两者最终都会线性扫描整个容器
    const auto it = std::find_if(h_models_.begin(), h_models_.end(), [&](const ObservationEntry& entry) {
        return entry.name == name;
    });

    if (it == h_models_.end()) {
        spdlog::warn("State::UpdateWithModel: model {} not registered", name);
        return;
    }

    ApplyObservationModel(*it);
}

template<int kObsDim, int kResDim>
void FilterStateTemplate<kObsDim, kResDim>::UpdateWithModels(const std::vector<std::string>& names)
{
    if (names.empty()) {
        spdlog::warn("State::UpdateWithModels: empty name list");
        return;
    }

    for (const auto& name : names) {
        UpdateWithModel(name);
    }
}

template<int kObsDim, int kResDim>
void FilterStateTemplate<kObsDim, kResDim>::ApplyObservationModel(const ObservationEntry& entry)
{
    if (!entry.model) {
        spdlog::warn("State::Update: skip empty observation model {}", entry.name);
        return;
    }

    const ProcessMatrix identity = ProcessMatrix::Identity();

    const BundleState X_predicted = X;
    const ProcessMatrix P_predicted = P;

    ObsH H;
    ObsZ z;
    NoiseDiag noise_diag_inv;
    bool has_observation = false;

    for (int iter = 0; iter < 4; ++iter) {
        noise_diag_inv.resize(0);
        entry.model(H, z, noise_diag_inv);
        if (H.rows() == 0 || z.rows() == 0) {
            break;
        }

        has_observation = true;

        ProcessMatrix J;
        Tangent dx = X.minus(X_predicted, J);  // Xu-2021, Eq. (35)
        const ProcessMatrix J_inv = J.inverse();
        const ProcessMatrix P_linearized = J_inv * P_predicted * J_inv.transpose();

        const Eigen::Matrix<double, DoFObs, Eigen::Dynamic> HTR_inv = H.transpose() * noise_diag_inv.asDiagonal();

        ProcessMatrix P_inv = P_linearized.inverse();
        const Eigen::Matrix<double, DoFObs, DoFObs> HTH = HTR_inv * H;

        P_inv.block<DoFObs, DoFObs>(0, 0) += HTH;
        P_inv = P_inv.inverse();

        const Tangent Kz = P_inv.block<DoF, DoFObs>(0, 0) * HTR_inv * z;

        ProcessMatrix KH = ProcessMatrix::Zero();
        KH.block<DoF, DoFObs>(0, 0) = P_inv.block<DoF, DoFObs>(0, 0) * HTH;

        dx = Kz + (KH - identity) * J_inv * dx;

        if ((dx.coeffs().array().abs() <= 0.0001).all() || iter == 3) {
            ProcessMatrix L;
            X = X.plus(dx, {}, L);

            ProcessMatrix cov = (identity - KH) * P_linearized;
            P = L * cov * L.transpose();
            break;
        }

        X = X.plus(dx);
    }

    if (!has_observation) {
        spdlog::info("State::Update: observation produced no residuals ({})", entry.name);
    }
}

template<int kObsDim, int kResDim>
typename FilterStateTemplate<kObsDim, kResDim>::Tangent FilterStateTemplate<kObsDim, kResDim>::f(const Eigen::Vector3d& ang_vel, const Eigen::Vector3d& lin_acc) const
{
    Tangent u = Tangent::Zero();
    u.element<0>().coeffs() = v();
    u.element<1>().coeffs() = ang_vel - b_g();
    u.element<2>().coeffs() = R() * (lin_acc - b_a()) + g();
    // u.element<3>().coeffs() = n_{b_w}
    // u.element<4>().coeffs() = n_{b_a}

    return u;
}

template<int kObsDim, int kResDim>
typename FilterStateTemplate<kObsDim, kResDim>::ProcessMatrix FilterStateTemplate<kObsDim, kResDim>::df_dx(const BundleInput& imu) const
{
    ProcessMatrix out = ProcessMatrix::Zero();
    const Eigen::Vector3d& in_acc = imu.element<1>().coeffs();

    // position
    out.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity();  // w.r.t v

    // rotation
    out.block<3, 3>(3, 9) = -Eigen::Matrix3d::Identity();  // w.r.t b_g

    // velocity
    out.block<3, 3>(6, 3) = -R() * manif::skew(in_acc - b_a());  // w.r.t R
    out.block<3, 3>(6, 12) = -R();                            // w.r.t b_a
    out.block<3, 3>(6, 15) = Eigen::Matrix3d::Identity();     // w.r.t g

    return out;
}

template<int kObsDim, int kResDim>
typename FilterStateTemplate<kObsDim, kResDim>::MappingMatrix FilterStateTemplate<kObsDim, kResDim>::df_dw(const BundleInput& imu) const
{
    // w = (n_g, n_a, n_{b_g}, n_{b_a})
    MappingMatrix out = MappingMatrix::Zero();

    out.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();  // R w.r.t n_g
    out.block<3, 3>(6, 3) = -R();                          // v w.r.t n_a
    out.block<3, 3>(9, 6) = Eigen::Matrix3d::Identity();   // b_g w.r.t n_{b_g}
    out.block<3, 3>(12, 9) = Eigen::Matrix3d::Identity();  // b_a w.r.t n_{b_a}

    return out;
}

template<int kObsDim, int kResDim>
void FilterStateTemplate<kObsDim, kResDim>::SetHModel(const ObservationModel& h_model)
{
    h_models_.clear();
    h_models_.push_back(ObservationEntry{std::string("obs_0"), h_model});
}

template<int kObsDim, int kResDim>
void FilterStateTemplate<kObsDim, kResDim>::SetHModels(const std::vector<ObservationModel>& h_models)
{
    h_models_.clear();
    h_models_.reserve(h_models.size());
    for (std::size_t i = 0; i < h_models.size(); ++i) {
        h_models_.push_back(ObservationEntry{std::string("obs_") + std::to_string(i), h_models[i]});
    }
}

template<int kObsDim, int kResDim>
void FilterStateTemplate<kObsDim, kResDim>::SetNamedHModels(const std::vector<std::pair<std::string, ObservationModel>>& named_models)
{
    h_models_.clear();
    h_models_.reserve(named_models.size());
    for (const auto& item : named_models) {
        h_models_.push_back(ObservationEntry{item.first, item.second});
    }
}

template<int kObsDim, int kResDim>
void FilterStateTemplate<kObsDim, kResDim>::AddHModel(const ObservationModel& h_model)
{
    const std::string name = "obs_" + std::to_string(h_models_.size());
    h_models_.push_back(ObservationEntry{name, h_model});
}

template<int kObsDim, int kResDim>
void FilterStateTemplate<kObsDim, kResDim>::AddHModel(const std::string& name, const ObservationModel& h_model)
{
    h_models_.push_back(ObservationEntry{name, h_model});
}

template<int kObsDim, int kResDim>
void FilterStateTemplate<kObsDim, kResDim>::ClearHModels()
{
    h_models_.clear();
}

template<int kObsDim, int kResDim>
CommonState FilterStateTemplate<kObsDim, kResDim>::ExportCommonState() const
{
    CommonState payload;
    payload.p(p());
    payload.R(R());
    payload.v(v());
    payload.b_g(b_g());
    payload.b_a(b_a());
    payload.g(g());
    payload.timestamp(timestamp());
    Eigen::Matrix<double, CommonState::DoF, CommonState::DoF> cov = P. template block<CommonState::DoF, CommonState::DoF>(0, 0);
    payload.cov(cov);
    return payload;
}

template class FilterStateTemplate<6, 1>;

}  // namespace ms_slam::slam_core
