#pragma once

#include <deque>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <manif/manif.h>

namespace ms_slam::slam_core
{

template<int kObsDim, int kResDim>
class FilterStateTemplate
{
  public:
    using BundleState = manif::Bundle<
        double,
        manif::R3,   // position
        manif::SO3,  // rotation
        manif::R3,   // velocity
        manif::R3,   // angular bias
        manif::R3,   // acceleartion bias
        manif::R3    // gravity
        >;

    using BundleInput = manif::Bundle<
        double,
        manif::R3,  // gyro
        manif::R3   // acc
        >;

    using Tangent = typename BundleState::Tangent;
    static constexpr int DoF = BundleState::DoF;     // DoF whole state
    static constexpr int DoFNoise = 12;              // b_w, b_a, n_{b_w}, n_{b_a}
    static constexpr int DoFObs = kObsDim;           // DoF observation equation
    static constexpr int DoFRes = kResDim;           // DoF residual equation

    using ProcessMatrix = Eigen::Matrix<double, DoF, DoF>;
    using MappingMatrix = Eigen::Matrix<double, DoF, DoFNoise>;
    using NoiseMatrix = Eigen::Matrix<double, DoFNoise, DoFNoise>;
    using ObsH = Eigen::Matrix<double, Eigen::Dynamic, DoFObs>;
    using ObsZ = Eigen::Matrix<double, Eigen::Dynamic, 1>;
    using NoiseDiag = Eigen::VectorXd;
    /**
     * @brief 观测模型回调类型
     * @param H 线性化雅可比
     * @param z 观测残差
     * @param noise_diag_inv 观测噪声协方差对角的逆；若 size==1 表示统一噪声
     */
    using ObservationModel = std::function<void(ObsH& H, ObsZ& z, NoiseDiag& noise_diag_inv)>;

    FilterStateTemplate();
    ~FilterStateTemplate() = default;

    void Predict(const BundleInput& imu, double dt, double timestamp);
    [[nodiscard]] std::optional<Eigen::Isometry3d> Predict(double timestamp) const;
    void Update();

    /**
     * @brief 仅更新指定名称的观测模型
     * @param name 观测模型名称
     */
    void UpdateWithModel(std::string_view name);

    /**
     * @brief 依序更新给定名称集合对应的观测模型
     * @param names 观测模型名称列表
     */
    void UpdateWithModels(const std::vector<std::string>& names);

    /**
     * @brief 计算离散时间预测所需的李代数增量
     * @param ang_vel 当前角速度测量（单位：rad/s）
     * @param lin_acc 当前线加速度测量（单位：m/s^2）
     * @return BundleState 对应的李代数增量
     */
    [[nodiscard]] Tangent f(const Eigen::Vector3d& ang_vel, const Eigen::Vector3d& lin_acc) const;
    /**
     * @brief ∂f(x ⊕ δx, u, 0) / ∂δx|_{δx = 0}
     *
     * @param imu
     * @return ProcessMatrix
     */
    [[nodiscard]] ProcessMatrix df_dx(const BundleInput& imu) const;
    /**
     * @brief ∂f(x, u, w) / ∂w|_{w = 0}
     *
     * @param imu
     * @return MappingMatrix
     */
    [[nodiscard]] MappingMatrix df_dw(const BundleInput& imu) const;

    // clang-format off
    inline Eigen::Map<const manif::R3d> ori_p()  const noexcept { return X.element<0>();}
    inline Eigen::Map<const manif::SO3d> ori_R() const noexcept { return X.element<1>();}

    inline Eigen::Vector3d p()       const noexcept { return X.element<0>().coeffs();   }
    inline Eigen::Matrix3d R()       const noexcept { return X.element<1>().rotation(); }
    inline Eigen::Quaterniond quat() const noexcept { return X.element<1>().quat();     }
    inline Eigen::Vector3d v()       const noexcept { return X.element<2>().coeffs();   }
    inline Eigen::Vector3d b_g()     const noexcept { return X.element<3>().coeffs();   }
    inline Eigen::Vector3d b_a()     const noexcept { return X.element<4>().coeffs();   }
    inline Eigen::Vector3d g()       const noexcept { return X.element<5>().coeffs();   }

    void quat(const Eigen::Quaterniond& in) { X.element<1>() = manif::SO3d(in); }
    void b_g(const Eigen::Vector3d& in)     { X.element<3>() = manif::R3d(in);  }
    void b_a(const Eigen::Vector3d& in)     { X.element<4>() = manif::R3d(in);  }
    void g(const Eigen::Vector3d& in)       { X.element<5>() = manif::R3d(in);  }
    // clang-format on

    inline Eigen::Isometry3d isometry3d() const
    {
        Eigen::Isometry3d T;
        T.linear() = R();
        T.translation() = p();
        return T;
    }

    inline ProcessMatrix cov() const noexcept { return P; }
    inline double timestamp() const noexcept { return stamp; }
    void timestamp(double in) noexcept { stamp = in; }

    /**
     * @brief 设置单一观测模型，旧接口保持兼容（内部清空后追加）
     * @param h_model 观测方程回调
     */
    void SetHModel(const ObservationModel& h_model);

    /**
     * @brief 批量设置观测模型
     * @param h_models 观测模型列表
     */
    void SetHModels(const std::vector<ObservationModel>& h_models);

    /**
     * @brief 使用自定义名称批量设置观测模型
     * @param named_models 名称与模型的组合
     */
    void SetNamedHModels(const std::vector<std::pair<std::string, ObservationModel>>& named_models);

    /**
     * @brief 追加单个观测模型
     * @param h_model 观测方程回调
     */
    void AddHModel(const ObservationModel& h_model);

    /**
     * @brief 追加带名称的观测模型
     * @param name 观测模型名称
     * @param h_model 观测方程回调
     */
    void AddHModel(const std::string& name, const ObservationModel& h_model);

    /**
     * @brief 清空全部观测模型
     */
    void ClearHModels();

  private:
    struct ObservationEntry
    {
        std::string name;
        ObservationModel model;
    };

    void ApplyObservationModel(const ObservationEntry& entry);

    BundleState X;
    ProcessMatrix P;
    NoiseMatrix Q;

    Eigen::Vector3d gyro;  // angular velocity (IMU input)
    Eigen::Vector3d acc;   // linear acceleration (IMU input)

    double stamp;

    std::vector<ObservationEntry> h_models_;
};

using FilterState = FilterStateTemplate<6, 1>;
using FilterStates = std::deque<FilterState>;
}  // namespace ms_slam::slam_core
