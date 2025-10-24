#pragma once

#include <deque>
#include <optional>

#include <manif/manif.h>

namespace ms_slam::slam_core
{

class State
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
    static constexpr int DoF = BundleState::DoF;  // DoF whole state
    static constexpr int DoFNoise = 12;           // b_w, b_a, n_{b_w}, n_{b_a}
    // TODO: DoFObs
    static constexpr int DoFObs = manif::SGal3<double>::DoF;  // DoF obsevation equation

    using ProcessMatrix = Eigen::Matrix<double, DoF, DoF>;
    using MappingMatrix = Eigen::Matrix<double, DoF, DoFNoise>;
    using NoiseMatrix = Eigen::Matrix<double, DoFNoise, DoFNoise>;

    State();
    ~State() = default;

    void Predict(const BundleInput& imu, double dt, double timestamp);
    [[nodiscard]] std::optional<Eigen::Isometry3d> Predict(double timestamp) const;

    [[nodiscard]] Tangent f(const Eigen::Vector3d& ang_vel, const Eigen::Vector3d& lin_acc) const;
    /**
     * @brief ∂f(x ⊕ δx, u, 0) / ∂δx|_{δx = 0}
     *
     * @param imu
     * @param dt
     * @return ProcessMatrix
     */
    [[nodiscard]] ProcessMatrix df_dx(const BundleInput& imu) const;
    /**
     * @brief ∂f(x, u, w) / ∂w|_{w = 0}
     *
     * @param imu
     * @param dt
     * @return MappingMatrix
     */
    [[nodiscard]] MappingMatrix df_dw(const BundleInput& imu) const;

    // clang-format off
    inline Eigen::Vector3d p()       const noexcept { return X.element<0>().coeffs();                  }
    inline Eigen::Matrix3d R()       const noexcept { return X.element<1>().quat().toRotationMatrix(); }
    inline Eigen::Quaterniond quat() const noexcept { return X.element<1>().quat();                    }
    inline Eigen::Vector3d v()       const noexcept { return X.element<2>().coeffs();                  }
    inline Eigen::Vector3d b_g()     const noexcept { return X.element<3>().coeffs();                  }
    inline Eigen::Vector3d b_a()     const noexcept { return X.element<4>().coeffs();                  }
    inline Eigen::Vector3d g()       const noexcept { return X.element<5>().coeffs();                  }

    void b_g(const Eigen::Vector3d& in) { X.element<3>() = manif::R3d(in); }
    void b_a(const Eigen::Vector3d& in) { X.element<4>() = manif::R3d(in); }
    void g(const Eigen::Vector3d& in)   { X.element<5>() = manif::R3d(in); }
    // clang-format on

    inline Eigen::Isometry3d isometry3d() const
    {
        Eigen::Isometry3d T;
        T.linear() = R();
        T.translation() = p();
        return T;
    }

    inline double timestamp() const noexcept { return stamp; }
    void timestamp(double in) noexcept { stamp = in; }

  private:
    BundleState X;
    ProcessMatrix P;
    NoiseMatrix Q;

    Eigen::Vector3d gyro;  // angular velocity (IMU input)
    Eigen::Vector3d acc;   // linear acceleration (IMU input)

    double stamp;
};

using States = std::deque<State>;

}  // namespace ms_slam::slam_core