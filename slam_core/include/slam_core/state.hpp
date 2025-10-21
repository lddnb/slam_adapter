#pragma once

#include <manif/manif.h>

namespace ms_slam::slam_core
{

class State
{
  public:
    using BundleT = manif::Bundle<
        double,
        manif::R3,   // position
        manif::SO3,  // rotation
        manif::R3,   // velocity
        manif::R3,   // angular bias
        manif::R3,   // acceleartion bias
        manif::R3    // gravity
        >;

    using Tangent = typename BundleT::Tangent;
    static constexpr int DoF = BundleT::DoF;  // DoF whole state
    static constexpr int DoFNoise = 12;       // b_w, b_a, n_{b_w}, n_{b_a}
    // TODO: DoFObs
    static constexpr int DoFObs = manif::SGal3<double>::DoF;  // DoF obsevation equation

    using ProcessMatrix = Eigen::Matrix<double, DoF, DoF>;
    using MappingMatrix = Eigen::Matrix<double, DoF, DoFNoise>;
    using NoiseMatrix = Eigen::Matrix<double, DoFNoise, DoFNoise>;

    // clang-format off
    inline Eigen::Vector3d p()       const { return X.element<0>().coeffs();                  }
    inline Eigen::Matrix3d R()       const { return X.element<1>().quat().toRotationMatrix(); }
    inline Eigen::Quaterniond quat() const { return X.element<1>().quat();                    }
    inline Eigen::Vector3d v()       const { return X.element<2>().coeffs();                  }
    inline Eigen::Vector3d b_g()     const { return X.element<3>().coeffs();                  }
    inline Eigen::Vector3d b_a()     const { return X.element<4>().coeffs();                  }
    inline Eigen::Vector3d g()       const { return X.element<5>().coeffs();                  }

    void b_g(const Eigen::Vector3d& in) { X.element<1>() = manif::R3d(in); }
    void b_a(const Eigen::Vector3d& in) { X.element<2>() = manif::R3d(in); }
    void g(const Eigen::Vector3d& in)   { X.element<3>() = manif::R3d(in); }
    // clang-format on

    inline Eigen::Isometry3d isometry3d() const
    {
        Eigen::Isometry3d T;
        T.linear() = R();
        T.translation() = p();
        return T;
    }

  private:
    BundleT X;
    ProcessMatrix P;
    NoiseMatrix Q;

    Eigen::Vector3d gyro;  // angular velocity (IMU input)
    Eigen::Vector3d acc;   // linear acceleration (IMU input)

    double stamp;
};
}  // namespace ms_slam::slam_core