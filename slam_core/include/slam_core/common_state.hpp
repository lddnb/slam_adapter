#pragma once

#include <Eigen/Geometry>
#include <spdlog/spdlog.h>
#include <manif/manif.h>

namespace ms_slam::slam_core
{
/**
 * @brief 贯穿里程计与建图的公共状态量载体
 */
class CommonState
{
  public:
    using BundleState = manif::Bundle<
        double,
        manif::SO3,  // rotation
        manif::R3,   // position
        manif::R3,   // velocity
        manif::R3,   // angular bias
        manif::R3    // acceleartion bias
        >;

    using Tangent = typename BundleState::Tangent;
    static constexpr int DoF = BundleState::DoF;

    CommonState()
    {
        Eigen::Vector3d zero_vec = Eigen::Vector3d::Zero();
        X_ = BundleState(                                 // X                    Tanget
            manif::SO3d(Eigen::Quaterniond::Identity()),  // R                     0
            manif::R3d(zero_vec),                         // p                     3
            manif::R3d(zero_vec),                         // v                     6
            manif::R3d(zero_vec),                         // b_g                   9
            manif::R3d(zero_vec)                          // b_a                   12
        );
        timestamp_ = -1.0;
        cov_ = Eigen::Matrix<double, DoF, DoF>::Identity();
    }

    // clang-format off
    inline Eigen::Matrix3d R()       const noexcept { return X_.element<0>().rotation(); }
    inline Eigen::Quaterniond quat() const noexcept { return X_.element<0>().quat();     }
    inline Eigen::Vector3d p()       const noexcept { return X_.element<1>().coeffs();   }
    inline Eigen::Vector3d v()       const noexcept { return X_.element<2>().coeffs();   }
    inline Eigen::Vector3d b_g()     const noexcept { return X_.element<3>().coeffs();   }
    inline Eigen::Vector3d b_a()     const noexcept { return X_.element<4>().coeffs();   }
    inline Eigen::Vector3d g()       const noexcept { return g_;                         }
    inline double timestamp()        const noexcept { return timestamp_;                 }
    inline Eigen::Matrix<double, DoF, DoF> cov() const noexcept { return cov_;           }

    void R(const Eigen::Matrix3d& in)       { X_.element<0>() = manif::SO3d(Eigen::Quaterniond(in)); }
    void quat(const Eigen::Quaterniond& in) { X_.element<0>() = manif::SO3d(in); }
    void p(const Eigen::Vector3d& in)       { X_.element<1>() = manif::R3d(in);  }
    void v(const Eigen::Vector3d& in)       { X_.element<2>() = manif::R3d(in);  }
    void b_g(const Eigen::Vector3d& in)     { X_.element<3>() = manif::R3d(in);  }
    void b_a(const Eigen::Vector3d& in)     { X_.element<4>() = manif::R3d(in);  }
    void g(const Eigen::Vector3d& in)       { g_ = in;                              }
    void timestamp(double in)               { timestamp_ = in;                      }
    void cov(const Eigen::Matrix<double, DoF, DoF>& in) { cov_ = in;                }
    // clang-format on

    inline Eigen::Isometry3d isometry3d() const
    {
        Eigen::Isometry3d T;
        T.linear() = R();
        T.translation() = p();
        return T;
    }

  private:
    BundleState X_;
    Eigen::Vector3d g_;
    double timestamp_;
    Eigen::Matrix<double, DoF, DoF> cov_;
};

}  // namespace ms_slam::slam_core
