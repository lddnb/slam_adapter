#pragma once

#include <deque>

#include <Eigen/Geometry>
#include "manif/manif.h"
#include <spdlog/spdlog.h>

#include "slam_core/common_state.hpp"
#include "slam_core/sensor/imu.hpp"

#define DIM 15

namespace ms_slam::slam_core::voxelslam
{

//! Don't forget to init
inline double imupre_scale_gravity = 1.0;
inline Eigen::Matrix<double, 6, 6> noiseMeas, noiseWalk;

class IMU_PRE
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Matrix3d R_delta;
  Eigen::Vector3d p_delta, v_delta;
  Eigen::Vector3d bg, ba;

  Eigen::Matrix3d R_bg;
  Eigen::Matrix3d p_bg, p_ba;
  Eigen::Matrix3d v_bg, v_ba;

  double dtime;

  Eigen::Vector3d dbg, dba;
  Eigen::Vector3d dbg_buf, dba_buf;

  Eigen::Matrix<double, DIM, DIM> cov;

  std::deque<IMU> _imus;

  IMU_PRE(const Eigen::Vector3d &bg1 = Eigen::Vector3d::Zero(), const Eigen::Vector3d &ba1 = Eigen::Vector3d::Zero())
  {
    bg = bg1; ba = ba1;
    R_delta.setIdentity();
    p_delta.setZero(); v_delta.setZero();

    R_bg.setZero();
    p_bg.setZero(); p_ba.setZero();
    v_bg.setZero(); v_ba.setZero();

    dtime = 0; // 累计时间

    dbg.setZero(); dba.setZero();
    dbg_buf.setZero(); dba_buf.setZero();

    cov.setZero();
  }

  /**
   * @brief 传入imu数据
   * 
   * @param imus 
   */
  void push_imu(std::deque<IMU> &imus)
  {
    _imus.insert(_imus.end(), imus.begin(), imus.end());
    Eigen::Vector3d cur_gyr, cur_acc;
    for(auto it_imu=imus.begin()+1; it_imu!=imus.end(); it_imu++)
    {
      IMU &imu1 = *(it_imu-1);
      IMU &imu2 = *it_imu;

      double dt = imu2.timestamp() - imu1.timestamp();

      cur_gyr << 0.5*(imu1.angular_velocity().x() + imu2.angular_velocity().x()),
                 0.5*(imu1.angular_velocity().y() + imu2.angular_velocity().y()),
                 0.5*(imu1.angular_velocity().z() + imu2.angular_velocity().z());
      cur_acc << 0.5*(imu1.linear_acceleration().x() + imu2.linear_acceleration().x()),
                 0.5*(imu1.linear_acceleration().y() + imu2.linear_acceleration().y()),
                 0.5*(imu1.linear_acceleration().z() + imu2.linear_acceleration().z());

      cur_gyr = cur_gyr - bg;
      cur_acc = cur_acc * imupre_scale_gravity - ba;

      add_imu(cur_gyr, cur_acc, dt);
    }
  }

  /**
   * @brief 计算预积分
   * 
   * @param cur_gyr 
   * @param cur_acc 
   * @param dt 
   */
  void add_imu(Eigen::Vector3d &cur_gyr, Eigen::Vector3d &cur_acc, double dt)
  {
    dtime += dt;
    Eigen::Matrix3d R_inc = manif::SO3Tangentd(cur_gyr * dt).exp().rotation();
    Eigen::Matrix3d R_jr(manif::SO3Tangentd(cur_gyr * dt).rjac());
    
    Eigen::Matrix3d R_dt = dt * R_delta;
    Eigen::Matrix3d R_dt2_2 = 0.5*dt*dt*R_delta;

    Eigen::Matrix3d acc_skew;
    acc_skew << manif::skew(cur_acc);

    p_ba = p_ba + v_ba*dt - R_dt2_2;
    p_bg = p_bg + v_bg*dt - R_dt2_2*acc_skew*R_bg;
    v_ba = v_ba - R_dt;
    v_bg = v_bg - R_dt * acc_skew * R_bg;
    R_bg = R_inc.transpose() * R_bg - R_jr*dt;
    
    // Eigen::Matrix<double, DIM, DIM> Ai;
    // Eigen::Matrix<double, DIM, DNOI> Bi;
    // Ai.setIdentity(); Bi.setZero();
    // Ai.block<3, 3>(0, 0) = R_inc.transpose();
    // Ai.block<3, 3>(0, 9) = -Eigen::Matrix3d::Identity() * dt;
    // // Ai.block<3, 3>(3, 0) = -R_dt2_2 * acc_skew;
    // Ai.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity() * dt;
    // // Ai.block<3, 3>(3, 12) = -R_dt2_2;
    // Ai.block<3, 3>(6, 0) = -R_dt * acc_skew;
    // Ai.block<3, 3>(6, 12) = -R_dt;
    // // Ai.block<3, 3>(6, 15) = Eigen::Matrix3d::Identity() * dt;

    // // Bi.block<3, 3>(0, 0) = R_jr * dt;
    // Bi.block<3, 3>(0, 0) = dt * Eigen::Matrix3d::Identity();
    // // Bi.block<3, 3>(3, 3) = R_dt2_2;
    // Bi.block<3, 3>(6, 3) = R_dt;
    // Bi.block<3, 3>(9, 6) = Eigen::Matrix3d::Identity() * dt;
    // Bi.block<3, 3>(12, 9) = Eigen::Matrix3d::Identity() * dt;

    // cov = Ai*cov*Ai.transpose() + Bi*noi_imu*Bi.transpose();

    Eigen::Matrix<double, 9, 9> A;
    Eigen::Matrix<double, 9, 6> B;
    A.setIdentity(); B.setZero();
    
    A.block<3, 3>(0, 0) = R_inc.transpose();
    A.block<3, 3>(3, 0) = -R_dt2_2 * acc_skew;
    A.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity() * dt;
    A.block<3, 3>(6, 0) = -R_dt * acc_skew;

    B.block<3, 3>(0, 0) = R_jr * dt;
    B.block<3, 3>(3, 3) = R_dt2_2;
    B.block<3, 3>(6, 3) = R_dt;

    // cov.block<9, 9>(0, 0) = A * cov.block<9, 9>(0, 0) * A.transpose() + B * noiseMeas * B.transpose() * dt * dt;
    cov.block<9, 9>(0, 0) = A * cov.block<9, 9>(0, 0) * A.transpose() + B * noiseMeas * B.transpose();
    // cov.block<6, 6>(9, 9) += noiseWalk * dt * dt;
    cov.block<6, 6>(9, 9) += noiseWalk * dt;

    p_delta += v_delta*dt + R_dt2_2*cur_acc;
    v_delta += R_dt * cur_acc;
    R_delta = R_delta * R_inc;
  }

  double give_evaluate(CommonState &st1, CommonState &st2, Eigen::MatrixXd &jtj, Eigen::VectorXd &gg, bool jac_enable)
  {
    Eigen::Matrix<double, DIM, DIM> joca, jocb;
    Eigen::Matrix<double, DIM, 1> rr;
    joca.setZero(); jocb.setZero(); rr.setZero();

    Eigen::Matrix3d R_correct = R_delta * manif::SO3Tangentd(R_bg * dbg).exp().rotation();
    Eigen::Vector3d t_correct = p_delta + p_bg*dbg + p_ba*dba;
    Eigen::Vector3d v_correct = v_delta + v_bg*dbg + v_ba*dba;

    Eigen::Matrix3d res_r = R_correct.transpose() * st1.R().transpose() * st2.R();
    Eigen::Vector3d exp_v = st1.R().transpose() * (st2.v() - st1.v() - dtime*st1.g());
    Eigen::Vector3d res_v = exp_v - v_correct;
    Eigen::Vector3d exp_t = st1.R().transpose() * (st2.p() - st1.p() - st1.v()*dtime - 0.5*dtime*dtime*st1.g());
    Eigen::Vector3d res_t = exp_t - t_correct;

    Eigen::Vector3d res_bg = st2.b_g() - st1.b_g();
    Eigen::Vector3d res_ba = st2.b_a() - st1.b_a();

    double b_wei = 1;

    rr.block<3, 1>(0, 0) = manif::SO3d(Eigen::Quaterniond(res_r)).log().coeffs();
    rr.block<3, 1>(3, 0) = res_t;
    rr.block<3, 1>(6, 0) = res_v;
    rr.block<3, 1>(9, 0) = res_bg*b_wei;
    rr.block<3, 1>(12, 0) = res_ba*b_wei;
    
    // rr.block<3, 1>(15, 0) = st2.g - st1.g;

    Eigen::Matrix<double, 15, 15> cov_inv = cov.inverse();

    if(jac_enable)
    {
      Eigen::Matrix3d JR_inv = manif::SO3d(Eigen::Quaterniond(res_r)).log().rjacinv();
      // joca.block<3, 3>(0, 0) = -JR_inv * st1.R.transpose() * st2.R;
      joca.block<3, 3>(0, 0) = -JR_inv * st2.R().transpose() * st1.R();
      jocb.block<3, 3>(0, 0) =  JR_inv;
      joca.block<3, 3>(0, 9) = -JR_inv * res_r.transpose() * manif::SO3Tangentd(R_bg*dbg).rjac() * R_bg;

      joca.block<3, 3>(3, 0) = manif::skew(exp_t);
      joca.block<3, 3>(3, 3) = -st1.R().transpose();
      joca.block<3, 3>(3, 6) = -st1.R().transpose() * dtime;
      joca.block<3, 3>(3, 9) = -p_bg;
      joca.block<3, 3>(3, 12) = -p_ba;
      jocb.block<3, 3>(3, 3) = st1.R().transpose();
      
      joca.block<3, 3>(6, 0) = manif::skew(exp_v);
      joca.block<3, 3>(6, 6) = -st1.R().transpose();
      joca.block<3, 3>(6, 9) = -v_bg;
      joca.block<3, 3>(6, 12) = -v_ba;
      jocb.block<3, 3>(6, 6) = st1.R().transpose();

      joca.block<3, 3>(9, 9) = -Eigen::Matrix3d::Identity()*b_wei;
      joca.block<3, 3>(12, 12) = -Eigen::Matrix3d::Identity()*b_wei;
      jocb.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity()*b_wei;
      jocb.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity()*b_wei;

      // joca.block<3, 3>(3, 15) = -0.5 * st1.R.transpose() * dtime * dtime;
      // joca.block<3, 3>(6, 15) = -st1.R.transpose() * dtime;
      // joca.block<3, 3>(15, 15) = -Eigen::Matrix3d::Identity();
      // jocb.block<3, 3>(15, 15) = Eigen::Matrix3d::Identity();

      Eigen::Matrix<double, DIM, 2*DIM> joc;
      joc.block<DIM, DIM>(0, 0) = joca;
      joc.block<DIM, DIM>(0, DIM) = jocb;

      // jtj = joc.transpose() * joc;
      // gg = joc.transpose() * rr;

      jtj = joc.transpose() * cov_inv * joc;
      gg = joc.transpose() * cov_inv * rr;
    }

    // return rr.squaredNorm();
    return rr.dot(cov_inv * rr);
  }

  void update_state(const Eigen::Matrix<double, DIM, 1> &dxi)
  {
    dbg_buf = dbg;
    dba_buf = dba;

    dbg += dxi.block<3, 1>(9, 0);
    dba += dxi.block<3, 1>(12, 0);
  }

  void merge(IMU_PRE &imu2)
  {
    p_bg += v_bg*imu2.dtime + R_delta*(imu2.p_bg-manif::skew(imu2.p_delta)*R_bg);
    p_ba += v_ba*imu2.dtime + R_delta*imu2.p_ba;
    v_bg += R_delta*(imu2.v_bg - manif::skew(imu2.v_delta)*R_bg);
    v_ba += R_delta*imu2.v_ba;
    R_bg = imu2.R_delta.transpose()*R_bg + imu2.R_bg;
    
    Eigen::Matrix<double, DIM, DIM> Ai, Bi;
    Ai.setIdentity(); Bi.setIdentity();
    Ai.block<3, 3>(0, 0) = imu2.R_delta.transpose();
    Ai.block<3, 3>(3, 0) = -R_delta * manif::skew(imu2.p_delta);
    Ai.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity() * imu2.dtime;
    Ai.block<3, 3>(6, 0) = -R_delta * manif::skew(imu2.v_delta);

    Bi.block<3, 3>(3, 3) = R_delta;
    Bi.block<3, 3>(6, 6) = R_delta;
    cov = Ai*cov*Ai.transpose() + Bi*imu2.cov*Bi.transpose();

    p_delta += v_delta*imu2.dtime + R_delta*imu2.p_delta;
    v_delta += R_delta*imu2.v_delta;
    R_delta = R_delta * imu2.R_delta;

    dtime += imu2.dtime;
  }

};

}  // namespace ms_slam::slam_core::voxelslam
