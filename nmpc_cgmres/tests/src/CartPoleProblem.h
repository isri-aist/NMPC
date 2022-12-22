/* Author: Masaki Murooka */

#pragma once

#include <cmath>
#include <functional>

#include <nmpc_cgmres/CgmresProblem.h>

/** \brief Problem of Cart-Pole. */
class CartPoleProblem : public nmpc_cgmres::CgmresProblem
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /** \brief Constructor. */
  CartPoleProblem(const RefFunc & ref_func = nullptr, bool with_input_bound = false)
  : ref_func_(ref_func), with_input_bound_(with_input_bound)
  {
    if(with_input_bound_)
    {
      dim_x_ = 4;
      dim_u_ = 2;
      dim_c_ = 1;
      dim_uc_ = dim_u_ + dim_c_;

      u_initial_.resize(dim_uc_);
      u_initial_ << 0, 1.0, 0.01;
    }
    else
    {
      dim_x_ = 4;
      dim_u_ = 1;
      dim_c_ = 0;
      dim_uc_ = dim_u_ + dim_c_;

      u_initial_.resize(dim_uc_);
      u_initial_ << 0;
    }

    x_initial_.resize(dim_x_);
    x_initial_ << 0, M_PI, 0, 0;

    // \f$ (m1, m2, l, f_max) \f$
    state_eq_param_.resize(4);
    state_eq_param_ << 1.0, 1.0, 1.0, 100.0;

    // \f$ (q_1, q_2, q_3, q_4, r_1, r_2) \f$
    obj_weight_.resize(6);
    obj_weight_ << 10, 100, 1, 10, 10, 0.01;

    // \f$ (s_{f1}, s_{f2}, s_{f3}, s_{f4}) \f$
    terminal_obj_weight_.resize(4);
    terminal_obj_weight_ << 100, 300, 1, 10;

    if(ref_func_ == nullptr)
    {
      ref_func_ = [&](double, // t
                      Eigen::Ref<Eigen::VectorXd> ref) { ref.setZero(); };
    }
    ref_.resize(dim_x_);
  }

  /** \brief Calculate the state equation. */
  virtual void stateEquation(double, // t
                             const Eigen::Ref<const Eigen::VectorXd> & xvec,
                             const Eigen::Ref<const Eigen::VectorXd> & uvec,
                             Eigen::Ref<Eigen::VectorXd> dotxvec) override
  {
    assert(dotxvec.size() == dim_x_);

    const double & m1 = state_eq_param_(0);
    const double & m2 = state_eq_param_(1);
    const double & l = state_eq_param_(2);

    // const double & x = xvec(0);
    const double & theta = xvec(1);
    const double & dx = xvec(2);
    const double & dtheta = xvec(3);
    const double & f = uvec(0);

    double sin_theta = std::sin(theta);
    double cos_theta = std::cos(theta);
    double denom = m1 + m2 * std::pow(sin_theta, 2);
    dotxvec(0) = dx;
    dotxvec(1) = dtheta;
    dotxvec(2) = (f - m2 * l * std::pow(dtheta, 2) * sin_theta + m2 * g_ * sin_theta * cos_theta) / denom;
    dotxvec(3) = (f * cos_theta - m2 * l * std::pow(dtheta, 2) * sin_theta * cos_theta + g_ * (m1 + m2) * sin_theta)
                 / (l * denom);
  }

  /** \brief Calculate the costate equation. */
  virtual void costateEquation(double t,
                               const Eigen::Ref<const Eigen::VectorXd> & lmd,
                               const Eigen::Ref<const Eigen::VectorXd> & xuvec,
                               Eigen::Ref<Eigen::VectorXd> dotlmd) override
  {
    assert(dotlmd.size() == dim_x_);

    const Eigen::Ref<const Eigen::VectorXd> & xvec = xuvec.head(dim_x_);
    const Eigen::Ref<const Eigen::VectorXd> & uvec = xuvec.tail(dim_uc_);

    const double & m1 = state_eq_param_(0);
    const double & m2 = state_eq_param_(1);
    const double & l = state_eq_param_(2);

    // const double & x = xvec(0);
    const double & theta = xvec(1);
    // const double & dx = xvec(2);
    const double & dtheta = xvec(3);
    const double & f = uvec(0);

    double sin_theta = std::sin(theta);
    double cos_theta = std::cos(theta);
    double dtheta2 = std::pow(dtheta, 2);
    double sin_theta2 = std::pow(sin_theta, 2);
    double cos_theta2 = std::pow(cos_theta, 2);
    double denom = m1 + m2 * sin_theta2;
    double denom_square = std::pow(denom, 2);

    ref_func_(t, ref_);

    dotlmd(0) = -(obj_weight_(0) * (xvec(0) - ref_(0)));
    dotlmd(1) = -(obj_weight_(1) * (xvec(1) - ref_(1))
                  + (lmd(2) / denom_square)
                        * (((-m2 * l * dtheta2 * cos_theta + m2 * g_ * (cos_theta2 - sin_theta2)) * denom)
                           - ((f - m2 * l * dtheta2 * sin_theta + m2 * g_ * sin_theta * cos_theta)
                              * (2 * m2 * sin_theta * cos_theta)))
                  + (lmd(3) / (std::pow(l, 2) * denom_square))
                        * (((-f * sin_theta - m2 * l * dtheta2 * (cos_theta2 - sin_theta2) + g_ * (m1 + m2) * cos_theta)
                            * l * denom)
                           - ((f * cos_theta - m2 * l * dtheta2 * sin_theta * cos_theta + g_ * (m1 + m2) * sin_theta)
                              * (2 * l * m2 * sin_theta * cos_theta))));
    dotlmd(2) = -(obj_weight_(2) * (xvec(2) - ref_(2)) + lmd(0));
    dotlmd(3) = -(obj_weight_(3) * (xvec(3) - ref_(3)) + lmd(1) + lmd(2) * (-2 * m2 * l * dtheta * sin_theta) / denom
                  + lmd(3) * (-2 * m2 * l * dtheta * sin_theta * cos_theta) / (l * denom));
  }

  /** \brief Calculate \f$ \frac{\partial \phi}{\partial x} \f$. */
  virtual void calcDphiDx(double t,
                          const Eigen::Ref<const Eigen::VectorXd> & xvec,
                          Eigen::Ref<Eigen::VectorXd> DphiDx) override
  {
    assert(DphiDx.size() == dim_x_);

    ref_func_(t, ref_);

    for(int i = 0; i < dim_x_; i++)
    {
      DphiDx(i) = terminal_obj_weight_(i) * (xvec(i) - ref_(i));
    }
  }

  /** \brief Calculate \f$ \frac{\partial h}{\partial u} \f$. */
  virtual void calcDhDu(double, // t
                        const Eigen::Ref<const Eigen::VectorXd> & xvec,
                        const Eigen::Ref<const Eigen::VectorXd> & uvec,
                        const Eigen::Ref<const Eigen::VectorXd> & lmd,
                        Eigen::Ref<Eigen::VectorXd> DhDu) override
  {
    const double & m1 = state_eq_param_(0);
    const double & m2 = state_eq_param_(1);
    const double & l = state_eq_param_(2);

    // const double & x = xvec(0);
    const double & theta = xvec(1);
    const double & f = uvec(0);
    const double & r1 = obj_weight_(4);

    double sin_theta = std::sin(theta);
    double cos_theta = std::cos(theta);
    double denom = m1 + m2 * std::pow(sin_theta, 2);

    assert(DhDu.size() == dim_uc_);
    DhDu(0) = (r1 * f + lmd(2) * (1.0 / denom) + lmd(3) * (cos_theta / (l * denom)));

    if(with_input_bound_)
    {
      const double & f_max = state_eq_param_(3);
      const double & r2 = obj_weight_(5);

      const double & f_dummy = uvec(1);
      const double & mu = uvec(2);

      DhDu(0) += 2 * mu * f;
      DhDu(1) = -r2 + 2 * mu * f_dummy;
      DhDu(2) = std::pow(f, 2) + std::pow(f_dummy, 2) - std::pow(f_max, 2);
    }
  }

public:
  // \f$ (q_1, q_2, q_3, q_4, r_1, r_2) \f$
  Eigen::VectorXd obj_weight_;

  // \f$ (s_{f1}, s_{f2}, s_{f3}, s_{f4}) \f$
  Eigen::VectorXd terminal_obj_weight_;

  // reference
  std::function<void(double, Eigen::Ref<Eigen::VectorXd>)> ref_func_;
  Eigen::VectorXd ref_;

  double g_ = 9.80665;

  bool with_input_bound_;
};
