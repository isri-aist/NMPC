/* Author: Masaki Murooka */

#pragma once

#include <cmath>

#include <nmpc_cgmres/CgmresProblem.h>

/** \brief Problem of semiactive damper. */
class SemiactiveDamperProblem : public nmpc_cgmres::CgmresProblem
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /** \brief Constructor. */
  SemiactiveDamperProblem()
  {
    dim_x_ = 2;
    dim_u_ = 2;
    dim_c_ = 1;
    dim_uc_ = dim_u_ + dim_c_;

    x_initial_.resize(dim_x_);
    x_initial_ << 2, 0;
    u_initial_.resize(dim_uc_);
    u_initial_ << 0.01, 0.9, 0.03;

    // \f$ (a, b, u_max) \f$
    state_eq_param_.resize(3);
    state_eq_param_ << -1, -1, 1;

    obj_weight_.resize(4);
    obj_weight_ << 1, 10, 1, 1e-1;

    terminal_obj_weight_.resize(2);
    terminal_obj_weight_ << 1, 10;
  }

  /** \brief Calculate the state equation. */
  virtual void stateEquation(double, // t
                             const Eigen::Ref<const Eigen::VectorXd> & x,
                             const Eigen::Ref<const Eigen::VectorXd> & u,
                             Eigen::Ref<Eigen::VectorXd> dotx) override
  {
    assert(dotx.size() == dim_x_);
    dotx(0) = x(1);
    dotx(1) = state_eq_param_(0) * x(0) + state_eq_param_(1) * x(1) * u(0);
  }

  /** \brief Calculate the costate equation. */
  virtual void costateEquation(double, // t
                               const Eigen::Ref<const Eigen::VectorXd> & lmd,
                               const Eigen::Ref<const Eigen::VectorXd> & xu,
                               Eigen::Ref<Eigen::VectorXd> dotlmd) override
  {
    double a = state_eq_param_(0);
    double b = state_eq_param_(1);
    double q1 = obj_weight_(0);
    double q2 = obj_weight_(1);

    const Eigen::Ref<const Eigen::VectorXd> & x = xu.head(dim_x_);
    const Eigen::Ref<const Eigen::VectorXd> & u = xu.tail(dim_u_);

    assert(dotlmd.size() == dim_x_);
    dotlmd(0) = -a * lmd(1) - q1 * x(0);
    dotlmd(1) = -b * lmd(1) * u(0) - q2 * x(1) - lmd(0);
  }

  /** \brief Calculate \f$ \frac{\partial \phi}{\partial x} \f$. */
  virtual void calcDphiDx(double, // t
                          const Eigen::Ref<const Eigen::VectorXd> & x,
                          Eigen::Ref<Eigen::VectorXd> DphiDx) override
  {
    double sf1 = terminal_obj_weight_(0);
    double sf2 = terminal_obj_weight_(1);

    assert(DphiDx.size() == dim_x_);
    DphiDx(0) = sf1 * x(0);
    DphiDx(1) = sf2 * x(1);
  }

  /** \brief Calculate \f$ \frac{\partial h}{\partial u} \f$. */
  virtual void calcDhDu(double, // t
                        const Eigen::Ref<const Eigen::VectorXd> & x,
                        const Eigen::Ref<const Eigen::VectorXd> & u,
                        const Eigen::Ref<const Eigen::VectorXd> & lmd,
                        Eigen::Ref<Eigen::VectorXd> DhDu) override
  {
    double b = state_eq_param_(1);
    double u_max = state_eq_param_(2);
    double r1 = obj_weight_(2);
    double r2 = obj_weight_(3);
    double mu = u(2);

    // note that u includes the Lagrange multiplier of the equality constraints
    // i.e. u = (u1, u2, mu)
    assert(DhDu.size() == dim_uc_);
    DhDu(0) = r1 * u(0) + b * lmd(1) * x(1) + mu * (2 * u(0) - u_max);
    DhDu(1) = -r2 + 2 * mu * u(1);
    DhDu(2) = std::pow((u(0) - u_max / 2.0), 2) + u(1) * u(1) - u_max * u_max / 4.0;
  }

public:
  // \f$ (q_1, q_2, r_1, r_2) \f$
  Eigen::VectorXd obj_weight_;

  // \f$ (s_{f1}, s_{f2}) \f$
  Eigen::VectorXd terminal_obj_weight_;
};
