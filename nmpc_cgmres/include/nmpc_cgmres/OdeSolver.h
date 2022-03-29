/* Author: Masaki Murooka */

#pragma once

#include <functional>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>

namespace nmpc_cgmres
{
/** \brief Virtual class to solve Ordinaly Diferential Equation. */
class OdeSolver
{
public:
  using StateEquation = std::function<void(double,
                                           const Eigen::Ref<const Eigen::VectorXd> &,
                                           const Eigen::Ref<const Eigen::VectorXd> &,
                                           Eigen::Ref<Eigen::VectorXd>)>;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  virtual void solve(const StateEquation & state_eq,
                     double t,
                     const Eigen::Ref<const Eigen::VectorXd> & x,
                     const Eigen::Ref<const Eigen::VectorXd> & u,
                     double dt,
                     Eigen::Ref<Eigen::VectorXd> ret) = 0;
};

/** \brief Class to solve Ordinaly Diferential Equation by Euler method. */
class EulerOdeSolver : public OdeSolver
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  virtual void solve(const StateEquation & state_eq,
                     double t,
                     const Eigen::Ref<const Eigen::VectorXd> & x,
                     const Eigen::Ref<const Eigen::VectorXd> & u,
                     double dt,
                     Eigen::Ref<Eigen::VectorXd> ret) override
  {
    Eigen::VectorXd dotx(x.size());
    state_eq(t, x, u, dotx);
    ret = x + dt * dotx;
  }
};

/** \brief Class to solve Ordinaly Diferential Equation by Runge-Kutta method. */
class RungeKuttaOdeSolver : public OdeSolver
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  virtual void solve(const StateEquation & state_eq,
                     double t,
                     const Eigen::Ref<const Eigen::VectorXd> & x,
                     const Eigen::Ref<const Eigen::VectorXd> & u,
                     double dt,
                     Eigen::Ref<Eigen::VectorXd> ret) override
  {
    double dt_half = dt / 2;
    Eigen::VectorXd k1(x.size()), k2(x.size()), k3(x.size()), k4(x.size());
    state_eq(t, x, u, k1);
    state_eq(t + dt_half, x + dt_half * k1, u, k2);
    state_eq(t + dt_half, x + dt_half * k2, u, k3);
    state_eq(t + dt, x + dt * k3, u, k4);
    ret = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
  }
};
} // namespace nmpc_cgmres
