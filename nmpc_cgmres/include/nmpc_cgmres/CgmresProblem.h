/* Author: Masaki Murooka */

#pragma once

#include <fstream>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>

namespace nmpc_cgmres
{
/** \brief C/GMRES problem. */
class CgmresProblem
{
public:
  /** \brief Type of function to return reference state. */
  using RefFunc = std::function<void(double, Eigen::Ref<Eigen::VectorXd>)>;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /** \brief Constructor. */
  CgmresProblem() {}

  /** \brief Calculate the state equation. */
  virtual void stateEquation(double t,
                             const Eigen::Ref<const Eigen::VectorXd> & x,
                             const Eigen::Ref<const Eigen::VectorXd> & u,
                             Eigen::Ref<Eigen::VectorXd> dotx) = 0;

  /** \brief Calculate the costate equation. */
  virtual void costateEquation(double t,
                               const Eigen::Ref<const Eigen::VectorXd> & lmd,
                               const Eigen::Ref<const Eigen::VectorXd> & xu,
                               Eigen::Ref<Eigen::VectorXd> dotlmd) = 0;

  /** \brief Calculate \f$ \frac{\partial \phi}{\partial x} \f$. */
  virtual void calcDphiDx(double t,
                          const Eigen::Ref<const Eigen::VectorXd> & x,
                          Eigen::Ref<Eigen::VectorXd> DphiDx) = 0;

  /** \brief Calculate \f$ \frac{\partial h}{\partial u} \f$. */
  virtual void calcDhDu(double t,
                        const Eigen::Ref<const Eigen::VectorXd> & x,
                        const Eigen::Ref<const Eigen::VectorXd> & u,
                        const Eigen::Ref<const Eigen::VectorXd> & lmd,
                        Eigen::Ref<Eigen::VectorXd> DhDu) = 0;

  /** \brief Dump model parameters. */
  virtual void dumpData(std::ofstream & ofs)
  {
    ofs << "\"state_eq_param\": [" << state_eq_param_.format(vecfmt_dump_) << "]," << std::endl;
  }

public:
  int dim_x_;
  int dim_u_;
  int dim_c_;
  int dim_uc_;

  Eigen::VectorXd state_eq_param_;

  Eigen::VectorXd x_initial_;
  Eigen::VectorXd u_initial_;

  const Eigen::IOFormat vecfmt_dump_ = Eigen::IOFormat(Eigen::StreamPrecision, 0, ", ", ", ", "", "", "", "");
};
} // namespace nmpc_cgmres
