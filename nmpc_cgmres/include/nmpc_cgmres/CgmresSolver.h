/* Author: Masaki Murooka */

#pragma once

#include <fstream>
#include <iostream>
#include <memory>

#include <nmpc_cgmres/CgmresProblem.h>
#include <nmpc_cgmres/OdeSolver.h>

namespace nmpc_cgmres
{
/** \brief C/GMRES solver.

    See the following articles about the C/GMRES method:
      - T Ohtsuka. Continuation/GMRES method for fast computation of nonlinear receding horizon control. Automatica. 2004.
      - https://www.coronasha.co.jp/np/isbn/9784339033182/
      - https://www.coronasha.co.jp/np/isbn/9784339032109/
 */
class CgmresSolver
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /** \brief Constructor. */
  CgmresSolver(std::shared_ptr<CgmresProblem> problem,
               std::shared_ptr<OdeSolver> ode_solver,
               std::shared_ptr<OdeSolver> sim_ode_solver = nullptr)
  : problem_(problem), ode_solver_(ode_solver), sim_ode_solver_(sim_ode_solver)
  {
    if(!sim_ode_solver_)
    {
      sim_ode_solver_ = ode_solver_;
    }
  }

  /** \brief Setup. */
  void setup();

  /** \brief Run NMPC. */
  void run();

  /** \brief Calculate the control input. */
  void calcControlInput(double t,
                        const Eigen::Ref<const Eigen::VectorXd> & x,
                        const Eigen::Ref<const Eigen::VectorXd> & next_x,
                        Eigen::Ref<Eigen::VectorXd> u);

  /** \brief Calculate the \f$ \frac{\partial h}{\partial u} \f$ list in the horizon. */
  void calcDhDuList(double t,
                    const Eigen::Ref<const Eigen::VectorXd> & x,
                    const Eigen::Ref<const Eigen::MatrixXd> & u_list,
                    Eigen::Ref<Eigen::MatrixXd> DhDu_list);

  /** \brief Function to return \f$ A * v \f$ where \f$ v \f$ is given. */
  Eigen::VectorXd eqAmulFunc(const Eigen::Ref<const Eigen::VectorXd> & vec);

public:
  std::shared_ptr<CgmresProblem> problem_;
  std::shared_ptr<OdeSolver> ode_solver_;
  std::shared_ptr<OdeSolver> sim_ode_solver_;

  //////// parameters of C/GMRES method ////////
  double sim_duration_ = 10;

  double steady_horizon_duration_ = 1.0;
  int horizon_divide_num_ = 25;
  double horizon_increase_ratio_ = 0.5;

  double dt_ = 0.001;

  double eq_zeta_ = 1000.0;
  int k_max_ = 5;

  double finite_diff_delta_ = 0.002;

  int dump_step_ = 5;

  //////// variables that are set during processing ////////
  Eigen::VectorXd x_;
  Eigen::VectorXd u_;

  double t_with_delta_;
  Eigen::VectorXd x_with_delta_;

  Eigen::MatrixXd x_list_;
  Eigen::MatrixXd lmd_list_;

  Eigen::MatrixXd u_list_;
  Eigen::MatrixXd u_list_Amul_func_;

  Eigen::MatrixXd DhDu_list_;
  Eigen::MatrixXd DhDu_list_with_delta_;
  Eigen::MatrixXd DhDu_list_Amul_func_;
  // Eigen::Map does not have a default constructor
  std::shared_ptr<Eigen::Map<Eigen::VectorXd>> DhDu_vec_;
  std::shared_ptr<Eigen::Map<Eigen::VectorXd>> DhDu_vec_with_delta_;
  std::shared_ptr<Eigen::Map<Eigen::VectorXd>> DhDu_vec_Amul_func_;

  Eigen::VectorXd delta_u_vec_;

  //////// variables for utility ////////
  std::ofstream ofs_x_;
  std::ofstream ofs_u_;
  std::ofstream ofs_err_;
  const Eigen::IOFormat vecfmt_dump_ = Eigen::IOFormat(Eigen::StreamPrecision, 0, ", ", ", ", "", "", "", "");
};
} // namespace nmpc_cgmres
