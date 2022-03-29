/* Author: Masaki Murooka */

#include <nmpc_cgmres/CgmresSolver.h>
#include <nmpc_cgmres/Gmres.h>

using namespace nmpc_cgmres;

void CgmresSolver::setup()
{
  double t_initial = 0;
  x_ = problem_->x_initial_;
  u_ = problem_->u_initial_;
  Eigen::VectorXd lmd_initial(problem_->dim_x_);
  Eigen::VectorXd DhDu(problem_->dim_uc_);

  // calc lmd_initial
  lmd_initial.resize(problem_->dim_x_);
  problem_->calcDphiDx(t_initial, x_, lmd_initial);

  // calc u by GMRES method
  Gmres gmres;
  Eigen::VectorXd DhDu_finite_diff(problem_->dim_uc_);
  Gmres::AmulFunc Amul_func = [&](const Eigen::Ref<const Eigen::VectorXd> & vec) {
    problem_->calcDhDu(t_initial, x_, u_ + finite_diff_delta_ * vec, lmd_initial, DhDu_finite_diff);
    return (DhDu_finite_diff - DhDu) / finite_diff_delta_;
  };

  Eigen::VectorXd delta_u = Eigen::VectorXd::Zero(problem_->dim_uc_);
  double DhDu_tol = 1e-6;
  for(int i = 0; i < 100; i++)
  {
    problem_->calcDhDu(t_initial, x_, u_, lmd_initial, DhDu);
    if(DhDu.norm() <= DhDu_tol)
    {
      break;
    }

    gmres.solve(Amul_func, -DhDu, delta_u, problem_->dim_uc_, 1e-10);
    u_ += delta_u;
  }
  if(DhDu.norm() > DhDu_tol)
  {
    std::cout << "failed to converge u in setup." << std::endl;
  }

  // setup variables
  x_with_delta_.resize(problem_->dim_x_);
  // since Eigen's matrix is column major order, it is better for rows to correspond
  // to time series in order to efficiently reshape a matrix into a vector.
  x_list_.resize(problem_->dim_x_, horizon_divide_num_ + 1);
  lmd_list_.resize(problem_->dim_x_, horizon_divide_num_ + 1);
  u_list_.resize(problem_->dim_uc_, horizon_divide_num_);
  u_list_Amul_func_.resize(problem_->dim_uc_, horizon_divide_num_);
  DhDu_list_.resize(problem_->dim_uc_, horizon_divide_num_);
  for(int i = 0; i < horizon_divide_num_; i++)
  {
    u_list_.col(i) = u_;
    DhDu_list_.col(i) = DhDu;
  }
  DhDu_list_with_delta_.resize(problem_->dim_uc_, horizon_divide_num_);
  DhDu_list_Amul_func_.resize(problem_->dim_uc_, horizon_divide_num_);
  delta_u_vec_.setZero(horizon_divide_num_ * problem_->dim_uc_);
}

void CgmresSolver::run()
{
  ofs_x_.open("/tmp/cgmres_x.dat");
  ofs_u_.open("/tmp/cgmres_u.dat");
  ofs_err_.open("/tmp/cgmres_err.dat");
  {
    std::ofstream ofs_param("/tmp/cgmres_param.dat");
    ofs_param << "{" << std::endl;
    ofs_param << "\"log_dt\": " << dt_ * dump_step_ << "," << std::endl;
    problem_->dumpData(ofs_param);
    ofs_param << "}" << std::endl;
  }

  setup();

  Eigen::VectorXd next_x(problem_->dim_x_);

  int i = 0;
  for(double t = 0; t <= sim_duration_; t += dt_)
  {
    // 1. simulate
    sim_ode_solver_->solve(std::bind(&CgmresProblem::stateEquation, problem_.get(), std::placeholders::_1,
                                     std::placeholders::_2, std::placeholders::_3, std::placeholders::_4),
                           t, x_, u_, dt_, next_x);

    // 2. calculate the control input
    calcControlInput(t, x_, next_x, u_);

    x_ = next_x;

    // 3. dump data
    if(i % dump_step_ == 0)
    {
      ofs_x_ << t << ", " << x_.format(vecfmt_dump_) << std::endl;
      ofs_u_ << t << ", " << u_.format(vecfmt_dump_) << std::endl;
      ofs_err_ << t << ", " << DhDu_vec_->norm() << std::endl;
    }
    i++;
  }

  ofs_x_.close();
  ofs_u_.close();
  ofs_err_.close();
}

void CgmresSolver::calcControlInput(double t,
                                    const Eigen::Ref<const Eigen::VectorXd> & x,
                                    const Eigen::Ref<const Eigen::VectorXd> & next_x,
                                    Eigen::Ref<Eigen::VectorXd> u)
{
  // 1.1 calculate DhDu_list_
  calcDhDuList(t, x, u_list_, DhDu_list_);

  // 1.2 calculate DhDu_list_with_delta_
  t_with_delta_ = t + finite_diff_delta_;
  x_with_delta_ = (1 - finite_diff_delta_ / dt_) * x + (finite_diff_delta_ / dt_) * next_x;
  calcDhDuList(t_with_delta_, x_with_delta_, u_list_, DhDu_list_with_delta_);

  // 2.1 calculate a vector of the linear equation
  // assume that the matrix is column major order, which is the default setting of Eigen
  DhDu_vec_ = std::make_shared<Eigen::Map<Eigen::VectorXd>>(DhDu_list_.data(), DhDu_list_.size());
  DhDu_vec_with_delta_ =
      std::make_shared<Eigen::Map<Eigen::VectorXd>>(DhDu_list_with_delta_.data(), DhDu_list_with_delta_.size());
  const Eigen::VectorXd & eq_b =
      ((1 - eq_zeta_ * finite_diff_delta_) * (*DhDu_vec_) - (*DhDu_vec_with_delta_)) / finite_diff_delta_;

  // 2.2 solve the linear equation by GMRES method
  Gmres gmres;
  gmres.solve(std::bind(&CgmresSolver::eqAmulFunc, this, std::placeholders::_1), eq_b, delta_u_vec_, k_max_, 1e-10);

  // 2.3 update u_list_ from delta_u_vec_
  for(int i = 0; i < horizon_divide_num_; i++)
  {
    u_list_.col(i) += dt_ * delta_u_vec_.segment(i * problem_->dim_uc_, problem_->dim_uc_);
  }

  // 3. set u_
  u = u_list_.col(0);
}

void CgmresSolver::calcDhDuList(double t,
                                const Eigen::Ref<const Eigen::VectorXd> & x,
                                const Eigen::Ref<const Eigen::MatrixXd> & u_list,
                                Eigen::Ref<Eigen::MatrixXd> DhDu_list)
{
  double horizon_duration = steady_horizon_duration_ * (1.0 - std::exp(-horizon_increase_ratio_ * t));
  double horizon_divide_step = horizon_duration / horizon_divide_num_;

  // 1.1 calculate x_list_[0]
  x_list_.col(0) = x;

  double tau = t;
  for(int i = 0; i < horizon_divide_num_; i++)
  {
    // 1.2 calculate x_list_[1, ..., horizon_divide_num_]
    ode_solver_->solve(std::bind(&CgmresProblem::stateEquation, problem_.get(), std::placeholders::_1,
                                 std::placeholders::_2, std::placeholders::_3, std::placeholders::_4),
                       tau, x_list_.col(i), u_list.col(i), horizon_divide_step, x_list_.col(i + 1));
    tau += horizon_divide_step;
  }

  // 2.1 calculate lmd_list_[horizon_divide_num_]
  problem_->calcDphiDx(tau, x_list_.col(horizon_divide_num_), lmd_list_.col(horizon_divide_num_));

  Eigen::VectorXd xu(problem_->dim_x_ + problem_->dim_uc_);
  for(int i = horizon_divide_num_ - 1; i >= 0; i--)
  {
    // 2.2 calculate lmd_list_[horizon_divide_num_-1, ..., 0]
    xu.head(problem_->dim_x_) = x_list_.col(i);
    xu.tail(problem_->dim_uc_) = u_list.col(i);
    ode_solver_->solve(std::bind(&CgmresProblem::costateEquation, problem_.get(), std::placeholders::_1,
                                 std::placeholders::_2, std::placeholders::_3, std::placeholders::_4),
                       tau, lmd_list_.col(i + 1), xu, -horizon_divide_step, lmd_list_.col(i));
    tau -= horizon_divide_step;

    // 3. DhDu_list[horizon_divide_num_-1, ..., 0]
    problem_->calcDhDu(tau, x_list_.col(i), u_list.col(i), lmd_list_.col(i + 1), DhDu_list.col(i));
  }
}

Eigen::VectorXd CgmresSolver::eqAmulFunc(const Eigen::Ref<const Eigen::VectorXd> & vec)
{
  // 1. calculate u_list_Amul_func_
  for(int i = 0; i < horizon_divide_num_; i++)
  {
    u_list_Amul_func_.col(i) =
        u_list_.col(i) + finite_diff_delta_ * vec.segment(i * problem_->dim_uc_, problem_->dim_uc_);
  }

  // 2. calculate DhDu_list_Amul_func_
  calcDhDuList(t_with_delta_, x_with_delta_, u_list_Amul_func_, DhDu_list_Amul_func_);

  // 3. calculate the finite difference
  DhDu_vec_Amul_func_ =
      std::make_shared<Eigen::Map<Eigen::VectorXd>>(DhDu_list_Amul_func_.data(), DhDu_list_Amul_func_.size());
  return ((*DhDu_vec_Amul_func_) - (*DhDu_vec_with_delta_)) / finite_diff_delta_;
}
