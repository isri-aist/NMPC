/* Author: Masaki Murooka */

#include <gtest/gtest.h>

#include <fstream>
#include <iostream>

#include <nmpc_fmpc/FmpcSolver.h>

using Variable = typename nmpc_fmpc::FmpcSolver<2, 1, 3>::Variable;

using Status = typename nmpc_fmpc::FmpcSolver<2, 1, 3>::Status;

/** \brief FMPC problem for Van der Pol oscillator.

    See https://web.casadi.org/docs/#a-simple-test-problem
 */
class FmpcProblemOscillator : public nmpc_fmpc::FmpcProblem<2, 1, 3>
{
public:
  FmpcProblemOscillator(double dt) : FmpcProblem(dt) {}

  virtual StateDimVector stateEq(double t, const StateDimVector & x, const InputDimVector & u) const override
  {
    return stateEq(t, x, u, dt_);
  }

  virtual StateDimVector stateEq(double, // t
                                 const StateDimVector & x,
                                 const InputDimVector & u,
                                 double dt) const
  {
    StateDimVector x_dot;
    x_dot << (1.0 - std::pow(x[1], 2)) * x[0] - x[1] + u[0], x[0];
    return x + dt * x_dot;
  }

  virtual double runningCost(double, // t
                             const StateDimVector & x,
                             const InputDimVector & u) const override
  {
    return 0.5 * (x.squaredNorm() + u.squaredNorm());
  }

  virtual double terminalCost(double, // t
                              const StateDimVector & // x
  ) const override
  {
    return 0;
  }

  virtual IneqDimVector ineqConst(double, // t
                                  const StateDimVector & x,
                                  const InputDimVector & u) const override
  {
    IneqDimVector g;
    g[0] = -1 * x[1] - 0.05;
    g[1] = -1 * u[0] - 1.0;
    g[2] = u[0] - 0.9;
    return g;
  }

  virtual void calcStateEqDeriv(double, // t
                                const StateDimVector & x,
                                const InputDimVector &, // u
                                Eigen::Ref<StateStateDimMatrix> state_eq_deriv_x,
                                Eigen::Ref<StateInputDimMatrix> state_eq_deriv_u) const override
  {
    state_eq_deriv_x.setZero();
    state_eq_deriv_x(0, 0) = 1.0 - std::pow(x[1], 2);
    state_eq_deriv_x(0, 1) = -2 * x[0] * x[1] - 1.0;
    state_eq_deriv_x(1, 0) = 1;
    state_eq_deriv_x *= dt_;
    state_eq_deriv_x.diagonal().array() += 1;

    state_eq_deriv_u.setZero();
    state_eq_deriv_u(0, 0) = 1;
    state_eq_deriv_u *= dt_;
  }

  virtual void calcRunningCostDeriv(double, // t
                                    const StateDimVector & x,
                                    const InputDimVector & u,
                                    Eigen::Ref<StateDimVector> running_cost_deriv_x,
                                    Eigen::Ref<InputDimVector> running_cost_deriv_u) const override
  {
    running_cost_deriv_x = x;
    running_cost_deriv_u = u;
  }

  virtual void calcRunningCostDeriv(double t,
                                    const StateDimVector & x,
                                    const InputDimVector & u,
                                    Eigen::Ref<StateDimVector> running_cost_deriv_x,
                                    Eigen::Ref<InputDimVector> running_cost_deriv_u,
                                    Eigen::Ref<StateStateDimMatrix> running_cost_deriv_xx,
                                    Eigen::Ref<InputInputDimMatrix> running_cost_deriv_uu,
                                    Eigen::Ref<StateInputDimMatrix> running_cost_deriv_xu) const override
  {
    calcRunningCostDeriv(t, x, u, running_cost_deriv_x, running_cost_deriv_u);
    running_cost_deriv_xx.setIdentity();
    running_cost_deriv_uu.setIdentity();
    running_cost_deriv_xu.setZero();
  }

  virtual void calcTerminalCostDeriv(double, // t
                                     const StateDimVector &, // x
                                     Eigen::Ref<StateDimVector> terminal_cost_deriv_x) const override
  {
    terminal_cost_deriv_x.setZero();
  }

  virtual void calcTerminalCostDeriv(double t,
                                     const StateDimVector & x,
                                     Eigen::Ref<StateDimVector> terminal_cost_deriv_x,
                                     Eigen::Ref<StateStateDimMatrix> terminal_cost_deriv_xx) const override
  {
    calcTerminalCostDeriv(t, x, terminal_cost_deriv_x);
    terminal_cost_deriv_xx.setZero();
  }

  virtual void calcIneqConstDeriv(double, // t
                                  const StateDimVector &, // x
                                  const InputDimVector &, // u
                                  Eigen::Ref<IneqStateDimMatrix> ineq_const_deriv_x,
                                  Eigen::Ref<IneqInputDimMatrix> ineq_const_deriv_u) const override
  {
    ineq_const_deriv_x.setZero();
    ineq_const_deriv_x(0, 1) = -1;

    ineq_const_deriv_u.setZero();
    ineq_const_deriv_u(1, 0) = -1;
    ineq_const_deriv_u(2, 0) = 1;
  }
};

TEST(TestFmpcOscillator, SolveMpc)
{
  double horizon_dt = 0.01; // [sec]
  double horizon_duration = 4.0; // [sec]
  int horizon_steps = static_cast<int>(horizon_duration / horizon_dt);
  double end_t = 10.0; // [sec]

  // Instantiate problem
  auto fmpc_problem = std::make_shared<FmpcProblemOscillator>(horizon_dt);

  // Instantiate solver
  auto fmpc_solver = std::make_shared<nmpc_fmpc::FmpcSolver<2, 1, 3>>(fmpc_problem);
  fmpc_solver->config().horizon_steps = horizon_steps;
  fmpc_solver->config().max_iter = 3;
  // The option "init_complementary_variable" is effective for problems in which state constraints are infasible.
  // fmpc_solver->config().init_complementary_variable = true;
  Variable variable(horizon_steps);
  variable.reset(0.0, 0.0, 0.0, 1e0, 1e0);

  // Initialize simulation
  double sim_dt = 0.005; // [sec]
  double current_t = 0; // [sec]
  FmpcProblemOscillator::StateDimVector current_x = FmpcProblemOscillator::StateDimVector(0.0, 1.0);

  // Run MPC loop
  bool first_iter = true;
  std::string file_path = "/tmp/TestFmpcOscillatorResult.txt";
  std::ofstream ofs(file_path);
  ofs << "time x[0] x[1] u[0] mpc_iter computation_time kkt_error" << std::endl;
  while(current_t < end_t)
  {
    // Solve
    auto status = fmpc_solver->solve(current_t, current_x, variable);
    EXPECT_TRUE(status == Status::Succeeded || status == Status::MaxIterationReached);
    if(first_iter)
    {
      first_iter = false;
      fmpc_solver->dumpTraceDataList("/tmp/TestFmpcOscillatorTraceData.txt");
    }

    // Check inequality constraints
    FmpcProblemOscillator::InputDimVector current_u = fmpc_solver->variable().u_list[0];
    FmpcProblemOscillator::IneqDimVector current_g = fmpc_problem->ineqConst(current_t, current_x, current_u);
    EXPECT_TRUE((current_g.array() <= 0).all()) << "Inequality constraints violated: " << current_g.transpose();

    // Dump
    ofs << current_t << " " << current_x.transpose() << " " << current_u.transpose() << " "
        << fmpc_solver->traceDataList().back().iter << " " << fmpc_solver->computationDuration().solve << " "
        << fmpc_solver->traceDataList().back().kkt_error << std::endl;

    // Update to next step
    current_x = fmpc_problem->stateEq(current_t, current_x, current_u, sim_dt);
    current_t += sim_dt;
    variable = fmpc_solver->variable();
  }

  // Check final convergence
  EXPECT_LT(std::abs(current_x[0]), 1e-2);
  EXPECT_LT(std::abs(current_x[1]), 1e-2);

  std::cout << "Run the following commands in gnuplot:\n"
            << "  set key autotitle columnhead\n"
            << "  set key noenhanced\n"
            << "  plot \"" << file_path << "\" u 1:2 w lp, \"\" u 1:3 w lp, \"\" u 1:4 w lp\n";
}

TEST(TestFmpcOscillator, CheckDerivative)
{
  double horizon_dt = 0.1; // [sec]
  auto fmpc_problem = std::make_shared<FmpcProblemOscillator>(horizon_dt);

  double t = 0;
  FmpcProblemOscillator::StateDimVector x;
  x << 0.1, -0.2;
  FmpcProblemOscillator::InputDimVector u;
  u << 0.3;
  constexpr double deriv_eps = 1e-6;

  {
    FmpcProblemOscillator::StateStateDimMatrix state_eq_deriv_x_analytical;
    FmpcProblemOscillator::StateInputDimMatrix state_eq_deriv_u_analytical;
    fmpc_problem->calcStateEqDeriv(t, x, u, state_eq_deriv_x_analytical, state_eq_deriv_u_analytical);

    FmpcProblemOscillator::StateStateDimMatrix state_eq_deriv_x_numerical;
    FmpcProblemOscillator::StateInputDimMatrix state_eq_deriv_u_numerical;
    for(int i = 0; i < fmpc_problem->stateDim(); i++)
    {
      state_eq_deriv_x_numerical.col(i) =
          (fmpc_problem->stateEq(t, x + deriv_eps * FmpcProblemOscillator::StateDimVector::Unit(i), u)
           - fmpc_problem->stateEq(t, x - deriv_eps * FmpcProblemOscillator::StateDimVector::Unit(i), u))
          / (2 * deriv_eps);
    }
    for(int i = 0; i < fmpc_problem->inputDim(); i++)
    {
      state_eq_deriv_u_numerical.col(i) =
          (fmpc_problem->stateEq(t, x, u + deriv_eps * FmpcProblemOscillator::InputDimVector::Unit(i))
           - fmpc_problem->stateEq(t, x, u - deriv_eps * FmpcProblemOscillator::InputDimVector::Unit(i)))
          / (2 * deriv_eps);
    }

    EXPECT_LT((state_eq_deriv_x_analytical - state_eq_deriv_x_numerical).norm(), 1e-6);
    EXPECT_LT((state_eq_deriv_u_analytical - state_eq_deriv_u_numerical).norm(), 1e-6);
  }

  {
    FmpcProblemOscillator::IneqStateDimMatrix ineq_const_deriv_x_analytical;
    FmpcProblemOscillator::IneqInputDimMatrix ineq_const_deriv_u_analytical;
    fmpc_problem->calcIneqConstDeriv(t, x, u, ineq_const_deriv_x_analytical, ineq_const_deriv_u_analytical);

    FmpcProblemOscillator::IneqStateDimMatrix ineq_const_deriv_x_numerical;
    FmpcProblemOscillator::IneqInputDimMatrix ineq_const_deriv_u_numerical;
    for(int i = 0; i < fmpc_problem->stateDim(); i++)
    {
      ineq_const_deriv_x_numerical.col(i) =
          (fmpc_problem->ineqConst(t, x + deriv_eps * FmpcProblemOscillator::StateDimVector::Unit(i), u)
           - fmpc_problem->ineqConst(t, x - deriv_eps * FmpcProblemOscillator::StateDimVector::Unit(i), u))
          / (2 * deriv_eps);
    }
    for(int i = 0; i < fmpc_problem->inputDim(); i++)
    {
      ineq_const_deriv_u_numerical.col(i) =
          (fmpc_problem->ineqConst(t, x, u + deriv_eps * FmpcProblemOscillator::InputDimVector::Unit(i))
           - fmpc_problem->ineqConst(t, x, u - deriv_eps * FmpcProblemOscillator::InputDimVector::Unit(i)))
          / (2 * deriv_eps);
    }

    EXPECT_LT((ineq_const_deriv_x_analytical - ineq_const_deriv_x_numerical).norm(), 1e-6);
    EXPECT_LT((ineq_const_deriv_u_analytical - ineq_const_deriv_u_numerical).norm(), 1e-6);
  }
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
