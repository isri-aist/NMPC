/* Author: Masaki Murooka */

#include <gtest/gtest.h>

#include <fstream>
#include <iostream>

#include <noc_ddp/DDP.h>

/** \brief DDP problem for vertical motion.

    State is [pos_z, vel_z]. Input is [force_z].
    Running cost is sum of the respective quadratic terms of state and input.
    Terminal cost is quadratic term of state.
 */
class DDPProblemVerticalMotion : public NOC::DDPProblem<2, Eigen::Dynamic>
{
public:
  struct CostWeight
  {
    CostWeight()
    {
      running_x << 1.0, 1e-1;
      running_u = 1e-2;
      terminal_x << 1.0, 1e-1;
    }

    StateDimVector running_x;
    double running_u;
    StateDimVector terminal_x;
  };

public:
  DDPProblemVerticalMotion(double dt,
                           const std::function<double(double)> & ref_pos_func,
                           const CostWeight & cost_weight = CostWeight())
  : DDPProblem(dt), ref_pos_func_(ref_pos_func), cost_weight_(cost_weight)
  {
  }

  using DDPProblem::inputDim;

  virtual int inputDim(double t) const override
  {
    // Add small values to avoid numerical instability at inequality bounds
    constexpr double epsilon_t = 1e-6;
    t += epsilon_t;
    if(4.5 < t && t < 5.0)
    {
      return 0;
    }
    else
    {
      return 1;
    }
  }

  virtual StateDimVector stateEq(double t, const StateDimVector & x, const InputDimVector & u) const override
  {
    StateDimVector x_dot;
    x_dot << x[1], u.sum() / mass_ - g_;
    return x + dt_ * x_dot;
  }

  virtual double runningCost(double t, const StateDimVector & x, const InputDimVector & u) const override
  {
    StateDimVector ref_x;
    ref_x << ref_pos_func_(t), 0;
    return 0.5 * cost_weight_.running_x.dot((x - ref_x).cwiseAbs2()) + 0.5 * cost_weight_.running_u * u.squaredNorm();
  }

  virtual double terminalCost(double t, const StateDimVector & x) const override
  {
    StateDimVector ref_x;
    ref_x << ref_pos_func_(t), 0;
    return 0.5 * cost_weight_.terminal_x.dot((x - ref_x).cwiseAbs2());
  }

  virtual void calcStatEqDeriv(double t,
                               const StateDimVector & x,
                               const InputDimVector & u,
                               Eigen::Ref<StateStateDimMatrix> state_eq_deriv_x,
                               Eigen::Ref<StateInputDimMatrix> state_eq_deriv_u) const override
  {
    state_eq_deriv_x << 0, 1, 0, 0;
    state_eq_deriv_x *= dt_;
    state_eq_deriv_x += StateStateDimMatrix::Identity();

    state_eq_deriv_u.row(0).setZero();
    state_eq_deriv_u.row(1).setConstant(1.0 / mass_);
    state_eq_deriv_u *= dt_;
  }

  virtual void calcStatEqDeriv(double t,
                               const StateDimVector & x,
                               const InputDimVector & u,
                               Eigen::Ref<StateStateDimMatrix> state_eq_deriv_x,
                               Eigen::Ref<StateInputDimMatrix> state_eq_deriv_u,
                               std::vector<StateStateDimMatrix> & state_eq_deriv_xx,
                               std::vector<InputInputDimMatrix> & state_eq_deriv_uu,
                               std::vector<StateInputDimMatrix> & state_eq_deriv_xu) const override
  {
    calcStatEqDeriv(t, x, u, state_eq_deriv_x, state_eq_deriv_u);

    if(state_eq_deriv_xx.size() != stateDim() || state_eq_deriv_uu.size() != stateDim()
       || state_eq_deriv_xu.size() != stateDim())
    {
      throw std::runtime_error("Vector size should be " + std::to_string(stateDim()) + " but "
                               + std::to_string(state_eq_deriv_xx.size()));
    }
    for(int i = 0; i < stateDim(); i++)
    {
      state_eq_deriv_xx[i].setZero();
      state_eq_deriv_uu[i].setZero();
      state_eq_deriv_xu[i].setZero();
    }
  }

  virtual void calcRunningCostDeriv(double t,
                                    const StateDimVector & x,
                                    const InputDimVector & u,
                                    Eigen::Ref<StateDimVector> running_cost_deriv_x,
                                    Eigen::Ref<InputDimVector> running_cost_deriv_u) const override
  {
    StateDimVector ref_x;
    ref_x << ref_pos_func_(t), 0;

    running_cost_deriv_x = cost_weight_.running_x.cwiseProduct(x - ref_x);
    running_cost_deriv_u = cost_weight_.running_u * u;
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
    StateDimVector ref_x;
    ref_x << ref_pos_func_(t), 0;

    running_cost_deriv_x = cost_weight_.running_x.cwiseProduct(x - ref_x);
    running_cost_deriv_u = cost_weight_.running_u * u;

    running_cost_deriv_xx = cost_weight_.running_x.asDiagonal();
    running_cost_deriv_uu.setIdentity();
    running_cost_deriv_uu *= cost_weight_.running_u;
    running_cost_deriv_xu.setZero();
  }

  virtual void calcTerminalCostDeriv(double t,
                                     const StateDimVector & x,
                                     Eigen::Ref<StateDimVector> terminal_cost_deriv_x) const override
  {
    StateDimVector ref_x;
    ref_x << ref_pos_func_(t), 0;

    terminal_cost_deriv_x = cost_weight_.terminal_x.cwiseProduct(x - ref_x);
  }

  virtual void calcTerminalCostDeriv(double t,
                                     const StateDimVector & x,
                                     Eigen::Ref<StateDimVector> terminal_cost_deriv_x,
                                     Eigen::Ref<StateStateDimMatrix> terminal_cost_deriv_xx) const override
  {
    StateDimVector ref_x;
    ref_x << ref_pos_func_(t), 0;

    terminal_cost_deriv_x = cost_weight_.terminal_x.cwiseProduct(x - ref_x);
    terminal_cost_deriv_xx = cost_weight_.terminal_x.asDiagonal();
  }

protected:
  static constexpr double g_ = 9.80665; // [m/s^2]
  std::function<double(double)> ref_pos_func_;
  CostWeight cost_weight_;
  double mass_ = 1.0; // [kg]
};

TEST(TestDDPVerticalMotion, TestCase1)
{
  double dt = 0.01; // [sec]
  double horizon_duration = 3.0; // [sec]
  int horizon_steps = static_cast<int>(horizon_duration / dt);
  double end_t = 10.0; // [sec]

  // Instantiate problem
  constexpr double epsilon_t = 1e-6;
  std::function<double(double)> ref_pos_func = [&](double t) {
    // Add small values to avoid numerical instability at inequality bounds
    t += epsilon_t;
    if(t < 8.0)
    {
      return 1.0; // [m]
    }
    else
    {
      return 1.5; // [m]
    }
  };
  auto ddp_problem = std::make_shared<DDPProblemVerticalMotion>(dt, ref_pos_func);

  // Instantiate solver
  auto ddp_solver = std::make_shared<NOC::DDPSolver<2, Eigen::Dynamic>>(ddp_problem);
  ddp_solver->config().horizon_steps = horizon_steps;

  // Initialize MPC
  double current_t = 0;
  DDPProblemVerticalMotion::StateDimVector current_x = DDPProblemVerticalMotion::StateDimVector(1.2, 0);
  std::vector<DDPProblemVerticalMotion::InputDimVector> current_u_list;
  for(int i = 0; i < horizon_steps; i++)
  {
    double t = current_t + i * dt;
    current_u_list.push_back(DDPProblemVerticalMotion::InputDimVector::Zero(ddp_problem->inputDim(t)));
  }

  // Run MPC loop
  bool first_iter = true;
  std::string file_path = "/tmp/TestDDPVerticalMotionResult.txt";
  std::ofstream ofs(file_path);
  ofs << "time pos vel force ref_pos iter" << std::endl;
  while(current_t < end_t)
  {
    // Solve
    ddp_solver->solve(current_t, current_x, current_u_list);
    if(first_iter)
    {
      first_iter = false;
      ddp_solver->dumpTraceDataList("/tmp/TestDDPVerticalMotionTraceData.txt");
    }

    // Check pos
    double planned_pos = ddp_solver->controlData().x_list[0][0];
    double ref_pos = ref_pos_func(current_t);
    EXPECT_LT(std::abs(planned_pos - ref_pos), 1.0);

    // Dump
    ofs << current_t << " " << ddp_solver->controlData().x_list[0].transpose() << " "
        << ddp_solver->controlData().u_list[0].sum() << " " << ref_pos << " " << ddp_solver->traceDataList().back().iter
        << std::endl;

    // Update to next step
    current_x = ddp_solver->controlData().x_list[1];
    current_u_list = ddp_solver->controlData().u_list;
    current_u_list.erase(current_u_list.begin());
    double terminal_t = current_t + horizon_steps * dt;
    int terminal_input_dim = ddp_problem->inputDim(terminal_t);
    if(current_u_list.back().size() == terminal_input_dim)
    {
      current_u_list.push_back(current_u_list.back());
    }
    else
    {
      current_u_list.push_back(DDPProblemVerticalMotion::InputDimVector::Zero(terminal_input_dim));
    }
    current_t += dt;
  }

  // Check final pos
  double ref_pos = ref_pos_func(current_t);
  EXPECT_LT(std::abs(current_x[0] - ref_pos), 1e-2);
  EXPECT_LT(std::abs(current_x[1]), 1e-2);

  std::cout << "Run the following commands in gnuplot:\n"
            << "  set key autotitle columnhead\n"
            << "  set key noenhanced\n"
            << "  plot \"" << file_path << "\" u 1:2 w lp, \"\" u 1:5 w l lw 3\n";
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
