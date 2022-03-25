/* Author: Masaki Murooka */

#include <gtest/gtest.h>

#include <fstream>
#include <iostream>

#include <noc_ddp/DDP.h>

/** \brief DDP problem for bipedal walking.

    State is [CoM_pos, CoM_vel]. Input is [ZMP].
    Running cost is CoM_vel^2 + (ZMP - ZMP_ref)^2.
    Terminal cost is (CoM_pos - ZMP_ref)^2 + CoM_vel^2.
 */
class DDPProblemBipedal : public NOC::DDPProblem<2, 1>
{
public:
  struct CostWeight
  {
    CostWeight() {}

    double running_vel = 1e-14;
    double running_zmp = 1e-1;
    double terminal_pos = 1e2;
    double terminal_vel = 1.0;
  };

public:
  DDPProblemBipedal(double dt,
                    const std::function<double(double)> & ref_zmp_func,
                    const std::function<double(double)> & omega2_func,
                    const CostWeight & cost_weight = CostWeight())
  : DDPProblem(dt, 2, 1), ref_zmp_func_(ref_zmp_func), omega2_func_(omega2_func), cost_weight_(cost_weight)
  {
  }

  virtual StateDimVector stateEq(double t, const StateDimVector & x, const InputDimVector & u) const override
  {
    return A(t) * x + B(t) * u;
  }

  virtual double runningCost(double t, const StateDimVector & x, const InputDimVector & u) const override
  {
    return cost_weight_.running_vel * 0.5 * std::pow(x[1], 2)
           + cost_weight_.running_zmp * 0.5 * std::pow(u[0] - ref_zmp_func_(t), 2);
  }

  virtual double terminalCost(double t, const StateDimVector & x) const override
  {
    return cost_weight_.terminal_pos * 0.5 * std::pow(x[0] - ref_zmp_func_(t), 2)
           + cost_weight_.terminal_vel * 0.5 * std::pow(x[1], 2);
  }

  virtual void calcStatEqDeriv(double t,
                               const StateDimVector & x,
                               const InputDimVector & u,
                               Eigen::Ref<StateStateDimMatrix> state_eq_deriv_x,
                               Eigen::Ref<StateInputDimMatrix> state_eq_deriv_u) const override
  {
    state_eq_deriv_x = A(t);
    state_eq_deriv_u = B(t);
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
    state_eq_deriv_xx.assign(stateDim(), StateStateDimMatrix::Zero());
    state_eq_deriv_uu.assign(stateDim(), InputInputDimMatrix::Zero());
    state_eq_deriv_xu.assign(stateDim(), StateInputDimMatrix::Zero());
  }

  virtual void calcRunningCostDeriv(double t,
                                    const StateDimVector & x,
                                    const InputDimVector & u,
                                    Eigen::Ref<StateDimVector> running_cost_deriv_x,
                                    Eigen::Ref<InputDimVector> running_cost_deriv_u) const override
  {
    running_cost_deriv_x[0] = 0;
    running_cost_deriv_x[1] = cost_weight_.running_vel * x[1];
    running_cost_deriv_u[0] = cost_weight_.running_zmp * (u[0] - ref_zmp_func_(t));
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
    running_cost_deriv_xx << 0, 0, 0, cost_weight_.running_vel;
    running_cost_deriv_uu << cost_weight_.running_zmp;
    running_cost_deriv_xu << 0, 0;
  }

  virtual void calcTerminalCostDeriv(double t,
                                     const StateDimVector & x,
                                     Eigen::Ref<StateDimVector> terminal_cost_deriv_x) const override
  {
    terminal_cost_deriv_x[0] = cost_weight_.terminal_pos * (x[0] - ref_zmp_func_(t));
    terminal_cost_deriv_x[1] = cost_weight_.terminal_vel * x[1];
  }

  virtual void calcTerminalCostDeriv(double t,
                                     const StateDimVector & x,
                                     Eigen::Ref<StateDimVector> terminal_cost_deriv_x,
                                     Eigen::Ref<StateStateDimMatrix> terminal_cost_deriv_xx) const override
  {
    calcTerminalCostDeriv(t, x, terminal_cost_deriv_x);
    terminal_cost_deriv_xx << cost_weight_.terminal_pos, 0, 0, cost_weight_.terminal_vel;
  }

protected:
  StateStateDimMatrix A(double t) const
  {
    StateStateDimMatrix A;
    double omega2 = omega2_func_(t);
    A << 1 + 0.5 * dt_ * dt_ * omega2, dt_, dt_ * omega2, 1;
    return A;
  }

  StateInputDimMatrix B(double t) const
  {
    StateInputDimMatrix B;
    double omega2 = omega2_func_(t);
    B << -0.5 * dt_ * dt_ * omega2, -1 * dt_ * omega2;
    return B;
  }

protected:
  std::function<double(double)> ref_zmp_func_;
  std::function<double(double)> omega2_func_;
  CostWeight cost_weight_;
};

TEST(TestDDPBipedal, TestCase1)
{
  double dt = 0.01; // [sec]
  double horizon_duration = 3.0; // [sec]
  int horizon_steps = static_cast<int>(horizon_duration / dt);
  double end_t = 20.0; // [sec]

  // Instantiate problem
  constexpr double epsilon_t = 1e-6;
  std::function<double(double)> ref_zmp_func = [&](double t) {
    // Add small values to avoid numerical instability at inequality bounds
    t += epsilon_t;
    if(t <= 1.5 || t >= end_t - 1.5)
    {
      return 0.0;
    }
    else
    {
      if(static_cast<int>(std::floor((t - 1.0) / 1.0)) % 2 == 0)
      {
        return 0.15; // [m]
      }
      else
      {
        return -0.15; // [m]
      }
    }
  };
  std::function<double(double)> omega2_func = [](double t) {
    // Add small values to avoid numerical instability at inequality bounds
    t += epsilon_t;
    double cog_pos_z_high = 1.0; // [m]
    double cog_pos_z_low = 0.3; // [m]
    double cog_pos_z = 0.0;
    double cog_acc_z = 0.0;
    if(t < 7.0)
    {
      cog_pos_z = cog_pos_z_high;
    }
    else if(t < 8.0)
    {
      double theta = M_PI * (t - 7.0) + M_PI;
      double scale = (cog_pos_z_low - cog_pos_z_high) / 2.0;
      cog_pos_z = scale * (std::cos(theta) + 1.0) + cog_pos_z_high;
      cog_acc_z = -1 * std::pow(M_PI, 2) * scale * std::cos(theta);
    }
    else if(t < 12.0)
    {
      cog_pos_z = cog_pos_z_low;
    }
    else if(t < 13.0)
    {
      double theta = M_PI * (t - 12.0) + M_PI;
      double scale = (cog_pos_z_high - cog_pos_z_low) / 2.0;
      cog_pos_z = scale * (std::cos(theta) + 1.0) + cog_pos_z_low;
      cog_acc_z = -1 * std::pow(M_PI, 2) * scale * std::cos(theta);
    }
    else
    {
      cog_pos_z = cog_pos_z_high;
    }
    constexpr double g = 9.80665;
    return (cog_acc_z + g) / cog_pos_z;
  };
  auto ddp_problem = std::make_shared<DDPProblemBipedal>(dt, ref_zmp_func, omega2_func);

  // Instantiate solver
  auto ddp_solver = std::make_shared<NOC::DDPSolver<2, 1>>(ddp_problem);
  ddp_solver->config().horizon_steps = horizon_steps;

  // Initialize MPC
  double current_t = 0;
  DDPProblemBipedal::StateDimVector current_x = DDPProblemBipedal::StateDimVector(0, 0);
  std::vector<DDPProblemBipedal::InputDimVector> current_u_list;
  current_u_list.assign(horizon_steps, DDPProblemBipedal::InputDimVector::Zero());

  // Run MPC loop
  bool first_iter = true;
  std::string file_path = "/tmp/TestDDPBipedalResult.txt";
  std::ofstream ofs(file_path);
  ofs << "time com_pos com_vel planned_zmp ref_zmp omega2 iter" << std::endl;
  while(current_t < end_t)
  {
    // Solve
    ddp_solver->solve(current_t, current_x, current_u_list);
    if(first_iter)
    {
      first_iter = false;
      ddp_solver->dumpTraceDataList("/tmp/TestDDPBipedalTraceData.txt");
    }

    // Check ZMP
    double planned_zmp = ddp_solver->controlData().u_list[0][0];
    double ref_zmp = ref_zmp_func(current_t);
    EXPECT_LT(std::abs(planned_zmp - ref_zmp), 1e-2);

    // Dump
    ofs << current_t << " " << ddp_solver->controlData().x_list[0].transpose() << " " << planned_zmp << " " << ref_zmp
        << " " << omega2_func(current_t) << " " << ddp_solver->traceDataList().back().iter << std::endl;

    // Update to next step
    current_t += dt;
    current_x = ddp_solver->controlData().x_list[1];
    current_u_list = ddp_solver->controlData().u_list;
    current_u_list.erase(current_u_list.begin());
    current_u_list.push_back(current_u_list.back());
  }

  // Check final CoM
  double ref_zmp = ref_zmp_func(current_t);
  EXPECT_LT(std::abs(current_x[0] - ref_zmp), 1e-2);
  EXPECT_LT(std::abs(current_x[1]), 1e-2);

  std::cout << "Run the following commands in gnuplot:\n"
            << "  set key autotitle columnhead\n"
            << "  set key noenhanced\n"
            << "  plot \"" << file_path << "\" u 1:2 w lp, \"\" u 1:4 w lp, \"\" u 1:5 w l lw 3\n";
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
