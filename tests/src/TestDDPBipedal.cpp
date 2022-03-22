/* Author: Masaki Murooka */

#include <gtest/gtest.h>

#include <noc_ddp/DDP.h>

class DDPProblemBipedal : public NOC::DDPProblem<2, 1>
{
public:
  DDPProblemBipedal(double dt,
                    const std::function<double(double)> & ref_zmp_func,
                    const std::function<double(double)> & omega2_func,
                    double running_vel_scale = 1e-6,
                    double terminal_vel_scale = 1e-6)
  : DDPProblem(dt, 2, 1), ref_zmp_func_(ref_zmp_func), omega2_func_(omega2_func), running_vel_scale_(running_vel_scale),
    terminal_vel_scale_(terminal_vel_scale)
  {
  }

  virtual StateDimVector stateEq(double t, const StateDimVector & x, const InputDimVector & u) const override
  {
    return A(t) * x + B(t) * u;
  }

  virtual double runningCost(double t, const StateDimVector & x, const InputDimVector & u) const override
  {
    return running_vel_scale_ * 0.5 * std::pow(x[1], 2) + 0.5 * std::pow(u[0] - ref_zmp_func_(t), 2);
  }

  virtual double terminalCost(double t, const StateDimVector & x) const override
  {
    return terminal_vel_scale_ * 0.5 * std::pow(x[1], 2);
  }

  virtual void calcStateqDeriv(double t,
                               const StateDimVector & x,
                               const InputDimVector & u,
                               Eigen::Ref<StateStateDimMatrix> state_eq_deriv_x,
                               Eigen::Ref<StateInputDimMatrix> state_eq_deriv_u) const override
  {
    state_eq_deriv_x = A(t);
    state_eq_deriv_u = B(t);
  }

  virtual void calcStateqDeriv(double t,
                               const StateDimVector & x,
                               const InputDimVector & u,
                               Eigen::Ref<StateStateDimMatrix> state_eq_deriv_x,
                               Eigen::Ref<StateInputDimMatrix> state_eq_deriv_u,
                               std::vector<StateStateDimMatrix> & state_eq_deriv_xx,
                               std::vector<InputInputDimMatrix> & state_eq_deriv_uu,
                               std::vector<StateInputDimMatrix> & state_eq_deriv_xu) const override
  {
    calcStateqDeriv(t, x, u, state_eq_deriv_x, state_eq_deriv_u);
    state_eq_deriv_xx.assign(2, StateStateDimMatrix::Zero());
    state_eq_deriv_uu.assign(2, InputInputDimMatrix::Zero());
    state_eq_deriv_xu.assign(2, StateInputDimMatrix::Zero());
  }

  virtual void calcRunningCostDeriv(double t,
                                    const StateDimVector & x,
                                    const InputDimVector & u,
                                    Eigen::Ref<StateDimVector> running_cost_deriv_x,
                                    Eigen::Ref<InputDimVector> running_cost_deriv_u) const override
  {
    running_cost_deriv_x[0] = 0;
    running_cost_deriv_x[1] = running_vel_scale_ * x[1];
    running_cost_deriv_u[0] = u[0] - ref_zmp_func_(t);
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
    running_cost_deriv_xx << 0, 0, 0, running_vel_scale_;
    running_cost_deriv_uu << 1;
    running_cost_deriv_xu << 0, 0;
  }

  virtual void calcTerminalCostDeriv(double t,
                                     const StateDimVector & x,
                                     Eigen::Ref<StateDimVector> terminal_cost_deriv_x) const override
  {
    terminal_cost_deriv_x[0] = 0;
    terminal_cost_deriv_x[1] = terminal_vel_scale_ * x[1];
  }

  virtual void calcTerminalCostDeriv(double t,
                                     const StateDimVector & x,
                                     Eigen::Ref<StateDimVector> terminal_cost_deriv_x,
                                     Eigen::Ref<StateStateDimMatrix> terminal_cost_deriv_xx) const override
  {
    calcTerminalCostDeriv(t, x, terminal_cost_deriv_x);
    terminal_cost_deriv_xx << 0, 0, 0, terminal_vel_scale_;
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
  std::function<double(double)> ref_zmp_func_ = nullptr;
  std::function<double(double)> omega2_func_ = nullptr;

  double running_vel_scale_ = 1e-6;
  double terminal_vel_scale_ = 1e-6;
};

TEST(TestDDPBipedal, TestCase1)
{
  double dt = 0.01; // [sec]
  double horizon_duration = 2.0; // [sec]
  int horizon_steps = static_cast<int>(horizon_duration / dt);

  std::function<double(double)> ref_zmp_func = [](double t) {
    if(t < 1.0)
    {
      return 0.0;
    }
    else
    {
      return 0.3;
    }
  };
  std::function<double(double)> omega2_func = [](double t) {
    double cog_pos_z = 1.0;
    double cog_acc_z = 0.0;
    constexpr double g = 9.80665;
    return (cog_acc_z + g) / cog_pos_z;
  };
  auto ddp_problem = std::make_shared<DDPProblemBipedal>(dt, ref_zmp_func, omega2_func);

  auto ddp_solver = std::make_shared<NOC::DDPSolver<2, 1>>(ddp_problem);
  ddp_solver->config().horizon_steps = horizon_steps;

  double current_t = 0;
  DDPProblemBipedal::StateDimVector current_x = DDPProblemBipedal::StateDimVector(0, 0);
  std::vector<DDPProblemBipedal::InputDimVector> initial_u_list;
  initial_u_list.assign(horizon_steps, DDPProblemBipedal::InputDimVector::Zero());

  ddp_solver->solve(current_t, current_x, initial_u_list);

  ddp_solver->dumpTraceDataList("/tmp/TestDDPBipedal.txt");
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
