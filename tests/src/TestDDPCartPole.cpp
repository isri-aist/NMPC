/* Author: Masaki Murooka */

#include <gtest/gtest.h>

#include <fstream>
#include <iostream>

#include <noc_ddp/DDP.h>

/** \brief DDP problem for cart-pole.

    State is [pos, theta, vel, omega]. Input is [force].
    Running cost is sum of the respective quadratic terms of state and input.
    Terminal cost is quadratic term of state.
 */
class DDPProblemCartPole : public NOC::DDPProblem<4, 1>
{
public:
  struct CostWeight
  {
    CostWeight()
    {
      running_x << 1e-1, 1.0, 1e-2, 1e-1;
      running_u << 1e-4;
      terminal_x << 1e-1, 1.0, 1e-2, 1e-1;
    }

    StateDimVector running_x;
    InputDimVector running_u;
    StateDimVector terminal_x;
  };

public:
  DDPProblemCartPole(double dt,
                     const std::function<double(double)> & ref_pos_func,
                     const CostWeight & cost_weight = CostWeight())
  : DDPProblem(dt, 4, 1), ref_pos_func_(ref_pos_func), cost_weight_(cost_weight)
  {
  }

  virtual StateDimVector stateEq(double t, const StateDimVector & x, const InputDimVector & u) const override
  {
    double pos = x[0];
    double theta = x[1];
    double vel = x[2];
    double omega = x[3];
    double f = u[0];

    double sin_theta = std::sin(theta);
    double cos_theta = std::cos(theta);
    double omega2 = std::pow(omega, 2);
    double denom = cart_mass_ + pole_mass_ * std::pow(sin_theta, 2);

    StateDimVector x_dot;
    // clang-format off
    x_dot[0] = vel;
    x_dot[1] = omega;
    x_dot[2] = (f - pole_mass_ * pole_length_ * omega2 * sin_theta + pole_mass_ * g_ * sin_theta * cos_theta) / denom;
    x_dot[3] = (f * cos_theta - pole_mass_ * pole_length_ * omega2 * sin_theta * cos_theta
                + g_ * (cart_mass_ + pole_mass_) * sin_theta) / (pole_length_ * denom);
    // clang-format on

    return x + dt_ * x_dot;
  }

  virtual double runningCost(double t, const StateDimVector & x, const InputDimVector & u) const override
  {
    StateDimVector ref_x;
    ref_x << ref_pos_func_(t), 0, 0, 0;
    return 0.5 * cost_weight_.running_x.dot((x - ref_x).cwiseAbs2()) + 0.5 * cost_weight_.running_u.dot(u.cwiseAbs2());
  }

  virtual double terminalCost(double t, const StateDimVector & x) const override
  {
    StateDimVector ref_x;
    ref_x << ref_pos_func_(t), 0, 0, 0;
    return 0.5 * cost_weight_.terminal_x.dot((x - ref_x).cwiseAbs2());
  }

  virtual void calcStatEqDeriv(double t,
                               const StateDimVector & x,
                               const InputDimVector & u,
                               Eigen::Ref<StateStateDimMatrix> state_eq_deriv_x,
                               Eigen::Ref<StateInputDimMatrix> state_eq_deriv_u) const override
  {
    double pos = x[0];
    double theta = x[1];
    double vel = x[2];
    double omega = x[3];
    double f = u[0];

    double sin_theta = std::sin(theta);
    double cos_theta = std::cos(theta);
    double omega2 = std::pow(omega, 2);
    double denom = cart_mass_ + pole_mass_ * std::pow(sin_theta, 2);

    state_eq_deriv_x.setZero();
    // clang-format off
    state_eq_deriv_x(0, 2) = 1;
    state_eq_deriv_x(1, 3) = 1;
    state_eq_deriv_x(2, 1) = ((-1 * pole_mass_ * pole_length_ * omega2 * cos_theta
                               + pole_mass_ * g_ * (1 - 2 * std::pow(sin_theta, 2))) * denom
                              + -1 * (f - pole_mass_ * pole_length_ * omega2 * sin_theta + pole_mass_ * g_ * sin_theta * cos_theta)
                              * (2 * pole_mass_ * sin_theta * cos_theta))
        / std::pow(denom, 2);
    state_eq_deriv_x(2, 3) = (-2 * pole_mass_ * pole_length_ * omega * sin_theta) / denom;
    state_eq_deriv_x(3, 1) = ((-1 * f * sin_theta + -1 * pole_mass_ * pole_length_ * omega2 * (1 - 2 * std::pow(sin_theta, 2))
                               + g_ * (cart_mass_ + pole_mass_) * cos_theta) * denom
                              + -1 * (f * cos_theta - pole_mass_ * pole_length_ * omega2 * sin_theta * cos_theta
                                      + g_ * (cart_mass_ + pole_mass_) * sin_theta) * (2 * pole_mass_ * sin_theta * cos_theta))
        / (pole_length_ * std::pow(denom, 2));
    state_eq_deriv_x(3, 3) = (-2 * pole_mass_ * pole_length_ * omega * sin_theta * cos_theta) / (pole_length_ * denom);
    // clang-format on
    state_eq_deriv_x *= dt_;
    state_eq_deriv_x += StateStateDimMatrix::Identity();

    state_eq_deriv_u.setZero();
    state_eq_deriv_u[2] = 1 / denom;
    state_eq_deriv_u[3] = cos_theta / (pole_length_ * denom);
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
    throw std::runtime_error("Second-order derivatives of state equation are not implemented.");
  }

  virtual void calcRunningCostDeriv(double t,
                                    const StateDimVector & x,
                                    const InputDimVector & u,
                                    Eigen::Ref<StateDimVector> running_cost_deriv_x,
                                    Eigen::Ref<InputDimVector> running_cost_deriv_u) const override
  {
    StateDimVector ref_x;
    ref_x << ref_pos_func_(t), 0, 0, 0;

    running_cost_deriv_x = cost_weight_.running_x.cwiseProduct(x - ref_x);
    running_cost_deriv_u = cost_weight_.running_u.cwiseProduct(u);
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
    ref_x << ref_pos_func_(t), 0, 0, 0;

    running_cost_deriv_x = cost_weight_.running_x.cwiseProduct(x - ref_x);
    running_cost_deriv_u = cost_weight_.running_u.cwiseProduct(u);

    running_cost_deriv_xx = cost_weight_.running_x.asDiagonal();
    running_cost_deriv_uu = cost_weight_.running_u.asDiagonal();
    running_cost_deriv_xu.setZero();
  }

  virtual void calcTerminalCostDeriv(double t,
                                     const StateDimVector & x,
                                     Eigen::Ref<StateDimVector> terminal_cost_deriv_x) const override
  {
    StateDimVector ref_x;
    ref_x << ref_pos_func_(t), 0, 0, 0;

    terminal_cost_deriv_x = cost_weight_.terminal_x.cwiseProduct(x - ref_x);
  }

  virtual void calcTerminalCostDeriv(double t,
                                     const StateDimVector & x,
                                     Eigen::Ref<StateDimVector> terminal_cost_deriv_x,
                                     Eigen::Ref<StateStateDimMatrix> terminal_cost_deriv_xx) const override
  {
    StateDimVector ref_x;
    ref_x << ref_pos_func_(t), 0, 0, 0;

    terminal_cost_deriv_x = cost_weight_.terminal_x.cwiseProduct(x - ref_x);
    terminal_cost_deriv_xx = cost_weight_.terminal_x.asDiagonal();
  }

protected:
  static constexpr double g_ = 9.80665; // [m/s^2]

  std::function<double(double)> ref_pos_func_;

  CostWeight cost_weight_;

  double cart_mass_ = 1.0; // [kg]
  double pole_mass_ = 0.5; // [kg]
  double pole_length_ = 1.0; // [m]
};

TEST(TestDDPCartPole, TestCase1)
{
  double dt = 0.01; // [sec]
  double horizon_duration = 2.0; // [sec]
  int horizon_steps = static_cast<int>(horizon_duration / dt);
  double end_t = 10.0; // [sec]

  // Instantiate problem
  constexpr double epsilon_t = 1e-6;
  std::function<double(double)> ref_pos_func = [&](double t) {
    // Add small values to avoid numerical instability at inequality bounds
    t += epsilon_t;
    if(t <= 6.0)
    {
      return 0.0;
    }
    else
    {
      return 0.5;
    }
  };
  auto ddp_problem = std::make_shared<DDPProblemCartPole>(dt, ref_pos_func);

  // Instantiate solver
  auto ddp_solver = std::make_shared<NOC::DDPSolver<4, 1>>(ddp_problem);
  ddp_solver->config().horizon_steps = horizon_steps;
  ddp_solver->config().max_iter = 3;

  // Initialize MPC
  double current_t = 0;
  DDPProblemCartPole::StateDimVector current_x;
  current_x << 0, M_PI, 0, 0;
  std::vector<DDPProblemCartPole::InputDimVector> current_u_list;
  current_u_list.assign(horizon_steps, DDPProblemCartPole::InputDimVector::Zero());

  // Run MPC loop
  bool first_iter = true;
  std::string file_path = "/tmp/TestDDPCartPoleResult.txt";
  std::ofstream ofs(file_path);
  ofs << "time pos theta vel omega force ref_pos iter" << std::endl;
  while(current_t < end_t)
  {
    // Solve
    ddp_solver->solve(current_t, current_x, current_u_list);
    if(first_iter)
    {
      first_iter = false;
      ddp_solver->dumpTraceDataList("/tmp/TestDDPCartPoleTraceData.txt");
    }

    // Check pos
    double planned_pos = ddp_solver->controlData().x_list[0][0];
    double ref_pos = ref_pos_func(current_t);
    EXPECT_LT(std::abs(planned_pos - ref_pos), 1e1);

    // Dump
    ofs << current_t << " " << ddp_solver->controlData().x_list[0].transpose() << " "
        << ddp_solver->controlData().u_list[0].transpose() << " " << ref_pos << " "
        << ddp_solver->traceDataList().back().iter << std::endl;

    // Update to next step
    current_t += dt;
    current_x = ddp_solver->controlData().x_list[1];
    current_u_list = ddp_solver->controlData().u_list;
    current_u_list.erase(current_u_list.begin());
    current_u_list.push_back(current_u_list.back());
  }

  // Check final pos
  double ref_pos = ref_pos_func(current_t);
  EXPECT_LT(std::abs(current_x[0] - ref_pos), 1e-2);
  EXPECT_LT(std::abs(current_x[1]), 1e-2);
  EXPECT_LT(std::abs(current_x[2]), 1e-2);
  EXPECT_LT(std::abs(current_x[3]), 1e-2);

  std::cout << "Run the following commands in gnuplot:\n"
            << "  set key autotitle columnhead\n"
            << "  set key noenhanced\n"
            << "  plot \"" << file_path << "\" u 1:2 w lp, \"\" u 1:3 w lp, \"\" u 1:7 w l lw 3\n";
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
