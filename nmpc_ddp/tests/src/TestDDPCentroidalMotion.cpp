/* Author: Masaki Murooka */

#include <gtest/gtest.h>

#include <fstream>
#include <iostream>

#include <nmpc_ddp/DDP.h>

/** \brief Calculate a matrix corresponding to the cross product. */
Eigen::Matrix3d crossMat(const Eigen::Vector3d & vec)
{
  Eigen::Matrix3d mat;
  mat << 0, -vec(2), vec(1), vec(2), 0, -vec(0), -vec(1), vec(0), 0;
  return mat;
}

/** \brief DDP problem for centroidal motion.

    State is [CoM_pos, linear_momentum, angular_momentum]. Input is [force_scale_1, ..., force_scale_N].
    Running cost is sum of the respective quadratic terms of state and input.
    Terminal cost is quadratic term of state.
 */
class DDPProblemCentroidalMotion : public nmpc_ddp::DDPProblem<9, Eigen::Dynamic>
{
public:
  struct StanceData
  {
    //! Contact vertices
    Eigen::Matrix3Xd vertices_mat;

    /** \brief Force direction (i.e., friction pyramid ridge)

        The number of columns is the same as that of vertices_mat.
    */
    Eigen::Matrix3Xd ridges_mat;
  };

  struct CostWeight
  {
    CostWeight()
    {
      running_x << Eigen::Vector3d::Constant(1.0), Eigen::Vector3d::Constant(0.0), Eigen::Vector3d::Constant(1.0);
      running_u = 1e-6;
      terminal_x << Eigen::Vector3d::Constant(1.0), Eigen::Vector3d::Constant(0.0), Eigen::Vector3d::Constant(1.0);
    }

    StateDimVector running_x;
    double running_u;
    StateDimVector terminal_x;
  };

public:
  DDPProblemCentroidalMotion(double dt,
                             const std::function<StanceData(double)> & ref_stance_func,
                             const std::function<Eigen::Vector3d(double)> & ref_pos_func,
                             const CostWeight & cost_weight = CostWeight())
  : DDPProblem(dt), ref_stance_func_(ref_stance_func), ref_pos_func_(ref_pos_func), cost_weight_(cost_weight)
  {
  }

  using DDPProblem::inputDim;

  virtual int inputDim(double t) const override
  {
    const StanceData & stance_data = ref_stance_func_(t);
    return stance_data.vertices_mat.cols();
  }

  virtual StateDimVector stateEq(double t, const StateDimVector & x, const InputDimVector & u) const override
  {
    const StanceData & stance_data = ref_stance_func_(t);
    const Eigen::Matrix3Xd & vertices_mat = stance_data.vertices_mat;
    const Eigen::Matrix3Xd & ridges_mat = stance_data.ridges_mat;

    const Eigen::Ref<const Eigen::Vector3d> & com = x.segment<3>(0);
    const Eigen::Ref<const Eigen::Vector3d> & linear_momentum = x.segment<3>(3);
    const Eigen::Ref<const Eigen::Vector3d> & angular_momentum = x.segment<3>(6);

    StateDimVector x_dot;
    Eigen::Ref<Eigen::Vector3d> com_dot = x_dot.segment<3>(0);
    Eigen::Ref<Eigen::Vector3d> linear_momentum_dot = x_dot.segment<3>(3);
    Eigen::Ref<Eigen::Vector3d> angular_momentum_dot = x_dot.segment<3>(6);
    com_dot = linear_momentum / mass_;
    linear_momentum_dot = ridges_mat * u - mass_ * g_;
    angular_momentum_dot.setZero();
    for(int i = 0; i < u.size(); i++)
    {
      angular_momentum_dot += u[i] * (vertices_mat.col(i) - com).cross(ridges_mat.col(i));
    }

    return x + dt_ * x_dot;
  }

  virtual double runningCost(double t, const StateDimVector & x, const InputDimVector & u) const override
  {
    StateDimVector x_diff;
    x_diff << x.head<3>() - ref_pos_func_(t), x.tail<6>();
    return 0.5 * cost_weight_.running_x.dot(x_diff.cwiseAbs2()) + 0.5 * cost_weight_.running_u * u.squaredNorm();
  }

  virtual double terminalCost(double t, const StateDimVector & x) const override
  {
    StateDimVector x_diff;
    x_diff << x.head<3>() - ref_pos_func_(t), x.tail<6>();
    return 0.5 * cost_weight_.terminal_x.dot(x_diff.cwiseAbs2());
  }

  virtual void calcStatEqDeriv(double t,
                               const StateDimVector & x,
                               const InputDimVector & u,
                               Eigen::Ref<StateStateDimMatrix> state_eq_deriv_x,
                               Eigen::Ref<StateInputDimMatrix> state_eq_deriv_u) const override
  {
    const StanceData & stance_data = ref_stance_func_(t);
    const Eigen::Matrix3Xd & vertices_mat = stance_data.vertices_mat;
    const Eigen::Matrix3Xd & ridges_mat = stance_data.ridges_mat;

    const Eigen::Ref<const Eigen::Vector3d> & com = x.segment<3>(0);
    const Eigen::Ref<const Eigen::Vector3d> & linear_momentum = x.segment<3>(3);
    const Eigen::Ref<const Eigen::Vector3d> & angular_momentum = x.segment<3>(6);

    state_eq_deriv_x.setZero();
    state_eq_deriv_x.block<3, 3>(0, 3).diagonal().setConstant(1 / mass_);
    state_eq_deriv_x.block<3, 3>(6, 0) = crossMat(ridges_mat * u);
    state_eq_deriv_x *= dt_;
    state_eq_deriv_x += StateStateDimMatrix::Identity();

    state_eq_deriv_u.setZero();
    state_eq_deriv_u.middleRows<3>(3) = ridges_mat;
    for(int i = 0; i < u.size(); i++)
    {
      state_eq_deriv_u.middleRows<3>(6).col(i) = (vertices_mat.col(i) - com).cross(ridges_mat.col(i));
    }
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
    StateDimVector x_diff;
    x_diff << x.head<3>() - ref_pos_func_(t), x.tail<6>();
    running_cost_deriv_x = cost_weight_.running_x.cwiseProduct(x_diff);
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
    calcRunningCostDeriv(t, x, u, running_cost_deriv_x, running_cost_deriv_u);

    running_cost_deriv_xx = cost_weight_.running_x.asDiagonal();
    running_cost_deriv_uu.setIdentity();
    running_cost_deriv_uu *= cost_weight_.running_u;
    running_cost_deriv_xu.setZero();
  }

  virtual void calcTerminalCostDeriv(double t,
                                     const StateDimVector & x,
                                     Eigen::Ref<StateDimVector> terminal_cost_deriv_x) const override
  {
    StateDimVector x_diff;
    x_diff << x.head<3>() - ref_pos_func_(t), x.tail<6>();
    terminal_cost_deriv_x = cost_weight_.terminal_x.cwiseProduct(x_diff);
  }

  virtual void calcTerminalCostDeriv(double t,
                                     const StateDimVector & x,
                                     Eigen::Ref<StateDimVector> terminal_cost_deriv_x,
                                     Eigen::Ref<StateStateDimMatrix> terminal_cost_deriv_xx) const override
  {
    calcTerminalCostDeriv(t, x, terminal_cost_deriv_x);
    terminal_cost_deriv_xx = cost_weight_.terminal_x.asDiagonal();
  }

protected:
  const Eigen::Vector3d g_ = Eigen::Vector3d(0, 0, 9.80665); // [m/s^2]
  std::function<StanceData(double)> ref_stance_func_;
  std::function<Eigen::Vector3d(double)> ref_pos_func_;
  CostWeight cost_weight_;
  double mass_ = 100.0; // [kg]
};

void checkDerivatives(const std::shared_ptr<DDPProblemCentroidalMotion> & ddp_problem)
{
  double t = 0;
  int state_dim = ddp_problem->stateDim();
  int input_dim = ddp_problem->inputDim(t);

  DDPProblemCentroidalMotion::StateDimVector x(state_dim);
  x << 1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0;
  DDPProblemCentroidalMotion::InputDimVector u(input_dim);
  u << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0;

  DDPProblemCentroidalMotion::StateStateDimMatrix state_eq_deriv_x_analytical(state_dim, state_dim);
  DDPProblemCentroidalMotion::StateInputDimMatrix state_eq_deriv_u_analytical(state_dim, input_dim);
  ddp_problem->calcStatEqDeriv(t, x, u, state_eq_deriv_x_analytical, state_eq_deriv_u_analytical);

  DDPProblemCentroidalMotion::StateStateDimMatrix state_eq_deriv_x_numerical(state_dim, state_dim);
  DDPProblemCentroidalMotion::StateInputDimMatrix state_eq_deriv_u_numerical(state_dim, input_dim);
  constexpr double deriv_eps = 1e-6;
  for(int i = 0; i < state_dim; i++)
  {
    state_eq_deriv_x_numerical.col(i) =
        (ddp_problem->stateEq(t, x + deriv_eps * DDPProblemCentroidalMotion::StateDimVector::Unit(state_dim, i), u)
         - ddp_problem->stateEq(t, x - deriv_eps * DDPProblemCentroidalMotion::StateDimVector::Unit(state_dim, i), u))
        / (2 * deriv_eps);
  }
  for(int i = 0; i < input_dim; i++)
  {
    state_eq_deriv_u_numerical.col(i) =
        (ddp_problem->stateEq(t, x, u + deriv_eps * DDPProblemCentroidalMotion::InputDimVector::Unit(input_dim, i))
         - ddp_problem->stateEq(t, x, u - deriv_eps * DDPProblemCentroidalMotion::InputDimVector::Unit(input_dim, i)))
        / (2 * deriv_eps);
  }

  EXPECT_LT((state_eq_deriv_x_analytical - state_eq_deriv_x_numerical).norm(), 1e-6);
  EXPECT_LT((state_eq_deriv_u_analytical - state_eq_deriv_u_numerical).norm(), 1e-6);
}

DDPProblemCentroidalMotion::StanceData makeStanceDataFromRect(const std::array<Eigen::Vector2d, 2> & rect_min_max)
{
  std::vector<Eigen::Vector3d> vertex_list(4);
  vertex_list[0] << rect_min_max[0], 0.0;
  vertex_list[1] << rect_min_max[0][0], rect_min_max[1][1], 0.0;
  vertex_list[2] << rect_min_max[1], 0.0;
  vertex_list[3] << rect_min_max[1][0], rect_min_max[0][1], 0.0;

  std::vector<Eigen::Vector3d> ridge_list(4);
  for(int i = 0; i < 4; i++)
  {
    double theta = 2 * M_PI * (static_cast<double>(i) / 4);
    ridge_list[i] << 0.5 * std::cos(theta), 0.5 * std::sin(theta), 1;
    ridge_list[i].normalize();
  }

  DDPProblemCentroidalMotion::StanceData stance_data;
  stance_data.vertices_mat.resize(3, 16);
  stance_data.ridges_mat.resize(3, 16);
  int col_idx = 0;
  for(const auto & vertex : vertex_list)
  {
    for(const auto & ridge : ridge_list)
    {
      stance_data.vertices_mat.col(col_idx) = vertex;
      stance_data.ridges_mat.col(col_idx) = ridge;
      col_idx++;
    }
  }

  return stance_data;
}

TEST(TestDDPCentroidalMotion, TestCase1)
{
  double dt = 0.03; // [sec]
  double horizon_duration = 3.0; // [sec]
  int horizon_steps = static_cast<int>(horizon_duration / dt);
  double end_t = 3.0; // [sec]

  // Instantiate problem
  constexpr double epsilon_t = 1e-6;
  std::function<DDPProblemCentroidalMotion::StanceData(double)> ref_stance_func = [&](double t) {
    // Add small values to avoid numerical instability at inequality bounds
    t += epsilon_t;
    if(t < 1.4)
    {
      return makeStanceDataFromRect({Eigen::Vector2d(-0.1, -0.1), Eigen::Vector2d(0.1, 0.1)});
    }
    else if(t < 1.6)
    {
      DDPProblemCentroidalMotion::StanceData stance_data;
      stance_data.vertices_mat.setZero(3, 0);
      stance_data.ridges_mat.setZero(3, 0);
      return stance_data;
    }
    else
    {
      return makeStanceDataFromRect({Eigen::Vector2d(0.4, -0.1), Eigen::Vector2d(0.6, 0.1)});
    }
  };
  std::function<Eigen::Vector3d(double)> ref_pos_func = [&](double t) {
    // Add small values to avoid numerical instability at inequality bounds
    t += epsilon_t;
    if(t < 1.5)
    {
      return Eigen::Vector3d(0.0, 0.0, 1.0); // [m]
    }
    else
    {
      return Eigen::Vector3d(0.5, 0.0, 1.0); // [m]
    }
  };
  auto ddp_problem = std::make_shared<DDPProblemCentroidalMotion>(dt, ref_stance_func, ref_pos_func);

  // Check derivatives
  checkDerivatives(ddp_problem);

  // Instantiate solver
  auto ddp_solver = std::make_shared<nmpc_ddp::DDPSolver<9, Eigen::Dynamic>>(ddp_problem);
  ddp_solver->config().horizon_steps = horizon_steps;

  // Initialize MPC
  double current_t = 0;
  DDPProblemCentroidalMotion::StateDimVector current_x;
  current_x << Eigen::Vector3d(0.0, 0.0, 1.0), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();
  std::vector<DDPProblemCentroidalMotion::InputDimVector> current_u_list;
  for(int i = 0; i < horizon_steps; i++)
  {
    double t = current_t + i * dt;
    current_u_list.push_back(DDPProblemCentroidalMotion::InputDimVector::Zero(ddp_problem->inputDim(t)));
  }

  // Run MPC loop
  bool first_iter = true;
  std::string file_path = "/tmp/TestDDPCentroidalMotionResult.txt";
  std::ofstream ofs(file_path);
  ofs << "time pos_x pos_y pos_z linear_momentum_x linear_momentum_y linear_momentum_z angular_momentum_x "
         "angular_momentum_y angular_momentum_z force_x force_y force_z ref_pos_x ref_pos_y ref_pos_z iter"
      << std::endl;
  while(current_t < end_t)
  {
    // Solve
    ddp_solver->solve(current_t, current_x, current_u_list);
    if(first_iter)
    {
      first_iter = false;
      ddp_solver->config().max_iter = 3;
      ddp_solver->dumpTraceDataList("/tmp/TestDDPCentroidalMotionTraceData.txt");
    }

    // Check pos
    const Eigen::Vector3d & planned_pos = ddp_solver->controlData().x_list[0].head<3>();
    const Eigen::Vector3d & ref_pos = ref_pos_func(current_t);
    EXPECT_LT((planned_pos - ref_pos).norm(), 1.0);

    // Dump
    const DDPProblemCentroidalMotion::StanceData & ref_stance = ref_stance_func(current_t);
    ddp_solver->controlData().u_list[0];
    ofs << current_t << " " << ddp_solver->controlData().x_list[0].transpose() << " "
        << (ref_stance.ridges_mat * ddp_solver->controlData().u_list[0]).transpose() << " " << ref_pos.transpose()
        << " " << ddp_solver->traceDataList().back().iter << std::endl;

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
      current_u_list.push_back(DDPProblemCentroidalMotion::InputDimVector::Zero(terminal_input_dim));
    }
    current_t += dt;
  }

  // Check final state
  EXPECT_LT((current_x.head<3>() - ref_pos_func(current_t)).norm(), 1e-2);
  EXPECT_LT(current_x.tail<6>().norm(), 1.0);

  std::cout << "Run the following commands in gnuplot:\n"
            << "  set key autotitle columnhead\n"
            << "  set key noenhanced\n"
            << "  plot \"" << file_path << "\" u 1:2 w lp, \"\" u 1:14 w l lw 3 # pos_x\n"
            << "  plot \"" << file_path << "\" u 1:3 w lp, \"\" u 1:15 w l lw 3 # pos_y\n"
            << "  plot \"" << file_path << "\" u 1:4 w lp, \"\" u 1:16 w l lw 3 # pos_z\n"
            << "  plot \"" << file_path << "\" u 1:9 w lp # angular_momentum_y\n"
            << "  plot \"" << file_path << "\" u 1:13 w lp # force_z\n"
            << "  plot \"" << file_path << "\" u 1:17 w lp # iter\n";
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
