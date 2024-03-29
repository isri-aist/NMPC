/* Author: Masaki Murooka */

#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>

#include <nmpc_fmpc/MathUtils.h>

#define CHECK_NAN(VAR, PRINT_PREFIX)                                                                      \
  if(VAR.array().isNaN().any() || VAR.array().isInf().any())                                              \
  {                                                                                                       \
    if(print_level >= 3)                                                                                  \
    {                                                                                                     \
      std::cout << PRINT_PREFIX << #VAR << " contains NaN or infinity: " << VAR.transpose() << std::endl; \
    }                                                                                                     \
    return true;                                                                                          \
  }

namespace
{
template<class Clock>
double calcDuration(const std::chrono::time_point<Clock> & start_time, const std::chrono::time_point<Clock> & end_time)
{
  return 1e3 * std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
}
} // namespace

namespace nmpc_fmpc
{
template<int StateDim, int InputDim, int IneqDim>
FmpcSolver<StateDim, InputDim, IneqDim>::Variable::Variable(int _horizon_steps) : horizon_steps(_horizon_steps)
{
  x_list.resize(horizon_steps + 1);
  u_list.resize(horizon_steps);
  lambda_list.resize(horizon_steps + 1);
  s_list.resize(horizon_steps);
  nu_list.resize(horizon_steps);
}

template<int StateDim, int InputDim, int IneqDim>
void FmpcSolver<StateDim, InputDim, IneqDim>::Variable::reset(double _x,
                                                              double _u,
                                                              double _lambda,
                                                              double _s,
                                                              double _nu)
{
  for(auto & x : x_list)
  {
    x.setConstant(_x);
  }
  for(auto & u : u_list)
  {
    u.setConstant(_u);
  }
  for(auto & lambda : lambda_list)
  {
    lambda.setConstant(_lambda);
  }
  for(auto & s : s_list)
  {
    s.setConstant(_s);
  }
  for(auto & nu : nu_list)
  {
    nu.setConstant(_nu);
  }
}

template<int StateDim, int InputDim, int IneqDim>
bool FmpcSolver<StateDim, InputDim, IneqDim>::Variable::containsNaN() const
{
  for(auto & x : x_list)
  {
    CHECK_NAN(x, "[FMPC/Variable] ");
  }
  for(auto & u : u_list)
  {
    CHECK_NAN(u, "[FMPC/Variable] ");
  }
  for(auto & lambda : lambda_list)
  {
    CHECK_NAN(lambda, "[FMPC/Variable] ");
  }
  for(auto & s : s_list)
  {
    CHECK_NAN(s, "[FMPC/Variable] ");
  }
  for(auto & nu : nu_list)
  {
    CHECK_NAN(nu, "[FMPC/Variable] ");
  }

  return false;
}

template<int StateDim, int InputDim, int IneqDim>
FmpcSolver<StateDim, InputDim, IneqDim>::Coefficient::Coefficient(int state_dim, int input_dim, int ineq_dim)
{
  A.resize(state_dim, state_dim);
  B.resize(state_dim, input_dim);
  C.resize(ineq_dim, state_dim);
  D.resize(ineq_dim, input_dim);

  Lx.resize(state_dim);
  Lu.resize(input_dim);
  Lxx.resize(state_dim, state_dim);
  Luu.resize(input_dim, input_dim);
  Lxu.resize(state_dim, input_dim);

  x_bar.resize(state_dim);
  g_bar.resize(ineq_dim);
  Lx_bar.resize(state_dim);
  Lu_bar.resize(input_dim);

  k.resize(input_dim);
  K.resize(input_dim, state_dim);
  s.resize(state_dim);
  P.resize(state_dim, state_dim);
}

template<int StateDim, int InputDim, int IneqDim>
FmpcSolver<StateDim, InputDim, IneqDim>::Coefficient::Coefficient(int state_dim)
{
  Lx.resize(state_dim);
  Lxx.resize(state_dim, state_dim);
  Lx_bar.resize(state_dim);

  s.resize(state_dim);
  P.resize(state_dim, state_dim);
}

template<int StateDim, int InputDim, int IneqDim>
bool FmpcSolver<StateDim, InputDim, IneqDim>::Coefficient::containsNaN() const
{
  CHECK_NAN(A, "[FMPC/Coefficient] ");
  CHECK_NAN(B, "[FMPC/Coefficient] ");
  CHECK_NAN(C, "[FMPC/Coefficient] ");
  CHECK_NAN(D, "[FMPC/Coefficient] ");
  CHECK_NAN(Lx, "[FMPC/Coefficient] ");
  CHECK_NAN(Lu, "[FMPC/Coefficient] ");
  CHECK_NAN(Lxx, "[FMPC/Coefficient] ");
  CHECK_NAN(Luu, "[FMPC/Coefficient] ");
  CHECK_NAN(Lxu, "[FMPC/Coefficient] ");
  CHECK_NAN(x_bar, "[FMPC/Coefficient] ");
  CHECK_NAN(g_bar, "[FMPC/Coefficient] ");
  CHECK_NAN(Lx_bar, "[FMPC/Coefficient] ");
  CHECK_NAN(Lu_bar, "[FMPC/Coefficient] ");
  CHECK_NAN(k, "[FMPC/Coefficient] ");
  CHECK_NAN(K, "[FMPC/Coefficient] ");
  CHECK_NAN(s, "[FMPC/Coefficient] ");
  CHECK_NAN(P, "[FMPC/Coefficient] ");

  return false;
}

template<int StateDim, int InputDim, int IneqDim>
typename FmpcSolver<StateDim, InputDim, IneqDim>::Status FmpcSolver<StateDim, InputDim, IneqDim>::solve(
    double current_t,
    const StateDimVector & current_x,
    const Variable & initial_variable)
{
  auto start_time = std::chrono::system_clock::now();

  // Initialize variables
  current_t_ = current_t;
  current_x_ = current_x;
  variable_ = initial_variable;
  variable_.print_level = config_.print_level;

  // Initialize complementarity variables
  if(config_.init_complementary_variable)
  {
    constexpr double initial_barrier_eps = 1e-4;
    constexpr double complementary_variable_margin_rate = 1e-2;
    constexpr double complementary_variable_min = 1e-2;

    barrier_eps_ = initial_barrier_eps;
    for(int i = 0; i < config_.horizon_steps; i++)
    {
      double t = current_t_ + i * problem_->dt();
      variable_.s_list[i] = (1.0 + complementary_variable_margin_rate)
                            * (-1 * problem_->ineqConst(t, variable_.x_list[i], variable_.u_list[i]))
                                  .cwiseMax(complementary_variable_min);
      variable_.nu_list[i] = (1.0 + complementary_variable_margin_rate)
                             * (barrier_eps_ * variable_.s_list[i].cwiseInverse()).cwiseMax(complementary_variable_min);
    }
  }

  // Check variable_
  checkVariable();

  // Setup delta_variable_
  if(delta_variable_.horizon_steps != config_.horizon_steps)
  {
    delta_variable_ = Variable(config_.horizon_steps);
    delta_variable_.print_level = config_.print_level;
  }

  // Setup coeff_list_
  if constexpr(InputDim == Eigen::Dynamic || IneqDim == Eigen::Dynamic)
  {
    coeff_list_.clear();
    for(int i = 0; i < config_.horizon_steps; i++)
    {
      double t = current_t_ + i * problem_->dt();
      coeff_list_.emplace_back(problem_->stateDim(), problem_->inputDim(t), problem_->ineqDim(t));
    }
  }
  else
  {
    // This assumes that the dimension is fixed, but it is efficient because it preserves existing elements
    coeff_list_.resize(config_.horizon_steps,
                       Coefficient(problem_->stateDim(), problem_->inputDim(), problem_->ineqDim()));
  }
  coeff_list_.emplace_back(problem_->stateDim());
  for(auto & coeff : coeff_list_)
  {
    coeff.print_level = config_.print_level;
  }

  // Clear trace_data_list_
  trace_data_list_.clear();

  // Setup computation_duration_
  computation_duration_ = ComputationDuration();

  auto setup_time = std::chrono::system_clock::now();
  computation_duration_.setup = calcDuration(start_time, setup_time);

  // Optimization loop
  Status status = Status::Uninitialized;
  for(int iter = 1; iter <= config_.max_iter; iter++)
  {
    status = procOnce(iter);
    if(status != Status::IterationContinued)
    {
      break;
    }
  }
  if(status == Status::IterationContinued)
  {
    status = Status::MaxIterationReached;
  }

  auto end_time = std::chrono::system_clock::now();
  computation_duration_.opt = calcDuration(setup_time, end_time);
  computation_duration_.solve = calcDuration(start_time, end_time);

  if(config_.print_level >= 3)
  {
    std::cout << "[FMPC] Solve finised. status: " << static_cast<int>(status)
              << ", iter: " << trace_data_list_.back().iter << std::endl;
  }

  return status;
}

template<int StateDim, int InputDim, int IneqDim>
void FmpcSolver<StateDim, InputDim, IneqDim>::dumpTraceDataList(const std::string & file_path) const
{
  std::ofstream ofs(file_path);
  // clang-format off
  ofs << "iter "
      << "kkt_error "
      << "duration_coeff "
      << "duration_backward "
      << "duration_forward "
      << "duration_update" << std::endl;
  // clang-format on
  for(const auto & trace_data : trace_data_list_)
  {
    // clang-format off
    ofs << trace_data.iter << " "
        << trace_data.kkt_error << " "
        << trace_data.duration_coeff << " "
        << trace_data.duration_backward << " "
        << trace_data.duration_forward << " "
        << trace_data.duration_update
        << std::endl;
    // clang-format on
  }
}
template<int StateDim, int InputDim, int IneqDim>
void FmpcSolver<StateDim, InputDim, IneqDim>::checkVariable() const
{
  // Check sequence length
  if(static_cast<int>(variable_.x_list.size()) != config_.horizon_steps + 1)
  {
    throw std::invalid_argument("[FMPC] x_list length should be " + std::to_string(config_.horizon_steps + 1) + " but "
                                + std::to_string(variable_.x_list.size()) + ".");
  }
  if(static_cast<int>(variable_.u_list.size()) != config_.horizon_steps)
  {
    throw std::invalid_argument("[FMPC] u_list length should be " + std::to_string(config_.horizon_steps) + " but "
                                + std::to_string(variable_.u_list.size()) + ".");
  }
  if(static_cast<int>(variable_.lambda_list.size()) != config_.horizon_steps + 1)
  {
    throw std::invalid_argument("[FMPC] lambda_list length should be " + std::to_string(config_.horizon_steps + 1)
                                + " but " + std::to_string(variable_.lambda_list.size()) + ".");
  }
  if(static_cast<int>(variable_.s_list.size()) != config_.horizon_steps)
  {
    throw std::invalid_argument("[FMPC] s_list length should be " + std::to_string(config_.horizon_steps) + " but "
                                + std::to_string(variable_.s_list.size()) + ".");
  }
  if(static_cast<int>(variable_.nu_list.size()) != config_.horizon_steps)
  {
    throw std::invalid_argument("[FMPC] nu_list length should be " + std::to_string(config_.horizon_steps) + " but "
                                + std::to_string(variable_.nu_list.size()) + ".");
  }

  // Check element dimension
  for(int i = 0; i < config_.horizon_steps; i++)
  {
    if constexpr(InputDim == Eigen::Dynamic)
    {
      double t = current_t_ + i * problem_->dt();
      int input_dim = problem_->inputDim(t);
      if(variable_.u_list[i].size() != input_dim)
      {
        throw std::runtime_error("[FMPC] u_list[i] dimension should be " + std::to_string(input_dim) + " but "
                                 + std::to_string(variable_.u_list[i].size()) + ". i: " + std::to_string(i)
                                 + ", time: " + std::to_string(t));
      }
    }
    if constexpr(IneqDim == Eigen::Dynamic)
    {
      double t = current_t_ + i * problem_->dt();
      int ineq_dim = problem_->ineqDim(t);
      if(variable_.s_list[i].size() != ineq_dim)
      {
        throw std::runtime_error("[FMPC] s_list[i] dimension should be " + std::to_string(ineq_dim) + " but "
                                 + std::to_string(variable_.s_list[i].size()) + ". i: " + std::to_string(i)
                                 + ", time: " + std::to_string(t));
      }
      if(variable_.nu_list[i].size() != ineq_dim)
      {
        throw std::runtime_error("[FMPC] nu_list[i] dimension should be " + std::to_string(ineq_dim) + " but "
                                 + std::to_string(variable_.nu_list[i].size()) + ". i: " + std::to_string(i)
                                 + ", time: " + std::to_string(t));
      }
    }
  }

  // Check non-negative
  for(int i = 0; i < config_.horizon_steps; i++)
  {
    double t = current_t_ + i * problem_->dt();
    if((variable_.s_list[i].array() < 0).any())
    {
      throw std::runtime_error("[FMPC] s_list[i] must be non-negative. i: " + std::to_string(i)
                               + ", time: " + std::to_string(t));
    }
    if((variable_.nu_list[i].array() < 0).any())
    {
      throw std::runtime_error("[FMPC] nu_list[i] must be non-negative. i: " + std::to_string(i)
                               + ", time: " + std::to_string(t));
    }
  }
}

template<int StateDim, int InputDim, int IneqDim>
typename FmpcSolver<StateDim, InputDim, IneqDim>::Status FmpcSolver<StateDim, InputDim, IneqDim>::procOnce(int iter)
{
  if(config_.print_level >= 3)
  {
    std::cout << "[FMPC] Start iteration " << iter << std::endl;
  }

  // Append trace data
  trace_data_list_.emplace_back();
  auto & trace_data = trace_data_list_.back();
  trace_data.iter = iter;

  // Update barrier parameter
  if(config_.update_barrier_eps)
  {
    // (19.19) in "Nocedal, Wright. Numerical optimization"
    double s_nu_ave = 0.0;
    double s_nu_min = std::numeric_limits<double>::max();
    int total_ineq_dim = 0;
    for(int i = 0; i < config_.horizon_steps; i++)
    {
      s_nu_ave += variable_.s_list[i].dot(variable_.nu_list[i]);
      s_nu_min = std::min(s_nu_min, variable_.s_list[i].cwiseProduct(variable_.nu_list[i]).minCoeff());
      total_ineq_dim += static_cast<int>(variable_.s_list[i].size());
    }
    s_nu_ave /= total_ineq_dim;

    double sigma = 0.5;
    // The following equations follow (19.20) in "Nocedal, Wright. Numerical optimization" but does not work
    // double xi = s_nu_min / s_nu_ave;
    // double sigma = 0.1 * std::pow(std::min(0.05 * (1.0 - xi) / xi, 2.0), 3);
    constexpr double barrier_eps_min = 1e-8;
    constexpr double barrier_eps_max = 1e6;
    barrier_eps_ = std::clamp(sigma * s_nu_ave, barrier_eps_min, barrier_eps_max);
  }

  // Step 1: calculate coefficients of linearized KKT condition
  {
    auto start_time = std::chrono::system_clock::now();

    double dt = problem_->dt();
    for(int i = 0; i < config_.horizon_steps; i++)
    {
      auto & coeff = coeff_list_[i];
      double t = current_t_ + i * dt;
      const StateDimVector & x = variable_.x_list[i];
      const StateDimVector & next_x = variable_.x_list[i + 1];
      const InputDimVector & u = variable_.u_list[i];
      const StateDimVector & lambda = variable_.lambda_list[i];
      const StateDimVector & next_lambda = variable_.lambda_list[i + 1];
      const IneqDimVector & s = variable_.s_list[i];
      const IneqDimVector & nu = variable_.nu_list[i];

      problem_->calcStateEqDeriv(t, x, u, coeff.A, coeff.B);
      problem_->calcIneqConstDeriv(t, x, u, coeff.C, coeff.D);
      problem_->calcRunningCostDeriv(t, x, u, coeff.Lx, coeff.Lu, coeff.Lxx, coeff.Luu, coeff.Lxu);

      coeff.x_bar = problem_->stateEq(t, x, u) - next_x; // (2.23c)
      coeff.g_bar = problem_->ineqConst(t, x, u) + s; // (2.23d)
      coeff.Lx_bar =
          -1 * lambda + dt * coeff.Lx + coeff.A.transpose() * next_lambda + coeff.C.transpose() * nu; // (2.25b)
      coeff.Lu_bar = dt * coeff.Lu + coeff.B.transpose() * next_lambda + coeff.D.transpose() * nu; // (2.25c)
    }
    {
      auto & terminal_coeff = coeff_list_[config_.horizon_steps];
      double terminal_t = current_t_ + config_.horizon_steps * dt;
      const StateDimVector & terminal_x = variable_.x_list[config_.horizon_steps];
      const StateDimVector & terminal_lambda = variable_.lambda_list[config_.horizon_steps];
      problem_->calcTerminalCostDeriv(terminal_t, terminal_x, terminal_coeff.Lx, terminal_coeff.Lxx);
      terminal_coeff.Lx_bar = terminal_coeff.Lx - terminal_lambda; // (2.25a)
    }

    double duration = calcDuration(start_time, std::chrono::system_clock::now());
    trace_data.duration_coeff = duration;
    computation_duration_.coeff += duration;
  }

  // Check KKT error
  double kkt_error = calcKktError(0.0);
  trace_data.kkt_error = kkt_error;
  if(kkt_error <= config_.kkt_error_thre)
  {
    return Status::Succeeded;
  }

  // Step 2: backward pass
  {
    auto start_time = std::chrono::system_clock::now();

    if(!backwardPass())
    {
      return Status::ErrorInBackward;
    }

    double duration = calcDuration(start_time, std::chrono::system_clock::now());
    trace_data.duration_backward = duration;
    computation_duration_.backward += duration;
  }

  // Step 3: forward pass
  {
    auto start_time = std::chrono::system_clock::now();

    if(!forwardPass())
    {
      return Status::ErrorInForward;
    }

    double duration = calcDuration(start_time, std::chrono::system_clock::now());
    trace_data.duration_forward = duration;
    computation_duration_.forward += duration;
  }

  // Step 4: update variables
  {
    auto start_time = std::chrono::system_clock::now();

    if(!updateVariables())
    {
      return Status::ErrorInUpdate;
    }

    double duration = calcDuration(start_time, std::chrono::system_clock::now());
    trace_data.duration_update = duration;
    computation_duration_.update += duration;
  }

  return Status::IterationContinued;
}

template<int StateDim, int InputDim, int IneqDim>
double FmpcSolver<StateDim, InputDim, IneqDim>::calcKktError(double barrier_eps) const
{
  double kkt_error = 0;

  {
    kkt_error += (current_x_ - variable_.x_list[0]).squaredNorm();
  }
  for(int i = 0; i < config_.horizon_steps; i++)
  {
    const auto & coeff = coeff_list_[i];
    kkt_error += coeff.x_bar.squaredNorm();
    kkt_error += coeff.g_bar.squaredNorm();
    kkt_error += coeff.Lx_bar.squaredNorm();
    kkt_error += coeff.Lu_bar.squaredNorm();
    kkt_error +=
        (variable_.s_list[i].array() * variable_.nu_list[i].array() - barrier_eps).max(0).matrix().squaredNorm();
  }
  {
    const auto & terminal_coeff = coeff_list_[config_.horizon_steps];
    kkt_error += terminal_coeff.Lx_bar.squaredNorm();
  }

  kkt_error = std::sqrt(kkt_error);

  return kkt_error;
}

template<int StateDim, int InputDim, int IneqDim>
bool FmpcSolver<StateDim, InputDim, IneqDim>::backwardPass()
{
  // To avoid repetitive memory allocation, the vector and matrix variables are created outside of loop
  StateStateDimMatrix Qxx_tilde;
  InputInputDimMatrix Quu_tilde;
  StateInputDimMatrix Qxu_tilde;
  StateDimVector Lx_tilde;
  InputDimVector Lu_tilde;

  StateStateDimMatrix F;
  StateInputDimMatrix H;
  InputInputDimMatrix G;

  InputDimVector k;
  InputStateDimMatrix K;
  StateDimVector s;
  StateStateDimMatrix P;
  StateStateDimMatrix P_symmetric;

  {
    auto & terminal_coeff = coeff_list_[config_.horizon_steps];
    s = -1 * terminal_coeff.Lx_bar; // (2.34)
    P = terminal_coeff.Lxx; // (2.34)
    terminal_coeff.s = s;
    terminal_coeff.P = P;
  }

  for(int i = config_.horizon_steps - 1; i >= 0; i--)
  {
    // Get coefficients
    double dt = problem_->dt();
    auto & coeff = coeff_list_[i];
    const StateStateDimMatrix & A = coeff.A;
    const StateInputDimMatrix & B = coeff.B;
    const IneqStateDimMatrix & C = coeff.C;
    const IneqInputDimMatrix & D = coeff.D;
    const StateStateDimMatrix & Lxx = coeff.Lxx;
    const InputInputDimMatrix & Luu = coeff.Luu;
    const StateInputDimMatrix & Lxu = coeff.Lxu;
    const StateDimVector & x_bar = coeff.x_bar;
    const IneqDimVector & g_bar = coeff.g_bar;
    const StateDimVector & Lx_bar = coeff.Lx_bar;
    const InputDimVector & Lu_bar = coeff.Lu_bar;

    // Pre-process for gain calculation
    {
      auto start_time_gain_pre = std::chrono::system_clock::now();

      IneqDimVector nu_s = (variable_.nu_list[i].array() / variable_.s_list[i].array()).matrix();
      IneqDimVector tilde_sub =
          nu_s.cwiseProduct(g_bar) - variable_.nu_list[i] + barrier_eps_ * variable_.s_list[i].cwiseInverse();
      Qxx_tilde.noalias() = dt * Lxx + C.transpose() * nu_s.asDiagonal() * C; // (2.28c)
      Quu_tilde.noalias() = dt * Luu + D.transpose() * nu_s.asDiagonal() * D; // (2.28e)
      Qxu_tilde.noalias() = dt * Lxu + C.transpose() * nu_s.asDiagonal() * D; // (2.28d)
      Lx_tilde.noalias() = Lx_bar + C.transpose() * tilde_sub; // (2.28f)
      Lu_tilde.noalias() = Lu_bar + D.transpose() * tilde_sub; // (2.28g)

      F.noalias() = Qxx_tilde + A.transpose() * P * A; // (2.35b)
      H.noalias() = Qxu_tilde + A.transpose() * P * B; // (2.35c)
      G.noalias() = Quu_tilde + B.transpose() * P * B; // (2.35d)

      computation_duration_.gain_pre += calcDuration(start_time_gain_pre, std::chrono::system_clock::now());
    }

    // Solve linear equation for gain calculation
    {
      auto start_time_gain_solve = std::chrono::system_clock::now();

      int input_dim = static_cast<int>(B.cols());
      if(input_dim > 0)
      {
        // In numerically difficult cases, LLT may diverge and LDLT may work.
        Eigen::LDLT<InputInputDimMatrix> llt_G(G);
        if(llt_G.info() == Eigen::Success)
        {
          k.noalias() = -1 * llt_G.solve(B.transpose() * (P * x_bar - s) + Lu_tilde); // (2.35e)
          K.noalias() = -1 * llt_G.solve(H.transpose()); // (2.35e)
        }
        else
        {
          if(config_.print_level >= 1)
          {
            std::cout << "[FMPC/Backward] G is not positive definite in Cholesky decomposition (LLT)." << std::endl;
          }
          if(config_.break_if_llt_fails)
          {
            return false;
          }
          else
          {
            Eigen::FullPivLU<InputInputDimMatrix> lu_G(G);
            k.noalias() = -1 * lu_G.solve(B.transpose() * (P * x_bar - s) + Lu_tilde); // (2.35e)
            K.noalias() = -1 * lu_G.solve(H.transpose()); // (2.35e)
          }
        }
      }
      else
      {
        k.setZero(0);
        K.setZero(0, problem_->stateDim());
      }

      computation_duration_.gain_solve += calcDuration(start_time_gain_solve, std::chrono::system_clock::now());
    }

    // Post-process for gain calculation
    {
      auto start_time_gain_post = std::chrono::system_clock::now();

      s = A.transpose() * (s - P * x_bar) - Lx_tilde - H * k; // (2.35a)
      P.noalias() = F - K.transpose() * G * K; // (2.35a)
      P_symmetric = 0.5 * (P + P.transpose()); // Enforce symmetric
      // Assigning directly to P without using the intermediate variable P_symmetric yields incorrect results!
      P = P_symmetric;

      computation_duration_.gain_post += calcDuration(start_time_gain_post, std::chrono::system_clock::now());
    }

    // Save gains
    coeff.k = k;
    coeff.K = K;
    coeff.s = s;
    coeff.P = P;
  }

  if(config_.check_nan)
  {
    for(const auto & coeff : coeff_list_)
    {
      if(coeff.containsNaN())
      {
        if(config_.print_level >= 1)
        {
          std::cout << "[FMPC/Backward] coeff contains NaN." << std::endl;
        }
        return false;
      }
    }
  }

  return true;
}

template<int StateDim, int InputDim, int IneqDim>
bool FmpcSolver<StateDim, InputDim, IneqDim>::forwardPass()
{
  delta_variable_.x_list[0] = current_x_ - variable_.x_list[0];

  for(int i = 0; i < config_.horizon_steps + 1; i++)
  {
    const auto & coeff = coeff_list_[i];

    delta_variable_.lambda_list[i].noalias() = coeff.P * delta_variable_.x_list[i] - coeff.s; // (2.33)

    if(i < config_.horizon_steps)
    {
      delta_variable_.u_list[i].noalias() = coeff.K * delta_variable_.x_list[i] + coeff.k; // (2.36)
      delta_variable_.x_list[i + 1].noalias() =
          coeff.A * delta_variable_.x_list[i] + coeff.B * delta_variable_.u_list[i] + coeff.x_bar; // (2.26b)
    }
  }

  for(int i = 0; i < config_.horizon_steps; i++)
  {
    const auto & coeff = coeff_list_[i];

    delta_variable_.s_list[i].noalias() =
        -1 * (coeff.C * delta_variable_.x_list[i] + coeff.D * delta_variable_.u_list[i] + coeff.g_bar); // (2.27a)
    delta_variable_.nu_list[i].noalias() =
        (-1 * (variable_.nu_list[i].array() * (delta_variable_.s_list[i] + variable_.s_list[i]).array() - barrier_eps_)
         / variable_.s_list[i].array())
            .matrix(); // (2.27b)
  }

  if(config_.check_nan && delta_variable_.containsNaN())
  {
    if(config_.print_level >= 1)
    {
      std::cout << "[FMPC/Forward] delta_variable contains NaN." << std::endl;
    }
    return false;
  }

  return true;
}

template<int StateDim, int InputDim, int IneqDim>
bool FmpcSolver<StateDim, InputDim, IneqDim>::updateVariables()
{
  // Fraction-to-boundary rule
  double alpha_s_max = 1.0;
  double alpha_nu_max = 1.0;
  {
    auto start_time_fraction = std::chrono::system_clock::now();

    constexpr double margin_ratio = 0.995;
    for(int i = 0; i < config_.horizon_steps; i++)
    {
      const IneqDimVector & s = variable_.s_list[i];
      const IneqDimVector & nu = variable_.nu_list[i];
      const IneqDimVector & delta_s = delta_variable_.s_list[i];
      const IneqDimVector & delta_nu = delta_variable_.nu_list[i];
      for(int ineq_idx = 0; ineq_idx < s.size(); ineq_idx++)
      {
        // (19.9) in "Nocedal, Wright. Numerical optimization"
        if(delta_s[ineq_idx] < 0)
        {
          alpha_s_max = std::min(alpha_s_max, -1 * margin_ratio * s[ineq_idx] / delta_s[ineq_idx]);
        }
        if(delta_nu[ineq_idx] < 0)
        {
          alpha_nu_max = std::min(alpha_nu_max, -1 * margin_ratio * nu[ineq_idx] / delta_nu[ineq_idx]);
        }
      }
    }
    if(!(alpha_s_max > 0.0 && alpha_s_max <= 1.0 && alpha_nu_max > 0.0 && alpha_nu_max <= 1.0))
    {
      if(config_.print_level >= 1)
      {
        std::cout << "[FMPC/Update] Invalid alpha. barrier_eps: " << barrier_eps_ << ", alpha_s_max: " << alpha_s_max
                  << ", alpha_nu_max: " << alpha_nu_max << std::endl;
      }
      return false;
    }

    computation_duration_.fraction += calcDuration(start_time_fraction, std::chrono::system_clock::now());
  }

  // Line search
  double alpha_s = alpha_s_max;
  double alpha_nu = alpha_nu_max;
  if(config_.enable_line_search)
  {
    setupMeritFunc();

    constexpr double armijo_scale = 1e-3;
    constexpr double alpha_s_update_ratio = 0.5;
    constexpr double alpha_s_min = 1e-10;
    Variable ls_variable = variable_;
    while(true)
    {
      if(alpha_s < alpha_s_min)
      {
        if(config_.print_level >= 1)
        {
          std::cout << "[FMPC/Update] alpha_s is too small in line search backtracking. alpha_s_max: " << alpha_s_max
                    << ", alpha_s: " << alpha_s << std::endl;
        }
        break;
      }

      for(int i = 0; i < config_.horizon_steps + 1; i++)
      {
        ls_variable.x_list[i] = variable_.x_list[i] + alpha_s * delta_variable_.x_list[i];

        if(i < config_.horizon_steps)
        {
          ls_variable.u_list[i] = variable_.u_list[i] + alpha_s * delta_variable_.u_list[i];
          ls_variable.s_list[i] = variable_.s_list[i] + alpha_s * delta_variable_.s_list[i];
        }
      }

      double merit_func_new = calcMeritFunc(ls_variable);
      if(merit_func_new < merit_func_ + armijo_scale * alpha_s * merit_deriv_)
      {
        break;
      }
      alpha_s *= alpha_s_update_ratio;
    }
  }

  if(config_.print_level >= 3)
  {
    std::cout << "[FMPC/update] barrier_eps: " << barrier_eps_ << ", alpha_s_max: " << alpha_s_max
              << ", alpha_nu_max: " << alpha_nu_max << ", alpha_s: " << alpha_s << ", alpha_nu: " << alpha_nu
              << std::endl;
  }

  for(int i = 0; i < config_.horizon_steps + 1; i++)
  {
    variable_.x_list[i] += alpha_s * delta_variable_.x_list[i];
    variable_.lambda_list[i] += alpha_nu * delta_variable_.lambda_list[i];

    if(i < config_.horizon_steps)
    {
      variable_.u_list[i] += alpha_s * delta_variable_.u_list[i];
      variable_.s_list[i] += alpha_s * delta_variable_.s_list[i];
      variable_.nu_list[i] += alpha_nu * delta_variable_.nu_list[i];

      constexpr double min_positive_value = std::numeric_limits<double>::lowest();
      if((variable_.s_list[i].array() < 0).any())
      {
        if(config_.print_level >= 1)
        {
          std::cout << "[FMPC/Update] Updated s is negative: " << variable_.s_list[i].transpose() << std::endl;
        }
        variable_.s_list[i] = variable_.s_list[i].array().max(min_positive_value).matrix();
      }
      if((variable_.nu_list[i].array() < 0).any())
      {
        if(config_.print_level >= 1)
        {
          std::cout << "[FMPC/Update] Updated nu is negative: " << variable_.nu_list[i].transpose() << std::endl;
        }
        variable_.nu_list[i] = variable_.nu_list[i].array().max(min_positive_value).matrix();
      }
    }
  }

  return true;
}

template<int StateDim, int InputDim, int IneqDim>
void FmpcSolver<StateDim, InputDim, IneqDim>::setupMeritFunc()
{
  double merit_func_obj = 0.0;
  double merit_func_const = 0.0;
  double merit_deriv_obj = 0.0;
  double merit_deriv_const = 0.0;
  double dt = problem_->dt();

  {
    StateDimVector const_func = current_x_ - variable_.x_list[0];
    merit_func_const += const_func.template lpNorm<1>();
    merit_deriv_const +=
        l1NormDirectionalDeriv(const_func, (-1 * StateStateDimMatrix::Identity()).eval(), delta_variable_.x_list[0]);
  }

  for(int i = 0; i < config_.horizon_steps; i++)
  {
    double t = current_t_ + i * dt;
    const StateDimVector & x = variable_.x_list[i];
    const InputDimVector & u = variable_.u_list[i];
    const IneqDimVector & s = variable_.s_list[i];
    const StateDimVector & next_x = variable_.x_list[i + 1];
    const StateDimVector & delta_x = delta_variable_.x_list[i];
    const InputDimVector & delta_u = delta_variable_.u_list[i];
    const IneqDimVector & delta_s = delta_variable_.s_list[i];
    const StateDimVector & delta_next_x = delta_variable_.x_list[i + 1];
    const auto & coeff = coeff_list_[i];

    {
      merit_func_obj += problem_->runningCost(t, x, u) * dt;
      merit_deriv_obj += (coeff.Lx.dot(delta_x) + coeff.Lu.dot(delta_u)) * dt;
    }

    {
      merit_func_obj += -1 * barrier_eps_ * s.array().log().sum();
      merit_deriv_obj += -1 * barrier_eps_ * s.cwiseInverse().dot(delta_s);
    }

    {
      StateDimVector const_func = problem_->stateEq(t, x, u) - next_x;
      merit_func_const += const_func.template lpNorm<1>();
      merit_deriv_const += l1NormDirectionalDeriv(const_func, coeff.A, delta_x);
      merit_deriv_const += l1NormDirectionalDeriv(const_func, coeff.B, delta_u);
      merit_deriv_const +=
          l1NormDirectionalDeriv(const_func, (-1 * StateStateDimMatrix::Identity()).eval(), delta_next_x);
    }

    {
      IneqDimVector const_func = problem_->ineqConst(t, x, u) + s;
      merit_func_const += const_func.template lpNorm<1>();
      merit_deriv_const += l1NormDirectionalDeriv(const_func, coeff.C, delta_x);
      merit_deriv_const += l1NormDirectionalDeriv(const_func, coeff.D, delta_u);
      merit_deriv_const += l1NormDirectionalDeriv(const_func, IneqIneqDimMatrix::Identity().eval(), delta_s);
    }
  }

  {
    double terminal_t = current_t_ + config_.horizon_steps * dt;
    const StateDimVector & terminal_x = variable_.x_list[config_.horizon_steps];
    const StateDimVector & terminal_delta_x = delta_variable_.x_list[config_.horizon_steps];
    const auto & terminal_coeff = coeff_list_[config_.horizon_steps];

    merit_func_obj += problem_->terminalCost(terminal_t, terminal_x);
    merit_deriv_obj += terminal_coeff.Lx.dot(terminal_delta_x);
  }

  constexpr double merit_const_scale_min = 1e-3;
  if(config_.merit_const_scale_from_lagrange_multipliers)
  {
    // (18.32) in "Nocedal, Wright. Numerical optimization"
    merit_const_scale_ = merit_const_scale_min;
    for(int i = 0; i < config_.horizon_steps + 1; i++)
    {
      merit_const_scale_ = std::max(merit_const_scale_, variable_.lambda_list[i].cwiseAbs().maxCoeff());

      if(i < config_.horizon_steps)
      {
        merit_const_scale_ = std::max(merit_const_scale_, variable_.nu_list[i].cwiseAbs().maxCoeff());
      }
    }
  }
  else
  {
    // (18.33) in "Nocedal, Wright. Numerical optimization"
    constexpr double rho = 0.5;
    merit_const_scale_ = std::max(merit_deriv_obj / ((1.0 - rho) * merit_func_const), merit_const_scale_min);
  }

  merit_func_ = merit_func_obj + merit_const_scale_ * merit_func_const;
  merit_deriv_ = merit_deriv_obj + merit_const_scale_ * merit_deriv_const;

  if(config_.print_level >= 3)
  {
    std::cout << "[FMPC/merit] merit_func: " << merit_func_ << ", merit_deriv: " << merit_deriv_
              << ", merit_const_scale: " << merit_const_scale_ << std::endl;
  }
}

template<int StateDim, int InputDim, int IneqDim>
double FmpcSolver<StateDim, InputDim, IneqDim>::calcMeritFunc(const Variable & variable) const
{
  double merit_func_obj = 0.0;
  double merit_func_const = 0.0;
  double dt = problem_->dt();

  {
    StateDimVector const_func = current_x_ - variable.x_list[0];
    merit_func_const += const_func.template lpNorm<1>();
  }

  for(int i = 0; i < config_.horizon_steps; i++)
  {
    double t = current_t_ + i * dt;
    const StateDimVector & x = variable.x_list[i];
    const InputDimVector & u = variable.u_list[i];
    const IneqDimVector & s = variable.s_list[i];
    const StateDimVector & next_x = variable.x_list[i + 1];

    {
      merit_func_obj += problem_->runningCost(t, x, u) * dt;
    }

    {
      merit_func_obj += -1 * barrier_eps_ * s.array().log().sum();
    }

    {
      StateDimVector const_func = problem_->stateEq(t, x, u) - next_x;
      merit_func_const += const_func.template lpNorm<1>();
    }

    {
      IneqDimVector const_func = problem_->ineqConst(t, x, u) + s;
      merit_func_const += const_func.template lpNorm<1>();
    }
  }

  {
    double terminal_t = current_t_ + config_.horizon_steps * dt;
    const StateDimVector & terminal_x = variable.x_list[config_.horizon_steps];

    merit_func_obj += problem_->terminalCost(terminal_t, terminal_x);
  }

  return merit_func_obj + merit_const_scale_ * merit_func_const;
}
} // namespace nmpc_fmpc

#undef CHECK_NAN
