/* Author: Masaki Murooka */

#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>

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
bool FmpcSolver<StateDim, InputDim, IneqDim>::solve(double current_t,
                                                    const StateDimVector & current_x,
                                                    const Variable & initial_variable)
{
  auto start_time = std::chrono::system_clock::now();

  // Initialize variables
  current_t_ = current_t;
  current_x_ = current_x;
  variable_ = initial_variable;

  // Check variable_
  checkVariable();

  // Setup delta_variable_
  if(delta_variable_.horizon_steps != config_.horizon_steps)
  {
    delta_variable_ = Variable(config_.horizon_steps);
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

  // Clear trace_data_list_
  trace_data_list_.clear();

  // Setup computation_duration_
  computation_duration_ = ComputationDuration();

  auto setup_time = std::chrono::system_clock::now();
  computation_duration_.setup = calcDuration(start_time, setup_time);

  // Optimization loop
  int retval = 0;
  for(int iter = 1; iter <= config_.max_iter; iter++)
  {
    retval = procOnce(iter);
    if(retval != 0)
    {
      break;
    }
  }

  auto end_time = std::chrono::system_clock::now();
  computation_duration_.opt = calcDuration(setup_time, end_time);
  computation_duration_.solve = calcDuration(start_time, end_time);

  if(config_.print_level >= 3)
  {
    std::cout << "[FMPC] Setup duration: " << computation_duration_.setup
              << " [ms], optimization duration: " << computation_duration_.opt << " [ms]." << std::endl;
  }

  return retval == 1;
}

template<int StateDim, int InputDim, int IneqDim>
void FmpcSolver<StateDim, InputDim, IneqDim>::checkVariable() const
{
  // Check sequence length
  if(variable_.x_list.size() != config_.horizon_steps + 1)
  {
    throw std::invalid_argument("[FMPC] x_list length should be " + std::to_string(config_.horizon_steps + 1) + " but "
                                + std::to_string(variable_.x_list.size()) + ".");
  }
  if(variable_.u_list.size() != config_.horizon_steps)
  {
    throw std::invalid_argument("[FMPC] u_list length should be " + std::to_string(config_.horizon_steps) + " but "
                                + std::to_string(variable_.u_list.size()) + ".");
  }
  if(variable_.lambda_list.size() != config_.horizon_steps + 1)
  {
    throw std::invalid_argument("[FMPC] lambda_list length should be " + std::to_string(config_.horizon_steps + 1)
                                + " but " + std::to_string(variable_.lambda_list.size()) + ".");
  }
  if(variable_.s_list.size() != config_.horizon_steps)
  {
    throw std::invalid_argument("[FMPC] s_list length should be " + std::to_string(config_.horizon_steps) + " but "
                                + std::to_string(variable_.s_list.size()) + ".");
  }
  if(variable_.nu_list.size() != config_.horizon_steps)
  {
    throw std::invalid_argument("[FMPC] nu_list length should be " + std::to_string(config_.horizon_steps) + " but "
                                + std::to_string(variable_.nu_list.size()) + ".");
  }

  // Check element dimension
  for(int i = 0; i < config_.horizon_steps; i++)
  {
    double t = current_t_ + i * problem_->dt();
    int input_dim = problem_->inputDim(t);
    int ineq_dim = problem_->ineqDim(t);
    if constexpr(InputDim == Eigen::Dynamic)
    {
      if(variable_.u_list[i].size() != input_dim)
      {
        throw std::runtime_error("[FMPC] u_list[i] dimension should be " + std::to_string(input_dim) + " but "
                                 + std::to_string(variable_.u_list[i].size()) + ". i: " + std::to_string(i)
                                 + ", time: " + std::to_string(t));
      }
    }
    if constexpr(IneqDim == Eigen::Dynamic)
    {
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
int FmpcSolver<StateDim, InputDim, IneqDim>::procOnce(int iter)
{
  if(config_.print_level >= 3)
  {
    std::cout << "[FMPC] Start iteration " << iter << std::endl;
  }

  // Append trace data
  trace_data_list_.emplace_back();
  auto & trace_data = trace_data_list_.back();
  trace_data.iter = iter;

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

      coeff.x_bar = problem_->stateEq(t, x, u) - next_x;
      coeff.g_bar = problem_->ineqConst(t, x, u) + s;
      coeff.Lx_bar = -1 * lambda + dt * coeff.Lx + coeff.A.transpose() * next_lambda + coeff.C.transpose() * nu;
      coeff.Lu_bar = dt * coeff.Lu + coeff.B.transpose() * next_lambda + coeff.D.transpose() * nu;
    }
    {
      auto & terminal_coeff = coeff_list_[config_.horizon_steps];
      double terminal_t = current_t_ + config_.horizon_steps * dt;
      const StateDimVector & terminal_x = variable_.x_list[config_.horizon_steps];
      const StateDimVector & terminal_lambda = variable_.lambda_list[config_.horizon_steps];
      problem_->calcTerminalCostDeriv(terminal_t, terminal_x, terminal_coeff.Lx_bar, terminal_coeff.Lxx);
      terminal_coeff.Lx_bar -= terminal_lambda;
    }

    double duration = calcDuration(start_time, std::chrono::system_clock::now());
    trace_data.duration_coeff = duration;
    computation_duration_.coeff += duration;
  }

  // Check KKT error
  double kkt_error = calcKktError();
  trace_data.kkt_error = kkt_error;
  if(kkt_error <= config_.kkt_error_thre)
  {
    return 1;
  }

  // Step 2: backward pass
  {
    auto start_time = std::chrono::system_clock::now();

    if(!backwardPass())
    {
      return -1;
    }

    double duration = calcDuration(start_time, std::chrono::system_clock::now());
    trace_data.duration_backward = duration;
    computation_duration_.backward += duration;
  }

  // Step 3: forward pass
  {
    auto start_time = std::chrono::system_clock::now();

    forwardPass();

    double duration = calcDuration(start_time, std::chrono::system_clock::now());
    trace_data.duration_forward = duration;
    computation_duration_.forward += duration;
  }

  // Step 4: update variables
  {
    auto start_time = std::chrono::system_clock::now();

    if(!updateVariables())
    {
      return -1;
    }

    double duration = calcDuration(start_time, std::chrono::system_clock::now());
    trace_data.duration_update = duration;
    computation_duration_.update += duration;
  }

  return 0;
}

template<int StateDim, int InputDim, int IneqDim>
double FmpcSolver<StateDim, InputDim, IneqDim>::calcKktError() const
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
        (variable_.s_list[i].array() * variable_.nu_list[i].array() - barrier_eps_).max(0).matrix().squaredNorm();
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

  {
    auto & terminal_coeff = coeff_list_[config_.horizon_steps];
    s = -1 * terminal_coeff.Lx_bar;
    P = terminal_coeff.Lxx;
    terminal_coeff.s = s;
    terminal_coeff.P = P;
  }

  for(int i = config_.horizon_steps - 1; i >= 0; i--)
  {
    // Get coefficients
    double dt = problem_->dt();
    double t = current_t_ + i * dt;
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
      Qxx_tilde.noalias() = dt * Lxx + C.transpose() * nu_s.asDiagonal() * C;
      Quu_tilde.noalias() = dt * Luu + D.transpose() * nu_s.asDiagonal() * D;
      Qxu_tilde.noalias() = dt * Lxu + C.transpose() * nu_s.asDiagonal() * D;
      Lx_tilde.noalias() = Lx_bar + C.transpose() * tilde_sub;
      Lu_tilde.noalias() = Lu_bar + D.transpose() * tilde_sub;

      F.noalias() = Qxx_tilde + A.transpose() * P * A;
      H.noalias() = Qxu_tilde + A.transpose() * P * B;
      G.noalias() = Quu_tilde + B.transpose() * P * B;

      computation_duration_.gain_pre += calcDuration(start_time_gain_pre, std::chrono::system_clock::now());
    }

    // Solve linear equation for gain calculation
    {
      auto start_time_gain_solve = std::chrono::system_clock::now();

      int input_dim = B.cols();
      if(input_dim > 0)
      {
        Eigen::LLT<InputInputDimMatrix> llt_G(G);
        if(llt_G.info() == Eigen::NumericalIssue)
        {
          if(config_.print_level >= 1)
          {
            std::cout << "[FMPC/Backward] G is not positive definite in Cholesky decomposition (LLT). current time: "
                      << current_t_ << ", horizon idx: " << i << " / " << config_.horizon_steps << std::endl;
          }
          return false;
        }
        k.noalias() = -1 * llt_G.solve(B.transpose() * (P * x_bar - s) + Lu_tilde);
        K.noalias() = -1 * llt_G.solve(H.transpose());
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

      s = A.transpose() * (s - P * x_bar) - Lx_tilde - H * k;
      P.noalias() = F - K.transpose() * G * K;
      P = 0.5 * (P + P.transpose()); // Enforce symmetric

      computation_duration_.gain_post += calcDuration(start_time_gain_post, std::chrono::system_clock::now());
    }

    // Save gains
    coeff.k = k;
    coeff.K = K;
    coeff.s = s;
    coeff.P = P;
  }

  return true;
}

template<int StateDim, int InputDim, int IneqDim>
void FmpcSolver<StateDim, InputDim, IneqDim>::forwardPass()
{
  delta_variable_.x_list[0] = current_x_ - variable_.x_list[0];

  for(int i = 0; i < config_.horizon_steps + 1; i++)
  {
    const auto & coeff = coeff_list_[i];

    delta_variable_.lambda_list[i].noalias() = coeff.P * delta_variable_.x_list[i] - coeff.s;

    if(i < config_.horizon_steps)
    {
      delta_variable_.u_list[i].noalias() = coeff.K * delta_variable_.x_list[i] + coeff.k;
      delta_variable_.x_list[i + 1].noalias() =
          coeff.A * delta_variable_.x_list[i] + coeff.B * delta_variable_.u_list[i] + coeff.x_bar;
    }
  }

  for(int i = 0; i < config_.horizon_steps; i++)
  {
    const auto & coeff = coeff_list_[i];

    delta_variable_.s_list[i].noalias() =
        -1 * (coeff.C * delta_variable_.x_list[i] + coeff.D * delta_variable_.u_list[i] + coeff.g_bar);
    delta_variable_.nu_list[i].noalias() =
        (-1 * (variable_.nu_list[i].array() * (delta_variable_.s_list[i] + variable_.s_list[i]).array() - barrier_eps_)
         / variable_.s_list[i].array())
            .matrix();
  }
}

template<int StateDim, int InputDim, int IneqDim>
bool FmpcSolver<StateDim, InputDim, IneqDim>::updateVariables()
{
  // Fraction-to-boundary rule
  double alpha = 1.0;
  {
    auto start_time_fraction = std::chrono::system_clock::now();

    constexpr double margin_ratio = 0.995;
    for(int i = 0; i < config_.horizon_steps; i++)
    {
      const IneqDimVector & s = variable_.s_list[i];
      const IneqDimVector & nu = variable_.nu_list[i];
      const IneqDimVector & delta_s = delta_variable_.s_list[i];
      const IneqDimVector & delta_nu = delta_variable_.nu_list[i];
      for(int ineq_idx = 0; ineq_idx < delta_s.size(); ineq_idx++)
      {
        if(delta_s[ineq_idx] < 0)
        {
          alpha = std::min(alpha, -1 * margin_ratio * s[ineq_idx] / delta_s[ineq_idx]);
        }
        if(delta_nu[ineq_idx] < 0)
        {
          alpha = std::min(alpha, -1 * margin_ratio * nu[ineq_idx] / delta_nu[ineq_idx]);
        }
      }
    }
    if(!(alpha > 0.0 && alpha <= 1.0))
    {
      if(config_.print_level >= 1)
      {
        std::cout << "[FMPC/Update] Invalid alpha: " + std::to_string(alpha) << std::endl;
      }
      return false;
    }

    computation_duration_.fraction += calcDuration(start_time_fraction, std::chrono::system_clock::now());
  }

  if(config_.print_level >= 3)
  {
    std::cout << "[FMPC/update] alpha: " << alpha << std::endl;
  }

  for(int i = 0; i < config_.horizon_steps + 1; i++)
  {
    variable_.x_list[i] += alpha * delta_variable_.x_list[i];
    variable_.lambda_list[i] += alpha * delta_variable_.lambda_list[i];

    if(i < config_.horizon_steps)
    {
      variable_.u_list[i] += alpha * delta_variable_.u_list[i];
      variable_.s_list[i] += alpha * delta_variable_.s_list[i];
      variable_.nu_list[i] += alpha * delta_variable_.nu_list[i];

      constexpr double min_positive_value = std::numeric_limits<double>::lowest();
      if((variable_.s_list[i].array() < 0).any())
      {
        if(config_.print_level >= 1)
        {
          std::cout << "[FMPC/Update] Updated s is negative: " << variable_.s_list[i].transpose() << std::endl;
        }
        variable_.s_list[i] = variable_.s_list[i].array().max(min_positive_value).matrix();
        return false;
      }
      if((variable_.nu_list[i].array() < 0).any())
      {
        if(config_.print_level >= 1)
        {
          std::cout << "[FMPC/Update] Updated nu is negative: " << variable_.nu_list[i].transpose() << std::endl;
        }
        variable_.nu_list[i] = variable_.nu_list[i].array().max(min_positive_value).matrix();
        return false;
      }
    }
  }

  return true;
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
} // namespace nmpc_fmpc
