/* Author: Masaki Murooka */

#include <chrono>
#include <fstream>
#include <iostream>

namespace nmpc_fmpc
{
template<int StateDim, int InputDim, int IneqDim>
FmpcSolver<StateDim, InputDim, IneqDim>::FmpcSolver(const std::shared_ptr<FmpcProblem<StateDim, InputDim>> & problem)
: problem_(problem)
{
}

template<int StateDim, int InputDim, int IneqDim>
bool FmpcSolver<StateDim, InputDim, IneqDim>::solve(double current_t,
                                                    const StateDimVector & current_x,
                                                    const Variable & initial_variable)
{
  computation_duration_ = ComputationDuration();

  auto start_time = std::chrono::system_clock::now();

  // Initialize variables
  current_t_ = current_t;
  current_x_ = current_x;
  variable_ = initial_variable;

  // Check optimization variables
  checkVariable();

  // Resize list
  if(delta_variable_.horizon_steps != config_.horizon_steps)
  {
    delta_variable_ = Variable(config_.horizon_steps);
  }
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
  k_list_.resize(config_.horizon_steps);
  K_list_.resize(config_.horizon_steps);
  s_list_.resize(config_.horizon_steps + 1);
  P_list_.resize(config_.horizon_steps + 1);

  // Initialize trace data
  trace_data_list_.clear();
  TraceData initial_trace_data;
  initial_trace_data.iter = 0;
  trace_data_list_.push_back(initial_trace_data);

  auto setup_time = std::chrono::system_clock::now();
  computation_duration_.setup =
      1e3 * std::chrono::duration_cast<std::chrono::duration<double>>(setup_time - start_time).count();

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
  computation_duration_.opt =
      1e3 * std::chrono::duration_cast<std::chrono::duration<double>>(end_time - setup_time).count();
  computation_duration_.solve =
      1e3 * std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();

  if(config_.print_level >= 3)
  {
    std::cout << "[FMPC] Setup duration: " << computation_duration_.setup << " [ms], optimization duration: " << computation_duration_.opt
              << " [ms]." << std::endl;
  }

  return retval == 1;
}

template<int StateDim, int InputDim, int IneqDim>
int FmpcSolver<StateDim, InputDim, IneqDim>::checkVariable() const
{
  if(initial_u_list.size() != config_.horizon_steps)
  {
    throw std::invalid_argument("initial_u_list length should be " + std::to_string(config_.horizon_steps) + " but "
                                + std::to_string(initial_u_list.size()) + ".");
  }
  if constexpr(InputDim == Eigen::Dynamic)
              {
                for(int i = 0; i < config_.horizon_steps; i++)
                {
                  double t = current_t_ + i * problem_->dt();
                  if(initial_u_list[i].size() != problem_->inputDim(t))
                  {
                    throw std::runtime_error("initial_u dimension should be " + std::to_string(problem_->inputDim(t)) + " but "
                                             + std::to_string(initial_u_list[i].size()) + ". i: " + std::to_string(i)
                                             + ", time: " + std::to_string(t));
                  }
                }
              }

  // Initialize state and cost sequence
  variable_.u_list = initial_u_list;
  variable_.x_list.resize(config_.horizon_steps + 1);
  variable_.x_list[0] = current_x;
  for(int i = 0; i < config_.horizon_steps; i++)
  {
    double t = current_t_ + i * problem_->dt();
    variable_.x_list[i + 1] = problem_->stateEq(t, variable_.x_list[i], variable_.u_list[i]);
  }
  double terminal_t = current_t_ + config_.horizon_steps * problem_->dt();
}

template<int StateDim, int InputDim, int IneqDim>
int FmpcSolver<StateDim, InputDim, IneqDim>::procOnce(int iter)
{
  if(config_.print_level >= 3)
  {
    std::cout << "[FMPC] Start iteration " << iter << std::endl;
  }

  // Append trace data
  trace_data_list_.push_back(TraceData());
  auto & trace_data = trace_data_list_.back();
  trace_data.iter = iter;

  // Step 1: calculate coefficients of linearized KKT condition
  {
    auto start_time = std::chrono::system_clock::now();

    for(int i = 0; i < config_.horizon_steps; i++)
    {
      auto & coeff = coeff_list_[i];
      double dt = problem_->dt();
      double t = current_t_ + i * dt;
      const StateDimVector & x = variable_.x_list[i];
      const StateDimVector & next_x = variable_.x_list[i + 1];
      const InputDimVector & u = variable_.u_list[i];
      const StateDimVector & lambda = variable_.lambda_list[i];
      const IneqDimVector & s = variable_.s_list[i];
      const IneqDimVector & nu = variable_.nu_list[i];

      problem_->calcStateEqDeriv(t, x, u, coeff.A, coeff.B);
      problem_->calcIneqConstDeriv(t, x, u, coeff.C, coeff.D);
      problem_->calcRunningCostDeriv(t, x, u, coeff.Lx, coeff.Lu, coeff.Lxx, coeff.Luu, coeff.Lxu);

      coeff.x_bar = problem_->stateEq(t, x, u) - next_x;
      coeff.g_bar = problem_->ineqConst(t, x, u) + s;
      coeff.Lx_bar = -1 * lambda + dt * coeff.Lx + A.transpose() * next_lambda + coeff.C.transpose() * nu;
      coeff.Lu_bar = dt * coeff.Lu + B.transpose() * next_lambda + coeff.D.transpose() * nu;
    }
    {
      auto & terminal_coeff = coeff_list_[config_.horizon_steps];
      double terminal_t = current_t_ + config_.horizon_steps * dt;
      const StateDimVector & terminal_x = variable_.x_list[config_.horizon_steps];
      const StateDimVector & terminal_lambda = variable_.lambda_list[config_.horizon_steps];
      problem_->calcTerminalCostDeriv(terminal_t, terminal_x, terminal_coeff.Lx_bar, terminal_coeff.Lxx);
      terminal_coeff.Lx_bar -= terminal_lambda;
    }

    double duration_coeff =
        1e3
        * std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::system_clock::now() - start_time)
              .count();
    trace_data.duration_coeff = duration_coeff;
    computation_duration_.coeff += duration_coeff;
  }

  // TODO check terminal

  // Step 2: backward pass, compute optimal control law and cost-to-go
  {
    auto start_time = std::chrono::system_clock::now();

    if(!backwardPass())
    {
      return -1;
    }

    double duration_backward =
        1e3
        * std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::system_clock::now() - start_time)
              .count();
    trace_data.duration_backward = duration_backward;
    computation_duration_.backward += duration_backward;
  }

  // Step 3: forward pass
  {
    auto start_time = std::chrono::system_clock::now();

    forwardPass();

    double duration_forward =
        1e3
        * std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::system_clock::now() - start_time)
              .count();
    trace_data.duration_forward = duration_forward;
    computation_duration_.forward += duration_forward;
  }

  // Step 4: update variables
  {
    auto start_time = std::chrono::system_clock::now();

    updateVariables();

    double duration_update =
        1e3
        * std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::system_clock::now() - start_time)
              .count();
    trace_data.duration_update = duration_update;
    computation_duration_.update += duration_update;
  }

  return 0;
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

  const auto & terminal_coeff = coeff_list_[config_.horizon_steps];
  InputDimVector k;
  InputStateDimMatrix K;
  StateDimVector s = -1 * terminal_coeff.Lx_bar;
  StateStateDimMatrix P = terminal_coeff.Lxx;

  s_list_[config_.horizon_steps] = s;
  P_list_[config_.horizon_steps] = P;

  for(int i = config_.horizon_steps - 1; i >= 0; i--)
  {
    // Get coefficients
    double dt = problem_->dt();
    double t = current_t_ + i * dt;
    const StateStateDimMatrix & A = coeff_list_[i].A;
    const StateInputDimMatrix & B = coeff_list_[i].B;
    const IneqStateDimMatrix & C = coeff_list_[i].C;
    const IneqInputDimMatrix & D = coeff_list_[i].D;
    const StateStateDimMatrix & Lxx = coeff_list_[i].Lxx;
    const InputInputDimMatrix & Luu = coeff_list_[i].Luu;
    const StateInputDimMatrix & Lxu = coeff_list_[i].Lxu;
    const StateDimVector & x_bar = coeff_list_[i].x_bar;
    const IneqDimVector & g_bar = coeff_list_[i].g_bar;
    const StateDimVector & Lx_bar = coeff_list_[i].Lx_bar;
    const InputDimVector & Lu_bar = coeff_list_[i].Lu_bar;

    // Pre-process for gain calculation
    {
      auto start_time_gain_pre = std::chrono::system_clock::now();

      IneqDimVector nu_s = (variable_.nu_list[i].array() / variable_.s_list[i].array()).matrix();
      IneqDimVector tilde_sub = nu_s.cwiseProduct(g_bar) - variable_.nu_list[i] + barrier_eps_ * variable_.s_list[i].cwiseInverse();
      Qxx_tilde.noalias() = dt * Lxx + C.transpose() * nu_s.asDiagonal() * C;
      Quu_tilde.noalias() = dt * Luu + D.transpose() * nu_s.asDiagonal() * D;
      Qxu_tilde.noalias() = dt * Lxu + C.transpose() * nu_s.asDiagonal() * D;
      Lx_tilde.noalias() = Lx_bar + C.transpose() * tilde_sub;
      Lu_tilde.noalias() = Lu_bar + D.transpose() * tilde_sub;

      F.noalias() = Qxx_tilde + A.transpose() * P * A;
      H.noalias() = Qxu_tilde + A.transpose() * P * B;
      G.noalias() = Quu_tilde + B.transpose() * P * B;

      computation_duration_.gain_pre += 1e3
          * std::chrono::duration_cast<std::chrono::duration<double>>(
              std::chrono::system_clock::now() - start_time_gain_pre)
          .count();
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
            std::cout << "[FMPC/Backward] G is not positive definite in Cholesky decomposition (LLT)." << std::endl;
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

      computation_duration_.gain_solve += 1e3
          * std::chrono::duration_cast<std::chrono::duration<double>>(
              std::chrono::system_clock::now() - start_time_gain_solve)
          .count();
    }

    // Post-process for gain calculation
    {
      auto start_time_gain_post = std::chrono::system_clock::now();

      s = A.transpose() * (s - P * x_bar) - Lx_tilde - H * k;
      P.noalias() = F - K.transpose() * G * K;

      computation_duration_.gain_post += 1e3
          * std::chrono::duration_cast<std::chrono::duration<double>>(
              std::chrono::system_clock::now() - start_time_gain_post)
          .count();
    }

    // Save gains
    k_list_[i] = k;
    K_list_[i] = K;
    s_list_[i] = s;
    P_list_[i] = P;
  }

  return true;
}

template<int StateDim, int InputDim, int IneqDim>
void FmpcSolver<StateDim, InputDim, IneqDim>::forwardPass()
{
  // Set initial state
  delta_variable_.x_list[0] = current_x_ - variable_.x_list[0];

  for(int i = 0; i < config_.horizon_steps + 1; i++)
  {
    delta_variable_.lambda_list[i].noalias() = P_list_[i] * delta_variable_.x_list[i] - s_list_[i];
    if(i == config_.horizon_steps)
    {
      break;
    }
    delta_variable_.u_list[i].noalias() = K_list_[i] * delta_variable_.x_list[i] + k_list_[i];
    delta_variable_.x_list[i + 1].noalias() = coeff_list_[i].A * delta_variable_.x_list[i]
                                                  + coeff_list_[i].B * delta_variable_.u_list[i] + x_bar;
  }

  for(int i = 0; i < config_.horizon_steps; i++)
  {
    delta_variable_.s_list[i].noalias() = -1
                                              * (coeff_list_[i].C * delta_variable_.x_list[i]
                                                 + coeff_list_[i].D * delta_variable_.u_list[i] + g_bar);
    delta_variable_.nu_list[i].noalias() =
        (-1
         * (variable_.nu_list[i].array() * (delta_variable_.s_list[i] + variable_.s_list[i]).array()
            - barrier_eps_)
         / variable_.s_list[i].array())
            .matrix();
  }
}

template<int StateDim, int InputDim, int IneqDim>
void FmpcSolver<StateDim, InputDim, IneqDim>::updateVariables()
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
    assert(alpha <= 1.0);
    assert(alpha > 0.0);

    computation_duration_.fraction += 1e3
        * std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::system_clock::now() - start_time_fraction)
        .count();
  }

  for(int i = 0; i < config_.horizon_steps; i++)
  {
    variable_.x_list[i] += alpha * delta_variable_.x_list[i];
    variable_.u_list[i] += alpha * delta_variable_.u_list[i];
    variable_.lambda_list[i] += alpha * delta_variable_.lambda_list[i];
    variable_.s_list[i] += alpha * delta_variable_.s_list[i];
    variable_.nu_list[i] += alpha * delta_variable_.nu_list[i];
  }
}

template<int StateDim, int InputDim, int IneqDim>
void FmpcSolver<StateDim, InputDim, IneqDim>::dumpTraceDataList(const std::string & file_path) const
{
  std::ofstream ofs(file_path);
  // clang-format off
  ofs << "iter "
      << "duration_coeff "
      << "duration_backward "
      << "duration_forward"
      << "duration_update" << std::endl;
  // clang-format on
  for(const auto & trace_data : trace_data_list_)
  {
    // clang-format off
    ofs << trace_data.iter << " "
        << trace_data.duration_coeff << " "
        << trace_data.duration_backward << " "
        << trace_data.duration_forward << " "
        << trace_data.duration_update
        << std::endl;
    // clang-format on
  }
}
} // namespace nmpc_fmpc
