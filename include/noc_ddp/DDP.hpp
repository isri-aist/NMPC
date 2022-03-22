/* Author: Masaki Murooka */

#include <chrono>
#include <iostream>
#include <fstream>

namespace NOC
{
template<int StateDim, int InputDim>
DDPProblem<StateDim, InputDim>::DDPProblem(double dt, int state_dim, int input_dim)
: dt_(dt), state_dim_(state_dim), input_dim_(input_dim)
{
  // Check dimension is positive
  if(state_dim_ <= 0)
  {
    throw std::runtime_error("state_dim must be positive: " + std::to_string(state_dim_) + " <= 0");
  }
  if(input_dim_ < 0)
  {
    throw std::runtime_error("input_dim must be non-negative: " + std::to_string(input_dim_) + " < 0");
  }

  // Check dimension consistency
  if constexpr(StateDim != Eigen::Dynamic)
  {
    if(state_dim_ != StateDim)
    {
      throw std::runtime_error("state_dim is inconsistent with template parameter: " + std::to_string(state_dim_)
                               + " != " + std::to_string(StateDim));
    }
  }
  if constexpr(InputDim != Eigen::Dynamic)
  {
    if(input_dim_ != InputDim)
    {
      throw std::runtime_error("input_dim is inconsistent with template parameter: " + std::to_string(input_dim_)
                               + " != " + std::to_string(InputDim));
    }
  }
}

template<int StateDim, int InputDim>
DDPSolver<StateDim, InputDim>::DDPSolver(const std::shared_ptr<DDPProblem<StateDim, InputDim>> & problem)
: problem_(problem)
{
}

template<int StateDim, int InputDim>
bool DDPSolver<StateDim, InputDim>::solve(double current_t,
                                          const StateDimVector & current_x,
                                          const std::vector<InputDimVector> & initial_u_list)
{
  // Initialize variables
  trace_data_list_.clear();
  current_t_ = current_t;
  lambda_ = config_.initial_lambda;
  dlambda_ = config_.initial_dlambda;

  // Resize list
  x_candidate_list_.resize(config_.horizon_steps + 1);
  u_candidate_list_.resize(config_.horizon_steps);
  cost_candidate_list_.resize(config_.horizon_steps + 1);
  derivative_list_.resize(config_.horizon_steps);
  for(auto & derivative : derivative_list_)
  {
    derivative.setStateDim(config_.use_state_eq_second_derivative ? problem_->stateDim() : 0);
  }
  k_list_.resize(config_.horizon_steps);
  K_list_.resize(config_.horizon_steps);

  // Initialize state and cost sequence
  u_list_ = initial_u_list;
  x_list_.resize(config_.horizon_steps + 1);
  cost_list_.resize(config_.horizon_steps + 1);
  x_list_[0] = current_x;
  for(int i = 0; i < config_.horizon_steps; i++)
  {
    double t = current_t_ + i * problem_->dt();
    x_list_[i + 1] = problem_->stateEq(t, x_list_[i], u_list_[i]);
    cost_list_[i] = problem_->runningCost(t, x_list_[i], u_list_[i]);
  }
  double terminal_t = current_t_ + config_.horizon_steps * problem_->dt();
  cost_list_[config_.horizon_steps] = problem_->terminalCost(terminal_t, x_list_[config_.horizon_steps]);

  // Optimization loop
  int retval = 0;
  for(int iter = 0; iter < config_.max_iter; iter++)
  {
    retval = procOnce(iter);
    if(retval != 0)
    {
      break;
    }
  }

  return retval == 1;
}

template<int StateDim, int InputDim>
int DDPSolver<StateDim, InputDim>::procOnce(int iter)
{
  // Append trace data
  trace_data_list_.push_back(TraceData());
  auto & trace_data = trace_data_list_.back();

  trace_data.iter = iter;

  // Step 1: differentiate dynamics and cost along new trajectory
  {
    auto start_time = std::chrono::system_clock::now();

    for(int i = 0; i < config_.horizon_steps; i++)
    {
      auto & derivative = derivative_list_[i];

      double t = current_t_ + i * problem_->dt();
      const StateDimVector & x = x_list_[i];
      const InputDimVector & u = u_list_[i];
      if(config_.use_state_eq_second_derivative)
      {
        problem_->calcStateqDeriv(t, x, u, derivative.Fx, derivative.Fu, derivative.Fxx, derivative.Fuu,
                                  derivative.Fxu);
      }
      else
      {
        problem_->calcStateqDeriv(t, x, u, derivative.Fx, derivative.Fu);
      }
      problem_->calcRunningCostDeriv(t, x, u, derivative.Lx, derivative.Lu, derivative.Lxx, derivative.Luu,
                                     derivative.Lxu);
    }
    double terminal_t = current_t_ + config_.horizon_steps * problem_->dt();
    problem_->calcTerminalCostDeriv(terminal_t, x_list_[config_.horizon_steps], last_Vx_, last_Vxx_);

    trace_data.duration_derivative =
        1e3
        * std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::system_clock::now() - start_time)
              .count();
  }

  // STEP 2: backward pass, compute optimal control law and cost-to-go
  {
    auto start_time = std::chrono::system_clock::now();

    while(!backwardPass())
    {
      // Increase lambda
      dlambda_ = std::max(dlambda_ * config_.lambda_factor, config_.lambda_factor);
      lambda_ = std::max(lambda_ * dlambda_, config_.lambda_min);
      if(lambda_ > config_.lambda_max)
      {
        if(config_.verbose_print)
        {
          std::cout << "[DDP] Failure due to large lambda." << std::endl;
        }
        return -1; // Failure
      }
    }

    trace_data.duration_backward =
        1e3
        * std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::system_clock::now() - start_time)
              .count();
  }

  // Check for termination due to small gradient
  double k_rel_norm = 0;
  for(int i = 0; i < config_.horizon_steps; i++)
  {
    k_rel_norm = std::max(k_rel_norm, k_list_[i].norm() / (u_list_[i].norm() + 1.0));
  }
  trace_data.k_rel_norm = k_rel_norm;
  if(k_rel_norm < config_.k_rel_norm_thre && lambda_ < config_.lambda_thre)
  {
    if(config_.verbose_print)
    {
      std::cout << "[DDP] Terminate due to small gradient." << std::endl;
    }
    return 1; // Terminate
  }

  // STEP 3: forward pass, line-search to find new control sequence, trajectory, cost
  // \todo Parallel line-search by broadcasting
  bool forward_pass_success = false;
  double cost_update_actual = 0;
  {
    auto start_time = std::chrono::system_clock::now();

    double cost_update_expected = 0;
    double cost_update_ratio = 0;
    for(int i = 0; i < config_.alpha_list.size(); i++)
    {
      double alpha = config_.alpha_list[i];

      forwardPass(alpha);

      cost_update_actual = cost_list_.sum() - cost_candidate_list_.sum();
      cost_update_expected = -1 * alpha * (dV_[0] + alpha * dV_[1]);
      cost_update_ratio = cost_update_actual / cost_update_expected;
      if(cost_update_expected < 0)
      {
        std::cout << "[DDP] Value is not expected to decrease." << std::endl;
        cost_update_ratio = (cost_update_actual >= 0 ? 1 : -1);
      }
      if(cost_update_ratio > config_.cost_update_ratio_thre)
      {
        forward_pass_success = true;
        trace_data.alpha = alpha;
        break;
      }

      if(i == config_.alpha_list.size() - 1)
      {
        trace_data.alpha = 0;
      }
    }
    trace_data.cost_update_actual = cost_update_actual;
    trace_data.cost_update_expected = cost_update_expected;
    trace_data.cost_update_ratio = cost_update_ratio;

    trace_data.duration_forward =
        1e3
        * std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::system_clock::now() - start_time)
              .count();
  }

  // STEP 4: accept step (or not)
  int retval = 0; // Continue
  if(forward_pass_success)
  {
    // Decrease lambda
    dlambda_ = std::min(dlambda_ / config_.lambda_factor, 1 / config_.lambda_factor);
    if(lambda_ > config_.lambda_min)
    {
      lambda_ *= dlambda_;
    }
    else
    {
      lambda_ = 0;
    }

    // Accept changes
    x_list_ = x_candidate_list_;
    u_list_ = u_candidate_list_;
    cost_list_ = cost_candidate_list_;

    // Check for termination due to small cost update
    if(cost_update_actual < config_.cost_update_thre)
    {
      if(config_.verbose_print)
      {
        std::cout << "[DDP] Terminate due to small cost update." << std::endl;
      }
      retval = 1; // Terminate
    }
  }
  else
  {
    // Increase lambda
    dlambda_ = std::max(dlambda_ * config_.lambda_factor, config_.lambda_factor);
    lambda_ = std::max(lambda_ * dlambda_, config_.lambda_min);
    if(lambda_ > config_.lambda_max)
    {
      if(config_.verbose_print)
      {
        std::cout << "[DDP] Failure due to large lambda." << std::endl;
      }
      retval = -1; // Failure
    }
  }

  trace_data.cost = cost_list_.sum();
  trace_data.lambda = lambda_;
  trace_data.dlambda = dlambda_;

  return retval;
}

template<int StateDim, int InputDim>
bool DDPSolver<StateDim, InputDim>::backwardPass()
{
  // To avoid memory allocation costs, the vector and matrix variables are created outside of loop
  StateDimVector Vx = last_Vx_;
  StateStateDimMatrix Vxx = last_Vxx_;
  StateStateDimMatrix Vxx_reg;

  InputDimVector Qu;
  StateDimVector Qx;
  InputStateDimMatrix Qux;
  InputInputDimMatrix Quu;
  StateStateDimMatrix Qxx;
  InputStateDimMatrix Qux_reg;
  InputInputDimMatrix Quu_F;

  InputStateDimMatrix VxFxu;
  InputInputDimMatrix VxFuu;

  InputDimVector k;
  InputStateDimMatrix K;

  dV_.setZero();

  for(int i = config_.horizon_steps - 1; i >= 0; i--)
  {
    // Get derivatives
    const StateStateDimMatrix & Fx = derivative_list_[i].Fx;
    const StateInputDimMatrix & Fu = derivative_list_[i].Fu;
    const std::vector<StateStateDimMatrix> & Fxx = derivative_list_[i].Fxx;
    const std::vector<InputInputDimMatrix> & Fuu = derivative_list_[i].Fuu;
    const std::vector<StateInputDimMatrix> & Fxu = derivative_list_[i].Fxu;
    const StateDimVector & Lx = derivative_list_[i].Lx;
    const InputDimVector & Lu = derivative_list_[i].Lu;
    const StateStateDimMatrix & Lxx = derivative_list_[i].Lxx;
    const InputInputDimMatrix & Luu = derivative_list_[i].Luu;
    const StateInputDimMatrix & Lxu = derivative_list_[i].Lxu;

    // Calculate Q
    Qu = Lu + Fu.transpose() * Vx;

    Qx = Lx + Fx.transpose() * Vx;

    Qux = Lxu.transpose() + Fu.transpose() * Vxx * Fx;
    if(config_.use_state_eq_second_derivative)
    {
      throw std::runtime_error("Vector-tensor product is not implemented yet.");
      // \todo Need operation to compute a matrix by vector and tensor product
      // VxFxu = Vx * Fxu;
      // Qux += VxFxu
    }

    Quu = Luu + Fu.transpose() * Vxx * Fu;
    if(config_.use_state_eq_second_derivative)
    {
      throw std::runtime_error("Vector-tensor product is not implemented yet.");
      // \todo Need operation to compute a matrix by vector and tensor product
      // VxFuu = Vx * Fuu;
      // Quu += VxFuu;
    }

    Qxx = Lxx + Fx.transpose() * Vxx * Fx;
    if(config_.use_state_eq_second_derivative)
    {
      throw std::runtime_error("Vector-tensor product is not implemented yet.");
      // \todo Need operation to compute a matrix by vector and tensor product
      // Qxx += Vx * Fxx;
    }

    // Calculate regularization
    Vxx_reg = Vxx;
    if(config_.reg_type == 2)
    {
      Vxx_reg += lambda_ * StateStateDimMatrix::Identity();
    }

    Qux_reg = Lxu.transpose() + Fu.transpose() * Vxx_reg * Fx;
    if(config_.use_state_eq_second_derivative)
    {
      Qux_reg += VxFxu;
    }

    Quu_F = Luu + Fu.transpose() * Vxx_reg * Fu;
    if(config_.use_state_eq_second_derivative)
    {
      Quu_F += VxFuu;
    }
    if(config_.reg_type == 1)
    {
      Quu_F += lambda_ * InputInputDimMatrix::Identity();
    }

    // Calculate gains
    if(config_.with_input_constraint)
    {
      // \todo Calculate gains with considering constraints
      throw std::runtime_error("Input constraint is not supported yet.");
    }
    else
    {
      Eigen::LLT<InputInputDimMatrix> llt_Quu_F(Quu_F);
      if(llt_Quu_F.info() == Eigen::NumericalIssue)
      {
        if(config_.verbose_print)
        {
          std::cout << "[DDP] Quu_F is not positive definite in Cholesky decomposition (LLT)." << std::endl;
        }
        return false;
      }
      k = -1 * llt_Quu_F.solve(Qu);
      K = -1 * llt_Quu_F.solve(Qux_reg);
    }

    // Update cost-to-go approximation
    dV_ += Eigen::Vector2d(k.dot(Qu), 0.5 * k.dot(Quu * k));
    Vx = Qx + K.transpose() * Quu * k + K.transpose() * Qu + Qux.transpose() * k;
    Vxx = Qxx + K.transpose() * Quu * K + K.transpose() * Qux + Qux.transpose() * K;
    Vxx = 0.5 * (Vxx + Vxx.transpose());

    // Save gains
    k_list_[i] = k;
    K_list_[i] = K;
  }

  return true;
}

template<int StateDim, int InputDim>
void DDPSolver<StateDim, InputDim>::forwardPass(double alpha)
{
  // Set initial state
  x_candidate_list_[0] = x_list_[0];

  for(int i = 0; i < config_.horizon_steps; i++)
  {
    // Calculate input
    u_candidate_list_[i] = u_list_[i] + alpha * k_list_[i] + K_list_[i] * (x_candidate_list_[i] - x_list_[i]);

    // \todo Impose constraints on input

    // Calculate next state and cost
    double t = current_t_ + i * problem_->dt();
    x_candidate_list_[i + 1] = problem_->stateEq(t, x_candidate_list_[i], u_candidate_list_[i]);
    cost_candidate_list_[i] = problem_->runningCost(t, x_candidate_list_[i], u_candidate_list_[i]);
  }
  double terminal_t = current_t_ + config_.horizon_steps * problem_->dt();
  cost_candidate_list_[config_.horizon_steps] =
      problem_->terminalCost(terminal_t, x_candidate_list_[config_.horizon_steps]);
}

template<int StateDim, int InputDim>
void DDPSolver<StateDim, InputDim>::dumpTraceDataList(const std::string & file_path) const
{
  std::ofstream ofs(file_path);
  ofs << "iter "
      << "cost "
      << "lambda "
      << "dlambda "
      << "alpha "
      << "k_rel_norm "
      << "cost_update_actual "
      << "cost_update_expected "
      << "cost_update_ratio "
      << "duration_derivative "
      << "duration_backward "
      << "duration_forward" << std::endl;
  for(const auto & trace_data : trace_data_list_)
  {
    ofs << trace_data.iter << " "
        << trace_data.cost << " "
        << trace_data.lambda << " "
        << trace_data.dlambda << " "
        << trace_data.alpha << " "
        << trace_data.k_rel_norm << " "
        << trace_data.cost_update_actual << " "
        << trace_data.cost_update_expected << " "
        << trace_data.cost_update_ratio << " "
        << trace_data.duration_derivative << " "
        << trace_data.duration_backward << " "
        << trace_data.duration_forward << std::endl;
  }
}
} // namespace NOC
