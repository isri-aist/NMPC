/* Author: Masaki Murooka */

#include <chrono>
#include <fstream>
#include <iostream>

#include <nmpc_ddp/BoxQP.h>

namespace
{
template<class Clock>
double calcDuration(const std::chrono::time_point<Clock> & start_time, const std::chrono::time_point<Clock> & end_time)
{
  return 1e3 * std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
}
} // namespace

namespace nmpc_ddp
{
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
  computation_duration_ = ComputationDuration();

  auto start_time = std::chrono::system_clock::now();

  // Initialize variables
  current_t_ = current_t;
  lambda_ = config_.initial_lambda;
  dlambda_ = config_.initial_dlambda;

  // Check initial_u_list
  if(static_cast<int>(initial_u_list.size()) != config_.horizon_steps)
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

  // Resize list
  candidate_control_data_.x_list.resize(config_.horizon_steps + 1);
  candidate_control_data_.u_list.resize(config_.horizon_steps);
  candidate_control_data_.cost_list.resize(config_.horizon_steps + 1);
  int outer_dim = config_.use_state_eq_second_derivative ? problem_->stateDim() : 0;
  if constexpr(InputDim == Eigen::Dynamic)
  {
    derivative_list_.clear();
    for(int i = 0; i < config_.horizon_steps; i++)
    {
      double t = current_t_ + i * problem_->dt();
      derivative_list_.push_back(Derivative(problem_->stateDim(), problem_->inputDim(t), outer_dim));
    }
  }
  else
  {
    // This assumes that the dimension is fixed, but it is efficient because it preserves existing elements
    derivative_list_.resize(config_.horizon_steps, Derivative(problem_->stateDim(), problem_->inputDim(), outer_dim));
  }
  k_list_.resize(config_.horizon_steps);
  K_list_.resize(config_.horizon_steps);

  // Initialize state and cost sequence
  control_data_.u_list = initial_u_list;
  control_data_.x_list.resize(config_.horizon_steps + 1);
  control_data_.cost_list.resize(config_.horizon_steps + 1);
  control_data_.x_list[0] = current_x;
  for(int i = 0; i < config_.horizon_steps; i++)
  {
    double t = current_t_ + i * problem_->dt();
    control_data_.x_list[i + 1] = problem_->stateEq(t, control_data_.x_list[i], control_data_.u_list[i]);
    control_data_.cost_list[i] = problem_->runningCost(t, control_data_.x_list[i], control_data_.u_list[i]);
  }
  double terminal_t = current_t_ + config_.horizon_steps * problem_->dt();
  control_data_.cost_list[config_.horizon_steps] =
      problem_->terminalCost(terminal_t, control_data_.x_list[config_.horizon_steps]);

  // Initialize trace data
  trace_data_list_.clear();
  TraceData initial_trace_data;
  initial_trace_data.iter = 0;
  initial_trace_data.cost = control_data_.cost_list.sum();
  initial_trace_data.lambda = lambda_;
  initial_trace_data.dlambda = dlambda_;
  trace_data_list_.push_back(initial_trace_data);

  if(config_.print_level >= 3)
  {
    std::cout << "[DDP] Initial cost: " << control_data_.cost_list.sum() << std::endl;
  }

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

  if(config_.print_level >= 3)
  {
    std::cout << "[DDP] Final cost: " << control_data_.cost_list.sum() << std::endl;
  }

  auto end_time = std::chrono::system_clock::now();
  computation_duration_.opt = calcDuration(setup_time, end_time);
  computation_duration_.solve = calcDuration(start_time, end_time);

  if(config_.print_level >= 3)
  {
    std::cout << "[DDP] Setup duration: " << computation_duration_.setup
              << " [ms], optimization duration: " << computation_duration_.opt << " [ms]." << std::endl;
  }

  return retval == 1;
}

template<int StateDim, int InputDim>
int DDPSolver<StateDim, InputDim>::procOnce(int iter)
{
  if(config_.print_level >= 3)
  {
    std::cout << "[DDP] Start iteration " << iter << std::endl;
  }

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
      const StateDimVector & x = control_data_.x_list[i];
      const InputDimVector & u = control_data_.u_list[i];
      if(config_.use_state_eq_second_derivative)
      {
        problem_->calcStateEqDeriv(t, x, u, derivative.Fx, derivative.Fu, derivative.Fxx, derivative.Fuu,
                                   derivative.Fxu);
      }
      else
      {
        problem_->calcStateEqDeriv(t, x, u, derivative.Fx, derivative.Fu);
      }
      problem_->calcRunningCostDeriv(t, x, u, derivative.Lx, derivative.Lu, derivative.Lxx, derivative.Luu,
                                     derivative.Lxu);
    }
    double terminal_t = current_t_ + config_.horizon_steps * problem_->dt();
    problem_->calcTerminalCostDeriv(terminal_t, control_data_.x_list[config_.horizon_steps], last_Vx_, last_Vxx_);

    double duration_derivative = calcDuration(start_time, std::chrono::system_clock::now());
    trace_data.duration_derivative = duration_derivative;
    computation_duration_.derivative += duration_derivative;
  }

  // Step 2: backward pass, compute optimal control law and cost-to-go
  {
    auto start_time = std::chrono::system_clock::now();

    while(!backwardPass())
    {
      // Increase lambda
      dlambda_ = std::max(dlambda_ * config_.lambda_factor, config_.lambda_factor);
      lambda_ = std::max(lambda_ * dlambda_, config_.lambda_min);
      if(lambda_ > config_.lambda_max)
      {
        if(config_.print_level >= 1)
        {
          std::cout << "[DDP/Backward] Failure due to large lambda. (time: " << current_t_ << ", iter: " << iter << ")"
                    << std::endl;
        }
        return -1; // Failure
      }
      if(config_.print_level >= 3)
      {
        std::cout << "[DDP/Backward] Increase lambda to " << lambda_ << std::endl;
      }
    }

    double duration_backward = calcDuration(start_time, std::chrono::system_clock::now());
    trace_data.duration_backward = duration_backward;
    computation_duration_.backward += duration_backward;
  }

  // Check for termination due to small gradient
  double k_rel_norm = 0;
  for(int i = 0; i < config_.horizon_steps; i++)
  {
    k_rel_norm = std::max(k_rel_norm, k_list_[i].norm() / (control_data_.u_list[i].norm() + 1.0));
  }
  trace_data.k_rel_norm = k_rel_norm;
  if(k_rel_norm < config_.k_rel_norm_thre && lambda_ < config_.lambda_thre)
  {
    if(config_.print_level >= 2)
    {
      std::cout << "[DDP] Terminate due to small gradient. (time: " << current_t_ << ", iter: " << iter << ")"
                << std::endl;
    }
    return 1; // Terminate
  }

  // Step 3: forward pass, line-search to find new control sequence, trajectory, cost
  bool forward_pass_success = false;
  double cost_update_actual = 0;
  {
    auto start_time = std::chrono::system_clock::now();

    double alpha = 0;
    double cost_update_expected = 0;
    double cost_update_ratio = 0;
    for(int i = 0; i < config_.alpha_list.size(); i++)
    {
      alpha = config_.alpha_list[i];

      forwardPass(alpha);

      cost_update_actual = control_data_.cost_list.sum() - candidate_control_data_.cost_list.sum();
      cost_update_expected = -1 * alpha * (dV_[0] + alpha * dV_[1]);
      cost_update_ratio = cost_update_actual / cost_update_expected;
      if(cost_update_expected < 0)
      {
        if((!config_.with_input_constraint && config_.print_level >= 0)
           || (config_.with_input_constraint && config_.print_level >= 2))
        {
          std::cout << "[DDP/Forward] Value is not expected to decrease." << std::endl;
        }
        cost_update_ratio = (cost_update_actual >= 0 ? 1 : -1);
      }
      if(cost_update_ratio > config_.cost_update_ratio_thre)
      {
        forward_pass_success = true;
        break;
      }
    }
    trace_data.alpha = alpha;
    trace_data.cost_update_actual = cost_update_actual;
    trace_data.cost_update_expected = cost_update_expected;
    trace_data.cost_update_ratio = cost_update_ratio;

    double duration_forward = calcDuration(start_time, std::chrono::system_clock::now());
    trace_data.duration_forward = duration_forward;
    computation_duration_.forward += duration_forward;
  }
  if(!forward_pass_success && config_.print_level >= 3)
  {
    std::cout << "[DDP] Forward pass failed." << std::endl;
  }

  // Step 4: accept step (or not)
  int retval = 0; // Continue
  if(forward_pass_success)
  {
    // Accept changes
    control_data_.x_list = candidate_control_data_.x_list;
    control_data_.u_list = candidate_control_data_.u_list;
    control_data_.cost_list = candidate_control_data_.cost_list;

    // Check for termination due to small cost update
    if(cost_update_actual < config_.cost_update_thre)
    {
      if(config_.print_level >= 2)
      {
        std::cout << "[DDP] Terminate due to small cost update. (time: " << current_t_ << ", iter: " << iter << ")"
                  << std::endl;
      }
      retval = 1; // Terminate
    }

    // Decrease lambda
    dlambda_ = std::min(dlambda_ / config_.lambda_factor, 1 / config_.lambda_factor);
    if(lambda_ >= config_.lambda_min)
    {
      lambda_ *= dlambda_;
    }
    else
    {
      lambda_ = 0;
    }
    if(config_.print_level >= 3)
    {
      std::cout << "[DDP/Forward] Decrease lambda to " << lambda_ << std::endl;
    }
  }
  else
  {
    // Increase lambda
    dlambda_ = std::max(dlambda_ * config_.lambda_factor, config_.lambda_factor);
    lambda_ = std::max(lambda_ * dlambda_, config_.lambda_min);
    if(lambda_ > config_.lambda_max)
    {
      if(config_.print_level >= 1)
      {
        std::cout << "[DDP/Forward] Failure due to large lambda. (time: " << current_t_ << ", iter: " << iter << ")"
                  << std::endl;
      }
      retval = -1; // Failure
    }
    if(config_.print_level >= 3)
    {
      std::cout << "[DDP/Forward] Increase lambda to " << lambda_ << std::endl;
    }
  }

  trace_data.cost = control_data_.cost_list.sum();
  trace_data.lambda = lambda_;
  trace_data.dlambda = dlambda_;

  return retval;
}

template<int StateDim, int InputDim>
bool DDPSolver<StateDim, InputDim>::backwardPass()
{
  // To avoid repetitive memory allocation, the vector and matrix variables are created outside of loop
  StateDimVector Vx = last_Vx_;
  StateStateDimMatrix Vxx = last_Vxx_;
  StateStateDimMatrix Vxx_reg;
  StateStateDimMatrix Vxx_symmetric;

  InputDimVector Qu;
  StateDimVector Qx;
  InputStateDimMatrix Qux;
  InputInputDimMatrix Quu;
  StateStateDimMatrix Qxx;
  InputStateDimMatrix Qux_reg;
  InputInputDimMatrix Quu_F;

  InputStateDimMatrix VxFux;
  InputInputDimMatrix VxFuu;

  InputDimVector k;
  InputStateDimMatrix K;

  dV_.setZero();

  for(int i = config_.horizon_steps - 1; i >= 0; i--)
  {
    // Get derivatives
    double t = current_t_ + i * problem_->dt();
    const StateStateDimMatrix & Fx = derivative_list_[i].Fx;
    const StateInputDimMatrix & Fu = derivative_list_[i].Fu;
    // const std::vector<StateStateDimMatrix> & Fxx = derivative_list_[i].Fxx;
    // const std::vector<InputInputDimMatrix> & Fuu = derivative_list_[i].Fuu;
    // const std::vector<StateInputDimMatrix> & Fxu = derivative_list_[i].Fxu;
    const StateDimVector & Lx = derivative_list_[i].Lx;
    const InputDimVector & Lu = derivative_list_[i].Lu;
    const StateStateDimMatrix & Lxx = derivative_list_[i].Lxx;
    const InputInputDimMatrix & Luu = derivative_list_[i].Luu;
    const StateInputDimMatrix & Lxu = derivative_list_[i].Lxu;
    int input_dim = static_cast<int>(Fu.cols());

    // Calculate Q
    auto start_time_Q = std::chrono::system_clock::now();

    Qu.noalias() = Lu + Fu.transpose() * Vx;

    Qx.noalias() = Lx + Fx.transpose() * Vx;

    Qux.noalias() = Lxu.transpose() + Fu.transpose() * Vxx * Fx;
    if(config_.use_state_eq_second_derivative)
    {
      throw std::runtime_error("Vector-tensor product is not implemented yet.");
      // \todo Need operation to compute a matrix by vector and tensor product
      // VxFux = Vx * Fxu.transpose();
      // Qux += VxFux
    }

    Quu.noalias() = Luu + Fu.transpose() * Vxx * Fu;
    if(config_.use_state_eq_second_derivative)
    {
      throw std::runtime_error("Vector-tensor product is not implemented yet.");
      // \todo Need operation to compute a matrix by vector and tensor product
      // VxFuu = Vx * Fuu;
      // Quu += VxFuu;
    }

    Qxx.noalias() = Lxx + Fx.transpose() * Vxx * Fx;
    if(config_.use_state_eq_second_derivative)
    {
      throw std::runtime_error("Vector-tensor product is not implemented yet.");
      // \todo Need operation to compute a matrix by vector and tensor product
      // Qxx += Vx * Fxx;
    }

    computation_duration_.Q += calcDuration(start_time_Q, std::chrono::system_clock::now());

    // Calculate regularization
    auto start_time_reg = std::chrono::system_clock::now();

    Vxx_reg = Vxx;
    if(config_.reg_type == 2)
    {
      Vxx_reg.diagonal().array() += lambda_;
    }

    Qux_reg.noalias() = Lxu.transpose() + Fu.transpose() * Vxx_reg * Fx;
    if(config_.use_state_eq_second_derivative)
    {
      Qux_reg += VxFux;
    }

    Quu_F.noalias() = Luu + Fu.transpose() * Vxx_reg * Fu;
    if(config_.use_state_eq_second_derivative)
    {
      Quu_F += VxFuu;
    }
    if(config_.reg_type == 1)
    {
      Quu_F.diagonal().array() += lambda_;
    }

    computation_duration_.reg += calcDuration(start_time_reg, std::chrono::system_clock::now());

    // Calculate gains
    auto start_time_gain = std::chrono::system_clock::now();

    if(input_dim > 0)
    {
      if(config_.with_input_constraint)
      {
        InputDimVector initial_k;
        if(i == config_.horizon_steps - 1)
        {
          initial_k.setZero(input_dim);
        }
        else
        {
          if(k_list_[i + 1].size() == input_dim)
          {
            initial_k = k_list_[i + 1];
          }
          else
          {
            initial_k.setZero(input_dim);
          }
        }

        BoxQP<Eigen::Dynamic> qp(static_cast<int>(Quu_F.cols()));
        const auto & u_limits = input_limits_func_(t);
        k = qp.solve(Quu_F, Qu, u_limits[0] - control_data_.u_list[i], u_limits[1] - control_data_.u_list[i],
                     initial_k);
        if(qp.retval_ < 0)
        {
          if(config_.print_level >= 1)
          {
            std::cout << "[DDP/Backward] Failed BoxQP: " << qp.retstr_.at(qp.retval_) << std::endl;
          }
          return false;
        }

        const auto & free_idxs = qp.free_idxs_;
        K.setZero(input_dim, problem_->stateDim());
        if(free_idxs.size() > 0)
        {
          Eigen::MatrixXd Qux_reg_free(free_idxs.size(), problem_->stateDim());
          for(size_t j = 0; j < free_idxs.size(); j++)
          {
            Qux_reg_free.row(j) = Qux_reg.row(free_idxs[j]);
          }
          Eigen::MatrixXd K_free = -1 * qp.llt_free_->solve(Qux_reg_free);
          for(size_t j = 0; j < free_idxs.size(); j++)
          {
            K.row(free_idxs[j]) = K_free.row(j);
          }
        }
      }
      else
      {
        Eigen::LLT<InputInputDimMatrix> llt_Quu_F(Quu_F);
        if(llt_Quu_F.info() == Eigen::NumericalIssue)
        {
          if(config_.print_level >= 1)
          {
            std::cout << "[DDP/Backward] Quu_F is not positive definite in Cholesky decomposition (LLT)." << std::endl;
          }
          return false;
        }
        k = -1 * llt_Quu_F.solve(Qu);
        K = -1 * llt_Quu_F.solve(Qux_reg);
      }
    }
    else
    {
      k.setZero(0);
      K.setZero(0, problem_->stateDim());
    }

    computation_duration_.gain += calcDuration(start_time_gain, std::chrono::system_clock::now());

    // Update cost-to-go approximation
    dV_ += Eigen::Vector2d(k.dot(Qu), 0.5 * k.dot(Quu * k));
    Vx.noalias() = Qx + K.transpose() * Quu * k + K.transpose() * Qu + Qux.transpose() * k;
    Vxx.noalias() = Qxx + K.transpose() * Quu * K + K.transpose() * Qux + Qux.transpose() * K;
    Vxx_symmetric = 0.5 * (Vxx + Vxx.transpose());
    Vxx = Vxx_symmetric;

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
  candidate_control_data_.x_list[0] = control_data_.x_list[0];

  for(int i = 0; i < config_.horizon_steps; i++)
  {
    // Calculate input
    candidate_control_data_.u_list[i] = control_data_.u_list[i] + alpha * k_list_[i]
                                        + K_list_[i] * (candidate_control_data_.x_list[i] - control_data_.x_list[i]);

    // \todo Impose constraints on input

    // Calculate next state and cost
    double t = current_t_ + i * problem_->dt();
    candidate_control_data_.x_list[i + 1] =
        problem_->stateEq(t, candidate_control_data_.x_list[i], candidate_control_data_.u_list[i]);
    candidate_control_data_.cost_list[i] =
        problem_->runningCost(t, candidate_control_data_.x_list[i], candidate_control_data_.u_list[i]);
  }
  double terminal_t = current_t_ + config_.horizon_steps * problem_->dt();
  candidate_control_data_.cost_list[config_.horizon_steps] =
      problem_->terminalCost(terminal_t, candidate_control_data_.x_list[config_.horizon_steps]);
}

template<int StateDim, int InputDim>
void DDPSolver<StateDim, InputDim>::dumpTraceDataList(const std::string & file_path) const
{
  std::ofstream ofs(file_path);
  // clang-format off
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
  // clang-format on
  for(const auto & trace_data : trace_data_list_)
  {
    // clang-format off
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
        << trace_data.duration_forward
        << std::endl;
    // clang-format on
  }
}
} // namespace nmpc_ddp
