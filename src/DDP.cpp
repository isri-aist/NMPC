/* Author: Masaki Murooka */

#include <chrono>
#include <stdlib.h>

#include <noc_ddp/DDP.h>

using namespace NOC;

DDP::DDP()
{
}

bool DDP::procOnce(int horizon_steps)
{
  // Step 1: differentiate dynamics and cost along new trajectory
  derivative_list_.resize(horizon_steps);
  for(int i = 0; i < horizon_steps; i++)
  {
    auto & derivative = derivative_list_[i];
    derivative.setStateDim(config_.use_state_eq_second_derivative ? problem_->stateDim() : 0);

    const StateDimVector & x = x_list_[i];
    const InputDimVector & u = u_list_[i];
    if(config_.use_state_eq_second_derivative)
    {
      problem_->calcStateqDeriv(x, u, derivative.Fx, derivative.Fu, derivative.Fxx, derivative.Fuu, derivative.Fxu);
    }
    else
    {
      problem_->calcStateqDeriv(x, u, derivative.Fx, derivative.Fu);
    }
    problem_->calcRunningCostDeriv(x, u, derivative.Lx, derivative.Lu, derivative.Lxx, derivative.Luu, derivative.Lxu);
  }
  problem_->calcTerminalCostDeriv(x_list_.back(), last_Vx_, last_Vxx_);

  // STEP 2: backward pass, compute optimal control law and cost-to-go
  while(!procBackwardPass())
  {
    dlambda_ = std::max(dlambda_ * config_.lambda_factor, config_.lambda_factor);
    lambda_ = std::max(lambda_ * dlambda_, config_.lambda_min);
    if(lambda_ > config_.lambda_max)
    {
      return false;
    }
  }

  // Check for termination due to small gradient
  double k_rel_norm = 0;
  for(int i = 0; i < horizon_steps; i++)
  {
    k_rel_norm = std::max(k_rel_norm, k_list_[i].norm() / (u_list_[i].norm() + 1.0));
  }
  if(k_rel_norm < config_.k_rel_norm_thre && lambda_ < config_.lambda_thre)
  {
    return true;
  }

  // STEP 3: line-search to find new control sequence, trajectory, cost
  // \todo Parallel line-search by broadcasting

  return true;
}

bool DDP::procBackwardPass()
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
  InputInputDimMatrix Quu_F;

  InputStateDimMatrix VxFxu;
  InputInputDimMatrix VxFuu;

  InputDimVector k;
  InputStateDimMatrix K;

  dV_.setZero();
  k_list_.resize(horizon_steps);
  K_list_.resize(horizon_steps);

  for(int i = horizon_steps - 1; i >= 0; i--)
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
      throw std::runtime_error("Input constraint is not supported yet.");
    }
    else
    {
      Eigen::LLT<InputInputDimMatrix> llt_Quu_F(Quu_F);
      if(llt_Quu_F.info() == Eigen::NumericalIssue)
      {
        if(config_.verbose_print)
        {
        std::cout << "Quu_F is not positive definite in Cholesky decomposition (LLT)" << std::endl;
        }
        return false;
      }
      k = -1 * llt_Quu_F.solve(Qu);
      K = -1 * llt_Quu_F.solve(Qux_reg);
    }

    // Update cost-to-go approximation
    dV_ += Eigen::Vector2d(k.transpose() * Qu, 0.5 * k.transpose() * Quu * k);
    Vx = Qx + K.transpose() * Quu * k + K.transpose() * Qu + Qux.transpose() * k;
    Vxx = Qxx + K.transpose() * Quu * K + K.transpose() * Qux + Qux.transpose() * K;
    Vxx = 0.5 * (Vxx + Vxx.transpose());

    // Save gains
    k_list_[i] = k;
    K_list_[i] = K;
  }

  return true;
}
