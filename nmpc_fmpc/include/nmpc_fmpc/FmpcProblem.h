/* Author: Masaki Murooka */

#pragma once

#include <nmpc_ddp/DDPProblem.h>

namespace nmpc_fmpc
{
/** \brief Fast MPC problem.
    \tparam StateDim state dimension (fixed only)
    \tparam InputDim input dimension (fixed or dynamic (i.e., Eigen::Dynamic))
    \tparam IneqDim inequality dimension (fixed or dynamic (i.e., Eigen::Dynamic))
 */
template<int StateDim, int InputDim, int IneqDim>
class FmpcProblem : public nmpc_ddp::DDPProblem<StateDim, InputDim>
{
public:
  /** \brief Type of vector of state dimension. */
  using StateDimVector = typename nmpc_ddp::DDPProblem<StateDim, InputDim>::StateDimVector;

  /** \brief Type of vector of input dimension. */
  using InputDimVector = typename nmpc_ddp::DDPProblem<StateDim, InputDim>::InputDimVector;

  /** \brief Type of vector of inequality dimension. */
  using IneqDimVector = Eigen::Matrix<double, IneqDim, 1>;

  /** \brief Type of matrix of state x state dimension. */
  using StateStateDimMatrix = typename nmpc_ddp::DDPProblem<StateDim, InputDim>::StateStateDimMatrix;

  /** \brief Type of matrix of input x input dimension. */
  using InputInputDimMatrix = typename nmpc_ddp::DDPProblem<StateDim, InputDim>::InputInputDimMatrix;

  /** \brief Type of matrix of state x input dimension. */
  using StateInputDimMatrix = typename nmpc_ddp::DDPProblem<StateDim, InputDim>::StateInputDimMatrix;

  /** \brief Type of matrix of input x state dimension. */
  using InputStateDimMatrix = typename nmpc_ddp::DDPProblem<StateDim, InputDim>::InputStateDimMatrix;

  /** \brief Type of matrix of inequality x state dimension. */
  using IneqStateDimMatrix = Eigen::Matrix<double, IneqDim, StateDim>;

  /** \brief Type of matrix of inequality x input dimension. */
  using IneqInputDimMatrix = Eigen::Matrix<double, IneqDim, InputDim>;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /** \brief Constructor.
      \param dt discretization timestep [sec]
   */
  FmpcProblem(double dt) : nmpc_ddp::DDPProblem<StateDim, InputDim>(dt)
  {
    // Check dimension
    static_assert(IneqDim >= 0 || IneqDim == Eigen::Dynamic,
                  "[FMPC] Template param IneqDim should be non-negative or Eigen::Dynamic.");
  }

  /** \brief Gets the inequality dimension.
      \note If inequality dimension is dynamic, this must not be called. Instead, ineqDim(t) must be called passing time
     as a parameter.
   */
  inline virtual int ineqDim() const
  {
    if constexpr(IneqDim == Eigen::Dynamic)
    {
      throw std::runtime_error("Since ineq dimension is dynamic, time must be passed to ineqDim().");
    }
    return IneqDim;
  }

  /** \brief Gets the inequality dimension.
      \param t time
      \note If inequality dimension is dynamic, this must be overridden.
  */
  inline virtual int ineqDim(double t) const
  {
    if constexpr(IneqDim == Eigen::Dynamic)
    {
      throw std::runtime_error("ineqDim(t) must be overridden if ineq dimension is dynamic.");
    }
    else
    {
      return ineqDim();
    }
  }

  /** \brief Calculate inequality constraints.
      \param t time [sec]
      \param x current state
      \param u current input
      \returns inequality constraints that must be less than or equal to zero
   */
  virtual IneqDimVector ineqConst(double t, const StateDimVector & x, const InputDimVector & u) const = 0;

  /** \brief Calculate first-order derivatives of inequality constraints.
      \param t time [sec]
      \param x state
      \param u input
      \param ineq_const_deriv_x first-order derivative of inequality constraints w.r.t. state
      \param ineq_const_deriv_u first-order derivative of inequality constraints w.r.t. input
  */
  virtual void calcIneqConstDeriv(double t,
                                  const StateDimVector & x,
                                  const InputDimVector & u,
                                  Eigen::Ref<IneqStateDimMatrix> ineq_const_deriv_x,
                                  Eigen::Ref<IneqInputDimMatrix> ineq_const_deriv_u) const = 0;

  using nmpc_ddp::DDPProblem<StateDim, InputDim>::calcStateEqDeriv;

private:
  /** \brief Calculate first-order and second-order derivatives of discrete state equation.
      \param t time [sec]
      \param x state
      \param u input
      \param state_eq_deriv_x first-order derivative of state equation w.r.t. state
      \param state_eq_deriv_u first-order derivative of state equation w.r.t. input
      \param state_eq_deriv_xx second-order derivative of state equation w.r.t. state
      \param state_eq_deriv_uu second-order derivative of state equation w.r.t. input
      \param state_eq_deriv_xu second-order derivative of state equation w.r.t. state and input
  */
  inline virtual void calcStateEqDeriv(double t,
                                       const StateDimVector & x,
                                       const InputDimVector & u,
                                       Eigen::Ref<StateStateDimMatrix> state_eq_deriv_x,
                                       Eigen::Ref<StateInputDimMatrix> state_eq_deriv_u,
                                       std::vector<StateStateDimMatrix> & state_eq_deriv_xx,
                                       std::vector<InputInputDimMatrix> & state_eq_deriv_uu,
                                       std::vector<StateInputDimMatrix> & state_eq_deriv_xu) const override
  {
    throw std::runtime_error("[FMPC] Second-order derivatives of state equation is not used.");
  }
};
} // namespace nmpc_fmpc
