/* Author: Masaki Murooka */

#pragma once

#include <vector>

#include <Eigen/Dense>

namespace NOC
{
/** \brief Class for non-linear optimal control problem. */
class NOCProblem
{
public:
  /** \brief Type of state vector. */
  using StateDimVector = Eigen::Matrix<double, StateDim, 1>;

  /** \brief Type of input vector. */
  using InputDimVector = Eigen::Matrix<double, InputDim, 1>;

  using StateStateDimMatrix = Eigen::Matrix<double, StateDim, StateDim>;

  using InputInputDimMatrix = Eigen::Matrix<double, InputDim, InputDim>;

  using StateInputDimMatrix = Eigen::Matrix<double, StateDim, InputDim>;

  using InputStateDimMatrix = Eigen::Matrix<double, InputDim, StateDim>;

public:
  NOCProblem() {}

  virtual StateDimVector stateEq(const StateDimVector & x, const InputDimVector & u) const = 0;

  virtual double runningCost(const StateDimVector & x, const InputDimVector & u) const = 0;

  virtual double terminalCost(const StateDimVector & x) const = 0;

  /** \brief Calculate first-order derivatives of state equation.
      \param x state
      \param u input
      \param state_eq_deriv_x first-order derivative of state equation w.r.t. state
      \param state_eq_deriv_u first-order derivative of state equation w.r.t. input
  */
  virtual void calcStateqDeriv(const StateDimVector & x,
                               const InputDimVector & u,
                               Eigen::Ref<StateStateDimMatrix> state_eq_deriv_x,
                               Eigen::Ref<StateInputDimMatrix> state_eq_deriv_u) const = 0;

  /** \brief Calculate first-order and second-order derivatives of state equation.
      \param x state
      \param u input
      \param state_eq_deriv_x first-order derivative of state equation w.r.t. state
      \param state_eq_deriv_u first-order derivative of state equation w.r.t. input
      \param state_eq_deriv_xx second-order derivative of state equation w.r.t. state
      \param state_eq_deriv_uu second-order derivative of state equation w.r.t. input
      \param state_eq_deriv_xu second-order derivative of state equation w.r.t. state and input
  */
  virtual void calcStateqDeriv(const StateDimVector & x,
                               const InputDimVector & u,
                               Eigen::Ref<StateStateDimMatrix> state_eq_deriv_x,
                               Eigen::Ref<StateInputDimMatrix> state_eq_deriv_u,
                               std::vector<StateStateDimMatrix> & state_eq_deriv_xx,
                               std::vector<InputInputDimMatrix> & state_eq_deriv_uu,
                               std::vector<StateInputDimMatrix> & state_eq_deriv_xu) const = 0;

  /** \brief Calculate first-order derivatives of running cost.
      \param x state
      \param u input
      \param running_cost_deriv_x first-order derivative of running cost w.r.t. state
      \param running_cost_deriv_u first-order derivative of running cost w.r.t. input
  */
  virtual void calcRunningCostDeriv(const StateDimVector & x,
                                    const InputDimVector & u,
                                    Eigen::Ref<StateDimVector> running_cost_deriv_x,
                                    Eigen::Ref<InputDimVector> running_cost_deriv_u) const = 0;

  /** \brief Calculate first-order and second-order derivatives of running cost.
      \param x state
      \param u input
      \param running_cost_deriv_x first-order derivative of running cost w.r.t. state
      \param running_cost_deriv_u first-order derivative of running cost w.r.t. input
      \param running_cost_deriv_xx second-order derivative of running cost w.r.t. state
      \param running_cost_deriv_uu second-order derivative of running cost w.r.t. input
      \param running_cost_deriv_xu second-order derivative of running cost w.r.t. state and input
  */
  virtual void calcRunningCostDeriv(const StateDimVector & x,
                                    const InputDimVector & u,
                                    Eigen::Ref<StateDimVector> running_cost_deriv_x,
                                    Eigen::Ref<InputDimVector> running_cost_deriv_u,
                                    Eigen::Ref<StateStateDimMatrix> running_cost_deriv_xx,
                                    Eigen::Ref<InputInputDimMatrix> running_cost_deriv_uu,
                                    Eigen::Ref<StateInputDimMatrix> running_cost_deriv_xu) const = 0;

  /** \brief Calculate first-order derivatives of terminal cost.
      \param x state
      \param terminal_cost_deriv_x first-order derivative of terminal cost w.r.t. state
   */
  virtual void calcTerminalCostDeriv(const StateDimVector & x,
                                     Eigen::Ref<StateDimVector> terminal_cost_deriv_x) const = 0;

  /** \brief Calculate first-order and second-order derivatives of terminal cost.
      \param x state
      \param terminal_cost_deriv_x first-order derivative of terminal cost w.r.t. state
      \param terminal_cost_deriv_xx second-order derivative of terminal cost w.r.t. state
   */
  virtual void calcTerminalCostDeriv(const StateDimVector & x,
                                     Eigen::Ref<StateDimVector> terminal_cost_deriv_x,
                                     Eigen::Ref<StateStateDimMatrix> terminal_cost_deriv_xx) const = 0;
};

/** \brief Class for algorithm of differential dynamic programming.

    Ref https://www.mathworks.com/matlabcentral/fileexchange/52069-ilqg-ddp-trajectory-optimization
 */
class DDP
{
public:
  /*! \brief Configuration. */
  struct Configuration
  {
    //! Whether to enable verbose print
    bool verbose_print = true;

    //! Whether to use second-order derivatives of state equation
    bool use_state_eq_second_derivative = false;

    //! Regularization type (1: Quu + lambda * I, 2: Vxx + lambda * I)
    int reg_type = 1;

    //! Increasing/decreasing factor of regularization coefficient scaling
    double lambda_factor = 1.6;

    //! Minimum regularization coefficient
    double lambda_min = 1e-6;

    //! Maximum regularization coefficient
    double lambda_max = 1e10;

    //! Termination threshold of relative norm of k
    double k_rel_norm_thre = 1e-4;

    //! Termination threshold of regularization coefficient
    double lambda_thre = 1e-5;
  };

  /*! \brief Derivatives of non-linear optimal control problem. */
  class NOCProblemDerivative
  {
  public:
    NOCProblemDerivative() {}

    NOCProblemDerivative(size_t state_dim)
    {
      setStateDim(state_dim);
    }

    void setStateDim(size_t state_dim)
    {
      Fxx.resize(state_dim);
      Fuu.resize(state_dim);
      Fxu.resize(state_dim);
    }

  public:
    //! First-order derivative of state equation w.r.t. state
    StateStateDimMatrix Fx;

    //! First-order derivative of state equation w.r.t. input
    StateInputDimMatrix Fu;

    //! Second-order derivative of state equation w.r.t. state
    std::vector<StateStateDimMatrix> Fxx;

    //! Second-order derivative of state equation w.r.t. input
    std::vector<InputInputDimMatrix> Fuu;

    //! Second-order derivative of state equation w.r.t. state and input
    std::vector<StateInputDimMatrix> Fxu;

    //! First-order derivative of running cost w.r.t. state
    StateDimVector Lx;

    //! First-order derivative of running cost w.r.t. input
    InputDimVector Lu;

    //! Second-order derivative of running cost w.r.t. state
    StateStateDimMatrix Lxx;

    //! Second-order derivative of running cost w.r.t. input
    InputInputDimMatrix Luu;

    //! Second-order derivative of running cost w.r.t. state and input
    StateInputDimMatrix Lxu;
  };

public:
  /** \brief Constructor.
      \param rb robot
  */
  DDP();

protected:
  /** \brief Constructor.
      \return whether the process is finished successfully
  */
  bool procBackwardPass();

public:
  //! Configuration
  Configuration config_;

  //! Non-linear optimal control problem
  std::shared_ptr<NOCProblem> problem_;

  //! Regularization coefficient
  double lambda_ = 1e-6;

  //! Scaling factor of regularization coefficient
  double dlambda_ = 1.0;

  //! Sequence of state (x[0], ..., x[N-1], x[N])
  std::vector<StateDimVector> x_list_;

  //! Sequence of input (u[0], ..., u[N-1])
  std::vector<InputDimVector> u_list_;

  //! Sequence of feedforward term for input (k[0], ..., k[N-1])
  std::vector<InputDimVector> k_list_;

  //! Sequence of feedback gain for input w.r.t. state error (K[0], ..., K[N-1])
  std::vector<InputStateDimMatrix> K_list_;

  //! Sequence of derivatives
  std::vector<NOCProblemDerivative> derivative_list_;

  //! First-order derivative of value in last step of horizon
  StateDimVector last_Vx_;

  //! Second-order derivative of value in last step of horizon
  StateStateDimMatrix last_Vxx_;
};
} // namespace NOC
