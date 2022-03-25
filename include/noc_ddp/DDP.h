/* Author: Masaki Murooka */

#pragma once

#include <memory>
#include <vector>

#include <Eigen/Dense>

namespace NOC
{
/** \brief DDP problem.
    \tparam StateDim state dimension (fixed size only)
    \tparam InputDim input dimension (fixed size or dynamic size (i.e., Eigen::Dynamic))
 */
template<int StateDim, int InputDim>
class DDPProblem
{
public:
  /** \brief Type of vector of state dimension. */
  using StateDimVector = Eigen::Matrix<double, StateDim, 1>;

  /** \brief Type of vector of input dimension. */
  using InputDimVector = Eigen::Matrix<double, InputDim, 1>;

  /** \brief Type of matrix of state x state dimension. */
  using StateStateDimMatrix = Eigen::Matrix<double, StateDim, StateDim>;

  /** \brief Type of matrix of input x input dimension. */
  using InputInputDimMatrix = Eigen::Matrix<double, InputDim, InputDim>;

  /** \brief Type of matrix of state x input dimension. */
  using StateInputDimMatrix = Eigen::Matrix<double, StateDim, InputDim>;

  /** \brief Type of matrix of input x state dimension. */
  using InputStateDimMatrix = Eigen::Matrix<double, InputDim, StateDim>;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /** \brief Constructor.
      \param dt discretization timestep [sec]
      \param state_dim state dimension (can be omitted for fixed size)
      \param input_dim input dimension (can be omitted for fixed size)
   */
  DDPProblem(double dt, int state_dim = StateDim, int input_dim = InputDim);

  /** \brief Whether state or input are dynamic size. */
  static inline constexpr bool isDynamicDim()
  {
    return StateDim == Eigen::Dynamic || InputDim == Eigen::Dynamic;
  }

  /** \brief Gets the state dimension. */
  inline int stateDim() const
  {
    return state_dim_;
  }

  /** \brief Gets the input dimension. */
  inline int inputDim() const
  {
    return input_dim_;
  }

  /** \brief Gets the discretization timestep [sec]. */
  inline double dt() const
  {
    return dt_;
  }

  /** \brief Calculate discrete state equation.
      \param t time [sec]
      \param x current state (x[k])
      \param u current input (u[k])
      \returns next state (x[k+1])
   */
  virtual StateDimVector stateEq(double t, const StateDimVector & x, const InputDimVector & u) const = 0;

  /** \brief Calculate running cost.
      \param t time [sec]
      \param x current state (x[k])
      \param u current input (u[k])
      \returns running cost (L[k])
   */
  virtual double runningCost(double t, const StateDimVector & x, const InputDimVector & u) const = 0;

  /** \brief Calculate terminal cost.
      \param t time [sec]
      \param x current state (x[k])
      \returns terminal cost (phi[k])
   */
  virtual double terminalCost(double t, const StateDimVector & x) const = 0;

  /** \brief Calculate first-order derivatives of discrete state equation.
      \param t time [sec]
      \param x state
      \param u input
      \param state_eq_deriv_x first-order derivative of state equation w.r.t. state
      \param state_eq_deriv_u first-order derivative of state equation w.r.t. input
  */
  virtual void calcStatEqDeriv(double t,
                               const StateDimVector & x,
                               const InputDimVector & u,
                               Eigen::Ref<StateStateDimMatrix> state_eq_deriv_x,
                               Eigen::Ref<StateInputDimMatrix> state_eq_deriv_u) const = 0;

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
  virtual void calcStatEqDeriv(double t,
                               const StateDimVector & x,
                               const InputDimVector & u,
                               Eigen::Ref<StateStateDimMatrix> state_eq_deriv_x,
                               Eigen::Ref<StateInputDimMatrix> state_eq_deriv_u,
                               std::vector<StateStateDimMatrix> & state_eq_deriv_xx,
                               std::vector<InputInputDimMatrix> & state_eq_deriv_uu,
                               std::vector<StateInputDimMatrix> & state_eq_deriv_xu) const = 0;

  /** \brief Calculate first-order derivatives of running cost.
      \param t time [sec]
      \param x state
      \param u input
      \param running_cost_deriv_x first-order derivative of running cost w.r.t. state
      \param running_cost_deriv_u first-order derivative of running cost w.r.t. input
  */
  virtual void calcRunningCostDeriv(double t,
                                    const StateDimVector & x,
                                    const InputDimVector & u,
                                    Eigen::Ref<StateDimVector> running_cost_deriv_x,
                                    Eigen::Ref<InputDimVector> running_cost_deriv_u) const = 0;

  /** \brief Calculate first-order and second-order derivatives of running cost.
      \param t time [sec]
      \param x state
      \param u input
      \param running_cost_deriv_x first-order derivative of running cost w.r.t. state
      \param running_cost_deriv_u first-order derivative of running cost w.r.t. input
      \param running_cost_deriv_xx second-order derivative of running cost w.r.t. state
      \param running_cost_deriv_uu second-order derivative of running cost w.r.t. input
      \param running_cost_deriv_xu second-order derivative of running cost w.r.t. state and input
  */
  virtual void calcRunningCostDeriv(double t,
                                    const StateDimVector & x,
                                    const InputDimVector & u,
                                    Eigen::Ref<StateDimVector> running_cost_deriv_x,
                                    Eigen::Ref<InputDimVector> running_cost_deriv_u,
                                    Eigen::Ref<StateStateDimMatrix> running_cost_deriv_xx,
                                    Eigen::Ref<InputInputDimMatrix> running_cost_deriv_uu,
                                    Eigen::Ref<StateInputDimMatrix> running_cost_deriv_xu) const = 0;

  /** \brief Calculate first-order derivatives of terminal cost.
      \param t time [sec]
      \param x state
      \param terminal_cost_deriv_x first-order derivative of terminal cost w.r.t. state
  */
  virtual void calcTerminalCostDeriv(double t,
                                     const StateDimVector & x,
                                     Eigen::Ref<StateDimVector> terminal_cost_deriv_x) const = 0;

  /** \brief Calculate first-order and second-order derivatives of terminal cost.
      \param t time [sec]
      \param x state
      \param terminal_cost_deriv_x first-order derivative of terminal cost w.r.t. state
      \param terminal_cost_deriv_xx second-order derivative of terminal cost w.r.t. state
  */
  virtual void calcTerminalCostDeriv(double t,
                                     const StateDimVector & x,
                                     Eigen::Ref<StateDimVector> terminal_cost_deriv_x,
                                     Eigen::Ref<StateStateDimMatrix> terminal_cost_deriv_xx) const = 0;

protected:
  //! State dimension
  const int state_dim_ = 0;

  //! Input dimension
  const int input_dim_ = 0;

  //! Discretization timestep [sec]
  const double dt_ = 0;
};

/** \brief DDP solver.
    \tparam StateDim state dimension
    \tparam InputDim input dimension

    See the following for a detailed algorithm.
      - Y Tassa, T Erez, E Todorov. Synthesis and stabilization of complex behaviors through online trajectory
   optimization. IROS2012.
      - Y Tassa, N Mansard, E Todorov. Control-limited differential dynamic programming. ICRA2014.
      - https://www.mathworks.com/matlabcentral/fileexchange/52069-ilqg-ddp-trajectory-optimization
 */
template<int StateDim, int InputDim>
class DDPSolver
{
public:
  /** \brief Type of vector of state dimension. */
  using StateDimVector = typename DDPProblem<StateDim, InputDim>::StateDimVector;

  /** \brief Type of vector of input dimension. */
  using InputDimVector = typename DDPProblem<StateDim, InputDim>::InputDimVector;

  /** \brief Type of matrix of state x state dimension. */
  using StateStateDimMatrix = typename DDPProblem<StateDim, InputDim>::StateStateDimMatrix;

  /** \brief Type of matrix of input x input dimension. */
  using InputInputDimMatrix = typename DDPProblem<StateDim, InputDim>::InputInputDimMatrix;

  /** \brief Type of matrix of state x input dimension. */
  using StateInputDimMatrix = typename DDPProblem<StateDim, InputDim>::StateInputDimMatrix;

  /** \brief Type of matrix of input x state dimension. */
  using InputStateDimMatrix = typename DDPProblem<StateDim, InputDim>::InputStateDimMatrix;

public:
  /*! \brief Configuration. */
  struct Configuration
  {
    /** \brief Constructor. */
    Configuration()
    {
      // Initialize alpha_list
      int list_size = 11;
      alpha_list.resize(list_size);
      Eigen::VectorXd alpha_exponent_list = Eigen::VectorXd::LinSpaced(list_size, 0, -3);
      for(int i = 0; i < list_size; i++)
      {
        alpha_list[i] = std::pow(10, alpha_exponent_list[i]);
      }
    }

    //! Print level (0: no print, 1: print only important, 2: print verbose, 3: print very verbose)
    int print_level = 1;

    // \todo Support use_state_eq_second_derivative
    //! Whether to use second-order derivatives of state equation
    bool use_state_eq_second_derivative = false;

    // \todo Support with_input_constraint
    //! Whether input has constraints
    bool with_input_constraint = false;

    //! Maximum iteration of optimization loop
    int max_iter = 500;

    //! Number of steps in horizon
    int horizon_steps = 100;

    //! Regularization type (1: Quu + lambda * I, 2: Vxx + lambda * I)
    int reg_type = 1;

    //! Initial regularization coefficient
    double initial_lambda = 1e-4; // 1.0

    //! Initial scaling factor of regularization coefficient
    double initial_dlambda = 1.0;

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

    //! List of alpha (scaling factor of k)
    Eigen::VectorXd alpha_list;

    //! Allowable threshold of cost update ratio
    double cost_update_ratio_thre = 0;

    //! Termination threshold of cost update
    double cost_update_thre = 1e-7;
  };

  /*! \brief Control data. */
  struct ControlData
  {
    //! Sequence of state (x[0], ..., x[N-1], x[N])
    std::vector<StateDimVector> x_list;

    //! Sequence of input (u[0], ..., u[N-1])
    std::vector<InputDimVector> u_list;

    //! Sequence of cost (L[0], ..., L[N-1], phi[N])
    Eigen::VectorXd cost_list;
  };

  /*! \brief Derivatives of DDP problem. */
  struct Derivative
  {
    /** \brief Constructor.
        \param state_dim state dimension
        \param input_dim input dimension
        \param outer_dim outer dimension of tensor
    */
    Derivative(int state_dim, int input_dim, int outer_dim)
    {
      Fx.resize(state_dim, state_dim);
      Fu.resize(state_dim, input_dim);
      Fxx.assign(outer_dim, StateStateDimMatrix(state_dim, state_dim));
      Fuu.assign(outer_dim, InputInputDimMatrix(input_dim, input_dim));
      Fxu.assign(outer_dim, StateInputDimMatrix(state_dim, input_dim));
      Lx.resize(state_dim);
      Lu.resize(input_dim);
      Lxx.resize(state_dim, state_dim);
      Luu.resize(input_dim, input_dim);
      Lxu.resize(state_dim, input_dim);
    }

    //! First-order derivative of state equation w.r.t. state
    StateStateDimMatrix Fx;

    //! First-order derivative of state equation w.r.t. input
    StateInputDimMatrix Fu;

    //! Second-order derivative of state equation w.r.t. state (tensor of rank 3)
    std::vector<StateStateDimMatrix> Fxx;

    //! Second-order derivative of state equation w.r.t. input (tensor of rank 3)
    std::vector<InputInputDimMatrix> Fuu;

    //! Second-order derivative of state equation w.r.t. state and input (tensor of rank 3)
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

  /*! \brief Data to trace optimization loop. */
  struct TraceData
  {
    //! Iteration of optimization loop
    int iter = 0;

    //! Total cost
    double cost = 0;

    //! Regularization coefficient
    double lambda = 0;

    //! Scaling factor of regularization coefficient
    double dlambda = 0;

    //! Scaling factor of k
    double alpha = 0;

    //! Norm of relative values of k and u
    double k_rel_norm = 0;

    //! Actual update value of cost
    double cost_update_actual = 0;

    //! Expected update value of cost
    double cost_update_expected = 0;

    //! Ratio of actual and expected update values of cost
    double cost_update_ratio = 0;

    //! Duration to calculate derivatives [msec]
    double duration_derivative = 0;

    //! Duration to process backward pass [msec]
    double duration_backward = 0;

    //! Duration to process forward pass [msec]
    double duration_forward = 0;
  };

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /** \brief Constructor.
      \param problem DDP problem
  */
  DDPSolver(const std::shared_ptr<DDPProblem<StateDim, InputDim>> & problem);

  /** \brief Accessor to configuration. */
  inline Configuration & config()
  {
    return config_;
  }

  /** \brief Const accessor to configuration. */
  inline const Configuration & config() const
  {
    return config_;
  }

  /** \brief Solve optimization.
      \param current_t current time [sec]
      \param current_x current state
      \param initial_u_list initial sequence of input
      \return whether the process is finished successfully
  */
  bool solve(double current_t, const StateDimVector & current_x, const std::vector<InputDimVector> & initial_u_list);

  /** \brief Const accessor to control data calculated by solve(). */
  inline const ControlData & controlData() const
  {
    return control_data_;
  }

  /** \brief Const accessor to trace data list. */
  const std::vector<TraceData> & traceDataList() const
  {
    return trace_data_list_;
  }

  /** \brief Dump trace data list.
      \param file_path path to output file
  */
  void dumpTraceDataList(const std::string & file_path) const;

protected:
  /** \brief Process one iteration.
      \param iter current iteration
      \return 0 for continue, 1 for terminate, -1 for failure
  */
  int procOnce(int iter);

  /** \brief Process backward pass.
      \return whether the process is finished successfully
  */
  bool backwardPass();

  /** \brief Process forward pass.
      \param alpha scaling factor of k
  */
  void forwardPass(double alpha);

protected:
  //! Configuration
  Configuration config_;

  //! DDP problem
  std::shared_ptr<DDPProblem<StateDim, InputDim>> problem_;

  //! Sequence of trace data
  std::vector<TraceData> trace_data_list_;

  //! Current time [sec]
  double current_t_ = 0;

  //! Regularization coefficient
  double lambda_ = 0;

  //! Scaling factor of regularization coefficient
  double dlambda_ = 0;

  //! Control data (sequence of state, input, and cost)
  ControlData control_data_;

  //! Candidate control data (sequence of state, input, and cost)
  ControlData candidate_control_data_;

  //! Sequence of feedforward term for input (k[0], ..., k[N-1])
  std::vector<InputDimVector> k_list_;

  //! Sequence of feedback gain for input w.r.t. state error (K[0], ..., K[N-1])
  std::vector<InputStateDimMatrix> K_list_;

  //! Sequence of derivatives
  std::vector<Derivative> derivative_list_;

  //! First-order derivative of value in last step of horizon
  StateDimVector last_Vx_;

  //! Second-order derivative of value in last step of horizon
  StateStateDimMatrix last_Vxx_;

  //! Expected update of value
  Eigen::Vector2d dV_;
};
} // namespace NOC

#include <noc_ddp/DDP.hpp>
