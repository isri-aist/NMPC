/* Author: Masaki Murooka */

#pragma once

#include <array>
#include <functional>
#include <memory>

#include <nmpc_fmpc/FmpcProblem.h>

namespace nmpc_fmpc
{
/** \brief FMPC solver.
    \tparam StateDim state dimension
    \tparam InputDim input dimension
    \tparam IneqDim inequality dimension

    See the following for a detailed algorithm.
      - S Katayama. Fast model predictive control of robotic systems with rigid contacts. Ph.D. thesis (section 2.2),
   Kyoto University, 2022.
 */
template<int StateDim, int InputDim, int IneqDim>
class FmpcSolver
{
public:
  /** \brief Type of vector of state dimension. */
  using StateDimVector = typename FmpcProblem<StateDim, InputDim, IneqDim>::StateDimVector;

  /** \brief Type of vector of input dimension. */
  using InputDimVector = typename FmpcProblem<StateDim, InputDim, IneqDim>::InputDimVector;

  /** \brief Type of vector of inequality dimension. */
  using IneqDimVector = typename FmpcProblem<StateDim, InputDim, IneqDim>::IneqDimVector;

  /** \brief Type of matrix of state x state dimension. */
  using StateStateDimMatrix = typename FmpcProblem<StateDim, InputDim, IneqDim>::StateStateDimMatrix;

  /** \brief Type of matrix of input x input dimension. */
  using InputInputDimMatrix = typename FmpcProblem<StateDim, InputDim, IneqDim>::InputInputDimMatrix;

  /** \brief Type of matrix of state x input dimension. */
  using StateInputDimMatrix = typename FmpcProblem<StateDim, InputDim, IneqDim>::StateInputDimMatrix;

  /** \brief Type of matrix of input x state dimension. */
  using InputStateDimMatrix = typename FmpcProblem<StateDim, InputDim, IneqDim>::InputStateDimMatrix;

  /** \brief Type of matrix of inequality x state dimension. */
  using IneqStateDimMatrix = typename FmpcProblem<StateDim, InputDim, IneqDim>::IneqStateDimMatrix;

  /** \brief Type of matrix of inequality x input dimension. */
  using IneqInputDimMatrix = typename FmpcProblem<StateDim, InputDim, IneqDim>::IneqInputDimMatrix;

public:
  /*! \brief Configuration. */
  struct Configuration
  {
    //! Print level (0: no print, 1: print only important, 2: print verbose, 3: print very verbose)
    int print_level = 1;

    //! Number of steps in horizon
    int horizon_steps = 100;

    //! Maximum iteration of optimization loop
    int max_iter = 10;

    //! Threshold of KKT condition error
    double kkt_error_thre = 1e-4;

    //! Whether to check NaN
    bool check_nan = true;

    //! Whether to break if LLT decomposition fails
    bool break_if_llt_fails = false;
  };

  /*! \brief Result status. */
  enum class Status
  {
    //! Uninitialized
    Uninitialized = 0,

    //! Succeeded
    Succeeded = 1,

    //! Error in forward
    ErrorInForward = 2,

    //! Error in backward
    ErrorInBackward = 3,

    //! Error in update
    ErrorInUpdate = 4,

    //! Maximum iteration reached
    MaxIterationReached = 5,

    //! Iteration continued (used internally only)
    IterationContinued = 6
  };

  /*! \brief Optimization variables. */
  struct Variable
  {
    /** \brief Constructor.
        \param horizon_steps number of steps in horizon
    */
    Variable(int _horizon_steps = 0);

    /** \brief Reset variables.
        \param _x x
        \param _u u
        \param _lambda lambda
        \param _s s (must be non-negative)
        \param _nu nu (must be non-negative)
    */
    void reset(double _x, double _u, double _lambda, double _s, double _nu);

    /** \brief Check whether NaN or infinity is containd.
        \return whether NaN or infinity is containd
    */
    bool containsNaN() const;

    //! Number of steps in horizon
    int horizon_steps;

    //! Sequence of state (x[0], ..., x[N-1], x[N])
    std::vector<StateDimVector> x_list;

    //! Sequence of input (u[0], ..., u[N-1])
    std::vector<InputDimVector> u_list;

    //! Sequence of Lagrange multipliers of equality constraints (lambda[0], ..., lambda[N-1], lambda[N])
    std::vector<StateDimVector> lambda_list;

    //! Sequence of slack variables of inequality constraints (s[0], ..., s[N-1])
    std::vector<IneqDimVector> s_list;

    //! Sequence of Lagrange multipliers of inequality constraints (nu[0], ..., nu[N-1])
    std::vector<IneqDimVector> nu_list;

    //! Print level (0: no print, 1: print only important, 2: print verbose, 3: print very verbose)
    int print_level = 1;
  };

  /*! \brief Coefficients of linearized KKT condition. */
  struct Coefficient
  {
    /** \brief Constructor.
        \param state_dim state dimension
        \param input_dim input dimension
        \param ineq_dim inequality dimension
    */
    Coefficient(int state_dim, int input_dim, int ineq_dim);

    /** \brief Constructor for terminal coefficient.
        \param state_dim state dimension
    */
    Coefficient(int state_dim);

    /** \brief Check whether NaN or infinity is containd.
        \return whether NaN or infinity is containd
    */
    bool containsNaN() const;

    //! First-order derivative of state equation w.r.t. state
    StateStateDimMatrix A;

    //! First-order derivative of state equation w.r.t. input
    StateInputDimMatrix B;

    //! First-order derivative of inequality constraints w.r.t. state
    IneqStateDimMatrix C;

    //! First-order derivative of inequality constraints w.r.t. input
    IneqInputDimMatrix D;

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

    //! Linearization offsets
    //! @{
    StateDimVector x_bar;
    IneqDimVector g_bar;
    StateDimVector Lx_bar;
    InputDimVector Lu_bar;
    //! @}

    //! Feedforward term for input
    InputDimVector k;

    //! Feedback gain for input w.r.t. state error
    InputStateDimMatrix K;

    //! Offset vector for lambda calculation
    StateDimVector s;

    //! Coefficient matrix for lambda calculation
    StateStateDimMatrix P;

    //! Print level (0: no print, 1: print only important, 2: print verbose, 3: print very verbose)
    int print_level = 1;
  };

  /*! \brief Data to trace optimization loop. */
  struct TraceData
  {
    //! Iteration of optimization loop
    int iter = 0;

    //! KKT condition error
    double kkt_error = 0;

    //! Duration to calculate coefficients [msec]
    double duration_coeff = 0;

    //! Duration to process backward pass [msec]
    double duration_backward = 0;

    //! Duration to process forward pass [msec]
    double duration_forward = 0;

    //! Duration to update variables [msec]
    double duration_update = 0;
  };

  /*! \brief Data of computation duration. */
  struct ComputationDuration
  {
    //! Duration to solve [msec]
    double solve = 0;

    //! Duration to setup (included in solve) [msec]
    double setup = 0;

    //! Duration of optimization loop (included in solve) [msec]
    double opt = 0;

    //! Duration to calculate coefficients (included in opt) [msec]
    double coeff = 0;

    //! Duration to process backward pass (included in opt) [msec]
    double backward = 0;

    //! Duration to process forward pass (included in opt) [msec]
    double forward = 0;

    //! Duration to update variables (included in opt) [msec]
    double update = 0;

    //! Duration of pre-process for gain calculation (included in backward) [msec]
    double gain_pre = 0;

    //! Duration to solve linear equation for gain calculation (included in backward) [msec]
    double gain_solve = 0;

    //! Duration of post-process for gain calculation (included in backward) [msec]
    double gain_post = 0;

    //! Duration to calculate fraction-to-boundary rule (included in update) [msec]
    double fraction = 0;
  };

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /** \brief Constructor.
      \param problem FMPC problem
  */
  FmpcSolver(const std::shared_ptr<FmpcProblem<StateDim, InputDim, IneqDim>> & problem) : problem_(problem) {}

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
      \param initial_variable initial guess of optimization variables
      \return result status
  */
  Status solve(double current_t, const StateDimVector & current_x, const Variable & initial_variable);

  /** \brief Const accessor to optimization variables. */
  inline const Variable & variable() const
  {
    return variable_;
  }

  /** \brief Const accessor to sequence of coefficients of linearized KKT condition. */
  inline const std::vector<Coefficient> & coeffList() const
  {
    return coeff_list_;
  }

  /** \brief Const accessor to trace data list. */
  inline const std::vector<TraceData> & traceDataList() const
  {
    return trace_data_list_;
  }

  /** \brief Const accessor to computation duration. */
  inline const ComputationDuration & computationDuration() const
  {
    return computation_duration_;
  }

  /** \brief Dump trace data list.
      \param file_path path to output file
  */
  void dumpTraceDataList(const std::string & file_path) const;

protected:
  /** \brief Check optimization variables. */
  void checkVariable() const;

  /** \brief Process one iteration.
      \param iter current iteration
      \return result status
  */
  Status procOnce(int iter);

  /** \brief Calculate KKT condition error.
      \param barrier_eps barrier parameter
  */
  double calcKktError(double barrier_eps) const;

  /** \brief Process backward pass a.k.a backward Riccati recursion.
      \return whether the process is finished successfully
  */
  bool backwardPass();

  /** \brief Process forward pass a.k.a forward Riccati recursion.
      \return whether the process is finished successfully
  */
  bool forwardPass();

  /** \brief Update optimization variables given Newton-step direction.
      \return whether the process is finished successfully
  */
  bool updateVariables();

protected:
  //! Configuration
  Configuration config_;

  //! FMPC problem
  std::shared_ptr<FmpcProblem<StateDim, InputDim, IneqDim>> problem_;

  //! Optimization variables
  Variable variable_;

  //! Update amount of optimization variables
  Variable delta_variable_;

  //! Sequence of coefficients of linearized KKT condition
  std::vector<Coefficient> coeff_list_;

  //! Sequence of trace data
  std::vector<TraceData> trace_data_list_;

  //! Computation duration data
  ComputationDuration computation_duration_;

  //! Current time [sec]
  double current_t_ = 0;

  //! Current state
  StateDimVector current_x_ = StateDimVector::Zero();

  //! Barrier parameter
  double barrier_eps_ = 1e-4;
};
} // namespace nmpc_fmpc

#include <nmpc_fmpc/FmpcSolver.hpp>
