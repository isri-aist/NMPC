/* Author: Masaki Murooka */

#pragma once

#include <memory>

#include <nmpc_ddp/DDPProblem.h>

namespace nmpc_ddp
{
/** \brief DDP solver.
    \tparam StateDim state dimension
    \tparam InputDim input dimension

    See the following for a detailed algorithm.
      - Y Tassa, T Erez, E Todorov. Synthesis and stabilization of complex behaviors through online trajectory
   optimization. IROS, 2012.
      - Y Tassa, N Mansard, E Todorov. Control-limited differential dynamic programming. ICRA, 2014.
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

  /*! \brief Data of computation duration. */
  struct ComputationDuration
  {
    //! Duration to setup [msec]
    double setup = 0;

    //! Duration of optimization loop [msec]
    double opt = 0;

    //! Duration to calculate derivatives (included in opt) [msec]
    double derivative = 0;

    //! Duration to process backward pass (included in opt) [msec]
    double backward = 0;

    //! Duration to process forward pass (included in opt) [msec]
    double forward = 0;

    //! Duration to calculate Q (included in backward) [msec]
    double Q = 0;

    //! Duration to calculate regularization (included in backward) [msec]
    double reg = 0;

    //! Duration to calculate gains (included in backward) [msec]
    double gain = 0;
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

  //! Computation duration data
  ComputationDuration computation_duration_;

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
} // namespace nmpc_ddp

#include <nmpc_ddp/DDPSolver.hpp>
