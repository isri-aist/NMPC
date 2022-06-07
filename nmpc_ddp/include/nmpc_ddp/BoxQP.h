/* Author: Masaki Murooka */

#pragma once

#include <vector>

#include <Eigen/Dense>

namespace nmpc_ddp
{
/** \brief Solver for quadratic programming problems with box constraints (i.e., only upper and lower bounds).
    \tparam VarDim dimension of decision variables

    See the following for a detailed algorithm.
      - Y Tassa, N Mansard, E Todorov. Control-limited differential dynamic programming. ICRA, 2014.
      - https://www.mathworks.com/matlabcentral/fileexchange/52069-ilqg-ddp-trajectory-optimization
 */
template<int VarDim>
class BoxQP
{
public:
  /** \brief Type of vector of variables dimension. */
  using VarDimVector = Eigen::Matrix<double, VarDim, 1>;

  /** \brief Type of matrix of variables x variables dimension. */
  using VarVarDimMatrix = Eigen::Matrix<double, VarDim, VarDim>;

  /** \brief Type of boolean array of variables dimension. */
  using VarDimArray = Eigen::Array<bool, VarDim, 1>;

public:
  /*! \brief Configuration. */
  struct Configuration
  {
    //! Print level (0: no print, 1: print only important, 2: print verbose, 3: print very verbose)
    int print_level = 1;

    //! Maximum iteration of optimization loop
    int max_iter = 500;

    //! Termination threshold of non-fixed gradient
    double grad_thre = 1e-8;

    //! Termination threshold of relative improvement
    double rel_improve_thre = 1e-8;

    //! Factor for decreasing stepsize
    double step_factor = 0.6;

    //! Minimum stepsize for linesearch
    double min_step = 1e-22;

    //! Armijo parameter (fraction of linear improvement required)
    double armijo_param = 0.1;
  };

  /*! \brief Data to trace optimization loop. */
  struct TraceData
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    //! Iteration of optimization loop
    int iter = 0;

    //! Decision variables
    VarDimVector x;

    //! Objective value
    double obj = 0;

    //! Search direction
    VarDimVector search_dir;

    //! Flag of clamped variables dimensions
    VarDimArray clamped_flag;

    //! Number of factorization
    int factorization_num = 0;

    //! Number of linesearch step
    int step_num = 0;

    /** \brief Constructor.
        \param var_dim dimension of decision variables
     */
    TraceData(int var_dim)
    {
      x.setZero(var_dim);
      search_dir.setZero(var_dim);
      clamped_flag.setZero(var_dim);
    }
  };

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /** \brief Constructor.
      \param var_dim dimension of decision variables
      \note dimensions in parameter can be omitted if a fixed value is given in the template value.
   */
  BoxQP(int var_dim = VarDim) : var_dim_(var_dim)
  {
    // Check dimension is positive
    if(var_dim_ <= 0)
    {
      throw std::runtime_error("var_dim must be positive: " + std::to_string(var_dim_) + " <= 0");
    }

    // Check dimension consistency
    if constexpr(VarDim != Eigen::Dynamic)
    {
      if(var_dim_ != VarDim)
      {
        throw std::runtime_error("var_dim is inconsistent with template parameter: " + std::to_string(var_dim_)
                                 + " != " + std::to_string(VarDim));
      }
    }
  }

  /** \brief Solve optimization.
      \param H Hessian matrix of objective function
      \param g gradient vector of objective function
      \param lower lower limit of decision variables
      \param upper upper limit of decision variables
   */
  inline VarDimVector solve(const VarVarDimMatrix & H,
                            const VarDimVector & g,
                            const VarDimVector & lower,
                            const VarDimVector & upper)
  {
    return solve(H, g, lower, upper, VarDimVector::Zero(var_dim_));
  }

  /** \brief Solve optimization.
      \param H Hessian matrix of objective function
      \param g gradient vector of objective function
      \param lower lower limit of decision variables
      \param upper upper limit of decision variables
      \param initial_x initial guess of decision variables
   */
  inline VarDimVector solve(const VarVarDimMatrix & H,
                            const VarDimVector & g,
                            const VarDimVector & lower,
                            const VarDimVector & upper,
                            const VarDimVector & initial_x)
  {
    // Initialize objective value
    VarDimVector x = initial_x.cwiseMin(upper).cwiseMax(lower);
    double obj = x.dot(g) + 0.5 * x.dot(H * x);
    double old_obj = obj;

    // Initialize trace data
    trace_data_list_.clear();
    TraceData initial_trace_data(var_dim_);
    initial_trace_data.iter = 0;
    initial_trace_data.x = x;
    initial_trace_data.obj = obj;
    trace_data_list_.push_back(initial_trace_data);

    // Main loop
    int retval = 0;
    int factorization_num = 0;
    VarDimVector grad = VarDimVector::Zero(var_dim_);
    VarDimArray clamped_flag = VarDimArray::Zero(var_dim_);
    VarDimArray old_clamped_flag = clamped_flag;
    int iter = 1;
    for(;; iter++)
    {
      // Append trace data
      trace_data_list_.push_back(TraceData(var_dim_));
      auto & trace_data = trace_data_list_.back();
      trace_data.iter = iter;

      // Check relative improvement
      if(iter > 1 && (old_obj - obj) < config_.rel_improve_thre * std::abs(old_obj))
      {
        retval = 4;
        break;
      }
      old_obj = obj;

      // Calculate gradient
      grad = g + H * x;

      // Find clamped dimensions
      old_clamped_flag = clamped_flag;
      clamped_flag.setConstant(false);
      clamped_flag =
          ((x.array() == lower.array() && grad.array() > 0) || (x.array() == upper.array() && grad.array() < 0))
              .select(true, clamped_flag);

      // Set clamped and free indices
      std::vector<int> clamped_idxs;
      std::vector<int> free_idxs;
      for(int i = 0; i < clamped_flag.size(); i++)
      {
        if(clamped_flag[i])
        {
          clamped_idxs.push_back(i);
        }
        else
        {
          free_idxs.push_back(i);
        }
      }

      // Check for all clamped
      if(clamped_flag.all())
      {
        retval = 6;
        break;
      }

      // Factorize if clamped has changed
      if(iter == 1 || (clamped_flag != old_clamped_flag).any())
      {
        // Set H_free
        Eigen::MatrixXd H_free(free_idxs.size(), free_idxs.size());
        for(int i = 0; i < free_idxs.size(); i++)
        {
          for(int j = 0; j < free_idxs.size(); j++)
          {
            H_free(i, j) = H(free_idxs[i], free_idxs[j]);
          }
        }

        // Cholesky decomposition
        llt_ = std::make_unique<Eigen::LLT<Eigen::MatrixXd>>(H_free);
        if(llt_->info() == Eigen::NumericalIssue)
        {
          if(config_.print_level >= 1)
          {
            std::cout << "[BoxQP] H_free is not positive definite in Cholesky decomposition (LLT)." << std::endl;
          }
          retval = -1;
          break;
        }

        factorization_num++;
      }

      // Check gradient norm
      double grad_norm = 0;
      for(int i = 0; i < free_idxs.size(); i++)
      {
        grad_norm += std::pow(grad[free_idxs[i]], 2);
      }
      if(grad_norm < std::pow(config_.grad_thre, 2))
      {
        retval = 5;
        break;
      }

      // Calculate search direction
      Eigen::VectorXd x_clamped(clamped_idxs.size());
      Eigen::VectorXd x_free(free_idxs.size());
      Eigen::VectorXd g_free(free_idxs.size());
      Eigen::MatrixXd H_free_clamped(free_idxs.size(), clamped_idxs.size());
      for(int i = 0; i < clamped_idxs.size(); i++)
      {
        x_clamped[i] = x[clamped_idxs[i]];
      }
      for(int i = 0; i < free_idxs.size(); i++)
      {
        x_free[i] = x[free_idxs[i]];
        g_free[i] = g[free_idxs[i]];
        for(int j = 0; j < clamped_idxs.size(); j++)
        {
          H_free_clamped(i, j) = H(free_idxs[i], clamped_idxs[j]);
        }
      }
      Eigen::VectorXd grad_free_clamped = g_free + H_free_clamped * x_clamped;
      Eigen::VectorXd search_dir_free = -1 * llt_->solve(grad_free_clamped) - x_free;
      VarDimVector search_dir = VarDimVector::Zero(var_dim_);
      for(int i = 0; i < free_idxs.size(); i++)
      {
        search_dir[free_idxs[i]] = search_dir_free[i];
      }

      // Check for descent direction
      double search_dir_grad = search_dir.dot(grad);
      if(search_dir_grad > 0) // This should not happen
      {
        if(config_.print_level >= 1)
        {
          std::cout << "[BoxQP] search_dir_grad is negative: " << search_dir_grad << std::endl;
        }
        retval = -2;
        break;
      }

      // Armijo linesearch
      double step = 1;
      int step_num = 0;
      VarDimVector x_candidate = (x + step * search_dir).cwiseMin(upper).cwiseMax(lower);
      double obj_candidate = x_candidate.dot(g) + 0.5 * x_candidate.dot(H * x_candidate);
      while((obj_candidate - old_obj) / (step * search_dir_grad) < config_.armijo_param)
      {
        step = step * config_.step_factor;
        step_num++;
        x_candidate = (x + step * search_dir).cwiseMin(upper).cwiseMax(lower);
        obj_candidate = x_candidate.dot(g) + 0.5 * x_candidate.dot(H * x_candidate);
        if(step < config_.min_step)
        {
          retval = 2;
          break;
        }
      }

      // Print
      if(config_.print_level >= 3)
      {
        std::cout << "[BoxQP] iter: " << iter << ", obj: " << obj << ", grad_norm: " << grad_norm
                  << ", obj_update, : " << old_obj - obj_candidate << ", step: " << step
                  << ", clamped_flag_num: " << clamped_idxs.size() << std::endl;
      }

      // Set trace data
      trace_data.x = x;
      trace_data.obj = obj;
      trace_data.search_dir = search_dir;
      trace_data.clamped_flag = clamped_flag;
      trace_data.factorization_num = factorization_num;
      trace_data.step_num = step_num;

      // Accept candidate
      x = x_candidate;
      obj = obj_candidate;

      // Check loop termination
      if(iter == config_.max_iter)
      {
        retval = 1;
        break;
      }
    }

    // Print
    std::unordered_map<int, std::string> retstr = {{-2, "Gradient of search direction is negative"},
                                                   {-1, "Hessian is not positive definite"},
                                                   {0, "No descent direction found"},
                                                   {1, "Maximum main iterations exceeded"},
                                                   {2, "Maximum line-search iterations exceeded"},
                                                   {3, "No bounds, returning Newton point"},
                                                   {4, "Improvement smaller than tolerance"},
                                                   {5, "Gradient norm smaller than tolerance"},
                                                   {6, "All dimensions are clamped"}};

    // Print
    if(config_.print_level >= 2)
    {
      std::cout << "[BoxQP] result: " << retval << " (" << retstr.at(retval) << "), iter: " << iter << ", obj: " << obj
                << ", factorization_num: " << factorization_num << std::endl;
    }

    return x;
  }

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

  /** \brief Const accessor to trace data list. */
  inline const std::vector<TraceData> & traceDataList() const
  {
    return trace_data_list_;
  }

public:
  //! Dimension of decision variables
  const int var_dim_ = 0;

  //! Cholesky decomposition (LLT) of free block of objective Hessian matrix
  std::unique_ptr<Eigen::LLT<Eigen::MatrixXd>> llt_;

protected:
  //! Configuration
  Configuration config_;

  //! Sequence of trace data
  std::vector<TraceData> trace_data_list_;
};
} // namespace nmpc_ddp
