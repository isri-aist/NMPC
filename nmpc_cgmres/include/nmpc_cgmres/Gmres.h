/* Author: Masaki Murooka */

#pragma once

#include <functional>
#include <iostream>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

namespace nmpc_cgmres
{
/** \brief GMRES method to solve a linear equation.

    See the following articles about the GMRES method:
      - C T Kelley. Iterative methods for linear and nonlinear equations. 1995.
      - https://www.coronasha.co.jp/np/isbn/9784339032109/
 */
class Gmres
{
public:
  /** \brief Type of function that returns x multiplied by A. */
  using AmulFunc = std::function<Eigen::VectorXd(const Eigen::Ref<const Eigen::VectorXd> &)>;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /** \brief Constructor. */
  Gmres() {}

  /** \brief Solve.
      \param A the matrix of linear equation
      \param b the vector of linear equation
      \param x the initial guess of solution, which is overwritten by the final solution
      \param k_max the maximum number of GMRES iteration
      \param eps the required solution tolerance

      Solve the linear equation: \f$ A x = b \f$.
   */
  inline void solve(const Eigen::Ref<const Eigen::MatrixXd> & A,
                    const Eigen::Ref<const Eigen::VectorXd> & b,
                    Eigen::Ref<Eigen::VectorXd> x,
                    int k_max = 100,
                    double eps = 1e-10)
  {
    assert(A.rows() == b.size());
    AmulFunc Amul_func = [&](const Eigen::Ref<const Eigen::VectorXd> & vec) { return A * vec; };
    solve(Amul_func, b, x, k_max, eps);
  }

  /** \brief Solve.
      \param Amul_func the function to return \f$ A * v \f$ where \f$ v \f$ is given
      \param b the vector of linear equation
      \param x the initial guess of solution, which is overwritten by the final solution
      \param k_max the maximum number of GMRES iteration
      \param eps the required solution tolerance

      Solve the linear equation: \f$ A x = b \f$.

      Refer to the Algorithm 3.5.1. in [1] for the case that make_triangular_ is true.
      Refer to the Algorithm 3.4.2. in [1] for the case that make_triangular_ is false.
      [1] Kelley, Carl T. Iterative methods for linear and nonlinear equations. Society for Industrial and Applied
     Mathematics, 1995.
   */
  inline void solve(const AmulFunc & Amul_func,
                    const Eigen::Ref<const Eigen::VectorXd> & b,
                    Eigen::Ref<Eigen::VectorXd> x,
                    int k_max = 100,
                    double eps = 1e-10)
  {
    k_max = std::min(k_max, static_cast<int>(x.size()));

    err_list_.clear();
    basis_.clear();

    // 1.
    Eigen::VectorXd r = b - Amul_func(x);
    basis_.push_back(r.normalized());
    double rho = r.norm();
    int k = 0;
    g_.setZero(k_max + 1);
    g_(0) = rho;

    double b_norm = b.norm();
    H_.setZero(k_max + 1, k_max);
    err_list_.push_back(rho);

    // 2.
    Eigen::VectorXd y_k;
    std::vector<double> c_list;
    std::vector<double> s_list;
    while(rho > eps * b_norm && k < k_max)
    {
      // (a).
      // note that k is 1 in the first iteration
      k++;

      // (b).
      // new_basis corresponds to $v_{k+1}$ in the paper
      Eigen::VectorXd Avk = Amul_func(basis_.back());
      Eigen::VectorXd new_basis = Avk;
      for(int j = 0; j < k; j++)
      {
        // i.
        H_(j, k - 1) = new_basis.dot(basis_[j]);
        // ii.
        new_basis = new_basis - H_(j, k - 1) * basis_[j];
      }

      // (c).
      double new_basis_norm = new_basis.norm();
      H_(k, k - 1) = new_basis_norm;

      // (d).
      if(apply_reorth_)
      {
        double Avk_norm = Avk.norm();
        if(Avk_norm + 1e-3 * new_basis_norm == Avk_norm)
        {
          // std::cout << "apply reorthogonalization. (loop: " << k << ")" << std::endl;
          for(int j = 0; j < k; j++)
          {
            double h_tmp = new_basis.dot(basis_[j]);
            H_(j, k - 1) += h_tmp;
            new_basis -= h_tmp * basis_[j];
          }
        }
      }

      // (e).
      basis_.push_back(new_basis.normalized());

      if(make_triangular_)
      {
        // (f).
        // i.
        for(int i = 0; i < k - 1; i++)
        {
          double h0 = H_(i, k - 1);
          double h1 = H_(i + 1, k - 1);
          double c = c_list[i];
          double s = s_list[i];
          H_(i, k - 1) = c * h0 - s * h1;
          H_(i + 1, k - 1) = s * h0 + c * h1;
        }

        // ii.
        double nu = std::sqrt(std::pow(H_(k - 1, k - 1), 2) + std::pow(H_(k, k - 1), 2));

        // iii.
        double c_k = H_(k - 1, k - 1) / nu;
        double s_k = -H_(k, k - 1) / nu;
        c_list.push_back(c_k);
        s_list.push_back(s_k);
        H_(k - 1, k - 1) = c_k * H_(k - 1, k - 1) - s_k * H_(k, k - 1);
        H_(k, k - 1) = 0;

        // iv.
        double g0 = g_(k - 1);
        double g1 = g_(k);
        g_(k - 1) = c_k * g0 - s_k * g1;
        g_(k) = s_k * g0 + c_k * g1;

        // (g).
        rho = std::abs(g_(k));
      }
      else
      {
        // (f).
        y_k = H_.topLeftCorner(k + 1, k).householderQr().solve(g_.head(k + 1));

        // (g).
        rho = (g_.head(k + 1) - H_.topLeftCorner(k + 1, k) * y_k).norm();
      }

      err_list_.push_back(rho);
    }

    if(make_triangular_)
    {
      // 3.
      y_k = H_.topLeftCorner(k, k).triangularView<Eigen::Upper>().solve(g_.head(k));
    }

    // 4.
    for(int i = 0; i < k; i++)
    {
      x += y_k(i) * basis_[i];
    }
  }

public:
  bool make_triangular_ = true;
  bool apply_reorth_ = true;

  Eigen::MatrixXd H_;
  Eigen::VectorXd g_;

  std::vector<double> err_list_;

  std::vector<Eigen::VectorXd> basis_;
};
} // namespace nmpc_cgmres
