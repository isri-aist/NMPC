/* Author: Masaki Murooka */

#pragma once

#include <Eigen/Dense>

namespace nmpc_fmpc
{
/** \brief Compute the directional derivative of the L1-norm of the function
    \param func function value
    \param jac jacobian matrix of function
    \param dir direction for directional derivative

    See (A.51) in "J Nocedal, S J Wright. Numerical optimization".
*/
template<int InputDim, int OutputDim>
double l1NormDirectionalDeriv(const Eigen::Matrix<double, OutputDim, 1> & func,
                              const Eigen::Matrix<double, OutputDim, InputDim> & jac,
                              const Eigen::Matrix<double, InputDim, 1> & dir)
{
  double deriv = 0.0;
  for(int i = 0; i < func.size(); i++)
  {
    if(func(i) > 0)
    {
      deriv += jac.row(i).transpose().dot(dir);
    }
    else if(func(i) < 0)
    {
      deriv += -1 * jac.row(i).transpose().dot(dir);
    }
    else
    {
      deriv += std::abs(jac.row(i).transpose().dot(dir));
    }
  }
  return deriv;
}
} // namespace nmpc_fmpc
