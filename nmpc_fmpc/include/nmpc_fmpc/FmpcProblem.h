/* Author: Masaki Murooka */

#pragma once

#include <nmpc_ddp/DDPProblem.h>

namespace nmpc_fmpc
{
/** \brief Fast MPC problem.
    \tparam StateDim state dimension (fixed only)
    \tparam InputDim input dimension (fixed or dynamic (i.e., Eigen::Dynamic))
 */
template<int StateDim, int InputDim>
using FmpcProblem = nmpc_ddp::DDPProblem<StateDim, InputDim>;
} // namespace nmpc_fmpc
