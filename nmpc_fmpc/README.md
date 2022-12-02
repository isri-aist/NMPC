# nmpc_fmpc
FMPC: Fast non-linear model predictive control (NMPC) combining the direct multiple shooting (DMS) method, the primal-dual interior point (PDIP) method, and Riccati recursion (RR)

## Features
- C++ header-only library
- Supports inequality constraints on state and control input
- Treats the dimensions of state, control input, and inequality constraints as template parameters
- Supports time-varying dimensions of control input  and inequality constraints

## Install
See [here](https://isri-aist.github.io/NMPC/doc/Install).

## Control method
See the following for a detailed algorithm.
- S Katayama. Fast model predictive control of robotic systems with rigid contacts. Ph.D. thesis (section 2.2), Kyoto University, 2022.
