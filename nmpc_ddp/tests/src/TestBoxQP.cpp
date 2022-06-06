/* Author: Masaki Murooka */

#include <gtest/gtest.h>

#include <iostream>

#include <nmpc_ddp/BoxQP.h>

template<int VarDim>
void solveOneQP(const Eigen::Matrix<double, VarDim, VarDim> & H,
                const Eigen::Matrix<double, VarDim, 1> & g,
                const Eigen::Matrix<double, VarDim, 1> & lower,
                const Eigen::Matrix<double, VarDim, 1> & upper,
                const Eigen::Matrix<double, VarDim, 1> & x_gt)
{
  nmpc_ddp::BoxQP<VarDim> qp;
  qp.config().print_level = 3;

  Eigen::Vector2d x_opt = qp.solve(H, g, lower, upper);
  EXPECT_LT((x_opt - x_gt).norm(), 1e-6) << "[TestBoxQP] QP solution is incorrect:\n"
                                         << "  solution: " << x_opt.transpose()
                                         << "\n  ground truth: " << x_gt.transpose()
                                         << "\n  error: " << (x_opt - x_gt).norm() << std::endl;
}

TEST(TestBoxQP, TestCase1)
{
  // Some problems are copied from
  // https://github.com/coin-or/qpOASES/blob/268b2f2659604df27c82aa6e32aeddb8c1d5cc7f/examples/example1b.cpp#L43-L52
  Eigen::Matrix2d H;
  H << 1.0, 0.0, 0.0, 0.5;

  solveOneQP<2>(H, Eigen::Vector2d(1.5, 1.0), Eigen::Vector2d(-10, -10), Eigen::Vector2d(10, 10),
                Eigen::Vector2d(-1.5, -2.0));

  solveOneQP<2>(H, Eigen::Vector2d(1.5, 1.0), Eigen::Vector2d(0.5, -2.0), Eigen::Vector2d(5.0, 2.0),
                Eigen::Vector2d(0.5, -2.0));

  solveOneQP<2>(H, Eigen::Vector2d(1.0, 1.5), Eigen::Vector2d(0.0, -1.0), Eigen::Vector2d(5.0, -0.5),
                Eigen::Vector2d(0.0, -1.0));

  solveOneQP<2>(H, Eigen::Vector2d(1.5, 1.0), Eigen::Vector2d(-5.0, -1.0), Eigen::Vector2d(-2.0, 2.0),
                Eigen::Vector2d(-2.0, -1.0));

  solveOneQP<2>(H, Eigen::Vector2d(1.0, 1.5), Eigen::Vector2d(-5.0, -10.0), Eigen::Vector2d(-2.0, 10.0),
                Eigen::Vector2d(-2.0, -3.0));
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
