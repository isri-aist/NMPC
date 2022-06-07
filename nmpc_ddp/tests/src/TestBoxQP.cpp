/* Author: Masaki Murooka */

#include <gtest/gtest.h>

#include <iostream>
#include <memory>

#include <nmpc_ddp/BoxQP.h>

template<int VarDim>
void solveOneQP(const Eigen::Matrix<double, VarDim, VarDim> & H,
                const Eigen::Matrix<double, VarDim, 1> & g,
                const Eigen::Matrix<double, VarDim, 1> & lower,
                const Eigen::Matrix<double, VarDim, 1> & upper,
                const Eigen::Matrix<double, VarDim, 1> & x_gt)
{
  std::shared_ptr<nmpc_ddp::BoxQP<VarDim>> qp;
  if constexpr(VarDim == Eigen::Dynamic)
  {
    qp = std::make_shared<nmpc_ddp::BoxQP<VarDim>>(g.size());
  }
  else
  {
    qp = std::make_shared<nmpc_ddp::BoxQP<VarDim>>();
  }
  qp->config().print_level = 3;

  Eigen::Vector2d x_opt = qp->solve(H, g, lower, upper);
  EXPECT_LT((x_opt - x_gt).norm(), 1e-6) << "[TestBoxQP] QP solution is incorrect:\n"
                                         << "  solution: " << x_opt.transpose()
                                         << "\n  ground truth: " << x_gt.transpose()
                                         << "\n  error: " << (x_opt - x_gt).norm() << std::endl;
}

TEST(TestBoxQP, TestFixedSize)
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

TEST(TestBoxQP, TestDynamicSize)
{
  // Some problems are copied from
  // https://github.com/coin-or/qpOASES/blob/268b2f2659604df27c82aa6e32aeddb8c1d5cc7f/examples/example1b.cpp#L43-L52
  Eigen::MatrixXd H(2, 2);
  H << 1.0, 0.0, 0.0, 0.5;
  Eigen::VectorXd g(2);
  Eigen::VectorXd lower(2);
  Eigen::VectorXd upper(2);
  Eigen::VectorXd x_gt(2);

  g << 1.5, 1.0;
  lower << -10, -10;
  upper << 10, 10;
  x_gt << -1.5, -2.0;
  solveOneQP<Eigen::Dynamic>(H, g, lower, upper, x_gt);

  g << 1.5, 1.0;
  lower << 0.5, -2.0;
  upper << 5.0, 2.0;
  x_gt << 0.5, -2.0;
  solveOneQP<Eigen::Dynamic>(H, g, lower, upper, x_gt);

  g << 1.0, 1.5;
  lower << 0.0, -1.0;
  upper << 5.0, -0.5;
  x_gt << 0.0, -1.0;
  solveOneQP<Eigen::Dynamic>(H, g, lower, upper, x_gt);

  g << 1.5, 1.0;
  lower << -5.0, -1.0;
  upper << -2.0, 2.0;
  x_gt << -2.0, -1.0;
  solveOneQP<Eigen::Dynamic>(H, g, lower, upper, x_gt);

  g << 1.0, 1.5;
  lower << -5.0, -10.0;
  upper << -2.0, 10.0;
  x_gt << -2.0, -3.0;
  solveOneQP<Eigen::Dynamic>(H, g, lower, upper, x_gt);
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
