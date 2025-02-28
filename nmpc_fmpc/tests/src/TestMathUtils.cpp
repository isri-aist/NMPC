/* Author: Masaki Murooka */

#include <gtest/gtest.h>

#include <nmpc_fmpc/MathUtils.h>

TEST(TestMathUtils, L1NormDirectionalDeriv)
{
  constexpr double deriv_eps = 1e-6;

  // Identity function
  {
    Eigen::Matrix4d jac = Eigen::Matrix4d::Identity();

    for(int i = 0; i < 1000; i++)
    {
      Eigen::Vector4d func = 100.0 * Eigen::Vector4d::Random();
      Eigen::Vector4d dir = Eigen::Vector4d::Random();
      if(i == 0)
      {
        func.setZero();
      }

      double deriv_analytical = nmpc_fmpc::l1NormDirectionalDeriv(func, jac, dir);
      double deriv_numerical = ((func + deriv_eps * dir).cwiseAbs().sum() - func.cwiseAbs().sum()) / deriv_eps;

      EXPECT_LT(std::abs(deriv_analytical - deriv_numerical), 1e-5)
          << "analytical: " << deriv_analytical << ", numerical: " << deriv_numerical << std::endl;
    }
  }

  // Nonlinear function
  {
    int input_dim = 4;
    int output_dim = 3;
    auto sample_func = [&](const Eigen::VectorXd & x)
    {
      Eigen::VectorXd func(output_dim);
      func[0] = x.squaredNorm() - 10.0;
      func[1] = std::pow(x[1], 3) + -5 * std::pow(x[2], 2) + 10 * x[3] + -20;
      func[2] = std::sin(x[0]) + std::cos(x[1]);
      return func;
    };
    auto sample_jac = [&](const Eigen::VectorXd & x) -> Eigen::MatrixXd
    {
      Eigen::MatrixXd jac(output_dim, input_dim);
      jac.row(0) = 2 * x.transpose();
      jac.row(1) << 0, 3 * std::pow(x[1], 2), -10 * x[2], 10;
      jac.row(2) << std::cos(x[0]), -1 * std::sin(x[1]), 0, 0;
      return jac;
    };

    for(int i = 0; i < 1000; i++)
    {
      Eigen::VectorXd x = 100.0 * Eigen::VectorXd::Random(input_dim);
      Eigen::VectorXd dir = Eigen::VectorXd::Random(input_dim);
      if(i == 0)
      {
        x.setZero();
      }

      double deriv_analytical = nmpc_fmpc::l1NormDirectionalDeriv(sample_func(x), sample_jac(x), dir);
      double deriv_numerical =
          (sample_func(x + deriv_eps * dir).cwiseAbs().sum() - sample_func(x).cwiseAbs().sum()) / deriv_eps;

      EXPECT_LT(std::abs(deriv_analytical - deriv_numerical), 1e-3)
          << "analytical: " << deriv_analytical << ", numerical: " << deriv_numerical << std::endl;
    }
  }
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
