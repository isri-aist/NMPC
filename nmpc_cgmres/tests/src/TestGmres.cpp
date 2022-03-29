/* Author: Masaki Murooka */

#include <gtest/gtest.h>

#include <ctime>

#include <nmpc_cgmres/Gmres.h>

class LinearSolver
{
public:
  virtual void solve(const Eigen::MatrixXd & A,
                     const Eigen::VectorXd & b,
                     Eigen::Ref<Eigen::VectorXd> x,
                     double & tm,
                     double & err) = 0;
};

class GmresLinearSolver : public LinearSolver
{
public:
  virtual void solve(const Eigen::MatrixXd & A,
                     const Eigen::VectorXd & b,
                     Eigen::Ref<Eigen::VectorXd> x,
                     double & tm,
                     double & err)
  {
    gmres_ = std::make_shared<nmpc_cgmres::Gmres>();
    gmres_->make_triangular_ = make_triangular_;
    gmres_->apply_reorth_ = apply_reorth_;
    gmres_->solve(A, b, x, k_max_);
  }

  std::shared_ptr<nmpc_cgmres::Gmres> gmres_;
  int k_max_ = 1000;
  bool make_triangular_ = true;
  bool apply_reorth_ = true;
};

class FullPivLuLinearSolver : public LinearSolver
{
public:
  virtual void solve(const Eigen::MatrixXd & A,
                     const Eigen::VectorXd & b,
                     Eigen::Ref<Eigen::VectorXd> x,
                     double & tm,
                     double & err)
  {
    x = A.fullPivLu().solve(b);
  }
};

class HouseholderQrLinearSolver : public LinearSolver
{
public:
  virtual void solve(const Eigen::MatrixXd & A,
                     const Eigen::VectorXd & b,
                     Eigen::Ref<Eigen::VectorXd> x,
                     double & tm,
                     double & err)
  {
    x = A.householderQr().solve(b);
  }
};

double eval(std::shared_ptr<LinearSolver> solver,
            const std::vector<Eigen::MatrixXd> & A_list,
            const std::vector<Eigen::VectorXd> & b_list)
{
  int trial_num = A_list.size();
  double tm = 0;
  double err = 0;

  for(int i = 0; i < trial_num; i++)
  {
    const Eigen::MatrixXd & A = A_list[i];
    const Eigen::VectorXd & b = b_list[i];
    Eigen::VectorXd x = Eigen::VectorXd::Zero(b.size());

    clock_t begin_clock = clock();
    solver->solve(A, b, x, tm, err);
    clock_t end_clock = clock();

    tm += static_cast<double>(end_clock - begin_clock) / CLOCKS_PER_SEC;
    err += (A * x - b).norm();
  }

  tm /= trial_num;
  err /= trial_num;
  std::cout << "ave time: " << 1e3 * tm << " [msec], ave err: " << err << std::endl;

  return err;
}

TEST(TestGmres, TestCase1)
{
  std::vector<int> eq_size_list = {10, 50, 100, 500};
  int trial_num = 10;

  for(int eq_size : eq_size_list)
  {
    // setup a problem list
    std::vector<Eigen::MatrixXd> A_list;
    std::vector<Eigen::VectorXd> b_list;
    for(int i = 0; i < trial_num; i++)
    {
      A_list.push_back(Eigen::MatrixXd::Random(eq_size, eq_size));
      b_list.push_back(Eigen::VectorXd::Random(eq_size));
    }

    std::cout << "======== eq_size: " << eq_size << " ========" << std::endl;

    // evaluate each method
    {
      std::cout << "== Gmres ==" << std::endl;
      auto solver = std::make_shared<GmresLinearSolver>();
      EXPECT_LT(eval(solver, A_list, b_list), 1e-10);

      if(eq_size <= 100)
      {
        std::cout << "== Gmres (no triangular) ==" << std::endl;
        solver->k_max_ = 1000;
        solver->make_triangular_ = false;
        solver->apply_reorth_ = true;
        EXPECT_LT(eval(solver, A_list, b_list), 1e-10);
      }

      std::cout << "== Gmres (no reorthogonalization) ==" << std::endl;
      solver->k_max_ = 1000;
      solver->make_triangular_ = true;
      solver->apply_reorth_ = false;
      EXPECT_LT(eval(solver, A_list, b_list), 1e-10);

      std::cout << "== Gmres (small iteration) ==" << std::endl;
      solver->k_max_ = 20;
      solver->make_triangular_ = true;
      solver->apply_reorth_ = true;
      EXPECT_LT(eval(solver, A_list, b_list), 1e2);
    }

    {
      auto solver = std::make_shared<FullPivLuLinearSolver>();
      std::cout << "== FullPivLu ==" << std::endl;
      EXPECT_LT(eval(solver, A_list, b_list), 1e-10);
    }

    {
      auto solver = std::make_shared<HouseholderQrLinearSolver>();
      std::cout << "== HouseholderQr ==" << std::endl;
      EXPECT_LT(eval(solver, A_list, b_list), 1e-10);
    }
  }
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
