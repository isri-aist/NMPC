/* Author: Masaki Murooka */

#include <gtest/gtest.h>

#include <nmpc_cgmres/CgmresSolver.h>

#include "CartPoleProblem.h"
#include "SemiactiveDamperProblem.h"

void testCgmresSolver(const std::shared_ptr<nmpc_cgmres::CgmresProblem> & problem, double x_thre)
{
  auto ode_solver = std::make_shared<nmpc_cgmres::EulerOdeSolver>();
  auto sim_ode_solver = std::make_shared<nmpc_cgmres::RungeKuttaOdeSolver>();
  auto solver = std::make_shared<nmpc_cgmres::CgmresSolver>(problem, ode_solver, sim_ode_solver);
  solver->sim_duration_ = 20.0;
  solver->run();
  EXPECT_LT(solver->x_.norm(), x_thre);
}

TEST(TestCgmresSolver, SemiactiveDamperProblem)
{
  testCgmresSolver(std::make_shared<SemiactiveDamperProblem>(), 0.1);
}

TEST(TestCgmresSolver, CartPoleProblem)
{
  testCgmresSolver(std::make_shared<CartPoleProblem>(nullptr, true), 0.1);
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
