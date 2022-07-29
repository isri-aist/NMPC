/* Author: Masaki Murooka */

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>

#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <std_srvs/Empty.h>

#include <nmpc_ddp/DDPSolver.h>

namespace Eigen
{
using Vector1d = Eigen::Matrix<double, 1, 1>;
}

/** \brief DDP problem for cart-pole.

    State is [pos, theta, vel, omega]. Input is [force].
    Running cost is sum of the respective quadratic terms of state and input.
    Terminal cost is quadratic term of state.
 */
class DDPProblemCartPole : public nmpc_ddp::DDPProblem<4, 1>
{
public:
  struct Param
  {
    Param() {}

    double cart_mass = 1.0; // [kg]
    double pole_mass = 0.5; // [kg]
    double pole_length = 2.0; // [m]
  };

  struct CostWeight
  {
    CostWeight()
    {
      running_x << 0.1, 1.0, 0.01, 0.1;
      running_u << 0.001;
      terminal_x << 0.1, 1.0, 0.01, 0.1;
    }

    StateDimVector running_x;
    InputDimVector running_u;
    StateDimVector terminal_x;
  };

public:
  DDPProblemCartPole(double dt,
                     const std::function<double(double)> & ref_pos_func,
                     const Param & param = Param(),
                     const CostWeight & cost_weight = CostWeight())
  : DDPProblem(dt), ref_pos_func_(ref_pos_func), param_(param), cost_weight_(cost_weight)
  {
  }

  virtual StateDimVector stateEq(double t, const StateDimVector & x, const InputDimVector & u) const override
  {
    double pos = x[0];
    double theta = x[1];
    double vel = x[2];
    double omega = x[3];
    double f = u[0];

    double m1 = param_.cart_mass;
    double m2 = param_.pole_mass;
    double l = param_.pole_length;

    double sin_theta = std::sin(theta);
    double cos_theta = std::cos(theta);
    double omega2 = std::pow(omega, 2);
    double denom = m1 + m2 * std::pow(sin_theta, 2);

    StateDimVector x_dot;
    // clang-format off
    x_dot[0] = vel;
    x_dot[1] = omega;
    x_dot[2] = (f - m2 * l * omega2 * sin_theta + m2 * g_ * sin_theta * cos_theta) / denom;
    x_dot[3] = (f * cos_theta - m2 * l * omega2 * sin_theta * cos_theta
                + g_ * (m1 + m2) * sin_theta) / (l * denom);
    // clang-format on

    return x + dt_ * x_dot;
  }

  virtual double runningCost(double t, const StateDimVector & x, const InputDimVector & u) const override
  {
    StateDimVector ref_x;
    ref_x << ref_pos_func_(t), 0, 0, 0;
    return 0.5 * cost_weight_.running_x.dot((x - ref_x).cwiseAbs2()) + 0.5 * cost_weight_.running_u.dot(u.cwiseAbs2());
  }

  virtual double terminalCost(double t, const StateDimVector & x) const override
  {
    StateDimVector ref_x;
    ref_x << ref_pos_func_(t), 0, 0, 0;
    return 0.5 * cost_weight_.terminal_x.dot((x - ref_x).cwiseAbs2());
  }

  virtual void calcStateEqDeriv(double t,
                                const StateDimVector & x,
                                const InputDimVector & u,
                                Eigen::Ref<StateStateDimMatrix> state_eq_deriv_x,
                                Eigen::Ref<StateInputDimMatrix> state_eq_deriv_u) const override
  {
    double pos = x[0];
    double theta = x[1];
    double vel = x[2];
    double omega = x[3];
    double f = u[0];

    double m1 = param_.cart_mass;
    double m2 = param_.pole_mass;
    double l = param_.pole_length;

    double sin_theta = std::sin(theta);
    double cos_theta = std::cos(theta);
    double omega2 = std::pow(omega, 2);
    double denom = m1 + m2 * std::pow(sin_theta, 2);

    state_eq_deriv_x.setZero();
    // clang-format off
    state_eq_deriv_x(0, 2) = 1;
    state_eq_deriv_x(1, 3) = 1;
    state_eq_deriv_x(2, 1) = ((-1 * m2 * l * omega2 * cos_theta
                               + m2 * g_ * (1 - 2 * std::pow(sin_theta, 2))) * denom
                              + -1 * (f - m2 * l * omega2 * sin_theta + m2 * g_ * sin_theta * cos_theta)
                              * (2 * m2 * sin_theta * cos_theta))
        / std::pow(denom, 2);
    state_eq_deriv_x(2, 3) = (-2 * m2 * l * omega * sin_theta) / denom;
    state_eq_deriv_x(3, 1) = ((-1 * f * sin_theta + -1 * m2 * l * omega2 * (1 - 2 * std::pow(sin_theta, 2))
                               + g_ * (m1 + m2) * cos_theta) * denom
                              + -1 * (f * cos_theta - m2 * l * omega2 * sin_theta * cos_theta
                                      + g_ * (m1 + m2) * sin_theta) * (2 * m2 * sin_theta * cos_theta))
        / (l * std::pow(denom, 2));
    state_eq_deriv_x(3, 3) = (-2 * m2 * l * omega * sin_theta * cos_theta) / (l * denom);
    // clang-format on
    state_eq_deriv_x *= dt_;
    state_eq_deriv_x.diagonal().array() += 1.0;

    state_eq_deriv_u.setZero();
    state_eq_deriv_u[2] = 1 / denom;
    state_eq_deriv_u[3] = cos_theta / (l * denom);
    state_eq_deriv_u *= dt_;
  }

  virtual void calcStateEqDeriv(double t,
                                const StateDimVector & x,
                                const InputDimVector & u,
                                Eigen::Ref<StateStateDimMatrix> state_eq_deriv_x,
                                Eigen::Ref<StateInputDimMatrix> state_eq_deriv_u,
                                std::vector<StateStateDimMatrix> & state_eq_deriv_xx,
                                std::vector<InputInputDimMatrix> & state_eq_deriv_uu,
                                std::vector<StateInputDimMatrix> & state_eq_deriv_xu) const override
  {
    throw std::runtime_error("Second-order derivatives of state equation are not implemented.");
  }

  virtual void calcRunningCostDeriv(double t,
                                    const StateDimVector & x,
                                    const InputDimVector & u,
                                    Eigen::Ref<StateDimVector> running_cost_deriv_x,
                                    Eigen::Ref<InputDimVector> running_cost_deriv_u) const override
  {
    StateDimVector ref_x;
    ref_x << ref_pos_func_(t), 0, 0, 0;

    running_cost_deriv_x = cost_weight_.running_x.cwiseProduct(x - ref_x);
    running_cost_deriv_u = cost_weight_.running_u.cwiseProduct(u);
  }

  virtual void calcRunningCostDeriv(double t,
                                    const StateDimVector & x,
                                    const InputDimVector & u,
                                    Eigen::Ref<StateDimVector> running_cost_deriv_x,
                                    Eigen::Ref<InputDimVector> running_cost_deriv_u,
                                    Eigen::Ref<StateStateDimMatrix> running_cost_deriv_xx,
                                    Eigen::Ref<InputInputDimMatrix> running_cost_deriv_uu,
                                    Eigen::Ref<StateInputDimMatrix> running_cost_deriv_xu) const override
  {
    StateDimVector ref_x;
    ref_x << ref_pos_func_(t), 0, 0, 0;

    running_cost_deriv_x = cost_weight_.running_x.cwiseProduct(x - ref_x);
    running_cost_deriv_u = cost_weight_.running_u.cwiseProduct(u);

    running_cost_deriv_xx = cost_weight_.running_x.asDiagonal();
    running_cost_deriv_uu = cost_weight_.running_u.asDiagonal();
    running_cost_deriv_xu.setZero();
  }

  virtual void calcTerminalCostDeriv(double t,
                                     const StateDimVector & x,
                                     Eigen::Ref<StateDimVector> terminal_cost_deriv_x) const override
  {
    StateDimVector ref_x;
    ref_x << ref_pos_func_(t), 0, 0, 0;

    terminal_cost_deriv_x = cost_weight_.terminal_x.cwiseProduct(x - ref_x);
  }

  virtual void calcTerminalCostDeriv(double t,
                                     const StateDimVector & x,
                                     Eigen::Ref<StateDimVector> terminal_cost_deriv_x,
                                     Eigen::Ref<StateStateDimMatrix> terminal_cost_deriv_xx) const override
  {
    StateDimVector ref_x;
    ref_x << ref_pos_func_(t), 0, 0, 0;

    terminal_cost_deriv_x = cost_weight_.terminal_x.cwiseProduct(x - ref_x);
    terminal_cost_deriv_xx = cost_weight_.terminal_x.asDiagonal();
  }

public:
  static constexpr double g_ = 9.80665; // [m/s^2]
  std::function<double(double)> ref_pos_func_;
  Param param_;
  CostWeight cost_weight_;
};

// Global variables
std::shared_ptr<nmpc_ddp::DDPSolver<4, 1>> ddp_solver = nullptr;
double current_t = 0;
DDPProblemCartPole::StateDimVector current_x = DDPProblemCartPole::StateDimVector::Zero();
DDPProblemCartPole::InputDimVector current_u = DDPProblemCartPole::InputDimVector::Zero();
std::vector<DDPProblemCartPole::InputDimVector> initial_u_list;
double dist_t = 0;
DDPProblemCartPole::InputDimVector dist_u = DDPProblemCartPole::InputDimVector::Zero();
double target_pos = std::numeric_limits<double>::quiet_NaN();
bool first_iter = true;
auto input_limits_func = [&](double t) -> std::array<Eigen::Vector1d, 2> {
  std::array<Eigen::Vector1d, 2> limits;
  limits[0].setConstant(-15.0);
  limits[1].setConstant(15.0);
  return limits;
};

void mpcTimerCallback(const ros::TimerEvent & event)
{
  // Solve
  ddp_solver->solve(current_t, current_x, initial_u_list);
  const auto & input_limits = input_limits_func(current_t);
  current_u = ddp_solver->controlData().u_list[0].cwiseMax(input_limits[0]).cwiseMin(input_limits[1]);
  initial_u_list = ddp_solver->controlData().u_list;

  // Dump
  if(first_iter)
  {
    first_iter = false;
    ddp_solver->dumpTraceDataList("/tmp/TestDDPCartPoleTraceData.txt");
  }
}

bool distCallback(std_srvs::Empty::Request & req, std_srvs::Empty::Response & res, double dist_force)
{
  dist_u << dist_force;
  dist_t = current_t + 0.5; // [sec]
  return true;
}

bool targetPosCallback(std_srvs::Empty::Request & req, std_srvs::Empty::Response & res, double pos)
{
  target_pos = pos;
  return true;
}

void checkDerivatives(const std::shared_ptr<DDPProblemCartPole> & ddp_problem)
{
  double t = 0;
  DDPProblemCartPole::StateDimVector x;
  x << 1.0, -2.0, 3.0, -4.0;
  DDPProblemCartPole::InputDimVector u;
  u << 10.0;

  DDPProblemCartPole::StateStateDimMatrix state_eq_deriv_x_analytical;
  DDPProblemCartPole::StateInputDimMatrix state_eq_deriv_u_analytical;
  ddp_problem->calcStateEqDeriv(t, x, u, state_eq_deriv_x_analytical, state_eq_deriv_u_analytical);

  DDPProblemCartPole::StateStateDimMatrix state_eq_deriv_x_numerical;
  DDPProblemCartPole::StateInputDimMatrix state_eq_deriv_u_numerical;
  constexpr double deriv_eps = 1e-6;
  for(int i = 0; i < ddp_problem->stateDim(); i++)
  {
    state_eq_deriv_x_numerical.col(i) =
        (ddp_problem->stateEq(t, x + deriv_eps * DDPProblemCartPole::StateDimVector::Unit(i), u)
         - ddp_problem->stateEq(t, x - deriv_eps * DDPProblemCartPole::StateDimVector::Unit(i), u))
        / (2 * deriv_eps);
  }
  for(int i = 0; i < ddp_problem->inputDim(); i++)
  {
    state_eq_deriv_u_numerical.col(i) =
        (ddp_problem->stateEq(t, x, u + deriv_eps * DDPProblemCartPole::InputDimVector::Unit(i))
         - ddp_problem->stateEq(t, x, u - deriv_eps * DDPProblemCartPole::InputDimVector::Unit(i)))
        / (2 * deriv_eps);
  }

  EXPECT_LT((state_eq_deriv_x_analytical - state_eq_deriv_x_numerical).norm(), 1e-6);
  EXPECT_LT((state_eq_deriv_u_analytical - state_eq_deriv_u_numerical).norm(), 1e-6);
}

visualization_msgs::MarkerArray makeMarkerArr(const DDPProblemCartPole::StateDimVector & x,
                                              const DDPProblemCartPole::InputDimVector & u,
                                              const std::shared_ptr<DDPProblemCartPole> & ddp_problem)
{
  std_msgs::Header header_msg;
  header_msg.frame_id = "world";
  header_msg.stamp = ros::Time::now();

  // Instantiate marker array
  visualization_msgs::MarkerArray marker_arr_msg;

  // Delete marker
  visualization_msgs::Marker del_marker;
  del_marker.action = visualization_msgs::Marker::DELETEALL;
  del_marker.header = header_msg;
  del_marker.id = marker_arr_msg.markers.size();
  marker_arr_msg.markers.push_back(del_marker);

  // Cart marker
  visualization_msgs::Marker cart_marker;
  cart_marker.header = header_msg;
  cart_marker.ns = "cart";
  cart_marker.id = marker_arr_msg.markers.size();
  cart_marker.type = visualization_msgs::Marker::CUBE;
  cart_marker.color.r = 0;
  cart_marker.color.g = 1;
  cart_marker.color.b = 0;
  cart_marker.color.a = 1;
  cart_marker.scale.x = 1.0;
  cart_marker.scale.y = 0.6;
  cart_marker.scale.z = 0.12;
  cart_marker.pose.position.x = x[0];
  cart_marker.pose.position.y = 0;
  cart_marker.pose.position.z = 0;
  cart_marker.pose.orientation.w = 1.0;
  marker_arr_msg.markers.push_back(cart_marker);

  // Mass marker
  visualization_msgs::Marker mass_marker;
  mass_marker.header = header_msg;
  mass_marker.ns = "mass";
  cart_marker.id = marker_arr_msg.markers.size();
  mass_marker.type = visualization_msgs::Marker::CYLINDER;
  mass_marker.color.r = 0;
  mass_marker.color.g = 0;
  mass_marker.color.b = 1;
  mass_marker.color.a = 1;
  mass_marker.scale.x = 0.6;
  mass_marker.scale.y = 0.6;
  mass_marker.scale.z = 0.1;
  mass_marker.pose.position.x = x[0] + ddp_problem->param_.pole_length * -1 * std::sin(x[1]);
  mass_marker.pose.position.y = ddp_problem->param_.pole_length * std::cos(x[1]);
  mass_marker.pose.position.z = 2.0;
  mass_marker.pose.orientation.w = 1.0;
  marker_arr_msg.markers.push_back(mass_marker);

  // Pole marker
  visualization_msgs::Marker pole_marker;
  pole_marker.header = header_msg;
  pole_marker.ns = "pole";
  pole_marker.id = marker_arr_msg.markers.size();
  pole_marker.type = visualization_msgs::Marker::LINE_LIST;
  pole_marker.color.r = 0;
  pole_marker.color.g = 0;
  pole_marker.color.b = 0;
  pole_marker.color.a = 1;
  pole_marker.scale.x = 0.2;
  pole_marker.pose.position.z = 1.0;
  pole_marker.pose.orientation.w = 1.0;
  pole_marker.points.resize(2);
  pole_marker.points[0].x = cart_marker.pose.position.x;
  pole_marker.points[0].y = cart_marker.pose.position.y;
  pole_marker.points[1].x = mass_marker.pose.position.x;
  pole_marker.points[1].y = mass_marker.pose.position.y;
  marker_arr_msg.markers.push_back(pole_marker);

  // Force marker
  constexpr double force_scale = 0.04;
  constexpr double force_thre = 1.0; // [N]
  if(std::abs(u[0]) > force_thre)
  {
    visualization_msgs::Marker force_marker;
    force_marker.header = header_msg;
    force_marker.ns = "force";
    force_marker.id = marker_arr_msg.markers.size();
    force_marker.type = visualization_msgs::Marker::ARROW;
    force_marker.color.r = 1;
    force_marker.color.g = 0;
    force_marker.color.b = 0;
    force_marker.color.a = 1;
    force_marker.scale.x = 0.2;
    force_marker.scale.y = 0.4;
    force_marker.scale.z = 0.2;
    force_marker.pose.position.z = 3.0;
    force_marker.pose.orientation.w = 1.0;
    force_marker.points.resize(2);
    force_marker.points[0].x = cart_marker.pose.position.x;
    force_marker.points[0].y = cart_marker.pose.position.y;
    force_marker.points[1].x = cart_marker.pose.position.x + force_scale * u[0];
    force_marker.points[1].y = cart_marker.pose.position.y;
    marker_arr_msg.markers.push_back(force_marker);
  }

  // Disturbance marker
  if(std::abs(dist_u[0]) > force_thre)
  {
    visualization_msgs::Marker dist_marker;
    dist_marker.header = header_msg;
    dist_marker.ns = "disturbance";
    dist_marker.id = marker_arr_msg.markers.size();
    dist_marker.type = visualization_msgs::Marker::ARROW;
    dist_marker.color.r = 1;
    dist_marker.color.g = 1;
    dist_marker.color.b = 0;
    dist_marker.color.a = 1;
    dist_marker.scale.x = 0.2;
    dist_marker.scale.y = 0.4;
    dist_marker.scale.z = 0.2;
    dist_marker.pose.position.z = 3.0;
    dist_marker.pose.orientation.w = 1.0;
    dist_marker.points.resize(2);
    dist_marker.points[0].x = cart_marker.pose.position.x;
    dist_marker.points[0].y = cart_marker.pose.position.y;
    dist_marker.points[1].x = cart_marker.pose.position.x + force_scale * dist_u[0];
    dist_marker.points[1].y = cart_marker.pose.position.y;
    marker_arr_msg.markers.push_back(dist_marker);
  }

  // Target marker
  double target_pos = ddp_problem->ref_pos_func_(current_t);
  visualization_msgs::Marker target_marker;
  target_marker.header = header_msg;
  target_marker.ns = "target";
  target_marker.id = marker_arr_msg.markers.size();
  target_marker.type = visualization_msgs::Marker::LINE_LIST;
  target_marker.color.r = 0;
  target_marker.color.g = 1;
  target_marker.color.b = 1;
  target_marker.color.a = 0.5;
  target_marker.scale.x = 0.05;
  target_marker.pose.position.z = -1.0;
  target_marker.pose.orientation.w = 1.0;
  target_marker.points.resize(2);
  target_marker.points[0].x = target_pos;
  target_marker.points[0].y = -1e3;
  target_marker.points[1].x = target_pos;
  target_marker.points[1].y = 1e3;
  marker_arr_msg.markers.push_back(target_marker);

  return marker_arr_msg;
}

TEST(TestDDPCartPole, TestCase1)
{
  // Setup ROS
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  ros::Publisher marker_arr_pub = nh.advertise<visualization_msgs::MarkerArray>("marker_arr", 1);
  constexpr double dist_force_small = 10; // [N]
  ros::ServiceServer dist_left_small_srv = nh.advertiseService<std_srvs::Empty::Request, std_srvs::Empty::Response>(
      "/dist_left_small", std::bind(distCallback, std::placeholders::_1, std::placeholders::_2, -1 * dist_force_small));
  ros::ServiceServer dist_right_small_srv = nh.advertiseService<std_srvs::Empty::Request, std_srvs::Empty::Response>(
      "/dist_right_small", std::bind(distCallback, std::placeholders::_1, std::placeholders::_2, dist_force_small));
  constexpr double dist_force_large = 30; // [N]
  ros::ServiceServer dist_left_large_srv = nh.advertiseService<std_srvs::Empty::Request, std_srvs::Empty::Response>(
      "/dist_left_large", std::bind(distCallback, std::placeholders::_1, std::placeholders::_2, -1 * dist_force_large));
  ros::ServiceServer dist_right_large_srv = nh.advertiseService<std_srvs::Empty::Request, std_srvs::Empty::Response>(
      "/dist_right_large", std::bind(distCallback, std::placeholders::_1, std::placeholders::_2, dist_force_large));
  ros::ServiceServer target_pos_m5_srv = nh.advertiseService<std_srvs::Empty::Request, std_srvs::Empty::Response>(
      "/target_pos_m5", std::bind(targetPosCallback, std::placeholders::_1, std::placeholders::_2, -5.0));
  ros::ServiceServer target_pos_0_srv = nh.advertiseService<std_srvs::Empty::Request, std_srvs::Empty::Response>(
      "/target_pos_0", std::bind(targetPosCallback, std::placeholders::_1, std::placeholders::_2, 0.0));
  ros::ServiceServer target_pos_p5_srv = nh.advertiseService<std_srvs::Empty::Request, std_srvs::Empty::Response>(
      "/target_pos_p5", std::bind(targetPosCallback, std::placeholders::_1, std::placeholders::_2, 5.0));

  double horizon_dt = 0.01; // [sec]
  double horizon_duration = 2.0; // [sec]
  double mpc_dt = 0.004; // [sec]
  double sim_dt = 0.002; // [sec]
  double end_t = 10.0; // [sec]
  pnh.getParam("control/horizon_dt", horizon_dt);
  pnh.getParam("control/horizon_duration", horizon_duration);
  pnh.getParam("control/mpc_dt", mpc_dt);
  pnh.getParam("control/sim_dt", sim_dt);

  // Instantiate problem for MPC
  constexpr double epsilon_t = 1e-6;
  std::function<double(double)> ref_pos_func = [&](double t) {
    // Add small values to avoid numerical instability at inequality bounds
    t += epsilon_t;
    if(std::isnan(target_pos))
    {
      return 0.0; // [m]
    }
    else
    {
      return target_pos;
    }
  };
  auto ddp_problem = std::make_shared<DDPProblemCartPole>(horizon_dt, ref_pos_func);
  pnh.getParam("param/cart_mass", ddp_problem->param_.cart_mass);
  pnh.getParam("param/pole_mass", ddp_problem->param_.pole_mass);
  pnh.getParam("param/pole_length", ddp_problem->param_.pole_length);
  std::vector<double> param_vec;
  if(pnh.getParam("cost/running_x", param_vec))
  {
    ddp_problem->cost_weight_.running_x = Eigen::Map<DDPProblemCartPole::StateDimVector>(param_vec.data());
  }
  if(pnh.getParam("cost/running_u", param_vec))
  {
    ddp_problem->cost_weight_.running_u = Eigen::Map<DDPProblemCartPole::InputDimVector>(param_vec.data());
  }
  if(pnh.getParam("cost/terminal_x", param_vec))
  {
    ddp_problem->cost_weight_.terminal_x = Eigen::Map<DDPProblemCartPole::StateDimVector>(param_vec.data());
  }

  // Check derivatives
  checkDerivatives(ddp_problem);

  // Instantiate solver
  ddp_solver = std::make_shared<nmpc_ddp::DDPSolver<4, 1>>(ddp_problem);
  ddp_solver->setInputLimitsFunc(input_limits_func);
  ddp_solver->config().with_input_constraint = true;
  int horizon_steps = static_cast<int>(horizon_duration / horizon_dt);
  ddp_solver->config().horizon_steps = horizon_steps;
  ddp_solver->config().max_iter = 3;

  // Instantiate simulation (only state equation is used and cost is ignored)
  auto sim = std::make_shared<DDPProblemCartPole>(sim_dt, ref_pos_func, ddp_problem->param_, ddp_problem->cost_weight_);

  // Setup simulation loop
  current_t = 0;
  current_x << 0, M_PI, 0, 0;
  current_u << 0;
  initial_u_list.assign(horizon_steps, DDPProblemCartPole::InputDimVector::Zero());
  dist_t = 0;
  dist_u << 0;
  std::string file_path = "/tmp/TestDDPCartPoleResult.txt";
  std::ofstream ofs(file_path);
  ofs << "time pos theta vel omega force ref_pos disturbance" << std::endl;
  ros::Rate rate(1.0 / sim_dt);
  bool no_exit = false;
  pnh.getParam("no_exit", no_exit);

  // Sleep to wait for Rviz to launch
  ros::Duration(1.0).sleep();

  // Run simulation loop
  ros::Timer mpc_timer = nh.createTimer(ros::Duration(mpc_dt), mpcTimerCallback);
  while(ros::ok() && (no_exit || current_t < end_t))
  {
    // Simulate one step
    if(dist_t < current_t)
    {
      dist_u << 0;
    }
    current_x = sim->stateEq(current_t, current_x, current_u + dist_u);
    current_t += sim_dt;

    // Check pos
    double current_pos = current_x[0];
    double ref_pos = ref_pos_func(current_t);
    EXPECT_LT(std::abs(current_pos - ref_pos), 1e2);

    // Dump
    ofs << current_t << " " << current_x.transpose() << " " << current_u.transpose() << " " << ref_pos << " "
        << dist_u.transpose() << std::endl;

    // Publish marker
    marker_arr_pub.publish(makeMarkerArr(current_x, current_u, ddp_problem));
    ros::spinOnce();
    rate.sleep();
  }
  mpc_timer.stop();

  // Check final pos
  double ref_pos = ref_pos_func(current_t);
  EXPECT_LT(std::abs(current_x[0] - ref_pos), 1.0);
  EXPECT_LT(std::abs(current_x[1]), 1e-1);
  EXPECT_LT(std::abs(current_x[2]), 1.0);
  EXPECT_LT(std::abs(current_x[3]), 1e-1);

  std::cout << "Run the following commands in gnuplot:\n"
            << "  set key autotitle columnhead\n"
            << "  set key noenhanced\n"
            << "  plot \"" << file_path << "\" u 1:2 w lp, \"\" u 1:3 w lp, \"\" u 1:7 w l lw 3 # State\n"
            << "  plot \"" << file_path << "\" u 1:6 w l lw 3 # Input\n";
}

int main(int argc, char ** argv)
{
  ros::init(argc, argv, "test_ddp_cart_pole");
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
