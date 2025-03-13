/* Author: Masaki Murooka */

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <chrono>

#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <std_srvs/srv/empty.hpp>

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
    return stateEq(t, x, u, dt_);
  }

  virtual StateDimVector stateEq(double, // t
                                 const StateDimVector & x,
                                 const InputDimVector & u,
                                 double dt) const
  {
    // double pos = x[0];
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

    return x + dt * x_dot;
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

  virtual void calcStateEqDeriv(double, // t
                                const StateDimVector & x,
                                const InputDimVector & u,
                                Eigen::Ref<StateStateDimMatrix> state_eq_deriv_x,
                                Eigen::Ref<StateInputDimMatrix> state_eq_deriv_u) const override
  {
    // double pos = x[0];
    double theta = x[1];
    // double vel = x[2];
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

  virtual void calcStateEqDeriv(double, // t
                                const StateDimVector &, // x
                                const InputDimVector &, // u
                                Eigen::Ref<StateStateDimMatrix>, // state_eq_deriv_x
                                Eigen::Ref<StateInputDimMatrix>, // state_eq_deriv_u
                                std::vector<StateStateDimMatrix> &, // state_eq_deriv_xx
                                std::vector<InputInputDimMatrix> &, // state_eq_deriv_uu
                                std::vector<StateInputDimMatrix> & // state_eq_deriv_xu
  ) const override
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

class TestDDPCartPole
{
public:
  TestDDPCartPole()
  {
    // Setup ROS
    marker_arr_pub_ = nh_->create_publisher<visualization_msgs::msg::MarkerArray>("marker_arr", 1);
    constexpr double dist_force_small = 10; // [N]
    dist_left_small_srv_ = nh_->create_service<std_srvs::srv::Empty>(
        "/dist_left_small", std::bind(&TestDDPCartPole::distCallback, this, std::placeholders::_1,
                                      std::placeholders::_2, -1 * dist_force_small));
    dist_right_small_srv_ = nh_->create_service<std_srvs::srv::Empty>(
        "/dist_right_small", std::bind(&TestDDPCartPole::distCallback, this, std::placeholders::_1,
                                       std::placeholders::_2, dist_force_small));
    constexpr double dist_force_large = 30; // [N]
    dist_left_large_srv_ = nh_->create_service<std_srvs::srv::Empty>(
        "/dist_left_large", std::bind(&TestDDPCartPole::distCallback, this, std::placeholders::_1,
                                      std::placeholders::_2, -1 * dist_force_large));
    dist_right_large_srv_ = nh_->create_service<std_srvs::srv::Empty>(
        "/dist_right_large", std::bind(&TestDDPCartPole::distCallback, this, std::placeholders::_1,
                                       std::placeholders::_2, dist_force_large));
    target_pos_m5_srv_ = nh_->create_service<std_srvs::srv::Empty>(
        "/target_pos_m5",
        std::bind(&TestDDPCartPole::targetPosCallback, this, std::placeholders::_1, std::placeholders::_2, -5.0));
    target_pos_0_srv_ = nh_->create_service<std_srvs::srv::Empty>(
        "/target_pos_0",
        std::bind(&TestDDPCartPole::targetPosCallback, this, std::placeholders::_1, std::placeholders::_2, 0.0));
    target_pos_p5_srv_ = nh_->create_service<std_srvs::srv::Empty>(
        "/target_pos_p5",
        std::bind(&TestDDPCartPole::targetPosCallback, this, std::placeholders::_1, std::placeholders::_2, 5.0));

    // Instantiate problem
    double horizon_dt = 0.01; // [sec]
    double horizon_duration = 2.0; // [sec]
    nh_->get_parameter("control/horizon_dt", horizon_dt);
    nh_->get_parameter("control/horizon_duration", horizon_duration);
    ddp_problem_ = std::make_shared<DDPProblemCartPole>(
        horizon_dt, std::bind(&TestDDPCartPole::getRefPos, this, std::placeholders::_1));
    nh_->get_parameter("param/cart_mass", ddp_problem_->param_.cart_mass);
    nh_->get_parameter("param/pole_mass", ddp_problem_->param_.pole_mass);
    nh_->get_parameter("param/pole_length", ddp_problem_->param_.pole_length);
    std::vector<double> param_vec;
    if(nh_->get_parameter("cost/running_x", param_vec))
    {
      ddp_problem_->cost_weight_.running_x = Eigen::Map<DDPProblemCartPole::StateDimVector>(param_vec.data());
    }
    if(nh_->get_parameter("cost/running_u", param_vec))
    {
      ddp_problem_->cost_weight_.running_u = Eigen::Map<DDPProblemCartPole::InputDimVector>(param_vec.data());
    }
    if(nh_->get_parameter("cost/terminal_x", param_vec))
    {
      ddp_problem_->cost_weight_.terminal_x = Eigen::Map<DDPProblemCartPole::StateDimVector>(param_vec.data());
    }

    // Instantiate solver
    ddp_solver_ = std::make_shared<nmpc_ddp::DDPSolver<4, 1>>(ddp_problem_);
    ddp_solver_->setInputLimitsFunc(std::bind(&TestDDPCartPole::getInputLimits, this, std::placeholders::_1));
    ddp_solver_->config().with_input_constraint = true;
    ddp_solver_->config().horizon_steps = static_cast<int>(horizon_duration / horizon_dt);
    ddp_solver_->config().max_iter = 3;
  }

  void run()
  {
    // Setup simulation loop
    double mpc_dt = 0.004; // [sec]
    double sim_dt = 0.002; // [sec]
    nh_->get_parameter("control/mpc_dt", mpc_dt);
    nh_->get_parameter("control/sim_dt", sim_dt);
    current_t_ = 0;
    current_x_ << 0, M_PI, 0, 0;
    current_u_ << 0;
    initial_u_list_.assign(ddp_solver_->config().horizon_steps, DDPProblemCartPole::InputDimVector::Zero());
    dist_t_ = 0;
    dist_u_ << 0;

    // Run simulation loop
    std::string file_path = "/tmp/TestDDPCartPoleResult.txt";
    std::ofstream ofs(file_path);
    ofs << "time pos theta vel omega force ref_pos disturbance" << std::endl;
    rclcpp::Rate rate(1.0 / sim_dt);
    bool no_exit = false;
    nh_->get_parameter("no_exit", no_exit);
    constexpr double end_t = 10.0; // [sec]
    rclcpp::TimerBase::SharedPtr mpc_timer = nh_->create_wall_timer(std::chrono::duration<double>(mpc_dt), std::bind(&TestDDPCartPole::mpcTimerCallback, this));
    while(rclcpp::ok() && (no_exit || current_t_ < end_t))
    {
      // Simulate one step
      if(dist_t_ < current_t_)
      {
        dist_u_ << 0;
      }
      current_x_ = ddp_problem_->stateEq(current_t_, current_x_, current_u_ + dist_u_, sim_dt);
      current_t_ += sim_dt;

      // Check pos
      double current_pos = current_x_[0];
      double ref_pos = getRefPos(current_t_);
      EXPECT_LT(std::abs(current_pos - ref_pos), 1e2);

      // Dump
      ofs << current_t_ << " " << current_x_.transpose() << " " << current_u_.transpose() << " " << ref_pos << " "
          << dist_u_.transpose() << std::endl;

      // Publish marker
      marker_arr_pub_->publish(makeMarkerArr());
      rclcpp::spin_some(nh_);
      rate.sleep();
    }
    mpc_timer->cancel();

    // Check final pos
    double ref_pos = getRefPos(current_t_);
    EXPECT_LT(std::abs(current_x_[0] - ref_pos), 1.0);
    EXPECT_LT(std::abs(current_x_[1]), 1e-1);
    EXPECT_LT(std::abs(current_x_[2]), 1.0);
    EXPECT_LT(std::abs(current_x_[3]), 1e-1);

    std::cout << "Run the following commands in gnuplot:\n"
              << "  set key autotitle columnhead\n"
              << "  set key noenhanced\n"
              << "  plot \"" << file_path << "\" u 1:2 w lp, \"\" u 1:3 w lp, \"\" u 1:7 w l lw 3 # State\n"
              << "  plot \"" << file_path << "\" u 1:6 w l lw 3 # Input\n";
  }

protected:
  double getRefPos(double t) const
  {
    // Add small values to avoid numerical instability at inequality bounds
    constexpr double epsilon_t = 1e-6;
    t += epsilon_t;
    if(std::isnan(target_pos_))
    {
      return 0.0; // [m]
    }
    else
    {
      return target_pos_;
    }
  }

  std::array<Eigen::Vector1d, 2> getInputLimits(double // t
  ) const
  {
    std::array<Eigen::Vector1d, 2> limits;
    limits[0].setConstant(-15.0);
    limits[1].setConstant(15.0);
    return limits;
  };

  void mpcTimerCallback(
  )
  {
    // Solve
    ddp_solver_->solve(current_t_, current_x_, initial_u_list_);
    const auto & input_limits = getInputLimits(current_t_);
    current_u_ = ddp_solver_->controlData().u_list[0].cwiseMax(input_limits[0]).cwiseMin(input_limits[1]);
    initial_u_list_ = ddp_solver_->controlData().u_list;

    // Dump
    if(first_iter_)
    {
      first_iter_ = false;
      ddp_solver_->dumpTraceDataList("/tmp/TestDDPCartPoleTraceData.txt");
    }
  }

  void distCallback(const std::shared_ptr<std_srvs::srv::Empty::Request> request, // req
                    std::shared_ptr<std_srvs::srv::Empty::Response> response,
                    double dist_force)
  {
    (void)request;
    (void)response;
    dist_u_ << dist_force;
    dist_t_ = current_t_ + 0.5; // [sec]
  }

  void targetPosCallback(const std::shared_ptr<std_srvs::srv::Empty::Request> request, // req
                        std::shared_ptr<std_srvs::srv::Empty::Response> response, // res
                         double pos)
  {
    (void)request;
    (void)response;
    target_pos_ = pos;
  }

  visualization_msgs::msg::MarkerArray makeMarkerArr() const
  {
    std_msgs::msg::Header header_msg;
    header_msg.frame_id = "world";
    rclcpp::Clock clock;
    rclcpp::Time time_now = clock.now();
    header_msg.stamp = time_now;

    // Instantiate marker array
    visualization_msgs::msg::MarkerArray marker_arr_msg;

    // Delete marker
    visualization_msgs::msg::Marker del_marker;
    del_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    del_marker.header = header_msg;
    del_marker.id = static_cast<int>(marker_arr_msg.markers.size());
    marker_arr_msg.markers.push_back(del_marker);

    // Cart marker
    visualization_msgs::msg::Marker cart_marker;
    cart_marker.header = header_msg;
    cart_marker.ns = "cart";
    cart_marker.id = static_cast<int>(marker_arr_msg.markers.size());
    cart_marker.type = visualization_msgs::msg::Marker::CUBE;
    cart_marker.color.r = 0;
    cart_marker.color.g = 1;
    cart_marker.color.b = 0;
    cart_marker.color.a = 1;
    cart_marker.scale.x = 1.0;
    cart_marker.scale.y = 0.6;
    cart_marker.scale.z = 0.12;
    cart_marker.pose.position.x = current_x_[0];
    cart_marker.pose.position.y = 0;
    cart_marker.pose.position.z = 0;
    cart_marker.pose.orientation.w = 1.0;
    marker_arr_msg.markers.push_back(cart_marker);

    // Mass marker
    visualization_msgs::msg::Marker mass_marker;
    mass_marker.header = header_msg;
    mass_marker.ns = "mass";
    cart_marker.id = static_cast<int>(marker_arr_msg.markers.size());
    mass_marker.type = visualization_msgs::msg::Marker::CYLINDER;
    mass_marker.color.r = 0;
    mass_marker.color.g = 0;
    mass_marker.color.b = 1;
    mass_marker.color.a = 1;
    mass_marker.scale.x = 0.6;
    mass_marker.scale.y = 0.6;
    mass_marker.scale.z = 0.1;
    mass_marker.pose.position.x = current_x_[0] + ddp_problem_->param_.pole_length * -1 * std::sin(current_x_[1]);
    mass_marker.pose.position.y = ddp_problem_->param_.pole_length * std::cos(current_x_[1]);
    mass_marker.pose.position.z = 2.0;
    mass_marker.pose.orientation.w = 1.0;
    marker_arr_msg.markers.push_back(mass_marker);

    // Pole marker
    visualization_msgs::msg::Marker pole_marker;
    pole_marker.header = header_msg;
    pole_marker.ns = "pole";
    pole_marker.id = static_cast<int>(marker_arr_msg.markers.size());
    pole_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
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
    if(std::abs(current_u_[0]) > force_thre)
    {
      visualization_msgs::msg::Marker force_marker;
      force_marker.header = header_msg;
      force_marker.ns = "force";
      force_marker.id = static_cast<int>(marker_arr_msg.markers.size());
      force_marker.type = visualization_msgs::msg::Marker::ARROW;
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
      force_marker.points[1].x = cart_marker.pose.position.x + force_scale * current_u_[0];
      force_marker.points[1].y = cart_marker.pose.position.y;
      marker_arr_msg.markers.push_back(force_marker);
    }

    // Disturbance marker
    if(std::abs(dist_u_[0]) > force_thre)
    {
      visualization_msgs::msg::Marker dist_marker;
      dist_marker.header = header_msg;
      dist_marker.ns = "disturbance";
      dist_marker.id = static_cast<int>(marker_arr_msg.markers.size());
      dist_marker.type = visualization_msgs::msg::Marker::ARROW;
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
      dist_marker.points[1].x = cart_marker.pose.position.x + force_scale * dist_u_[0];
      dist_marker.points[1].y = cart_marker.pose.position.y;
      marker_arr_msg.markers.push_back(dist_marker);
    }

    // Target marker
    double target_pos = getRefPos(current_t_);
    visualization_msgs::msg::Marker target_marker;
    target_marker.header = header_msg;
    target_marker.ns = "target";
    target_marker.id = static_cast<int>(marker_arr_msg.markers.size());
    target_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
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

protected:
  std::shared_ptr<DDPProblemCartPole> ddp_problem_;
  std::shared_ptr<nmpc_ddp::DDPSolver<4, 1>> ddp_solver_;

  double current_t_ = 0; // [sec]
  DDPProblemCartPole::StateDimVector current_x_ = DDPProblemCartPole::StateDimVector::Zero();
  DDPProblemCartPole::InputDimVector current_u_ = DDPProblemCartPole::InputDimVector::Zero();
  std::vector<DDPProblemCartPole::InputDimVector> initial_u_list_;

  double dist_t_ = 0;
  DDPProblemCartPole::InputDimVector dist_u_ = DDPProblemCartPole::InputDimVector::Zero();
  double target_pos_ = std::numeric_limits<double>::quiet_NaN();

  bool first_iter_ = true;

  std::shared_ptr<rclcpp::Node> nh_ = rclcpp::Node::make_shared("test_ddp_cart_pole");
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_arr_pub_;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr dist_left_small_srv_;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr dist_right_small_srv_;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr dist_left_large_srv_;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr dist_right_large_srv_;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr target_pos_m5_srv_;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr target_pos_0_srv_;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr target_pos_p5_srv_;
};

TEST(TestDDPCartPole, SolveMpc)
{
  TestDDPCartPole test;

  // Sleep to wait for Rviz to launch
  rclcpp::sleep_for(std::chrono::seconds(1));

  test.run();
}

TEST(TestDDPCartPole, CheckDerivative)
{
  double dt = 0.01; // [sec]
  std::function<double(double)> ref_pos_func = [&](double // t
                                               ) {
    return 0.0; // [m]
  };
  auto ddp_problem = std::make_shared<DDPProblemCartPole>(dt, ref_pos_func);

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

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}