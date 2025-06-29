//
// Created by bluewolf on 14.05.24.
//
// sample.cpp

#include "qp_solver_collection/QpSolverCollection.h"
#include <iostream>
#include "mpc_wrapper/MpcSolver.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <limits>


int main() {
  const int nx = 12;
  const int nu = 4;
  const int N = 20;  // Prediction horizon

  // System matrices
  Eigen::MatrixXd Ad(nx, nx);
  Ad << 1., 0., 0., 0., 0., 0., 0.1, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.1, 0., 0.,
      0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.1, 0., 0., 0., 0.0488, 0., 0., 1., 0., 0., 0.0016,
      0., 0., 0.0992, 0., 0., 0., -0.0488, 0., 0., 1., 0., 0., -0.0016, 0., 0., 0.0992, 0., 0.,
      0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.0992, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
      0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
      0., 0., 0.9734, 0., 0., 0., 0., 0., 0.0488, 0., 0., 0.9846, 0., 0., 0., -0.9734, 0., 0., 0.,
      0., 0., -0.0488, 0., 0., 0.9846, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.9846;

  Eigen::MatrixXd Bd(nx, nu);
  Bd << 0., -0.0726, 0., 0.0726, -0.0726, 0., 0.0726, 0., -0.0152, 0.0152, -0.0152, 0.0152, -0.,
      -0.0006, -0., 0.0006, 0.0006, 0., -0.0006, 0.0000, 0.0106, 0.0106, 0.0106, 0.0106, 0,
      -1.4512, 0., 1.4512, -1.4512, 0., 1.4512, 0., -0.3049, 0.3049, -0.3049, 0.3049, -0.,
      -0.0236, 0., 0.0236, 0.0236, 0., -0.0236, 0., 0.2107, 0.2107, 0.2107, 0.2107;


  // Constraints
  double u0 = 10.5916;
  Eigen::VectorXd umin(nu), umax(nu), xmin(nx), xmax(nx);
  umin << 9.6, 9.6, 9.6, 9.6;
  umax << 13.0, 13.0, 13.0, 13.0;
  umin.array() -= u0;
  umax.array() -= u0;
  xmin << -M_PI / 6, -M_PI / 6, -std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(), -1.0, -std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity();
  xmax << M_PI / 6, M_PI / 6, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity();

  // Objective function
  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(nx, nx);
  Q.diagonal() << 0., 0., 10., 10., 10., 10., 0., 0., 0., 5., 5., 5.;
  Eigen::MatrixXd QN = Q;
  Eigen::MatrixXd R = 0.1 * Eigen::MatrixXd::Identity(nu, nu);

  // Initial and reference states
  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nx);
  Eigen::VectorXd xr(nx);
  xr << 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.;

  // Formulate QP matrices
  int dim_var = (N + 1) * nx + N * nu; // number of variables
  int dim_eq = (N + 1) * nx;           // number of equality constraints

  QpSolverCollection::QpCoeff qp_coeff;
  qp_coeff.setup(dim_var, dim_eq, 0);

//  Eigen::MatrixXd P = Eigen::MatrixXd::Zero(dim_var, dim_var);
//  Eigen::VectorXd q = Eigen::VectorXd::Zero(dim_var);

//  for (int i = 0; i < N; ++i) {
//    P.block(i * nx, i   * nx, nx, nx) = Q;
//    P.block((N + 1) * nx + i * nu, (N + 1) * nx + i * nu, nu, nu) = R;
//    q.segment(i * nx, nx) = -Q * xr;
//  }
//  P.block(N * nx, N * nx, nx, nx) = QN;
//  q.segment(N * nx, nx) = -QN * xr;

  Eigen::SparseMatrix<double> P((N + 1) * nx + N * nu, (N + 1) * nx + N * nu); // Hessian matrix
  Eigen::VectorXd q = Eigen::VectorXd::Zero((N + 1) * nx + N * nu); // Gradient vector

  // Populate the Hessian matrix P
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < nx; ++j) {
      P.insert(i * nx + j, i * nx + j) = Q.diagonal()[j]; // Insert Q diagonal elements
    }
    for (int j = 0; j < nu; ++j) {
      P.insert((N + 1) * nx + i * nu + j, (N + 1) * nx + i * nu + j) = R.diagonal()[j]; // Insert R diagonal elements
    }
  }
  // Insert terminal weight matrix QN
  for (int j = 0; j < nx; ++j) {
    P.insert(N * nx + j, N * nx + j) = QN(j, j); // Insert QN diagonal elements
  }

  // Populate the gradient vector q
  Eigen::Matrix<double, 12, 1> Qx_ref = Q * (-xr); // Compute Q * (-xr)

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < nx; ++j) {
      q(i * nx + j) = Qx_ref(j); // Populate gradient vector
    }
  }
  // Populate terminal gradient segment
  Eigen::Matrix<double, 12, 1> QN_xr = QN * (-xr);
  for (int j = 0; j < nx; ++j) {
    q(N * nx + j) = QN_xr(j); // Populate terminal gradient vector
  }

//  qp_coeff.obj_mat_ = P.sparseView();
  qp_coeff.obj_mat_ = P;
  qp_coeff.obj_vec_ = q;

  Eigen::MatrixXd Aeq = Eigen::MatrixXd::Zero(dim_var, dim_var);
  Eigen::VectorXd leq = Eigen::VectorXd::Zero(dim_eq);
  Eigen::VectorXd ueq = Eigen::VectorXd::Zero(dim_eq);

  for (int i = 0; i < N; ++i) {
    Aeq.block(i * nx, i * nx, nx, nx) = -Eigen::MatrixXd::Identity(nx, nx);
    Aeq.block(i * nx, (i + 1) * nx, nx, nx) = Ad;
    Aeq.block(i * nx, (N + 1) * nx + i * nu, nx, nu) = Bd;
  }
  leq.segment(0, nx) = -x0;
  ueq.segment(0, nx) = -x0;

  qp_coeff.eq_mat_ = Aeq.sparseView();
  qp_coeff.eq_vec_ = leq;

  // Constraints
  Eigen::VectorXd x_min = Eigen::VectorXd::Constant(dim_var, -std::numeric_limits<double>::infinity());
  Eigen::VectorXd x_max = Eigen::VectorXd::Constant(dim_var, std::numeric_limits<double>::infinity());

  for (int i = 0; i < N + 1; i++) {
    x_min.segment(i * nx, nx) = xmin;
    x_max.segment(i * nx, nx) = xmax;
  }
  for (int i = 0; i < N; i++) {
    x_min.segment((N + 1) * nx + i * nu, nu) = umin;
    x_max.segment((N + 1) * nx + i * nu, nu) = umax;
  }

  qp_coeff.x_min_ = x_min;
  qp_coeff.x_max_ = x_max;

  // Solve the QP problem
  auto qp_solver = QpSolverCollection::allocateQpSolver(QpSolverCollection::QpSolverType::OSQP);
  int nsim = 10;
  Eigen::VectorXd solution, ctrl;
  for (int i = 0; i < nsim; ++i)
  {
    std::cout << "x0 at start of iteration: " << i+1 << ":" << x0.transpose() << std::endl;
    // Update the constraints for the new initial state
    leq.segment(0, nx) = -x0;
    ueq.segment(0, nx) = -x0;
    qp_coeff.eq_vec_ = leq;
    // Solve the QP problem
    solution = qp_solver->solve(qp_coeff);
    // Extract the control input
    ctrl = solution.segment((N+1)* nx, nu);
    std::cout << "Iteration " << i+1 << ": Solution = " << solution.transpose() << std::endl;
    std::cout << ": Control Input = " << ctrl.transpose() << std::endl;
    std::cout << "Objective value: " << 0.5 * solution.transpose() * P * solution + q.transpose() * solution << std::endl;
    x0 = Ad * x0 + Bd * ctrl;
    std::cout << "x0 at end of iteration: " << i+1 << ":" << x0.transpose() << std::endl;
//    auto x0Data = x0.data();
  }

  std::cout << "Closed-loop simulation completed successfully." << std::endl;

  return 0;
}















