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
  // System dimensions
  const int nx = 12;
  const int nu = 4;
  const int N = 20;  // Prediction horizon
  const double INFTY = 1e+30;

  // System matrices
  Eigen::MatrixXd Ad(nx, nx);
  Ad << 1., 0., 0., 0., 0., 0., 0.1, 0., 0., 0., 0., 0.,
      0., 1., 0., 0., 0., 0., 0., 0.1, 0., 0., 0., 0.,
      0., 0., 1., 0., 0., 0., 0., 0., 0.1, 0., 0., 0.,
      0.0488, 0., 0., 1., 0., 0., 0.0016, 0., 0., 0.0992, 0., 0.,
      0., -0.0488, 0., 0., 1., 0., 0., -0.0016, 0., 0., 0.0992, 0.,
      0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.0992,
      0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
      0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
      0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
      0.9734, 0., 0., 0., 0., 0., 0.0488, 0., 0., 0.9846, 0., 0.,
      0., -0.9734, 0., 0., 0., 0., 0., -0.0488, 0., 0., 0.9846, 0.,
      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.9846;

  Eigen::MatrixXd Bd(nx, nu);
  Bd << 0., -0.0726, 0., 0.0726,
      -0.0726, 0., 0.0726, 0.,
      -0.0152, 0.0152, -0.0152, 0.0152,
      -0., -0.0006, -0., 0.0006,
      0.0006, 0., -0.0006, 0.0000,
      0.0106, 0.0106, 0.0106, 0.0106,
      0, -1.4512, 0., 1.4512,
      -1.4512, 0., 1.4512, 0.,
      -0.3049, 0.3049, -0.3049, 0.3049,
      -0., -0.0236, 0., 0.0236,
      0.0236, 0., -0.0236, 0.,
      0.2107, 0.2107, 0.2107, 0.2107;

  // Objective matrices
  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(nx, nx);
  Q.diagonal() << 0., 0., 10., 10., 10., 10., 0., 0., 0., 5., 5., 5.;
  Eigen::MatrixXd QN = Q;
  Eigen::MatrixXd R = 0.1 * Eigen::MatrixXd::Identity(nu, nu);

  // Constraints
  double u0 = 10.5916;
  Eigen::VectorXd umin(nu), umax(nu), xmin(nx), xmax(nx);
  umin << 9.6, 9.6, 9.6, 9.6;
  umax << 13.0, 13.0, 13.0, 13.0;
  umin.array() -= u0;
  umax.array() -= u0;
  xmin << -M_PI / 6, -M_PI / 6, -INFTY, -INFTY,
      -INFTY, -1.0, -INFTY, -INFTY,
      -INFTY, -INFTY, -INFTY, -INFTY;
  xmax << M_PI / 6, M_PI / 6, INFTY, INFTY,
      INFTY, INFTY, INFTY, INFTY,
      INFTY, INFTY, INFTY, INFTY;

  // Initial and reference states
//  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nx);
  Eigen::VectorXd x0(nx);
  x0 << 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.;

  Eigen::VectorXd xr(nx);
  xr << 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.;

  // Create MPC solver
  MpcSolver mpc(nx, nu, N, Ad, Bd, Q, QN, R);

  // Set constraints
  mpc.setConstraints(xmin, xmax, umin, umax);

  // Set initial state
  mpc.setInitialState(x0);

  // Set reference state
  mpc.setReferenceState(xr);

  // Solve MPC
  int nsim = 10;
  Eigen::VectorXd solution = mpc.solveMPC(nsim);

  std::cout << "Final solution: " << solution.transpose() << std::endl;

  return 0;
}
