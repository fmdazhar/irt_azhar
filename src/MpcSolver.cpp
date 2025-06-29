#include "mpc_wrapper/MpcSolver.h"

#include <iostream>
// Define INFTY
constexpr double INFTY = 1e+30;

MpcSolver::MpcSolver(int nx, int nu, int N, const Eigen::MatrixXd& Ad, const Eigen::MatrixXd& Bd, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& QN, const Eigen::MatrixXd& R)
: nx_(nx), nu_(nu), N_(N), Ad_(Ad), Bd_(Bd), Q_(Q), QN_(QN), R_(R) {
  qp_solver_ = QpSolverCollection::allocateQpSolver(QpSolverCollection::QpSolverType::OSQP);
  qp_coeff_.setup((N + 1) * nx + N * nu, (N + 1) * nx, 0);
}

void MpcSolver::setConstraints(const Eigen::VectorXd& xmin, const Eigen::VectorXd& xmax, const Eigen::VectorXd& umin, const Eigen::VectorXd& umax) {
  xmin_ = xmin;
  xmax_ = xmax;
  umin_ = umin;
  umax_ = umax;
}

void MpcSolver::setInitialState(const Eigen::VectorXd& x0) {
  x0_ = x0;
}

void MpcSolver::setReferenceState(const Eigen::VectorXd& xr) {
  xr_ = xr;
}

void MpcSolver::setupQpProblem() {
  int dim_var = (N_ + 1) * nx_ + N_ * nu_;
  int dim_eq = (N_ + 1) * nx_;


  Eigen::MatrixXd P = Eigen::MatrixXd::Zero(dim_var, dim_var);
  Eigen::VectorXd q = Eigen::VectorXd::Zero(dim_var);

  for (int i = 0; i < N_; ++i) {
    P.block(i * nx_, i * nx_, nx_, nx_) = Q_;
    P.block((N_ + 1) * nx_ + i * nu_, (N_ + 1) * nx_ + i * nu_, nu_, nu_) = R_;
    q.segment(i * nx_, nx_) = -Q_ * xr_;
  }
  P.block(N_ * nx_, N_ * nx_, nx_, nx_) = QN_;
  q.segment(N_ * nx_, nx_) = -QN_ * xr_;

  qp_coeff_.obj_mat_ = P.sparseView();
  qp_coeff_.obj_vec_ = q;

  Eigen::MatrixXd Aeq = Eigen::MatrixXd::Zero(dim_eq, dim_var);
  Eigen::VectorXd leq = Eigen::VectorXd::Zero(dim_eq);
//  Eigen::VectorXd ueq = Eigen::VectorXd::Zero(dim_eq);

  for (int i = 0; i < N_; i++) {
    Aeq.block(i * nx_, i * nx_, nx_, nx_) = -Eigen::MatrixXd::Identity(nx_, nx_);
    Aeq.block((i+1) * nx_, (i ) * nx_, nx_, nx_) = Ad_;
    Aeq.block((i+1) * nx_, (N_ + 1) * nx_ + i * nu_, nx_, nu_) = Bd_;
  }
  leq.segment(0, nx_) = -x0_;
//  ueq.segment(0, nx_) = -x0_;

  qp_coeff_.eq_mat_ = Aeq.sparseView();
  qp_coeff_.eq_vec_ = leq;

  Eigen::VectorXd x_min = Eigen::VectorXd::Constant(dim_var, -INFTY);
  Eigen::VectorXd x_max = Eigen::VectorXd::Constant(dim_var, INFTY);

  for (int i = 0; i < N_ + 1; ++i) {
    x_min.segment(i * nx_, nx_) = xmin_;
    x_max.segment(i * nx_, nx_) = xmax_;
  }
  for (int i = 0; i < N_; ++i) {
    x_min.segment((N_ + 1) * nx_ + i * nu_, nu_) = umin_;
    x_max.segment((N_ + 1) * nx_ + i * nu_, nu_) = umax_;
  }

  qp_coeff_.x_min_ = x_min;
  qp_coeff_.x_max_ = x_max;
}

Eigen::VectorXd MpcSolver::solveMPC(int nsim) {
  setupQpProblem();
  Eigen::VectorXd solution;
  for (int i = 0; i < nsim; ++i) {
    solution = qp_solver_->solve(qp_coeff_);
    Eigen::VectorXd ctrl = solution.segment((N_ + 1) * nx_, nu_);
    std::cout << "Iteration " << i + 1 << ": Solution = " << solution.transpose() << std::endl;
    std::cout << "Control Input = " << ctrl.transpose() << std::endl;
    std::cout << "Objective value: " << 0.5 * solution.transpose() * qp_coeff_.obj_mat_ * solution + qp_coeff_.obj_vec_.transpose() * solution << std::endl;
//    std::cout << "x0 at start of iteration: " << i+1 << ":" << x0_.transpose() << std::endl;
    x0_ = Ad_ * x0_ + Bd_ * ctrl;
//    std::cout << "x0 at end of iteration: " << i+1 << ":" << x0_.transpose() << std::endl;
    qp_coeff_.eq_vec_.segment(0, nx_) = -x0_;
  }
  return solution;
}