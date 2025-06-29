#ifndef MPC_SOLVER_H
#define MPC_SOLVER_H

#include <qp_solver_collection/QpSolverCollection.h>
#include <Eigen/Dense>
#include <memory>

#define MPCSOLVER_H


class MpcSolver {
public:
  MpcSolver(int nx, int nu, int N, const Eigen::MatrixXd& Ad, const Eigen::MatrixXd& Bd, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& QN, const Eigen::MatrixXd& R);
  void setConstraints(const Eigen::VectorXd& xmin, const Eigen::VectorXd& xmax, const Eigen::VectorXd& umin, const Eigen::VectorXd& umax);
  void setInitialState(const Eigen::VectorXd& x0);
  void setReferenceState(const Eigen::VectorXd& xr);
  Eigen::VectorXd solveMPC(int nsim);

private:
  int nx_, nu_, N_;
  Eigen::MatrixXd Ad_, Bd_, Q_, QN_, R_;
  Eigen::VectorXd xmin_, xmax_, umin_, umax_, x0_, xr_;
  QpSolverCollection::QpCoeff qp_coeff_;
  std::shared_ptr<QpSolverCollection::QpSolver> qp_solver_;

  void setupQpProblem();
};

#endif // MPCSOLVER_H