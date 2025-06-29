//
// Created by bluewolf on 14.05.24.
//
// sample.cpp

#include "qp_solver_collection/QpSolverCollection.h"

int main()
{
  int dim_var = 6;
  int dim_eq = 3;
  int dim_ineq = 2;
  QpSolverCollection::QpCoeff qp_coeff;
  qp_coeff.setup(dim_var, dim_eq, dim_ineq);
  qp_coeff.obj_mat_.setIdentity();
  qp_coeff.obj_vec_ << 1., 2., 3., 4., 5., 6.;
  qp_coeff.eq_mat_ << 1., -1., 1., 0., 3., 1., -1., 0., -3., -4., 5., 6., 2., 5., 3., 0., 1., 0.;
  qp_coeff.eq_vec_ << 1., 2., 3.;
  qp_coeff.ineq_mat_ << 0., 1., 0., 1., 2., -1., -1., 0., 2., 1., 1., 0.;
  qp_coeff.ineq_vec_ << -1., 2.5;
  qp_coeff.x_min_ << -1000., -10000., 0., -1000., -1000., -1000.;
  qp_coeff.x_max_ << 10000., 100., 1.5, 100., 100., 1000.;

  auto qp_solver = QpSolverCollection::allocateQpSolver(QpSolverCollection::QpSolverType::IPOPT);

  Eigen::VectorXd solution = qp_solver->solve(qp_coeff);
  std::cout << "solution: " << solution.transpose() << std::endl;

  return 0;
}