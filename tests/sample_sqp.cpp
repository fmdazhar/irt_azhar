#include "qp_solver_collection/QpSolverCollection.h"
#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>

// Define the problem structure with automatic differentiation
class ConstrainedRosenbrock {
public:
  using scalar_t = double;
  using variable_t = Eigen::Matrix<scalar_t, 2, 1>;
  using eq_constraint_t = Eigen::Matrix<scalar_t, 1, 1>;
  using ineq_constraint_t = Eigen::Matrix<scalar_t, 0, 1>;

  using ad_scalar_t = Eigen::AutoDiffScalar<Eigen::VectorXd>;
  using ad_variable_t = Eigen::Matrix<ad_scalar_t, 2, 1>;
  using ad_eq_constraint_t = Eigen::Matrix<ad_scalar_t, 1, 1>;

  static constexpr int NX = 2;
  static constexpr int NE = 1;
  static constexpr int NI = 0;

  const scalar_t a = 1;
  const scalar_t b = 100;

  scalar_t obj_func(const variable_t &x) const {
    return (a - x(0)) * (a - x(0)) + b * (x(1) - x(0) * x(0)) * (x(1) - x(0) * x(0));
  }

  template<typename T>
  T obj_func(const Eigen::Matrix<T, NX, 1> &x) const {
    return (a - x(0)) * (a - x(0)) + b * (x(1) - x(0) * x(0)) * (x(1) - x(0) * x(0));
  }

  void grad_obj_func(const variable_t &x, variable_t &grad) const {
    ad_variable_t x_ad;
    for (int i = 0; i < NX; ++i) {
      x_ad(i) = x(i);
      x_ad(i).derivatives() = Eigen::VectorXd::Unit(NX, i);
    }

    ad_scalar_t f_ad = obj_func(x_ad);

    for (int i = 0; i < NX; ++i) {
      grad(i) = f_ad.derivatives()(i);
    }
  }

  void hess_obj_func(const variable_t &x, Eigen::Matrix<scalar_t, NX, NX> &hess) const {
    using ad_ad_scalar_t = Eigen::AutoDiffScalar<Eigen::Matrix<ad_scalar_t, NX, 1>>;
    using ad_ad_variable_t = Eigen::Matrix<ad_ad_scalar_t, NX, 1>;

    ad_ad_variable_t x_ad;
    for (int i = 0; i < NX; ++i) {
      x_ad(i) = x(i);
      x_ad(i).derivatives() = Eigen::VectorXd::Unit(NX, i);
      x_ad(i).value().derivatives() = Eigen::VectorXd::Unit(NX, i);
    }

    ad_ad_scalar_t f_ad = obj_func(x_ad);

    for (int i = 0; i < NX; ++i) {
      for (int j = 0; j < NX; ++j) {
        hess(i, j) = f_ad.derivatives()(i).derivatives()(j);
      }
    }
  }

  void eq_constraints(const variable_t &x, eq_constraint_t &constraints) const {
    constraints(0) =  x.squaredNorm() - 1;
  }

  template<typename T>
  void eq_constraints(const Eigen::Matrix<T, NX, 1> &x, Eigen::Matrix<T, NE, 1> &constraints) const {
    constraints(0) = x.squaredNorm() - 1;
  }

  void jac_eq_constraints(const variable_t &x, Eigen::Matrix<scalar_t, NE, NX> &J) const {
    ad_variable_t x_ad;
    for (int i = 0; i < NX; ++i) {
      x_ad(i) = x(i);
      x_ad(i).derivatives() = Eigen::VectorXd::Unit(NX, i);
    }

    ad_eq_constraint_t g_ad;
    eq_constraints(x_ad, g_ad);

    for (int i = 0; i < NE; ++i) {
      for (int j = 0; j < NX; ++j) {
        J(i, j) = g_ad(i).derivatives()(j);
      }
    }
  }
};

// SQP solver class
class SQPSolver {
public:
  using scalar_t = ConstrainedRosenbrock::scalar_t;
  using variable_t = ConstrainedRosenbrock::variable_t;
  using eq_constraint_t = ConstrainedRosenbrock::eq_constraint_t;
  using ineq_constraint_t = ConstrainedRosenbrock::ineq_constraint_t;

  struct Settings {
    int max_iter = 50;
    int line_search_max_iter = 5;
    scalar_t tol = 1e-6;
    scalar_t tau = 0.5; // Line search step decrease
    scalar_t eta = 0.01; // Line search sufficient decrease condition
    scalar_t rho = 0.1; // Merit function parameter
  };

  struct Info {
    int iter = 0;
  };

  SQPSolver() {
    settings_ = Settings();
    info_ = Info();
  }

  void solve(const variable_t& x0, const eq_constraint_t& y0) {
    variable_t x = x0;
    eq_constraint_t lambda_eq = y0;

    auto qp_solver = QpSolverCollection::allocateQpSolver(QpSolverCollection::QpSolverType::OSQP);

    for (info_.iter = 0; info_.iter < settings_.max_iter; ++info_.iter) {
      // Formulate QP subproblem
      QpSolverCollection::QpCoeff qp_coeff;
      qp_coeff.setup(ConstrainedRosenbrock::NX, ConstrainedRosenbrock::NE, ConstrainedRosenbrock::NI);

      // Hessian (approximate or exact)
      Eigen::Matrix<scalar_t, ConstrainedRosenbrock::NX, ConstrainedRosenbrock::NX> hess;
      problem_.hess_obj_func(x, hess);
      qp_coeff.obj_mat_ = hess;

      // Gradient of the objective
      variable_t grad_obj;
      problem_.grad_obj_func(x, grad_obj);
      qp_coeff.obj_vec_ = grad_obj;

      // Equality constraints
      Eigen::Matrix<scalar_t, ConstrainedRosenbrock::NE, ConstrainedRosenbrock::NX> eq_mat;
      problem_.jac_eq_constraints(x, eq_mat);
      qp_coeff.eq_mat_ = eq_mat;

      eq_constraint_t eq_val;
      problem_.eq_constraints(x, eq_val);
      qp_coeff.eq_vec_ = -eq_val;

      // Inequality constraints (none for this problem)
      // qp_coeff.ineq_mat_ = Eigen::MatrixXd::Zero(ConstrainedRosenbrock::NI, ConstrainedRosenbrock::NX);
      // qp_coeff.ineq_vec_ = Eigen::VectorXd::Zero(ConstrainedRosenbrock::NI);

      // Solve the QP subproblem
      Eigen::VectorXd delta_x = qp_solver->solve(qp_coeff);

      // Line search using merit function
      scalar_t alpha = step_size_selection(x, grad_obj, delta_x, hess);

      // Update variables
      x += alpha * delta_x;

      // Check for convergence
      if (delta_x.norm() < settings_.tol) {
        break;
      }

      std::cout << "Iteration " << info_.iter << ": x = " << x.transpose() << std::endl;
    }

    primal_solution_ = x;
  }

  scalar_t constraints_violation(const variable_t& x) const {
    eq_constraint_t eq_val;
    problem_.eq_constraints(x, eq_val);
    return eq_val.template lpNorm<Eigen::Infinity>();
  }

  scalar_t step_size_selection(const variable_t& x, const variable_t& grad_obj, const Eigen::VectorXd& p, const Eigen::Matrix<scalar_t, ConstrainedRosenbrock::NX, ConstrainedRosenbrock::NX>& hess) const {
    scalar_t mu, phi_l1, Dp_phi_l1;
    const scalar_t tau = settings_.tau; // line search step decrease
    const scalar_t eta = settings_.eta; // sufficient decrease condition
    const scalar_t rho = settings_.rho; // merit function parameter

    scalar_t constr_l1 = constraints_violation(x);

    // Get mu from merit function model using Hessian of Lagrangian
    const scalar_t quad_term = p.dot(hess * p);
    const scalar_t qt = quad_term >= 0 ? scalar_t(0.5) * quad_term : 0;
    mu = std::abs(grad_obj.dot(p)) / ((1 - rho) * constr_l1 + qt);
    // mu = 1.0; // Initial scaling factor for merit function, can be tuned


    scalar_t cost_1 = problem_.obj_func(x);

    phi_l1 = cost_1 + mu * constr_l1;
    Dp_phi_l1 = grad_obj.dot(p) - mu * constr_l1;

    scalar_t alpha = scalar_t(1.0);
    scalar_t cost_step;
    variable_t x_step;
    for (int i = 1; i < settings_.line_search_max_iter; i++) {
      x_step.noalias() = x + alpha * p;
      cost_step = problem_.obj_func(x_step);
      scalar_t phi_l1_step = cost_step + mu * constraints_violation(x_step);

      if (phi_l1_step <= (phi_l1 + alpha * eta * Dp_phi_l1))
        return alpha;
      else
        alpha *= tau;
    }

    return alpha;
  }

  variable_t primal_solution() const { return primal_solution_; }
  const Settings& settings() const { return settings_; }
  Settings& settings() { return settings_; }
  const Info& info() const { return info_; }

private:
  Settings settings_;
  Info info_;
  variable_t primal_solution_;
  ConstrainedRosenbrock problem_;
};

// Test case for Constrained Rosenbrock
int main() {
  ConstrainedRosenbrock problem;
  SQPSolver solver;

  solver.settings().max_iter = 50;
  solver.settings().line_search_max_iter = 50;

  SQPSolver::variable_t x0;
  SQPSolver::eq_constraint_t y0;
  y0.setZero();
  x0 << 0.00, 0.00;

  solver.solve(x0, y0);
  auto x = solver.primal_solution();

  std::cout << "iter " << solver.info().iter << std::endl;
  std::cout << "Solution " << x.transpose() << std::endl;
  double obj_value_x0 = problem.obj_func(x0);
  SQPSolver::variable_t x_opt;
  x_opt << 0.7864, 0.6177;
  // Evaluate and print the objective function at the solution
  double obj_value = problem.obj_func(x);
  double obj_value_x_opt = problem.obj_func(x_opt);
  std::cout << "Objective function value at solution: " << obj_value << std::endl;
  std::cout << "Objective function value at initial: " << obj_value_x0 << std::endl;
  std::cout << "Objective function value at optimal(0.7864, 0.6177): " << obj_value_x_opt << std::endl;

  return 0;
}
