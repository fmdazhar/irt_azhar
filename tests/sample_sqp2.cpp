#include "qp_solver_collection/QpSolverCollection.h"
#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>

// Define the HS071 problem structure
class HS071_2 {
public:
    using scalar_t = double;
    using variable_t = Eigen::Matrix<scalar_t, 4, 1>;
    using eq_constraint_t = Eigen::Matrix<scalar_t, 1, 1>;
    using ineq_constraint_t = Eigen::Matrix<scalar_t, 1, 1>;

    using ad_scalar_t = Eigen::AutoDiffScalar<Eigen::VectorXd>;
    using ad_variable_t = Eigen::Matrix<ad_scalar_t, 4, 1>;
    using ad_eq_constraint_t = Eigen::Matrix<ad_scalar_t, 1, 1>;
    using ad_ineq_constraint_t = Eigen::Matrix<ad_scalar_t, 1, 1>;

    static constexpr int NX = 4;
    static constexpr int NE = 1;
    static constexpr int NI = 1;

    Eigen::Matrix<scalar_t, 4, 1> SOLUTION = {1.00000000, 4.74299963, 3.82114998, 1.37940829};

    scalar_t obj_func(const variable_t &x) const {
        return x(0) * x(3) * (x(0) + x(1) + x(2)) + x(2);
    }

    template<typename T>
    T obj_func(const Eigen::Matrix<T, NX, 1> &x) const {
        return x(0) * x(3) * (x(0) + x(1) + x(2)) + x(2);
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
        constraints(0) = x.squaredNorm() - 40;
    }

    template<typename T>
    void eq_constraints(const Eigen::Matrix<T, NX, 1> &x, Eigen::Matrix<T, NE, 1> &constraints) const {
        constraints(0) = x.squaredNorm() - 40;
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

    void ineq_constraints(const variable_t &x, ineq_constraint_t &constraints) const {
        constraints(0) = x(0) * x(1) * x(2) * x(3);
    }

    template<typename T>
    void ineq_constraints(const Eigen::Matrix<T, NX, 1> &x, Eigen::Matrix<T, NI, 1> &constraints) const {
        constraints(0) = x(0) * x(1) * x(2) * x(3);
    }

    void jac_ineq_constraints(const variable_t &x, Eigen::Matrix<scalar_t, NI, NX> &J) const {
        ad_variable_t x_ad;
        for (int i = 0; i < NX; ++i) {
            x_ad(i) = x(i);
            x_ad(i).derivatives() = Eigen::VectorXd::Unit(NX, i);
        }

        ad_ineq_constraint_t g_ad;
        ineq_constraints(x_ad, g_ad);

        for (int i = 0; i < NI; ++i) {
            for (int j = 0; j < NX; ++j) {
                J(i, j) = g_ad(i).derivatives()(j);
            }
        }
    }
};

// SQP solver class
template<typename Problem>
class SQPSolver {
public:
    using scalar_t = typename Problem::scalar_t;
    using variable_t = typename Problem::variable_t;
    using eq_constraint_t = typename Problem::eq_constraint_t;
    using ineq_constraint_t = typename Problem::ineq_constraint_t;
    using hessian_t = Eigen::Matrix<scalar_t, Problem::NX, Problem::NX>;

    struct Settings {
        int max_iter = 50;
        int line_search_max_iter = 5;
        scalar_t tol = 1e-6;
        scalar_t tau = 0.5; // Line search step decrease
        scalar_t eta = 0.25; // Line search sufficient decrease condition
        scalar_t rho = 0.5; // Merit function parameter
    };

    struct Info {
        int iter = 0;
    };

    SQPSolver() {
        settings_ = Settings();
        info_ = Info();
    }

    void solve(const variable_t& x0, const eq_constraint_t& y0, const Eigen::VectorXd& x_min, const Eigen::VectorXd& x_max, const Eigen::VectorXd& ineq_min, const Eigen::VectorXd& ineq_max) {
        variable_t x = x0;
        eq_constraint_t lambda_eq = y0;
        ineq_constraint_t lambda_ineq;
        lambda_ineq.setZero();

        auto qp_solver = QpSolverCollection::allocateQpSolver(QpSolverCollection::QpSolverType::OSQP);
        QpSolverCollection::QpCoeff qp_coeff;

        for (info_.iter = 0; info_.iter < settings_.max_iter; ++info_.iter) {
            // Formulate QP subproblem
            qp_coeff.setup(Problem::NX, Problem::NE, Problem::NI * 2); // NI * 2 to handle both lower and upper bounds

            // Hessian (approximate or exact)
            hessian_t hess;
            problem_.hess_obj_func(x, hess);
            hessian_regularisation_dense_impl(hess); // Regularize the Hessian
            qp_coeff.obj_mat_ = hess;

            // Gradient of the objective
            variable_t grad_obj;
            problem_.grad_obj_func(x, grad_obj);
            qp_coeff.obj_vec_ = grad_obj;

            // Equality constraints
            Eigen::Matrix<scalar_t, Problem::NE, Problem::NX> eq_mat;
            problem_.jac_eq_constraints(x, eq_mat);
            qp_coeff.eq_mat_ = eq_mat;

            eq_constraint_t eq_val;
            problem_.eq_constraints(x, eq_val);
            qp_coeff.eq_vec_ = -eq_val;

            // Inequality constraints
            Eigen::Matrix<scalar_t, Problem::NI, Problem::NX> ineq_mat;
            problem_.jac_ineq_constraints(x, ineq_mat);
            qp_coeff.ineq_mat_.block(0, 0, Problem::NI, Problem::NX) = ineq_mat;
            qp_coeff.ineq_mat_.block(Problem::NI, 0, Problem::NI, Problem::NX) = -ineq_mat;

            ineq_constraint_t ineq_val;
            problem_.ineq_constraints(x, ineq_val);
            qp_coeff.ineq_vec_.head(Problem::NI) = ineq_max - ineq_val;
            qp_coeff.ineq_vec_.tail(Problem::NI) = ineq_val - ineq_min ;

            // Bounds on variables
            qp_coeff.x_min_ = x_min - x;
            qp_coeff.x_max_ = x_max - x;

            Eigen::VectorXd delta_x = qp_solver->solve(qp_coeff);

            // Line search using merit function
            scalar_t alpha = step_size_selection(x, ineq_min, ineq_max, grad_obj, delta_x, hess);

            // Update variables
            x += alpha * delta_x;

          // Check for convergence
          scalar_t constraint_violation = constraints_violation(x, ineq_min, ineq_max);
          if (delta_x.norm() < settings_.tol && constraint_violation < settings_.tol) {
            std::cout << "Convergence criteria met." << std::endl;
            break;
          }

            std::cout << "Iteration " << info_.iter << ": x = " << x.transpose() << ", Constraint Violation = " << constraint_violation << std::endl;
        }

        primal_solution_ = x;
    }

    void hessian_regularisation_dense_impl(Eigen::Ref<hessian_t> lag_hessian) noexcept {
        Eigen::EigenSolver<hessian_t> es;
        es.compute(lag_hessian);

        scalar_t minEigValue = es.eigenvalues().real().minCoeff();
        if (minEigValue <= 0) {
            Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic> Deig = es.eigenvalues().real().asDiagonal();
            for (int i = 0; i < Deig.rows(); i++) {
                if (Deig(i, i) <= 0) { Deig(i, i) = -1 * Deig(i, i) + 0.1; } // Mirror regularization
            }
            lag_hessian.noalias() = es.eigenvectors().real() * Deig * es.eigenvectors().real().transpose(); // V*D*V^-1 with V^-1 ~= V'
        }
    }

  scalar_t constraints_violation(const variable_t& x, const ineq_constraint_t& ineq_min, const ineq_constraint_t& ineq_max) const {
      eq_constraint_t eq_val;
      ineq_constraint_t ineq_val;
      problem_.eq_constraints(x, eq_val);
      problem_.ineq_constraints(x, ineq_val);

      // Equality constraint violation
      scalar_t eq_violation = eq_val.cwiseAbs().sum();

      // Inequality constraint violation with bounds
      scalar_t ineq_violation = 0.0;
      for (int i = 0; i < ineq_val.size(); ++i) {
        if (ineq_val(i) < ineq_min(i)) {
          ineq_violation += (ineq_min(i) - ineq_val(i));
        }
        if (ineq_val(i) > ineq_max(i)) {
          ineq_violation += (ineq_val(i) - ineq_max(i));
        }
      }

      return eq_violation + ineq_violation;
    }

    scalar_t step_size_selection(const variable_t& x, const ineq_constraint_t& ineq_min, const ineq_constraint_t& ineq_max, const variable_t& grad_obj, const Eigen::VectorXd& p, const hessian_t& hess) const {
        scalar_t mu, phi_l1, Dp_phi_l1;
        const scalar_t tau = settings_.tau; // line search step decrease
        const scalar_t eta = settings_.eta; // sufficient decrease condition
        const scalar_t rho = settings_.rho; // merit function parameter

        scalar_t constr_l1 = constraints_violation(x, ineq_min, ineq_max);

        // // Get mu from merit function model using Hessian of Lagrangian
        // const scalar_t quad_term = p.dot(hess * p);
        // const scalar_t qt = quad_term >= 0 ? scalar_t(0.5) * quad_term : 0;
        // mu = std::abs(grad_obj.dot(p)) / ((1 - rho) * constr_l1 + qt);
        mu = 1.0; // Initial scaling factor for merit function, can be tuned

        scalar_t cost_1 = problem_.obj_func(x);

        phi_l1 = cost_1 + mu * constr_l1;
        Dp_phi_l1 = grad_obj.dot(p) - mu * constr_l1;

        scalar_t alpha = scalar_t(1.0);
        scalar_t cost_step;
        variable_t x_step;
        for (int i = 1; i < settings_.line_search_max_iter; i++) {
            x_step.noalias() = x + alpha * p;
            cost_step = problem_.obj_func(x_step);
            scalar_t phi_l1_step = cost_step + mu * constraints_violation(x_step, ineq_min, ineq_max);

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
    Problem problem_;
};

// Test case for HS071_2 problem
int main() {
    HS071_2 problem;
    SQPSolver<HS071_2> solver;

    solver.settings().max_iter = 500;
    solver.settings().line_search_max_iter = 5;

    HS071_2::variable_t x0;
    HS071_2::eq_constraint_t y0;
    y0.setZero();
    x0 << 1.0, 5.0, 5.0, 1.0;

    // Input bounds for variables and inequality constraints
    Eigen::VectorXd x_min(HS071_2::NX);
    Eigen::VectorXd x_max(HS071_2::NX);
    Eigen::VectorXd ineq_min(HS071_2::NI);
    Eigen::VectorXd ineq_max(HS071_2::NI);

    x_min << 1.0, 1.0, 1.0, 1.0;
    x_max << 5.0, 5.0, 5.0, 5.0;
    ineq_min << 25.0;
    ineq_max << std::numeric_limits<double>::infinity();

    solver.solve(x0, y0, x_min, x_max, ineq_min, ineq_max);
    auto x = solver.primal_solution();

    std::cout << "iter " << solver.info().iter << std::endl;
    std::cout << "Solution " << x.transpose() << std::endl;
    double obj_value_x0 = problem.obj_func(x0);
    double obj_value = problem.obj_func(x);
    double obj_value_opt = problem.obj_func(problem.SOLUTION);
    std::cout << "Objective function value at solution: " << obj_value << std::endl;
    std::cout << "Objective function value at initial: " << obj_value_x0 << std::endl;
    std::cout << "Objective function value at optimal solution: " << obj_value_opt << std::endl;
    // Calculate and print constraint violation for the optimal solution
    double constraint_violation_opt = solver.constraints_violation(problem.SOLUTION, ineq_min, ineq_max);
    std::cout << "Constraint violation at optimal solution: " << constraint_violation_opt << std::endl;
    return 0;
}
