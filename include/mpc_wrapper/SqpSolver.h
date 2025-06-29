#ifndef SQPSOLVER_H
#define SQPSOLVER_H

#include "qp_solver_collection/QpSolverCollection.h"
#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>

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
        scalar_t tol = 1e-3;
        scalar_t tau = 0.5; // Line search step decrease
        scalar_t eta = 0.25; // Line search sufficient decrease condition
        scalar_t rho = 0.5; // Merit function parameter
        QpSolverCollection::QpSolverType solver_type = QpSolverCollection::QpSolverType::OSQP; // QP solver type
    };

    struct Info {
        int iter = 0;
    };

    SQPSolver();

    void solve(const variable_t& x0, const eq_constraint_t& y0, const Eigen::VectorXd& x_min, const Eigen::VectorXd& x_max, const Eigen::VectorXd& ineq_min, const Eigen::VectorXd& ineq_max);

    scalar_t constraints_violation(const variable_t& x, const ineq_constraint_t& ineq_min, const ineq_constraint_t& ineq_max) const;

    scalar_t step_size_selection(const variable_t& x, const ineq_constraint_t& ineq_min, const ineq_constraint_t& ineq_max, const variable_t& grad_obj, const Eigen::VectorXd& p, const hessian_t& hess) const;

    variable_t primal_solution() const { return primal_solution_; }
    const Settings& settings() const { return settings_; }
    Settings& settings() { return settings_; }
    const Info& info() const { return info_; }

private:
    void hessian_regularisation_dense_impl(Eigen::Ref<hessian_t> lag_hessian) noexcept;

    void grad_obj_func(const variable_t& x, variable_t& grad) const;
    void hess_obj_func(const variable_t& x, hessian_t& hess) const;
    void jac_eq_constraints(const variable_t& x, Eigen::Matrix<scalar_t, Problem::NE, Problem::NX>& J) const;
    void jac_ineq_constraints(const variable_t& x, Eigen::Matrix<scalar_t, Problem::NI, Problem::NX>& J) const;

    Settings settings_;
    Info info_;
    variable_t primal_solution_;
    Problem problem_;
};

// Template definitions

template<typename Problem>
SQPSolver<Problem>::SQPSolver() {
    settings_ = Settings();
    info_ = Info();
}

template<typename Problem>
void SQPSolver<Problem>::solve(const variable_t& x0, const eq_constraint_t& y0, const Eigen::VectorXd& x_min, const Eigen::VectorXd& x_max, const Eigen::VectorXd& ineq_min, const Eigen::VectorXd& ineq_max) {
    variable_t x = x0;
    eq_constraint_t lambda_eq = y0;
    ineq_constraint_t lambda_ineq;
    lambda_ineq.setZero();

    auto qp_solver = QpSolverCollection::allocateQpSolver(settings_.solver_type);
    QpSolverCollection::QpCoeff qp_coeff;

    for (info_.iter = 0; info_.iter < settings_.max_iter; ++info_.iter) {
        // Formulate QP subproblem
        qp_coeff.setup(Problem::NX, Problem::NE, Problem::NI * 2); // NI * 2 to handle both lower and upper bounds

        // Hessian (approximate or exact)
        hessian_t hess;
        hess_obj_func(x, hess);
        hessian_regularisation_dense_impl(hess); // Regularize the Hessian
        qp_coeff.obj_mat_ = hess;

        // Gradient of the objective
        variable_t grad_obj;
        grad_obj_func(x, grad_obj);
        qp_coeff.obj_vec_ = grad_obj;

        // Equality constraints
        Eigen::Matrix<scalar_t, Problem::NE, Problem::NX> eq_mat;
        jac_eq_constraints(x, eq_mat);
        qp_coeff.eq_mat_ = eq_mat;

        eq_constraint_t eq_val;
        problem_.eq_constraints(x, eq_val);
        qp_coeff.eq_vec_ = -eq_val;

        // Inequality constraints
        Eigen::Matrix<scalar_t, Problem::NI, Problem::NX> ineq_mat;
        jac_ineq_constraints(x, ineq_mat);
        qp_coeff.ineq_mat_.block(0, 0, Problem::NI, Problem::NX) = ineq_mat;
        qp_coeff.ineq_mat_.block(Problem::NI, 0, Problem::NI, Problem::NX) = -ineq_mat;

        ineq_constraint_t ineq_val;
        problem_.ineq_constraints(x, ineq_val);
        qp_coeff.ineq_vec_.head(Problem::NI) = ineq_max - ineq_val;
        qp_coeff.ineq_vec_.tail(Problem::NI) = ineq_val - ineq_min;

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

template<typename Problem>
void SQPSolver<Problem>::hessian_regularisation_dense_impl(Eigen::Ref<hessian_t> lag_hessian) noexcept {
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

template<typename Problem>
typename SQPSolver<Problem>::scalar_t SQPSolver<Problem>::constraints_violation(const variable_t& x, const ineq_constraint_t& ineq_min, const ineq_constraint_t& ineq_max) const {
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

template<typename Problem>
typename SQPSolver<Problem>::scalar_t SQPSolver<Problem>::step_size_selection(const variable_t& x, const ineq_constraint_t& ineq_min, const ineq_constraint_t& ineq_max, const variable_t& grad_obj, const Eigen::VectorXd& p, const hessian_t& hess) const {
    scalar_t mu, phi_l1, Dp_phi_l1;
    const scalar_t tau = settings_.tau; // line search step decrease
    const scalar_t eta = settings_.eta; // sufficient decrease condition
    const scalar_t rho = settings_.rho; // merit function parameter

    scalar_t constr_l1 = constraints_violation(x, ineq_min, ineq_max);

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

template<typename Problem>
void SQPSolver<Problem>::grad_obj_func(const variable_t& x, variable_t& grad) const {
    using ad_scalar_t = typename Problem::ad_scalar_t;
    using ad_variable_t = typename Problem::ad_variable_t;

    ad_variable_t x_ad;
    for (int i = 0; i < Problem::NX; ++i) {
        x_ad(i) = x(i);
        x_ad(i).derivatives() = Eigen::VectorXd::Unit(Problem::NX, i);
    }

    ad_scalar_t f_ad = problem_.obj_func(x_ad);

    for (int i = 0; i < Problem::NX; ++i) {
        grad(i) = f_ad.derivatives()(i);
    }
}

template<typename Problem>
void SQPSolver<Problem>::hess_obj_func(const variable_t& x, hessian_t& hess) const {
    using ad_ad_scalar_t = Eigen::AutoDiffScalar<Eigen::Matrix<typename Problem::ad_scalar_t, Problem::NX, 1>>;
    using ad_ad_variable_t = Eigen::Matrix<ad_ad_scalar_t, Problem::NX, 1>;

    ad_ad_variable_t x_ad;
    for (int i = 0; i < Problem::NX; ++i) {
        x_ad(i) = x(i);
        x_ad(i).derivatives() = Eigen::VectorXd::Unit(Problem::NX, i);
        x_ad(i).value().derivatives() = Eigen::VectorXd::Unit(Problem::NX, i);
    }

    ad_ad_scalar_t f_ad = problem_.obj_func(x_ad);

    for (int i = 0; i < Problem::NX; ++i) {
        for (int j = 0; j < Problem::NX; ++j) {
            hess(i, j) = f_ad.derivatives()(i).derivatives()(j);
        }
    }
}

template<typename Problem>
void SQPSolver<Problem>::jac_eq_constraints(const variable_t& x, Eigen::Matrix<scalar_t, Problem::NE, Problem::NX>& J) const {
    using ad_scalar_t = typename Problem::ad_scalar_t;
    using ad_variable_t = typename Problem::ad_variable_t;
    using ad_eq_constraint_t = typename Problem::ad_eq_constraint_t;

    ad_variable_t x_ad;
    for (int i = 0; i < Problem::NX; ++i) {
        x_ad(i) = x(i);
        x_ad(i).derivatives() = Eigen::VectorXd::Unit(Problem::NX, i);
    }

    ad_eq_constraint_t g_ad;
    problem_.eq_constraints(x_ad, g_ad);

    for (int i = 0; i < Problem::NE; ++i) {
        for (int j = 0; j < Problem::NX; ++j) {
            J(i, j) = g_ad(i).derivatives()(j);
        }
    }
}

template<typename Problem>
void SQPSolver<Problem>::jac_ineq_constraints(const variable_t& x, Eigen::Matrix<scalar_t, Problem::NI, Problem::NX>& J) const {
    using ad_scalar_t = typename Problem::ad_scalar_t;
    using ad_variable_t = typename Problem::ad_variable_t;
    using ad_ineq_constraint_t = typename Problem::ad_ineq_constraint_t;

    ad_variable_t x_ad;
    for (int i = 0; i < Problem::NX; ++i) {
        x_ad(i) = x(i);
        x_ad(i).derivatives() = Eigen::VectorXd::Unit(Problem::NX, i);
    }

    ad_ineq_constraint_t g_ad;
    problem_.ineq_constraints(x_ad, g_ad);

    for (int i = 0; i < Problem::NI; ++i) {
        for (int j = 0; j < Problem::NX; ++j) {
            J(i, j) = g_ad(i).derivatives()(j);
        }
    }
}

#endif // SQPSOLVER_H
