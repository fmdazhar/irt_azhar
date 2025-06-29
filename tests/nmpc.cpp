#ifndef ROBOT_OCP_H
#define ROBOT_OCP_H

#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>

class RobotOCP {
public:
    using scalar_t = double;
    using state_t = Eigen::Matrix<scalar_t, 5, 1>; // Combined state and control
    using control_t = Eigen::Matrix<scalar_t, 2, 1>;
    using parameter_t = Eigen::Matrix<scalar_t, 1, 1>;
    using static_parameter_t = Eigen::Matrix<scalar_t, 1, 1>;

    using ad_scalar_t = Eigen::AutoDiffScalar<Eigen::VectorXd>;
    using ad_state_t = Eigen::Matrix<ad_scalar_t, 5, 1>; // Combined state and control
    using ad_control_t = Eigen::Matrix<ad_scalar_t, 2, 1>;
    using ad_parameter_t = Eigen::Matrix<ad_scalar_t, 1, 1>;

    Eigen::DiagonalMatrix<scalar_t, 3> Q{1, 1, 1};
    Eigen::DiagonalMatrix<scalar_t, 2> R{1, 1};
    Eigen::DiagonalMatrix<scalar_t, 3> QN{1, 1, 1};

    template<typename T>
    void dynamics(const Eigen::Matrix<T, 3, 1>& x, const Eigen::Matrix<T, 2, 1>& u, const Eigen::Matrix<T, 1, 1>& p, Eigen::Matrix<T, 3, 1>& xdot) const {
        xdot(0) = u(0) * cos(x(2)) * cos(u(1));
        xdot(1) = u(0) * sin(x(2)) * cos(u(1));
        xdot(2) = u(0) * sin(u(1)) / p(0);
    }

  template<typename T>
  T obj_func(const Eigen::Matrix<T, 5, 1>& xu) const {
      Eigen::Matrix<T, 3, 3> Qm = Q.toDenseMatrix().template cast<T>();
      Eigen::Matrix<T, 2, 2> Rm = R.toDenseMatrix().template cast<T>();
      Eigen::Matrix<T, 3, 3> QNm = QN.toDenseMatrix().template cast<T>();

      Eigen::Matrix<T, 3, 1> x = xu.template head<3>();
      Eigen::Matrix<T, 2, 1> u = xu.template tail<2>();

      T cost = x.dot(Qm * x) + u.dot(Rm * u) + x.dot(QNm * x);
      return cost;
    }

  scalar_t obj_func(const state_t& xu) const {
      return obj_func<scalar_t>(xu);
    }

    template<typename T>
    void equality_constraints_impl(const Eigen::Matrix<T, 5, 1>& xu_prev, const Eigen::Matrix<T, 5, 1>& xu, const Eigen::Matrix<T, 1, 1>& p, Eigen::Matrix<T, 3, 1>& eq) const {
        Eigen::Matrix<T, 3, 1> xdot;
        Eigen::Matrix<T, 3, 1> x_prev = xu_prev.template head<3>();
        Eigen::Matrix<T, 2, 1> u_prev = xu_prev.template tail<2>();
        Eigen::Matrix<T, 3, 1> x = xu.template head<3>();
        Eigen::Matrix<T, 2, 1> u = xu.template tail<2>();
        dynamics(x_prev, u_prev, p, xdot);
        eq =  (x - x_prev) - xdot ;
    }

    template<typename T>
    void inequality_constraints(const Eigen::Matrix<T, 5, 1>& xu, Eigen::Matrix<T, 1, 1>& constraints) const {
        constraints(0) = xu(3) * xu(3) * cos(xu(4));
    }

    void set_Q_coeff(const scalar_t& coeff) {
        Q.diagonal() << coeff, coeff, coeff;
    }
};

#endif // ROBOT_OCP_H

#include "qp_solver_collection/QpSolverCollection.h"
#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>
#include <limits> // for INFTY

constexpr double INFTY = std::numeric_limits<double>::infinity();

template<typename Problem>
class SQPSolver {
public:
    using scalar_t = typename Problem::scalar_t;
    using state_t = typename Problem::state_t;
    using control_t = typename Problem::control_t;
    using parameter_t = typename Problem::parameter_t;
    using static_parameter_t = typename Problem::static_parameter_t;
    using ad_scalar_t = typename Problem::ad_scalar_t;
    using ad_state_t = typename Problem::ad_state_t;
    using hessian_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;

    struct Settings {
        int max_iter = 50;
        int line_search_max_iter = 5;
        scalar_t tol = 1e-6;
        scalar_t tau = 0.5;
        scalar_t eta = 0.25;
        scalar_t rho = 0.5;
    };

    struct Info {
        int iter = 0;
    };

    SQPSolver() {
        settings_ = Settings();
        info_ = Info();
    }

    void solve(const state_t& x0, const Eigen::VectorXd& x_min, const Eigen::VectorXd& x_max, const Eigen::VectorXd& lbg, const Eigen::VectorXd& ubg, int N, int nsim) {
        state_t xu_prev = x0;
        state_t xu = x0;

        auto qp_solver = QpSolverCollection::allocateQpSolver(QpSolverCollection::QpSolverType::OSQP);
        QpSolverCollection::QpCoeff qp_coeff;

        const int num_eq = (N + 1) * 3; // Number of equality constraints (dynamics)
        const int num_ineq = (N + 1) * 1 ; // Number of inequality constraints

        qp_coeff.setup((N + 1) * 5, num_eq, num_ineq* 2); //num_ineq * 2 to handle both lower and upper bounds

        for (int sim_iter = 0; sim_iter < nsim; ++sim_iter) {
            for (info_.iter = 0; info_.iter < settings_.max_iter; ++info_.iter) {
                // Hessian (using automatic differentiation)
                hessian_t hess = hessian_ad(xu, N);
                hessian_regularisation_dense_impl(hess);
                qp_coeff.obj_mat_ = hess.sparseView();

                // Gradient of the objective
                Eigen::VectorXd grad_obj = gradient_ad(xu, N);
                qp_coeff.obj_vec_ = grad_obj;

                // Equality constraints
                Eigen::MatrixXd eq_mat = Eigen::MatrixXd::Zero(num_eq, (N + 1) * 5);
                equality_jacobian_ad(xu_prev, xu, eq_mat, N);
                qp_coeff.eq_mat_ = eq_mat.sparseView();

                Eigen::VectorXd eq_val(num_eq);
                equality_constraints(xu_prev, xu, eq_val, N);
                qp_coeff.eq_vec_ = -eq_val;

                // Inequality constraints
                Eigen::MatrixXd ineq_mat = Eigen::MatrixXd::Zero(num_ineq, (N + 1) * 5);
                inequality_jacobian_ad(xu, ineq_mat, N);
                qp_coeff.ineq_mat_.block(0, 0, num_ineq, ineq_mat.cols()) = ineq_mat;
                qp_coeff.ineq_mat_.block(num_ineq, 0, num_ineq, ineq_mat.cols()) = -ineq_mat;

                Eigen::VectorXd ineq_val(num_ineq);
                inequality_constraints(xu, ineq_val, N);
                qp_coeff.ineq_vec_.head(num_ineq) = ubg.head(num_ineq) - ineq_val;
                qp_coeff.ineq_vec_.tail(num_ineq) = ineq_val - lbg.head(num_ineq);

                // Bounds on variables
                Eigen::VectorXd x_min_total = Eigen::VectorXd::Constant((N + 1) * 5, -INFTY);
                Eigen::VectorXd x_max_total = Eigen::VectorXd::Constant((N + 1) * 5, INFTY);

                for (int i = 0; i < N + 1; ++i) {
                    x_min_total.segment(i * 5, 3) = x_min.head(3); // Only for state part
                    x_max_total.segment(i * 5, 3) = x_max.head(3); // Only for state part
                }

                qp_coeff.x_min_ = x_min_total;
                qp_coeff.x_max_ = x_max_total;

                Eigen::VectorXd delta_x = qp_solver->solve(qp_coeff);

                // Line search using merit function
                scalar_t alpha = step_size_selection(xu, grad_obj, delta_x, hess);

                // Update variables
                xu += alpha * delta_x;

                // Check for convergence
                scalar_t constraint_violation = constraints_violation(xu);
                if (delta_x.norm() < settings_.tol && constraint_violation < settings_.tol) {
                    std::cout << "Convergence criteria met." << std::endl;
                    break;
                }

                std::cout << "Iteration " << info_.iter << ": xu = " << xu.transpose() << ", Constraint Violation = " << constraint_violation << std::endl;
            }

            // Update state variables for the next simulation step
            Eigen::Matrix<scalar_t, 3, 1> xdot;
            Eigen::Matrix<scalar_t, 3, 1> x_prev = xu_prev.template head<3>();
            Eigen::Matrix<scalar_t, 2, 1> u_curr = xu.template tail<2>();
            Eigen::Matrix<scalar_t, 1, 1> p;
            p << 2.0;
            problem_.dynamics(x_prev, u_curr, p, xdot);
            // Using forward difference method for differentiation
            Eigen::Matrix<scalar_t, 3, 1> eq_val_new = (xu.template head<3>() - xu_prev.template head<3>()) - xdot;
            qp_coeff.eq_vec_.segment(0, 3) = -eq_val_new;

            xu_prev = xu;
        }

        primal_solution_xu_ = xu;
    }

    void hessian_regularisation_dense_impl(Eigen::Ref<hessian_t> lag_hessian) noexcept {
        Eigen::EigenSolver<hessian_t> es;
        es.compute(lag_hessian);

        scalar_t minEigValue = es.eigenvalues().real().minCoeff();
        if (minEigValue <= 0) {
            Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic> Deig = es.eigenvalues().real().asDiagonal();
            for (int i = 0; i < Deig.rows(); i++) {
                if (Deig(i, i) <= 0) { Deig(i, i) = -1 * Deig(i, i) + 0.1; }
            }
            lag_hessian.noalias() = es.eigenvectors().real() * Deig * es.eigenvectors().real().transpose();
        }
    }

    scalar_t constraints_violation(const state_t& xu) const {
        scalar_t violation = 0;

        Eigen::Matrix<scalar_t, 3, 1> eq_constraints;
        Eigen::Matrix<scalar_t, 1, 1> p = parameter_t::Constant(2.0);
        problem_.equality_constraints_impl(xu, xu, p, eq_constraints);  // Correct call with xu twice
        violation += eq_constraints.template lpNorm<1>();

        Eigen::Matrix<scalar_t, 1, 1> ineq_constraints;
        problem_.inequality_constraints(xu, ineq_constraints);
        violation += (ineq_constraints.array() < 0).select(-ineq_constraints, 0).sum();

        return violation;
    }

    scalar_t step_size_selection(const state_t& xu, const Eigen::VectorXd& grad_obj, const Eigen::VectorXd& delta, const hessian_t& hess) const {
        scalar_t mu, phi_l1, Dp_phi_l1;
        const scalar_t tau = settings_.tau;
        const scalar_t eta = settings_.eta;
        const scalar_t rho = settings_.rho;

        scalar_t constr_l1 = constraints_violation(xu);
        mu = 1.0;

        scalar_t cost_1 = problem_.obj_func(xu);

        phi_l1 = cost_1 + mu * constr_l1;
        Dp_phi_l1 = grad_obj.dot(delta) - mu * constr_l1;

        scalar_t alpha = scalar_t(1.0);
        scalar_t cost_step;
        state_t xu_step;
        for (int i = 1; i < settings_.line_search_max_iter; i++) {
            xu_step.noalias() = xu + alpha * delta;
            cost_step = problem_.obj_func(xu_step);
            scalar_t phi_l1_step = cost_step + mu * constraints_violation(xu_step);

            if (phi_l1_step <= (phi_l1 + alpha * eta * Dp_phi_l1))
                return alpha;
            else
                alpha *= tau;
        }

        return alpha;
    }

    state_t primal_solution_xu() const { return primal_solution_xu_; }
    const Settings& settings() const { return settings_; }
    Settings& settings() { return settings_; }
    const Info& info() const { return info_; }

private:

  hessian_t hessian_ad(const state_t& xu, int N) {
    using ad_ad_scalar_t = Eigen::AutoDiffScalar<Eigen::Matrix<ad_scalar_t, Eigen::Dynamic, 1>>;
    using ad_variable_t = Eigen::Matrix<ad_ad_scalar_t, Eigen::Dynamic, 1>;

    int total_size = (N + 1) * xu.size();
    hessian_t hess = hessian_t::Zero(total_size, total_size);

    ad_variable_t xu_ad;
    for (int i = 0; i < total_size; ++i) {
      xu_ad(i) = xu(i);
      xu_ad(i).derivatives() = Eigen::VectorXd::Unit(total_size, i);
      xu_ad(i).value().derivatives() = Eigen::VectorXd::Unit(total_size, i);
    }

    // ad_variable_t obj_func_ad = problem_.obj_func(xu_ad);
    ad_ad_scalar_t obj_func_ad = problem_.template obj_func<ad_scalar_t>(xu_ad);


    for (int j = 0; j < total_size; ++j) {
      for (int k = 0; k < total_size; ++k) {
        hess(j, k) = obj_func_ad.derivatives()(j).derivatives()(k);
      }
    }

    return hess;
  }


  Eigen::VectorXd gradient_ad(const state_t& xu, int N) {
    using ad_scalar_t = typename Problem::ad_scalar_t;
    using ad_variable_t = Eigen::Matrix<ad_scalar_t, Eigen::Dynamic, 1>;

    int total_size = (N + 1) * xu.size();
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(total_size);

    ad_variable_t xu_ad(total_size);
    for (int i = 0; i < total_size; ++i) {
      xu_ad(i) = xu(i);
      xu_ad(i).derivatives() = Eigen::VectorXd::Unit(total_size, i);
    }

    ad_scalar_t obj_func_ad = problem_.obj_func(xu_ad);

    for (int i = 0; i < total_size; ++i) {
      grad(i) = obj_func_ad.derivatives()(i);
    }

    return grad;
  }


  void equality_constraints(const state_t& xu_prev, const state_t& xu, Eigen::VectorXd& constraints, int N) const {
        constraints.resize(N * 3); // Adjusted for 3 equality constraints per time step
        Eigen::Matrix<scalar_t, 1, 1> p = Eigen::Matrix<scalar_t, 1, 1>::Constant(2.0);
        for (int i = 0; i < N; ++i) {
            Eigen::Matrix<scalar_t, 3, 1> eq;
            problem_.equality_constraints_impl(xu_prev, xu, p, eq);
            constraints.segment(i * 3, 3) = eq;
        }
    }

    void inequality_constraints(const state_t& xu, Eigen::VectorXd& constraints, int N) const {
        constraints.resize(N);
        for (int i = 0; i < N; ++i) {
            Eigen::Matrix<scalar_t, 1, 1> ineq;
            problem_.inequality_constraints(xu, ineq);
            constraints(i) = ineq(0);
        }
    }

    void equality_jacobian_ad(const state_t& xu_prev, const state_t& xu, Eigen::MatrixXd& J, int N) const {
        using ad_scalar_t = typename Problem::ad_scalar_t;
        using ad_state_t = typename Problem::ad_state_t;
        using ad_control_t = typename Problem::ad_control_t;

        J.setZero(N * 3, (N + 1) * 5);

        for (int i = 0; i < N; ++i) {
            ad_state_t xu_prev_ad, xu_ad;
            for (int j = 0; j < xu.size(); ++j) {
                xu_prev_ad(j) = ad_scalar_t(xu_prev(j), Eigen::VectorXd::Unit((N + 1) * xu.size(), j));
                xu_ad(j) = ad_scalar_t(xu(j), Eigen::VectorXd::Unit((N + 1) * xu.size(), j));
            }

            Eigen::Matrix<ad_scalar_t, 3, 1> eq_ad;
            Eigen::Matrix<ad_scalar_t, 1, 1> p_ad = parameter_t::Constant(2.0);
            problem_.equality_constraints_impl(xu_prev_ad, xu_ad, p_ad, eq_ad);

            for (int j = 0; j < 3; ++j) {
                J.block(i * 3 + j, 0, 1, (N + 1) * 5) = eq_ad(j).derivatives().transpose();
            }
        }
    }

    void inequality_jacobian_ad(const state_t& xu, Eigen::MatrixXd& J, int N) const {
        using ad_scalar_t = typename Problem::ad_scalar_t;
        using ad_state_t = typename Problem::ad_state_t;
        using ad_control_t = typename Problem::ad_control_t;

        J.setZero(N, (N + 1) * 5);

        for (int i = 0; i < N; ++i) {
            ad_state_t xu_ad;
            for (int j = 0; j < xu.size(); ++j) {
                xu_ad(j) = ad_scalar_t(xu(j), Eigen::VectorXd::Unit((N + 1) * xu.size(), j));
            }

            Eigen::Matrix<ad_scalar_t, 1, 1> ineq_ad;
            problem_.inequality_constraints(xu_ad, ineq_ad);

            J.block(i, 0, 1, (N + 1) * 5) = ineq_ad(0).derivatives().transpose();
        }
    }

    Settings settings_;
    Info info_;
    state_t primal_solution_xu_;
    Problem problem_;
};

int main() {
    RobotOCP problem;
    SQPSolver<RobotOCP> solver;

    problem.set_Q_coeff(2.0);
    solver.settings().max_iter = 50;
    solver.settings().line_search_max_iter = 5;

    RobotOCP::state_t xu0;
    xu0 << 0.5, 0.5, 0.5, 0.0, 0.0;

    Eigen::VectorXd x_min(5);
    Eigen::VectorXd x_max(5);
    Eigen::VectorXd lbg(1);  // Inequality constraint bounds
    Eigen::VectorXd ubg(1);  // Inequality constraint bounds

    x_min << -std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(), -1.5, -0.75;
    x_max << std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), 1.5, 0.75;
    lbg << -10;  // Lower bound for inequality constraint
    ubg << 10;   // Upper bound for inequality constraint

    int N = 10; // Prediction horizon
    int nsim = 10;
    solver.solve(xu0, x_min, x_max, lbg, ubg, N, nsim);
    auto xu = solver.primal_solution_xu();

    std::cout << "iter " << solver.info().iter << std::endl;
    std::cout << "Solution xu: " << xu.transpose() << std::endl;

    return 0;
}
