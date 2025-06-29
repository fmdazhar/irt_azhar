#include "mpc_wrapper/SqpSolver.h"
#include "qp_solver_collection/QpSolverCollection.h"
#include <iostream>

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

    void eq_constraints(const variable_t &x, eq_constraint_t &constraints) const {
        constraints(0) = x.squaredNorm() - 40;
    }

    template<typename T>
    void eq_constraints(const Eigen::Matrix<T, NX, 1> &x, Eigen::Matrix<T, NE, 1> &constraints) const {
        constraints(0) = x.squaredNorm() - 40;
    }

    void ineq_constraints(const variable_t &x, ineq_constraint_t &constraints) const {
        constraints(0) = x(0) * x(1) * x(2) * x(3);
    }

    template<typename T>
    void ineq_constraints(const Eigen::Matrix<T, NX, 1> &x, Eigen::Matrix<T, NI, 1> &constraints) const {
        constraints(0) = x(0) * x(1) * x(2) * x(3);
    }
};


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

    // Set solver type (e.g., OSQP or other supported type)
    solver.settings().solver_type = QpSolverCollection::QpSolverType::OSQP;

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
