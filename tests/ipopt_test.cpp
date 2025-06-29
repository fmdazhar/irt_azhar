#define HAVE_CSTDDEF
#include "ipopt/helpers.hpp"
#include "ipopt/nlproblem.hpp"
#include "ipopt/ipopt_interface.hpp"
#undef HAVE_CSTDDEF

#include <iostream>
#include <limits>
#include <Eigen/Core>


// Constrained Rosenbrock Function
POLYMPC_FORWARD_NLP_DECLARATION(/*Name*/ ConstrainedRosenbrock, /*NX*/ 2, /*NE*/1, /*NI*/0, /*NP*/0, /*Type*/double);
class ConstrainedRosenbrock : public ProblemBase<ConstrainedRosenbrock>
{
public:
    const scalar_t a = 1;
    const scalar_t b = 100;
    Eigen::Matrix<scalar_t, 2, 1> SOLUTION = {0.7864, 0.6177};

    template<typename T>
    EIGEN_STRONG_INLINE void cost_impl(const Eigen::Ref<const variable_t<T>>& x, const Eigen::Ref<const static_parameter_t>& p, T& cost) const noexcept
    {
        //Eigen::Array<T,2,1> lox;
        // (a-x)^2 + b*(y-x^2)^2
        //cost = pow(a - x(0), 2) + b * pow(x(1) - pow(x(0), 2), 2);
        polympc::ignore_unused_var(p);
        cost = (a - x(0)) * (a - x(0)) + b * (x(1) - x(0)*x(0)) * (x(1) - x(0)*x(0));
    }

    template<typename T>
    EIGEN_STRONG_INLINE void equality_constraints_impl(const Eigen::Ref<const variable_t<T>>& x, const Eigen::Ref<const static_parameter_t>& p,
                                                       Eigen::Ref<constraint_t<T>> constraint) const noexcept
    {
        // x^2 + y^2 == 1
        constraint << x.squaredNorm() - 1;
        polympc::ignore_unused_var(p);
    }
};

/** HS071 problem as in Ipopt tutorial */
POLYMPC_FORWARD_NLP_DECLARATION(/*Name*/ HS071, /*NX*/ 4, /*NE*/0, /*NI*/2, /*NP*/0, /*Type*/double);
class HS071 : public ProblemBase<HS071>
{
public:
    Eigen::Matrix<scalar_t, 4, 1> SOLUTION = {1.00000000, 4.74299963, 3.82114998, 1.37940829};

    template<typename T>
    EIGEN_STRONG_INLINE void cost_impl(const Eigen::Ref<const variable_t<T>>& x, const Eigen::Ref<const static_parameter_t>& p, T& cost) const noexcept
    {
        cost = x(0)*x(3)*(x(0) + x(1) + x(2)) + x(2);
        polympc::ignore_unused_var(p);
    }

    template<typename T>
    EIGEN_STRONG_INLINE void inequality_constraints_impl(const Eigen::Ref<const variable_t<T>>& x, const Eigen::Ref<const static_parameter_t>& p,
                                                         Eigen::Ref<ineq_constraint_t<T>> constraint) const noexcept
    {
        // 25 <= x^2 + y^2 <= Inf -> will set bounds later once the problem is instantiated
        constraint(1) = x(0)*x(1)*x(2)*x(3);
        constraint(0) = x.squaredNorm() - 40;
        polympc::ignore_unused_var(p);
    }
};
/** HS071 problem as in Ipopt tutorial */
POLYMPC_FORWARD_NLP_DECLARATION(/*Name*/ HS071_2, /*NX*/ 4, /*NE*/1, /*NI*/1, /*NP*/0, /*Type*/double);
class HS071_2 : public ProblemBase<HS071_2>
{
public:
    Eigen::Matrix<scalar_t, 4, 1> SOLUTION = {1.00000000, 4.74299963, 3.82114998, 1.37940829};

    template<typename T>
    EIGEN_STRONG_INLINE void cost_impl(const Eigen::Ref<const variable_t<T>>& x, const Eigen::Ref<const static_parameter_t>& p, T& cost) const noexcept
    {
        cost = x(0)*x(3)*(x(0) + x(1) + x(2)) + x(2);
        polympc::ignore_unused_var(p);
    }

    template<typename T>
    EIGEN_STRONG_INLINE void inequality_constraints_impl(const Eigen::Ref<const variable_t<T>>& x, const Eigen::Ref<const static_parameter_t>& p,
                                                         Eigen::Ref<ineq_constraint_t<T>> constraint) const noexcept
    {
        // 25 <= x^2 + y^2 <= Inf -> will set bounds later once the problem is instantiated
        constraint << x(0)*x(1)*x(2)*x(3);
        polympc::ignore_unused_var(p);
    }

    template<typename T>
    EIGEN_STRONG_INLINE void equality_constraints_impl(const Eigen::Ref<const variable_t<T>>& x, const Eigen::Ref<const static_parameter_t>& p,
                                                       Eigen::Ref<eq_constraint_t<T>> constraint) const noexcept
    {
        // x(0)^2 + x(1)^2 + x(2)^2 + x(3)^2 == 40
        constraint << x.squaredNorm() - 40;
        polympc::ignore_unused_var(p);
    }
};
void TestConstrainedRosenbrock()
{
// will be using the default
using Solver = IpoptInterface<ConstrainedRosenbrock>;
ConstrainedRosenbrock problem;
Solver solver;
Solver::nlp_variable_t x0, x;
Solver::nlp_dual_t y0;
y0.setZero();

x0 << 0.0, 0.0;
solver.solve(x0, y0);

x = solver.primal_solution();

    std::cout << "Solving ConstrainedRosenbrock \n";


std::cout << "iter " << solver.info().iter << std::endl;
std::cout << "Solution " << x.transpose() << std::endl;

    if (x.isApprox(problem.SOLUTION, 1e-2)) {
        std::cout << "Rosenbrock passed\n";
    } else {
        std::cout << "Rosenbrock failed\n";
    }}


void TestHS071()
{
    // will be using the default
    using Solver = IpoptInterface<HS071>;
    HS071 problem;
    Solver solver;

    // try to solve the problem
    solver.lower_bound_x() << 1.0, 1.0, 1.0, 1.0;
    solver.upper_bound_x() << 5.0, 5.0, 5.0, 5.0;
    solver.lower_bound_g() << 0.0, 25.0;
    solver.upper_bound_g() << 0.0, std::numeric_limits<double>::infinity();
    solver.primal_solution() << 1.0, 5.0, 5.0, 1.0;

    std::cout << "Solving HS071 \n";

    Solver::nlp_variable_t x0, x;
    Solver::nlp_dual_t y0;
//    x.setZero();

    x0 << 1.0, 5.0, 5.0, 1.0;
    y0.setZero();

    solver.solve(x0, y0);
    x = solver.primal_solution();

    std::cout << "iter " << solver.info().iter << std::endl;
    std::cout << "Solution " << x.transpose() << std::endl;

    if (x.isApprox(problem.SOLUTION, 1e-2)) {
        std::cout << "TestHS071 passed\n";
    } else {
        std::cout << "TestHS071 failed\n";
    }

}



void TestHS071_2()
{
    // will be using the default
    using Solver = IpoptInterface<HS071_2>;
    HS071_2 problem;
    Solver solver;

    // try to solve the problem
    solver.lower_bound_x() << 1.0, 1.0, 1.0, 1.0;
    solver.upper_bound_x() << 5.0, 5.0, 5.0, 5.0;
    solver.lower_bound_g() << 25.0;
    solver.upper_bound_g() << std::numeric_limits<double>::infinity();
    solver.primal_solution() << 1.0, 5.0, 5.0, 1.0;
    solver.settings().SetIntegerValue("max_iter", 50);


    std::cout << "Solving HS071_2\n";

    Solver::nlp_variable_t x0, x;
    Solver::nlp_dual_t y0;

    // Provide initial guesses for x and lambda
    x0 << 0.0, 0.0, 0.0, 0.0;
    y0.setZero();

    solver.solve(x0, y0);
    x = solver.primal_solution();
    std::cout << "iter " << solver.info().iter << std::endl;
    std::cout << "Solution " << x.transpose() << std::endl;
    if (x.isApprox(problem.SOLUTION, 1e-2)) {
        std::cout << "TestHS071_2 passed\n";
    } else {
        std::cout << "TestHS071_2 failed\n";
    }
}

int main()
{
    TestHS071();
    TestHS071_2();
    TestConstrainedRosenbrock();
    return 0;
}
