#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>
#include <iostream>

// Define the RobotOCP class
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

    static constexpr int NX = 5;


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

    void set_Q_coeff(const scalar_t& coeff) {
        Q.diagonal() << coeff, coeff, coeff;
    }
};

// Define the HessianGradientCalculator class
template<typename Problem>
class HessianGradientCalculator {
public:
    using scalar_t = typename Problem::scalar_t;
    using state_t = typename Problem::state_t;
    using ad_scalar_t = typename Problem::ad_scalar_t;
    using ad_state_t = typename Problem::ad_state_t;
    using hessian_t = Eigen::Matrix<scalar_t, Problem::NX, Problem::NX>;
    static constexpr int NX = Problem::NX;
    HessianGradientCalculator(const Problem& problem) : problem_(problem) {}

  void grad_obj_func(const state_t &x, state_t &grad) const {
      ad_state_t x_ad;
      for (int i = 0; i < NX; ++i) {
        x_ad(i) = x(i);
        x_ad(i).derivatives() = Eigen::VectorXd::Unit(NX, i);
      }

      ad_scalar_t f_ad = problem_.obj_func(x_ad);

      for (int i = 0; i < NX; ++i) {
        grad(i) = f_ad.derivatives()(i);
      }
    }

  void hess_obj_func(const state_t &x, Eigen::Matrix<scalar_t, NX, NX> &hess) const {
      using ad_ad_scalar_t = Eigen::AutoDiffScalar<Eigen::Matrix<ad_scalar_t, NX, 1>>;
      using ad_ad_state_t = Eigen::Matrix<ad_ad_scalar_t, NX, 1>;

      ad_ad_state_t x_ad;
      for (int i = 0; i < NX; ++i) {
        x_ad(i) = x(i);
        x_ad(i).derivatives() = Eigen::VectorXd::Unit(NX, i);
        x_ad(i).value().derivatives() = Eigen::VectorXd::Unit(NX, i);
      }

      ad_ad_scalar_t f_ad = problem_.obj_func(x_ad);

      for (int i = 0; i < NX; ++i) {
        for (int j = 0; j < NX; ++j) {
          hess(i, j) = f_ad.derivatives()(i).derivatives()(j);
        }
      }
    }

private:
    const Problem& problem_;
};

// Test the Hessian and Gradient calculation
int main() {
    RobotOCP problem;
    HessianGradientCalculator<RobotOCP> calculator(problem);

    RobotOCP::state_t xu;
    xu << 0.5, 0.5, 0.5, 0.0, 0.0;

    Eigen::Matrix<double, RobotOCP::state_t::RowsAtCompileTime, 1> grad;
    calculator.grad_obj_func(xu, grad);

    Eigen::Matrix<double, RobotOCP::state_t::RowsAtCompileTime, RobotOCP::state_t::RowsAtCompileTime> hess;
    calculator.hess_obj_func(xu, hess);

    std::cout << "Gradient:\n" << grad << std::endl;
    std::cout << "Hessian:\n" << hess << std::endl;

    return 0;
}
