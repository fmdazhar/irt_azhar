#include <qp_solver_collection/QpSolverCollection.h>

#if ENABLE_IPOPT
#define HAVE_CSTDDEF
#include <IpTNLP.hpp>
#undef HAVE_CSTDDEF
#include <coin/IpSmartPtr.hpp>
#include "IpSolveStatistics.hpp"
#include <coin/IpIpoptApplication.hpp>
#include <memory>
#include <iostream>
#include <list>

using namespace QpSolverCollection;

class QpSolverIpoptNLP : public Ipopt::TNLP
{
public:
  QpSolverIpoptNLP(int dim_var,
                   int dim_eq,
                   int dim_ineq,
                   Eigen::Ref<Eigen::MatrixXd> Q,
                   const Eigen::Ref<const Eigen::VectorXd>& c,
                   const Eigen::Ref<const Eigen::MatrixXd>& A,
                   const Eigen::Ref<const Eigen::VectorXd>& b,
                   const Eigen::Ref<const Eigen::MatrixXd>& C,
                   const Eigen::Ref<const Eigen::VectorXd>& d,
                   const Eigen::Ref<const Eigen::VectorXd>& x_min,
                   const Eigen::Ref<const Eigen::VectorXd>& x_max)
  : dim_var_(dim_var), dim_eq_(dim_eq), dim_ineq_(dim_ineq), Q_(Q), c_(c), A_(A), b_(b), C_(C), d_(d), x_min_(x_min), x_max_(x_max)
  {
//    std::cout << "QpSolverIpoptNLP initialized with dimensions: " << dim_var << ", " << dim_eq << ", " << dim_ineq << std::endl;
//    std::cout << "Q: " << Q_ << std::endl;
//    std::cout << "c: " << c_ << std::endl;
//    std::cout << "A: " << A_ << std::endl;
//    std::cout << "b: " << b_ << std::endl;
//    std::cout << "C: " << C_ << std::endl;
//    std::cout << "d: " << d_ << std::endl;
//    std::cout << "x_min: " << x_min_ << std::endl;
//    std::cout << "x_max: " << x_max_ << std::endl;
  }

  virtual ~QpSolverIpoptNLP() {}

  virtual bool get_nlp_info(Ipopt::Index& n, Ipopt::Index& m, Ipopt::Index& nnz_jac_g, Ipopt::Index& nnz_h_lag, IndexStyleEnum& index_style)
  {
    n = dim_var_;
    m = dim_eq_ + dim_ineq_;
    nnz_jac_g = dim_eq_ * dim_var_ + dim_ineq_ * dim_var_;
    nnz_h_lag = dim_var_ * (dim_var_ + 1) / 2;
    index_style = TNLP::C_STYLE;
//    std::cout << "get_nlp_info: n=" << n << ", m=" << m << ", nnz_jac_g=" << nnz_jac_g << ", nnz_h_lag=" << nnz_h_lag << std::endl;
    return true;
  }

  virtual bool get_bounds_info(Ipopt::Index n, Ipopt::Number* x_l, Ipopt::Number* x_u, Ipopt::Index m, Ipopt::Number* g_l, Ipopt::Number* g_u)
  {
    for (Ipopt::Index i = 0; i < n; ++i)
    {
      x_l[i] = x_min_[i];
      x_u[i] = x_max_[i];
    }
    for (Ipopt::Index i = 0; i < dim_eq_; ++i)
    {
      g_l[i] = b_[i];
      g_u[i] = b_[i];
    }
    for (Ipopt::Index i = 0; i < dim_ineq_; ++i)
    {
      g_l[dim_eq_ + i] = -1e19;
      g_u[dim_eq_ + i] = d_[i];
    }
//    std::cout << "get_bounds_info: x_l=" << Eigen::Map<Eigen::VectorXd>(x_l, n).transpose() << ", x_u=" << Eigen::Map<Eigen::VectorXd>(x_u, n).transpose() << std::endl;
    return true;
  }

  virtual bool get_starting_point(Ipopt::Index n, bool init_x, Ipopt::Number* x, bool init_z, Ipopt::Number* z_L, Ipopt::Number* z_U, Ipopt::Index m, bool init_lambda, Ipopt::Number* lambda)
  {
    for (Ipopt::Index i = 0; i < n; ++i)
    {
      x[i] = 0.0;
    }
//    std::cout << "get_starting_point: x=" << Eigen::Map<Eigen::VectorXd>(x, n).transpose() << std::endl;
    return true;
  }

  virtual bool eval_f(Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Number& obj_value)
  {
    assert(n == dim_var_); // Ensure n matches dim_var_
    Eigen::Map<const Eigen::VectorXd> x_vec(x, n);
    obj_value = (0.5 * x_vec.transpose() * Q_ * x_vec + c_.transpose() * x_vec)(0);
//    std::cout << "eval_f: obj_value=" << obj_value << std::endl;
    return true;
  }




  virtual bool eval_grad_f(Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Number* grad_f)
  {
    Eigen::Map<const Eigen::VectorXd> x_vec(x, n);
    Eigen::Map<Eigen::VectorXd> grad_f_vec(grad_f, n);
    grad_f_vec = Q_ * x_vec + c_;
//    std::cout << "eval_grad_f: grad_f=" << grad_f_vec.transpose() << std::endl;
    return true;
  }

  virtual bool eval_g(Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Index m, Ipopt::Number* g) override
  {
    Eigen::Map<const Eigen::VectorXd> x_vec(x, n);
    Eigen::Map<Eigen::VectorXd> g_vec(g, m);

    // Print x vector
//    std::cout << "eval_g: x=" << x_vec.transpose() << std::endl;

    // Evaluate equality constraints
    g_vec.head(dim_eq_) = A_ * x_vec ;
//    std::cout << "eval_g: A*x - b = " << (A_ * x_vec - b_).transpose() << std::endl;

    // Evaluate inequality constraints
    g_vec.tail(dim_ineq_) = C_ * x_vec ;
//    std::cout << "eval_g: C*x - d = " << (C_ * x_vec - d_).transpose() << std::endl;
//    std::cout << "eval_g: g=" << g_vec.transpose() << std::endl;

    return true;
  }

  virtual bool eval_jac_g(Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Index m, Ipopt::Index nele_jac, Ipopt::Index* iRow, Ipopt::Index* jCol, Ipopt::Number* values)
  {
    if (values == nullptr)
    {
      Ipopt::Index idx = 0;
      for (Ipopt::Index i = 0; i < dim_eq_; ++i)
      {
        for (Ipopt::Index j = 0; j < dim_var_; ++j)
        {
          iRow[idx] = i;
          jCol[idx] = j;
          ++idx;
        }
      }
      for (Ipopt::Index i = 0; i < dim_ineq_; ++i)
      {
        for (Ipopt::Index j = 0; j < dim_var_; ++j)
        {
          iRow[idx] = dim_eq_ + i;
          jCol[idx] = j;
          ++idx;
        }
      }
//      std::cout << "eval_jac_g: Jacobian structure set" << std::endl;

    }
    else
    {
      Ipopt::Index idx = 0;
      for (Ipopt::Index i = 0; i < dim_eq_; ++i)
      {
        for (Ipopt::Index j = 0; j < dim_var_; ++j)
        {
          values[idx] = A_(i, j);
          ++idx;
        }
      }
      for (Ipopt::Index i = 0; i < dim_ineq_; ++i)
      {
        for (Ipopt::Index j = 0; j < dim_var_; ++j)
        {
          values[idx] = C_(i, j);
          ++idx;
        }
      }
//      std::cout << "eval_jac_g: Jacobian values set" << std::endl;

    }

    return true;
  }

  virtual bool eval_h(Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Number obj_factor, Ipopt::Index m, const Ipopt::Number* lambda, bool new_lambda, Ipopt::Index nele_hess, Ipopt::Index* iRow, Ipopt::Index* jCol, Ipopt::Number* values)
  {
    if (values == nullptr)
    {
      Ipopt::Index idx = 0;
      for (Ipopt::Index i = 0; i < dim_var_; ++i)
      {
        for (Ipopt::Index j = 0; j <= i; ++j)
        {
          iRow[idx] = i;
          jCol[idx] = j;
          ++idx;
        }
      }
//      std::cout << "eval_h: Hessian structure set" << std::endl;

    }
    else
    {
      Ipopt::Index idx = 0;
      for (Ipopt::Index i = 0; i < dim_var_; ++i)
      {
        for (Ipopt::Index j = 0; j <= i; ++j)
        {
          values[idx] = obj_factor * Q_(i, j);
          ++idx;
        }
      }
//      std::cout << "eval_h: Hessian values set" << std::endl;

    }
    return true;
  }

  virtual void finalize_solution(Ipopt::SolverReturn status,
                                 Ipopt::Index n,
                                 const Ipopt::Number* x,
                                 const Ipopt::Number* z_L,
                                 const Ipopt::Number* z_U,
                                 Ipopt::Index m,
                                 const Ipopt::Number* g,
                                 const Ipopt::Number* lambda,
                                 Ipopt::Number obj_value,
                                 const Ipopt::IpoptData* ip_data,
                                 Ipopt::IpoptCalculatedQuantities* ip_cq)
  {
//    std::cout << "finalize_solution: status=" << status << std::endl;
    solution_ = Eigen::Map<const Eigen::VectorXd>(x, n);
//    std::cout << "finalize_solution: solution=" << solution_.transpose() << std::endl;
        }

  Eigen::VectorXd get_solution() const { return solution_; }

private:
  int dim_var_, dim_eq_, dim_ineq_;
  Eigen::MatrixXd Q_;
  Eigen::VectorXd c_;
  Eigen::MatrixXd A_;
  Eigen::VectorXd b_;
  Eigen::MatrixXd C_;
  Eigen::VectorXd d_;
  Eigen::VectorXd x_min_;
  Eigen::VectorXd x_max_;
  Eigen::VectorXd solution_;
};

QpSolverIpopt::QpSolverIpopt()
{  type_ = QpSolverType::IPOPT;
  ipopt_app_ = std::unique_ptr<Ipopt::IpoptApplication>(IpoptApplicationFactory());
  // Specify the library containing HSL routines
  ipopt_app_->Options()->SetStringValue("hsllib", "/usr/local/lib/libcoinhsl.so");
//  ipopt_app_->Options()->SetStringValue("linear_solver", "HSL_MA97");
//  ipopt_app_->Options()->SetIntegerValue("print_level", 10); // corrected the option name
  ipopt_app_->Options()->SetNumericValue("tol", 1e-3);

  std::cout << "QpSolverIpopt initialized with linear solver: ma27" << std::endl;

}

 Eigen::VectorXd QpSolverIpopt::solve(int dim_var,
                         int dim_eq,
                         int dim_ineq,
                         Eigen::Ref<Eigen::MatrixXd> Q,
                         const Eigen::Ref<const Eigen::VectorXd> & c,
                         const Eigen::Ref<const Eigen::MatrixXd> & A,
                         const Eigen::Ref<const Eigen::VectorXd> & b,
                         const Eigen::Ref<const Eigen::MatrixXd> & C,
                         const Eigen::Ref<const Eigen::VectorXd> & d,
                         const Eigen::Ref<const Eigen::VectorXd> & x_min,
                         const Eigen::Ref<const Eigen::VectorXd> & x_max)
{
  Eigen::VectorXd x(dim_var);
  x.setZero();

  Ipopt::SmartPtr<Ipopt::TNLP> mynlp = new QpSolverIpoptNLP(dim_var, dim_eq, dim_ineq, Q, c, A, b, C, d, x_min, x_max);

  Ipopt::ApplicationReturnStatus status = ipopt_app_->Initialize();
  if (status != Ipopt::Solve_Succeeded) {
    std::cerr << "Error during IPOPT initialization!" << std::endl;
    solve_failed_ = true;
    return x;
  }
  status = ipopt_app_->OptimizeTNLP(mynlp);
  if (status ==  Ipopt::Solve_Succeeded) {
    solve_failed_ = false;
    std::cout << "The problem solved successfully!" << std::endl;
    x = static_cast<QpSolverIpoptNLP*>(Ipopt::GetRawPtr(mynlp))->get_solution();
  }
  else {
    solve_failed_ = true;
    std::cout << "The problem not solved!" << std::endl;
  }
  return x;
}

namespace QpSolverCollection {

std::shared_ptr<QpSolverIpopt> allocateQpSolverIpopt() {
  return std::make_shared<QpSolverIpopt>();
}

} // namespace QpSolverCollection
#endif // ENABLE_IPOPT