# QpSolverCollection – extended fork

> **Fork of [`isri-aist/QpSolverCollection`](https://github.com/isri-aist/QpSolverCollection/tree/ros1)** with additional solvers, utilities and examples developed during my work at RWTH‑IRT.

---

## What’s new in this fork?

| Category             | Additions                                                                                                                                                                                                                                                                                   |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Algorithms**       | • **Sequential Quadratic Programming (SQP)** solver powered by [IPOPT](https://github.com/coin-or/Ipopt) and Eigen‐based automatic differentiation.<br>• **Model‑Predictive Control (MPC) wrapper** that formulates linear MPC problems as QPs and works with any solver in the collection. |
| **Code**             | • New `include/ipopt` and `src/QpSolverIpopt.cpp` implementing the IPOPT back‑end.<br>• New `include/mpc_wrapper` and `src/MpcSolver.cpp`.                                                                                                                                                  |
| **Examples & tests** | • `sample_sqp*.cpp` – constrained Rosenbrock & HS071 problems.<br>• `SampleMPC*.cpp` – cart‑pole and vessel‑dynamics MPC demos.<br>• `nmpc*.cpp` – non‑linear MPC with the SQP solver.                                                                                                      |
| **Build system**     | • ROS 1 **catkin** workflow (replacing upstream ROS 2/colcon).<br>• New CMake option `ENABLE_IPOPT` (default **OFF**).                                                                                                                                                                      |

---

## Supported solvers

| Solver          | Flag                   | Notes                               |
| --------------- | ---------------------- | ----------------------------------- |
| QLD             | `-DENABLE_QLD=ON`      | Classic active‑set QP               |
| QuadProg        | `-DENABLE_QUADPROG=ON` | Goldfarb–Idnani implementation      |
| JRLQP           | `-DENABLE_JRLQP=ON`    | Active set (Eigen)                  |
| qpOASES         | `-DENABLE_QPOASES=ON`  | Online active‑set                   |
| OSQP            | `-DENABLE_OSQP=ON`     | Operator‑splitting (ADMM)           |
| NASOQ           | `-DENABLE_NASOQ=ON`    | Interior‑point                      |
| HPIPM           | `-DENABLE_HPIPM=ON`    | Structure‑exploiting interior‑point |
| ProxQP          | `-DENABLE_PROXQP=ON`   | Proximal QP                         |
| qpmad           | `-DENABLE_QPMAD=ON`    | Reduced Hessian                     |
| **IPOPT (NEW)** | `-DENABLE_IPOPT=ON`    | Uses IPOPT for SQP / NLP            |
| LSSOL (private) | `-DENABLE_LSSOL=ON`    | Lawson–Hanson                       |

---

## Quick installation (ROS 1 – Catkin)

### 1 · Prerequisites

```bash
sudo apt update && sudo apt install \
    build-essential cmake git pkg-config \
    libeigen3-dev coinor-libipopt-dev      # IPOPT
# plus the libraries for any solvers you plan to enable
```

### 2 · Workspace setup

```bash
mkdir -p ~/ros/ws_qp_solver_collection/src
cd ~/ros/ws_qp_solver_collection/src

git clone --recursive https://github.com/fmdazhar/irt_azhar.git -b qpsolvercollection
cd ..

catkin config --cmake-args \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DENABLE_IPOPT=ON          \
    -DENABLE_QPOASES=ON        # (example – enable what you need)

catkin build
source devel/setup.bash
```

> **Tip:** If no `ENABLE_*` flag is passed **and** `DEFAULT_ENABLE_ALL` is **ON**, all available solvers will be built automatically.

---

## Minimal examples

### 1 · Solve a quadratic program (unchanged)

```cpp
#include <qp_solver_collection/QpSolverCollection.h>
// ... (see original README)
```

### 2 · Solve a nonlinear program with SQP

```cpp
#include <mpc_wrapper/SqpSolver.h>
#include <ipopt/nlproblem.hpp>

// Define your NLP in a class derived from `Ipopt::TNLP` or using the
// helper in `nlproblem.hpp`.

auto nlp = makeMyRosenbrockProblem();            // user‑defined
SqpSolver sqp(nlp);

Eigen::VectorXd x0(2); x0 << -1.2, 1.0;
Eigen::VectorXd solution = sqp.solve(x0);
std::cout << "Optimal x = " << solution.transpose() << std::endl;
```

### 3 · Linear MPC

```cpp
#include <mpc_wrapper/MpcSolver.h>
// Build (A, B) model, weights (Q, R, Q_N), horizon N, constraints ...
MpcSolver mpc(nx, nu, N, Ad, Bd, Q, QN, R);
mpc.setInitialState(x0);
Eigen::VectorXd u = mpc.solveMPC(1);
```

Complete examples are in [`tests/`](tests/).

---

## Directory layout

```
include/
├─ ipopt/            # IPOPT interface & AutoDiff utilities
├─ mpc_wrapper/      # MPC & SQP high‑level APIs
└─ qp_solver_collection/ (upstream headers)

src/                 # corresponding implementation files
cmake/               # FindIpopt.cmake etc.
tests/               # SampleQP, MPC, SQP and vessel‑dynamics demos
```

---

## License

BSD 2‑Clause – identical to the upstream project.

---

## Acknowledgements

This work extends the excellent **QpSolverCollection** by AIST‑CNRS JRL. IPOPT and Eigen are © their respective authors.
