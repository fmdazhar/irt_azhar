# Vessel Simulation & Interactive Control

This repository contains a **self‑contained C++17 project** that models the 3‑DOF planar motion of a marine vessel (surge, sway, yaw) and lets you interactively apply control forces from the keyboard in real time.  The code demonstrates

* parameter‑identified hydrodynamic & aerodynamic force models,
* adaptive state scaling utilities,
* an extensible Heun / RK4 / Euler integrator,
* seamless Eigen/Boost math integration, and
* a tiny terminal UI for exploration.

---

## Quick Start

```bash
# 1.  Clone and build (out‑of‑source):
$ git clone <repo‑url> vessel_sim && cd vessel_sim
$ mkdir build && cd build
$ cmake .. && make -j$(nproc)

# 2.  Run the demo (keyboard driven):
$ ./Vessel
```

ℹ️  The executable prints the initial state vector and then waits for **live keyboard commands** (see below). All required hydrodynamic data tables ship in *include/*. No extra downloads needed.

---

## Keyboard Controls (in `keyboard_input_handler`)

| Key | Action                                                 |
| --- | ------------------------------------------------------ |
| `1` | Increment/decrement **τ₁** (surge force)               |
| `2` | Increment/decrement **τ₂** (sway force)                |
| `3` | Increment/decrement **τ₃** (yaw moment)                |
| `s` | Toggle sign (adds or subtracts the selected increment) |
| `a` | Cycle through increment scales (1000/100/10 …)         |
| `q` | Quit                                                   |

> After each keypress the program re‑integrates the ODE for one step and prints the updated state `[x  y  ψ  u  v  r  0]`.

---

## Repository Layout

```
.
├── include/            # Hydrodynamic tables, wind coeffs & images
│   ├── *.txt           # Resistance / drift data (XFN, XBETA, …)
│   ├── windcoeff.dat   # Wind coefficients (α, C_Fx, C_Fy, C_Mz)
│   └── Horst_*.png     # Visualisations used in the README
├── main.cpp            # Simulation, integrator and UI logic
├── CMakeLists.txt      # Build script (C++17, Eigen3, Boost)
└── README.md           # You are here
```

---

## Dependencies

| Library    | Tested Version | Notes                                           |
| ---------- | -------------- | ----------------------------------------------- |
| **Eigen3** | ≥ 3.4          | Header‑only, found via `find_package(Eigen3)`   |
| **Boost**  | ≥ 1.65         | Uses *Boost.Math* for barycentric interpolation |
| **glibc**  | —              | POSIX termios + `std::filesystem` need C++17    |

### Custom include path

If Boost headers live in a non‑standard location, tweak

```cmake
set(Boost_INCLUDE_DIR "/path/to/boost/include")
```

in *CMakeLists.txt* before configuring.

---

## Model Overview

* **Hydrodynamics** – Resistance, drift and yaw terms derived from tabulated model‑test data (files in *include/*).  The lookup uses **barycentric rational interpolation** for smooth forces across the entire beta/γ domain.
* **Aerodynamics** – Wind drag & moment from `windcoeff.dat`.
* **Rigid‑Body & Added‑Mass** – Parametrised via the `params` vector; two predefined sets (default vs. system‑identified).
* **Integrator** – Choose Euler, Heun (default) or classic RK4 by changing the `integrator_scheme` string.
* **Scaling** – All inputs, states and derivatives can be non‑dimensionalised with respect to user‑supplied `Scaling`.

---

## TODO

1. **Other integrators** – Implement a new branch in `vessel_model_integrator`.
2. **6‑DOF motion** – Add heave/roll/pitch states and expand force models.
3. **GUI visualisation** – Publish state updates over a socket and plot with Python or a browser.
4. **Controller design** – Replace the manual keyboard loop with an MPC/NN controller.

Pull requests and discussion are welcome 🛳️.

---


Enjoy exploring vessel dynamics! 🚢
