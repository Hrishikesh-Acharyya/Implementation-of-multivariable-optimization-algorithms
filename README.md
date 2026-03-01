# Multivariable Optimization Algorithm Suite

This repository contains a comprehensive implementation of classic and modern multivariable optimization algorithms. Developed as part of an academic deep-dive into computational engineering and optimization theory, the suite features robust mathematical hardening, advanced line search techniques, and standardized statistical benchmarking using the **Dolan-Moré** method.

---

## Features

* **Three Core Solvers**: 
    * **Steepest Descent**: First-order baseline with linear convergence.
    * **Modified Newton's Method**: Second-order solver with a **Levenberg-Marquardt** eigenvalue shift to escape saddle points.
    * **Damped BFGS**: Quasi-Newton solver utilizing **Powell’s Damping** for numerical stability on non-convex surfaces.
* **Line Search Engine**: Supports Backtracking (Armijo), Exact (Quadratic), and Strong Wolfe (Pseudo-implementation) routines.
* **Scientific Benchmarking**: Automated testing against standard topologies (Rosenbrock, Wood's, Eason-Fenton) with randomized starting points.
* **Interactive Visualization**: 
    * **3D WebGL** surface plots with animated trajectories (Plotly).
    * **Log-Linear Convergence Curves** for diagnostic analysis.
    * **Performance Profiles** for statistical solver comparison.

---

##  Project Structure

| File | Description |
| :--- | :--- |
| `optimization.py` | Implementation of Steepest Descent, Newton, and BFGS solvers. |
| `objective_functions.py` | Mathematical definitions for test functions (Gradients & Hessians). |
| `line_search.py` | Step-size determination logic (Armijo, Wolfe, Exact). |
| `benchmark.py` | The Dolan-Moré statistical engine and convergence rate plotter. |
| `main.py` | Interactive CLI for individual optimization runs and 3D visualization. |


---

##  Algorithms Used

### 1. Modified Newton's Method
To solve the "Saddle Point Trap" where the Hessian is not positive-definite, this implementation uses a Levenberg-Marquardt shift.
* **Mechanic**: It extracts eigenvalues of the exact Hessian.
* **Modification**: If `λ_min < 1e-5`, it applies a shift `τ = -λ_min + 1e-3` to the diagonal (`H_mod = H + τI`).
* **Result**: This forces a safe descent direction while maintaining quadratic convergence near the minimum.

### 2. Damped BFGS (Quasi-Newton)
While standard BFGS is fragile on non-convex surfaces, this suite implements **Powell's Damping** to maintain matrix stability.
* **The Curvature Condition**: BFGS requires `s_k^T y_k > 0` to remain positive-definite.
* **The Fix**: If curvature is too low, it creates a synthetic gradient difference vector `r_k` to "dampen" the update.
* **Note**: This implementation currently utilizes `np.linalg.inv` for damping calculations `O(N^3)`, which is suitable for the provided low-dimensional benchmarks.

---

## 📊 Benchmarking & Topologies

We utilize standard benchmarks to stress-test the algorithms:

* **Rosenbrock (2D)**: The "Banana Function" with a narrow, curved valley.
* **Wood's Function (4D)**: A highly non-convex 4-dimensional space with multiple saddle points.
* **Eason-Fenton (2D)**: A multimodal function with steep walls and symmetric minima at approximately `(±1.74, ±2.72)` with a value of `≈ -1.71`.

### How to Read Performance Profiles
The `benchmark.py` script generates **Dolan-Moré Performance Profiles**.
* **Y-axis (ρ):** Fraction of problems solved.
* **X-axis (τ):** Multiple of the best solver's performance (Log Scale).
* **Interpretation**: The curve highest and furthest to the left represents the most robust and efficient algorithm.

---

## Usage

### 1. Run Interactive Optimization
