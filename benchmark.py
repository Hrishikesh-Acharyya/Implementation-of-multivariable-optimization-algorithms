import numpy as np
import time #Built-in timing module for measuring CPU time
import matplotlib.pyplot as plt
import os
import contextlib

from optimization import steepest_descent, newton, bfgs
from objective_functions import Rosenbrock, Woods, Eason_Fenton


GLOBAL_MINIMA = {
    'Rosenbrock': 0.0,
    'Woods': 0.0,
    'EasonFenton': 1.77101,  # Approximate minimum
    'Quadratic': 0.0
}

class TrackedFunction:
    """
    A decorator-style wrapper that intercepts and increments a counter every time 
    an optimization algorithm evaluates the objective function, gradient, or Hessian.

    Attributes:
    -----------
    func_obj : object
        The underlying mathematical objective function (e.g., Rosenbrock).
    calls : int
        The cumulative count of all mathematical evaluations performed.
    """
    def __init__(self, func_obj):
        self.func_obj = func_obj
        self.calls = 0

    def evaluate(self, x):
        self.calls += 1
        return self.func_obj.evaluate(x)

    def gradient(self, x):
        self.calls += 1
        return self.func_obj.gradient(x)

    def hessian(self, x):
        self.calls += 1
        return self.func_obj.hessian(x)
        
    @property
    def name(self):
        return self.func_obj.__class__.__name__

def plot_convergence_rates(solvers, functions):
    """
    Generates a multi-panel figure showing the logarithmic error decay of each 
    solver across different topologies from a fixed starting point.

    Visualization Logic:
    --------------------
    * X-axis: Iteration Number.
    * Y-axis: Log10(f(x)).
    * Mathematical Interpretation: 
        - A straight diagonal slope indicates Linear Convergence (Steepest Descent).
        - An accelerating downward curve indicates Superlinear (BFGS) or 
          Quadratic (Newton) Convergence.
    """
    print("\n--- Generating Convergence Rate Curves ---")
    fig, axes = plt.subplots(1, len(functions), figsize=(16, 5))
    
    for ax, func_raw in zip(axes, functions):
        # Use a standardized difficult starting point for the curves
        dim = 4 if func_raw.__class__.__name__ == 'Woods' else 2
        start_pt = np.full(dim, -1.2) 
        
        for solver in solvers:
            
            with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f): # Mutes terminal output
                _, _, path_f = solver(func_raw, start_pt, max_iter=2000) # discard optimal point and path_x, only keep path_f for plotting
            
            # Plot Log10(f(x)). Added 1e-15 to prevent log(0) domain errors
            ax.plot(np.log10(np.array(path_f) + 1e-15), 
                    label=solver.__name__, linewidth=2)

        ax.set_title(f"{func_raw.__class__.__name__}", fontweight='bold')
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Log10 Function Value")
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend()

    plt.suptitle("Algorithmic Convergence Rates (Standard Start Point)", fontsize=16)
    plt.tight_layout()
    plt.show()

def run_comprehensive_dolan_more(solvers, functions, runs_per_func=50):
    """
    Executes a large-scale statistical benchmark by running all solvers across 
    multiple randomized starting points and aggregating performance data.

    Benchmark Methodology:
    ----------------------
    1. Randomized Starts: For each function, 'n' starting points are generated 
       uniformly between [-5.0, 5.0].
    2. Data Aggregation: The script records CPU time, iteration counts, and 
       total function calls (via TrackedFunction) for every run.
    3. Error Handling: Failed optimizations or numerical explosions (NaN) are 
       assigned a value of 'inf', representing a failure to solve the problem.
    """

    # solvers: list of  optimizer functions
    # functions: list of  objective functions
    # runs_per_func: default 30 random starts per function

    print(f"\n--- Initiating Comprehensive Dolan-Moré Benchmark ---")
    
    total_problems = len(functions) * runs_per_func #3*30 = 90
    print(f"Total topological problems to solve: {total_problems} per algorithm...")
    
    # Storage matrices: shape (total_problems, num_solvers) ((3*30,3)  = (90,3) in this case)
    #initialize with np.inf to represent unsolved problems by default

    times = np.full((total_problems, len(solvers)), np.inf)
    iters = np.full((total_problems, len(solvers)), np.inf)
    calls = np.full((total_problems, len(solvers)), np.inf)
    
    problem_idx = 0
    

    # Outer loop: Different function topologies (variety)
    # Middle loop: Different starting points 
    # Inner loop: Different algorithms (comparison)

    for func_raw in functions:
        dim = 4 if func_raw.__class__.__name__ == 'Woods' else 2
        starts = np.random.uniform(-5.0, 5.0, (runs_per_func, dim))
    
        for start_point in starts:
            for s_idx, solver in enumerate(solvers):
                # Wrap the function to reset the counter to 0 for this run
                tracked_target = TrackedFunction(func_raw)
                
                start_time = time.perf_counter()
                try:
                    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
                        x_opt, path_x, path_f = solver(tracked_target, start_point, max_iter=5000)
                    
                    # Verify convergence using function-aware tolerance
                    func_name = func_raw.__class__.__name__
                    true_min = GLOBAL_MINIMA.get(func_name, 0.0)
                    
                    # Use relative tolerance (within 20% of global minimum)
                    # For zero-minimum functions: absolute tolerance of 0.1
                    if true_min == 0.0:
                        converged = path_f[-1] < 0.1
                    else:
                        relative_error = abs(path_f[-1] - true_min) / abs(true_min)
                        converged = relative_error < 0.2  # Within 20% of true minimum
                    
                    if converged:
                        times[problem_idx, s_idx] = time.perf_counter() - start_time
                        iters[problem_idx, s_idx] = len(path_f) - 1
                        calls[problem_idx, s_idx] = tracked_target.calls
                except Exception:
                    pass # Leave as np.inf if crashed
                    
            problem_idx += 1

    #  Extracting Function Names for the Titles 
    solver_names = [s.__name__ for s in solvers]
    
    # Dynamically build a string of the test function names
    func_names_str = ", ".join([f.__class__.__name__ for f in functions])
    
    # Inject the function names into the titles using a newline character for a clean subtitle
    title_cpu = f"Performance Profile (CPU Time)\nAggregated Topologies: {func_names_str}"
    title_iters = f"Performance Profile (Iterations)\nAggregated Topologies: {func_names_str}"
    title_calls = f"Performance Profile (Total Function/Gradient Calls)\nAggregated Topologies: {func_names_str}"

    plot_single_profile(times, solver_names, title_cpu)
    plot_single_profile(iters, solver_names, title_iters)
    plot_single_profile(calls, solver_names, title_calls)

def plot_single_profile(data_matrix, solver_names, title):
    """
    Calculates and renders a Dolan-Moré Performance Profile.

    What is a Dolan-Moré Profile?
    -----------------------------
    A performance profile is a Cumulative Distribution Function (CDF) used to 
    compare the relative performance of solvers.

    How to Interpret the Graphs:
    ----------------------------
    * Performance Ratio (tau) [X-axis]: Represents the multiple of the best 
      solver's performance. (e.g., tau=2 means the solver was twice as slow 
      as the fastest solver for that problem).
    * Efficiency (rho) [Y-axis]: The fraction of the total problem set solved 
      within a specific tau.
    * Key Insights:
        - The Y-intercept (rho at tau=1) shows how often a solver was the 
          absolute best performer in the set.
        - The point where a line hits 1.0 shows the threshold at which the 
          solver successfully completed all problems.
        - Lines that stay higher and further to the left represent the most 
          robust and efficient algorithms.
    """
    min_vals = np.min(data_matrix, axis=1)
    
    # Filter out problems where EVERY solver failed
    valid = min_vals < np.inf 
    data_matrix = data_matrix[valid]
    min_vals = min_vals[valid].reshape(-1, 1)
    
    if len(data_matrix) == 0:
        print(f"Skipping {title}: All solvers failed all problems.")
        return
        
    ratios = data_matrix / min_vals
    taus = np.logspace(0, 3, 500) # Evenly spaced on log scale in [10^0, 10^3]
    rho = np.zeros((len(taus), len(solver_names)))
    
    #CDF computation: For each solver, calculate the fraction of problems solved within each tau threshold
    for s_idx in range(len(solver_names)):
        for t_idx, tau in enumerate(taus):
            rho[t_idx, s_idx] = np.sum(ratios[:, s_idx] <= tau) / len(data_matrix)
            
    plt.figure(figsize=(10, 6)) # Slightly wider to accommodate the subtitle
    colors, styles = ['#1f77b4', '#ff7f0e', '#2ca02c'], ['-', '--', '-.']
    
    for s_idx in range(len(solver_names)):
        plt.plot(taus, rho[:, s_idx], color=colors[s_idx], linestyle=styles[s_idx], 
                 linewidth=2.5, label=solver_names[s_idx])
                 
    plt.xscale('log')
    plt.xlabel('Performance Ratio $\\tau$ (Log Scale)')
    plt.ylabel('Fraction of Problems Solved $\\rho_s(\\tau)$')
    plt.title(title, fontweight='bold', fontsize=12) 
    plt.legend(loc='lower right')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_functions = [Rosenbrock(), Woods(), Eason_Fenton()]
    test_solvers = [steepest_descent, newton, bfgs]
    
    # 1. Show the mathematical convergence properties
    plot_convergence_rates(test_solvers, test_functions)
    
    # 2. Show the statistical robustness (150 combined tests)
    run_comprehensive_dolan_more(test_solvers, test_functions, runs_per_func=30)