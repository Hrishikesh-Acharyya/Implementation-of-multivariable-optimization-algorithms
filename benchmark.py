import numpy as np
import matplotlib.pyplot as plt
from objective_functions import Rosenbrock, Woods, EasonFenton
from optimization import steepest_descent, newton, bfgs

def run_benchmark(func_obj, starting_point, ls_type='backtracking', function_name="Objective"):
    print(f"\n{'='*50}")
    print(f"BENCHMARKING: {function_name} Function")
    print(f"Starting Point: {starting_point}")
    print(f"Line Search: {ls_type.capitalize()}")
    print(f"{'='*50}")

    # 1. Run Steepest Descent
    print("Running Steepest Descent...")
    x_sd, _, path_f_sd = steepest_descent(func_obj, starting_point, ls_type=ls_type)
    
    # 2. Run Newton's Method
    print("Running Newton's Method...")
    x_nt, _, path_f_nt = newton(func_obj, starting_point, ls_type=ls_type)
    
    # 3. Run BFGS
    print("Running BFGS (Quasi-Newton)...")
    x_bfgs, _, path_f_bfgs = bfgs(func_obj, starting_point, ls_type=ls_type)

    # Print Summary Table
    print(f"\n{'-'*50}")
    print(f"{'Algorithm':<20} | {'Iterations':<10} | {'Final f(x)':<15}")
    print(f"{'-'*50}")
    print(f"{'Steepest Descent':<20} | {len(path_f_sd)-1:<10} | {path_f_sd[-1]:.6e}")
    print(f"{'Newton''s Method':<20} | {len(path_f_nt)-1:<10} | {path_f_nt[-1]:.6e}")
    print(f"{'BFGS':<20} | {len(path_f_bfgs)-1:<10} | {path_f_bfgs[-1]:.6e}")
    print(f"{'-'*50}")

    # Visualization: Convergence Plot (Log Scale)
    plt.figure(figsize=(10, 6))
    
    # We use semilogy because optimization errors drop exponentially
    plt.semilogy(path_f_sd, 'b-', linewidth=2, label=f'Steepest Descent ({len(path_f_sd)-1} iters)')
    plt.semilogy(path_f_nt, 'r--', linewidth=2, label=f'Newton\'s Method ({len(path_f_nt)-1} iters)')
    plt.semilogy(path_f_bfgs, 'g-.', linewidth=2, label=f'BFGS ({len(path_f_bfgs)-1} iters)')
    
    plt.title(f'Convergence Comparison on {function_name} Function\n({ls_type.capitalize()} Line Search)')
    plt.xlabel('Iteration Number')
    plt.ylabel('Function Value f(x) (Log Scale)')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test 1: 2D Rosenbrock
    rosen = Rosenbrock()
    run_benchmark(rosen, starting_point=[-1.2, 1.0], ls_type='backtracking', function_name="Rosenbrock (2D)")
    
    # Test 2: 4D Wood's Function (Uncomment to test higher dimensions!)
    # woods = Woods()
    # run_benchmark(woods, starting_point=[-3.0, -1.0, -3.0, -1.0], ls_type='backtracking', function_name="Wood's (4D)")