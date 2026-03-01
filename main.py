import numpy as np
import matplotlib.pyplot as plt
import sys
from objective_functions import Rosenbrock, EasonFenton, Woods, Quadratic
from optimization import steepest_descent, newton, bfgs

def get_float_input(prompt, default_val):
    """Helper to get robust float inputs from the terminal."""
    user_input = input(f"{prompt} [Default: {default_val}]: ").strip()
    if not user_input:
        return default_val
    try:
        return float(user_input)
    except ValueError:
        print("Invalid input. Using default.")
        return default_val

def plot_2d_function(func_obj, path_x, path_f, title_prefix):
    """Plots 3D Surface and Contour for 2D functions."""
    x1_range = np.linspace(-3, 3, 50)
    x2_range = np.linspace(-3, 3, 50)
    X1, X2 = np.meshgrid(x1_range, x2_range)

    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = func_obj.evaluate([X1[i, j], X2[i, j]])

    fig = plt.figure(figsize=(15, 6))

    # 3D Surface
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)
    ax1.plot(path_x[:, 0], path_x[:, 1], path_f, 'r.-', linewidth=2, markersize=8, label='Path')
    ax1.scatter(path_x[0, 0], path_x[0, 1], path_f[0], color='green', s=100, label='Start')
    ax1.scatter(path_x[-1, 0], path_x[-1, 1], path_f[-1], color='red', s=100, label='End')
    ax1.set_title(f'{title_prefix} - 3D Surface')
    ax1.legend()

    # Contour
    ax2 = fig.add_subplot(1, 2, 2)
    contour = ax2.contour(X1, X2, Z, levels=30)
    ax2.plot(path_x[:, 0], path_x[:, 1], 'r.-', linewidth=2, markersize=6, label='Path')
    ax2.scatter(path_x[0, 0], path_x[0, 1], color='green', s=100, label='Start', zorder=5)
    ax2.scatter(path_x[-1, 0], path_x[-1, 1], color='red', s=100, label='End', zorder=5)
    ax2.set_title(f'{title_prefix} - Contour')
    ax2.legend()

    plt.tight_layout()
    plt.show()

def plot_convergence(path_f, title_prefix):
    """Plots a convergence graph for higher dimensional functions (like Wood's)."""
    plt.figure(figsize=(8, 5))
    plt.semilogy(path_f, 'b.-', linewidth=2)
    plt.title(f'{title_prefix} - Convergence Curve')
    plt.xlabel('Iteration Number')
    plt.ylabel('Function Value f(x) (Log Scale)')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.show()

def main_menu():
    while True:
        print("\n" + "="*50)
        print(" MULTIVARIABLE OPTIMIZATION ALGORITHM SUITE")
        print("="*50)
        
        # 1. Select Function
        print("\nSelect Objective Function:")
        print("1. Rosenbrock (2D)")
        print("2. Eason-Fenton (2D)")
        print("3. Wood's Function (4D)")
        print("4. Quadratic (2D)")
        print("5. Exit")
        
        func_choice = input("Enter choice (1-5): ").strip()
        
        if func_choice == '5':
            print("Exiting program. Goodbye!")
            sys.exit()
            
        if func_choice == '1':
            print("\nConfigure Rosenbrock Parameters:")
            a = get_float_input("Enter parameter 'a'", 1.0)
            b = get_float_input("Enter parameter 'b'", 100.0)
            func = Rosenbrock(a=a, b=b)
            func_name = f"Rosenbrock (a={a}, b={b})"
            dim = 2
        elif func_choice == '2':
            func = EasonFenton()
            func_name = "Eason-Fenton"
            dim = 2
        elif func_choice == '3':
            func = Woods()
            func_name = "Wood's Function"
            dim = 4
        elif func_choice == '4':
            print("\nConfigure Quadratic Parameters:")
            a = get_float_input("Enter parameter 'a'", 100.0)
            b = get_float_input("Enter parameter 'b'", 1.0)
            func = Quadratic(a=a, b=b) 
            func_name = f"Quadratic (a={a}, b={b})"
            dim = 2
        else:
            print("Invalid choice. Try again.")
            continue # This continue correctly restarts the main loop for function selection

        # 2. Select Optimizer (Wrapped in its own loop)
        while True:
            print("\nSelect Optimization Method:")
            print("1. Steepest Descent")
            print("2. Newton's Method")
            print("3. Quasi-Newton (BFGS)")
            
            opt_choice = input("Enter choice (1-3): ").strip()
            opt_map = {'1': (steepest_descent, "Steepest Descent"), 
                       '2': (newton, "Newton's Method"), 
                       '3': (bfgs, "BFGS")}
            
            if opt_choice in opt_map:
                optimizer, opt_name = opt_map[opt_choice]
                break # Valid choice, break out of the optimizer selection loop
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")

        # 3. Select Line Search (Wrapped in its own loop)
        while True:
            print("\nSelect Line Search Method:")
            print("1. Backtracking (Armijo)")
            print("2. Exact Line Search")
            
            ls_choice = input("Enter choice (1-2): ").strip()
            if ls_choice in ['1', '2']:
                ls_type = 'exact' if ls_choice == '2' else 'backtracking'
                ls_name = "Exact" if ls_choice == '2' else "Backtracking"
                break # Valid choice, break out of the line search selection loop
            else:
                print("Invalid choice. Please enter 1 or 2.")

        # 4. Get Starting Point
        print(f"\nEnter Starting Point ({dim} dimensions required):")
        starting_point = []
        default_starts = [-1.2, 1.0, -3.0, -1.0] # Common defaults
        for i in range(dim):
            val = get_float_input(f"x{i+1}", default_starts[i])
            starting_point.append(val)

        # 5. Execute Optimization
        print(f"\n[{opt_name} | {ls_name} | {func_name}]")
        print("Optimizing... please wait.")
        
        optimal_x, path_x, path_f = optimizer(func, starting_point, ls_type=ls_type)

        print("\n" + "-"*30)
        print(" RESULTS")
        print("-"*30)
        print(f"Optimal Point: {np.round(optimal_x, 6)}")
        print(f"Final Value:   {path_f[-1]:.10e}")
        print(f"Iterations:    {len(path_x) - 1}")
        
        # 6. Visualization
        print("\nGenerating plots... (Close the plot window to continue)")
        title_prefix = f"{opt_name} on {func_name}"
        if dim == 2:
            plot_2d_function(func, path_x, path_f, title_prefix)
        else:
            plot_convergence(path_f, title_prefix)
            
        # 7. Repeat or Exit
        again = input("\nWould you like to run another optimization? (y/n): ").strip().lower()
        if again != 'y':
            print("Exiting program. Goodbye!")
            break

if __name__ == "__main__":
    main_menu()
    input("\nPress Enter to close...")  # Keeps window open after completion