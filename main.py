import numpy as np
import sys
from objective_functions import Rosenbrock, Eason_Fenton, Woods, Quadratic
from optimization import steepest_descent, newton, bfgs
import plotly.graph_objects as go
import matplotlib.pyplot as plt

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

def plot_plotly_3d(func_obj, path_x, path_f, title="Optimization Path"):
    """
    Renders an interactive 3D WebGL surface plot of the objective function 
    topology and overlays the optimization trajectory in the default web browser

    Visual Components:
    ------------------
    * Surface Mesh: A high-resolution (100x100) grid dynamically sized to the 
      algorithm's travel range.
    * Trajectory: A thick red line connecting each iterative step (path_x).
    * Start Point: Indicated by a large green circle.
    * Convergence Point: Indicated by a gold diamond (Global/Local Minimum).

    Technical Features:
    -------------------
    * Z-Clipping: Automatically caps the vertical axis at 2x the maximum path height 
      to prevent topological singularities (like those in Eason-Fenton) from 
      distorting the plot scale.
    * Meshgrid Evaluation: Dynamically computes Z-heights across the mesh using 
      the provided func_obj.evaluate() method.
    """
    #  Define the spatial boundaries for the mesh grid
    # We dynamically pad the grid based on where the algorithm traveled
    x_min, x_max = min(path_x[:, 0]) - 1, max(path_x[:, 0]) + 1
    y_min, y_max = min(path_x[:, 1]) - 1, max(path_x[:, 1]) + 1
    
    # Create the high-resolution mesh grid
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate the Z heights for the surface
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func_obj.evaluate([X[i, j], Y[i, j]])
            
    # Clip extreme Z values for functions like Eason-Fenton so the plot doesn't distort
    z_max_plot = max(path_f) * 2  # Cap the height at 2x the highest point in our path
    Z = np.clip(Z, a_min=None, a_max=z_max_plot)

    # Build the Plotly Figure
    fig = go.Figure()

    # Add the 3D Topographical Surface
    fig.add_trace(go.Surface(
        z=Z, x=X, y=Y, 
        colorscale='Viridis', 
        opacity=0.8,
        name='Objective Topology',
        showscale=False
    ))

    # Add the Optimization Path (Thick Red Line)
    fig.add_trace(go.Scatter3d(
        x=path_x[:, 0], y=path_x[:, 1], z=path_f,
        mode='lines+markers',
        line=dict(color='red', width=6),
        marker=dict(size=4, color='red'),
        name='Algorithm Path'
    ))

    # Add a massive Green Dot for the Start Point
    fig.add_trace(go.Scatter3d(
        x=[path_x[0, 0]], y=[path_x[0, 1]], z=[path_f[0]],
        mode='markers',
        marker=dict(size=8, color='green', symbol='circle'),
        name='Start'
    ))

    # Add a massive Gold Star for the End Point (Minimum)
    fig.add_trace(go.Scatter3d(
        x=[path_x[-1, 0]], y=[path_x[-1, 1]], z=[path_f[-1]],
        mode='markers',
        marker=dict(size=10, color='gold', symbol='diamond'),
        name='Convergence Point'
    ))

    # 4. Update the layout aesthetics
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        scene=dict(
            xaxis_title='X1',
            yaxis_title='X2',
            zaxis_title='f(X1, X2)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2) # Default viewing angle
            )
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Launch the interactive HTML viewer in the default web browser
    fig.show()

def plot_convergence(path_f, title_prefix):
    """
    Generates a 2D Log-Linear convergence graph (Function Value vs. Iterations).
    
    Purpose:
    --------
    Essential for high-dimensional functions (dim > 2) like the Wood's function. 
    It allows for the visual verification of the mathematical convergence rate 
    (Linear, Superlinear, or Quadratic) by observing the slope of the decay on 
    a logarithmic scale. Dimensionality reducing algorithms like PCA may be used in the future to visualize 
    high-dimensional trajectories, but for now, this plot serves as a critical diagnostic tool for convergence behavior.
    """
    plt.figure(figsize=(8, 5))
    plt.semilogy(path_f, 'b.-', linewidth=2)
    plt.title(f'{title_prefix} - Convergence Curve')
    plt.xlabel('Iteration Number')
    plt.ylabel('Function Value f(x) (Log Scale)')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.show()

def main_menu():

    """
    The main interactive CLI loop for the optimization suite.
    
    Logic Flow:
    -----------
    1. Function Selection: User chooses between Rosenbrock, Eason-Fenton, 
       Wood's, or Quadratic topologies.
    2. Parameter Injection: Prompts the user for domain-specific constants 
       (e.g., 'a' and 'b' for Rosenbrock).
    3. Optimizer Selection: Routes the problem to Steepest Descent, Newton, 
       or BFGS.
    4. Line Search Routing: Configures the solver to use either the 
       Backtracking (Armijo) or Exact Line Search modules.
    5. Result Synthesis: Outputs final coordinates, total iterations, and 
       the final objective value before launching the visualization engine.
    """
    while True:
        print("\n" + "="*50)
        print(" MULTIVARIABLE OPTIMIZATION ALGORITHM SUITE")
        print("="*50)
        
        # Select Function
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
            func = Eason_Fenton()
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

        # Select Optimizer
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

        # Select Line Search 
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

        #  Get Starting Point
        print(f"\nEnter Starting Point ({dim} dimensions required):")
        starting_point = []
        default_starts = [-1.2, 1.0, -3.0, -1.0] # Common defaults
        for i in range(dim):
            val = get_float_input(f"x{i+1}", default_starts[i])
            starting_point.append(val)

        # Execute Optimization
        print(f"\n[{opt_name} | {ls_name} | {func_name}]")
        print("Optimizing... please wait.")
        
        optimal_x, path_x, path_f = optimizer(func, starting_point, ls_type=ls_type)

        print("\n" + "-"*30)
        print(" RESULTS")
        print("-"*30)
        print(f"Optimal Point: {np.round(optimal_x, 6)}")
        print(f"Final Value:   {path_f[-1]:.10e}")
        print(f"Iterations:    {len(path_x) - 1}")
        
        # Visualization
        print("\nGenerating plots... (Close the plot window to continue)")
        title_prefix = f"{opt_name} on {func_name}"
        if dim == 2:
            plot_plotly_3d(func, path_x, path_f, title_prefix)
        else:
            plot_convergence(path_f, title_prefix)
            
        # Repeat or Exit
        again = input("\nWould you like to run another optimization? (y/n): ").strip().lower()
        if again != 'y':
            print("Exiting program. Goodbye!")
            break

if __name__ == "__main__":
    main_menu()