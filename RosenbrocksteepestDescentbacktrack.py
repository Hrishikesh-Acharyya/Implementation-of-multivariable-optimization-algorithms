import numpy as np
import matplotlib.pyplot as plt
import math

# Note: For backtracking line search, we don't need user input for alpha
# The algorithm finds a suitable alpha using the Armijo condition
starting_point = np.zeros(2)
starting_point[0] = float(input("Enter the starting point (x1): "))
starting_point[1] = float(input("Enter the starting point (x2): "))

def f(x1, x2):
    """Rosenbrock function: f(x1,x2) = 100*(x2-x1²)² + (1-x1)²"""
    return 100*(x2 - x1**2)**2 + (1 - x1)**2

def gradient(x1,x2):
    """Gradient of Rosenbrock function"""
    # ∂f/∂x1 = -400*x1*(x2-x1²) - 2*(1-x1)
    # ∂f/∂x2 = 200*(x2-x1²)
    return np.array([-400*x1*(x2 - x1**2) - 2*(1-x1), 200*(x2 - x1**2)])

def calcAlpha(x):
    # Backtracking line search parameters
    beta = 0.0001  # Armijo parameter (typically between 1e-4 and 0.5)
    rho = 0.9    # Backtracking factor (typically 0.1 to 0.8)
    alpha = 1.0  # Initial step size
    
    grad = gradient(x[0], x[1])
    while (f(x[0] - alpha * grad[0], x[1] - alpha * grad[1]) > f(x[0], x[1]) - beta * alpha * np.dot(grad, grad)):
        alpha *= rho
        
        # Safety check to prevent alpha from becoming too small
        if alpha < 1e-8:
            print(f"Warning: Alpha became very small ({alpha}), stopping backtracking")
            break

    return alpha

def steepest_descent(starting_point, tolerance = 1e-8):
  x0 = starting_point
  count = 0
  
  # Store the optimization path
  path_x1 = [x0[0]]  # x1 coordinates
  path_x2 = [x0[1]]  # x2 coordinates
  path_f = [f(x0[0], x0[1])]  # function values
  
  while(True):
     count += 1
     grad = gradient(x0[0],x0[1])
     alpha = calcAlpha(x0)

     x1 = x0 - alpha*grad
     
     # Store the new point in the path
     path_x1.append(x1[0])
     path_x2.append(x1[1])
     path_f.append(f(x1[0], x1[1]))
     
     # Calculate actual values for better display
     f_x0 = f(x0[0], x0[1])
     f_x1 = f(x1[0], x1[1])
     deviation = abs(f_x1 - f_x0)
     
     if deviation <= tolerance:
         print(f"Converged after {count} iterations!")
         print(f"Final point: x = [{x1[0]:.15f}, {x1[1]:.15f}]")
         print(f"Final function value: f(x) = {f_x1:.15f}")
         print(f"Final deviation: {deviation:.15f}")
         return x1, path_x1, path_x2, path_f

     print(f"Iteration {count}: x0 = [{x0[0]:.15f}, {x0[1]:.15f}], f(x0) = {f_x0:.12f}, alpha = {alpha:.15f}, deviation: {deviation:.15f}")
     x0 = x1
     
     # Add safety check to prevent infinite loops
     if count >= 100000:
         print("Maximum iterations (100000) reached!")
         print(f"Current point: x = [{x1[0]:.15f}, {x1[1]:.15f}]")  # Better formatting
         print(f"Current function value: f(x) = {f_x1:.15f}")
         print(f"Final deviation: {deviation:.15f}")  # Show final deviation
         return x1, path_x1, path_x2, path_f

# Run the optimization
result, path_x1, path_x2, path_f = steepest_descent(starting_point)
print(f"\nOptimization complete!")
print(f"Optimal point: {result}")
print(f"Optimal value: {f(result[0], result[1])}")
print(f"Total path length: {len(path_x1)} points")


# Step 1: Create a grid of points
print("Step 1: Creating a grid of (x1, x2) points...")
x1_range = np.linspace(-2, 2, 50)  # 50 points from -2 to 2 (better for Rosenbrock)
x2_range = np.linspace(-1, 3, 50)  # 50 points from -1 to 3 (better for Rosenbrock)

# Create meshgrid - this creates all combinations of x1 and x2
X1, X2 = np.meshgrid(x1_range, x2_range)
print(f"Grid shape: {X1.shape} (50x50 = 2500 points)")

# Step 2: Calculate function values at all grid points
print("Step 2: Calculating function values...")
Z = f(X1, X2)  # This calculates f(x1,x2) for all points
print(f"Function values range from {Z.min():.2f} to {Z.max():.2f}")

# Step 3: Create different types of plots
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(18, 6))

# Plot 1: 3D Surface with optimization path
print("Step 3a: Creating 3D surface plot...")
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surface = ax1.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('f(x1, x2)')
ax1.set_title('3D Surface: Rosenbrock Function')

# Add optimization path to 3D plot
ax1.plot(path_x1, path_x2, path_f, 'r.-', linewidth=2, markersize=8, label='Optimization Path')
ax1.scatter(path_x1[0], path_x2[0], path_f[0], color='green', s=100, label='Start')
ax1.scatter(path_x1[-1], path_x2[-1], path_f[-1], color='red', s=100, label='End')
ax1.legend()

# Plot 2: Contour plot with optimization path
print("Step 3b: Creating contour plot...")
ax2 = fig.add_subplot(1, 2, 2)
contour = ax2.contour(X1, X2, Z, levels=15)
ax2.clabel(contour, inline=True, fontsize=8)
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_title('Contour Plot: Rosenbrock Function')
ax2.grid(True, alpha=0.3)

# Add optimization path to contour plot
ax2.plot(path_x1, path_x2, 'r.-', linewidth=2, markersize=6, label='Optimization Path')
ax2.scatter(path_x1[0], path_x2[0], color='green', s=100, label='Start', zorder=5)
ax2.scatter(path_x1[-1], path_x2[-1], color='red', s=100, label='End', zorder=5)

# Mark the minimum point - Rosenbrock minimum is at (1, 1)
ax2.plot(1, 1, 'r*', markersize=15, label='Global Minimum (1,1)')
ax2.legend()



plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("OPTIMIZATION PATH ANALYSIS")
print("="*60)
print(f"Starting point: [{path_x1[0]:.6f}, {path_x2[0]:.6f}]")
print(f"Final point: [{path_x1[-1]:.6f}, {path_x2[-1]:.6f}]")
print(f"Initial function value: {path_f[0]:.6f}")
print(f"Final function value: {path_f[-1]:.6f}")
print(f"Total reduction: {path_f[0] - path_f[-1]:.6f}")
print(f"Total iterations: {len(path_f) - 1}")

# Show first few and last few steps
print(f"\nFirst 5 iterations:")
for i in range(min(5, len(path_f))):
    print(f"  Iter {i}: x = [{path_x1[i]:.6f}, {path_x2[i]:.6f}], f(x) = {path_f[i]:.6f}")

if len(path_f) > 10:
    print(f"\nLast 5 iterations:")
    for i in range(max(0, len(path_f)-5), len(path_f)):
        iter_num = i
        print(f"  Iter {iter_num}: x = [{path_x1[i]:.6f}, {path_x2[i]:.6f}], f(x) = {path_f[i]:.6f}")

## OScillates between 2 values

