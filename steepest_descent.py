import numpy as np
import matplotlib.pyplot as plt
import math

alpha = float(input("Enter the step size (alpha): "))
starting_point = np.zeros(2)
starting_point[0] = float(input("Enter the starting point (x1): "))
starting_point[1] = float(input("Enter the starting point (x2): "))

def f(x1, x2):
    return x1**2 + x2**2

def gradient(x1,x2):
    return np.array([2*x1,2*x2])

def steepest_descent(starting_point, alpha, tolerance = 1e-8):
  x0 = starting_point
  count = 0
  while(True):
     count += 1
     grad = gradient(x0[0],x0[1])
     x1 = x0 - alpha*grad
     
     # Calculate actual values for better display
     f_x0 = f(x0[0], x0[1])
     f_x1 = f(x1[0], x1[1])
     deviation = abs(f_x1 - f_x0)
     
     if deviation <= tolerance:
         print(f"Converged after {count} iterations!")
         print(f"Final point: x = [{x1[0]:.15f}, {x1[1]:.15f}]")
         print(f"Final function value: f(x) = {f_x1:.15f}")
         print(f"Final deviation: {deviation:.15f}")
         return x1
     
     print(f"Iteration {count}: x0 = [{x0[0]:.15f}, {x0[1]:.15f}], f(x0) = {f_x0:.12f}, deviation: {deviation:.15f}")
     x0 = x1
     
     # Add safety check to prevent infinite loops
     if count >= 1000000:  # Changed from > to >=
         print("Maximum iterations (1000000) reached!")
         print(f"Current point: x = [{x1[0]:.15f}, {x1[1]:.15f}]")  # Better formatting
         print(f"Current function value: f(x) = {f_x1:.15f}")
         print(f"Final deviation: {deviation:.15f}")  # Show final deviation
         return x1

# Run the optimization
result = steepest_descent(starting_point, alpha)
print(f"\nOptimization complete!")
print(f"Optimal point: {result}")
print(f"Optimal value: {f(result[0], result[1])}")


# Step 1: Create a grid of points
print("Step 1: Creating a grid of (x1, x2) points...")
x1_range = np.linspace(-5, 5, 50)  # 50 points from -5 to 5
x2_range = np.linspace(-5, 5, 50)  # 50 points from -5 to 5

# Create meshgrid - this creates all combinations of x1 and x2
X1, X2 = np.meshgrid(x1_range, x2_range)
print(f"Grid shape: {X1.shape} (50x50 = 2500 points)")

# Step 2: Calculate function values at all grid points
print("Step 2: Calculating function values...")
Z = f(X1, X2)  # This calculates f(x1,x2) for all points
print(f"Function values range from {Z.min():.2f} to {Z.max():.2f}")

# Step 3: Create different types of plots
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(15, 5))

# Plot 1: 3D Surface
print("Step 3a: Creating 3D surface plot...")
ax1 = fig.add_subplot(1, 2, 1, projection='3d') # adds a subplot(individual plot area = 1 row of plots 3 columns this is 1st plot)
surface = ax1.plot_surface(X1, X2, Z, cmap='viridis', alpha=1.0) # cmap is colormap, alpha is transparency
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('f(x1, x2)')
ax1.set_title('3D Surface: f(x1,x2) = x1² + x2²')

# Plot 2: Contour plot (2D view from above)
print("Step 3b: Creating contour plot...")
ax2 = fig.add_subplot(1, 2, 2)
contour = ax2.contour(X1, X2, Z, levels=15)
ax2.clabel(contour, inline=True, fontsize=8)
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_title('Contour Plot (Top View)')
ax2.grid(True, alpha=0.3)

# Mark the minimum point
ax2.plot(0, 0, 'r*', markersize=15, label='Global Minimum (0,0)')
ax2.legend()


plt.tight_layout()
plt.show()

