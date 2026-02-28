import numpy as np
import matplotlib.pyplot as plt
import math

# Note: For backtracking line search, we don't need user input for alpha
# The algorithm finds a suitable alpha using the Armijo condition
starting_point = np.zeros(4)
starting_point[0] = float(input("Enter the starting point (x1): "))
starting_point[1] = float(input("Enter the starting point (x2): "))
starting_point[2] = float(input("Enter the starting point (x3): "))
starting_point[3] = float(input("Enter the starting point (x4): "))

def f(x1, x2, x3, x4):
    return (
        100 * (x1**2 - x2)**2 +
        (x1 - 1)**2 +
        (x3 - 1)**2 +
        90 * (x3**2 - x4)**2 +
        10.1 * ((x2 - 1)**2 + (x4 - 1)**2) +
        19.8 * (x2 - 1) * (x4 - 1)
    )

def gradient(x1, x2, x3, x4):
    df_dx1 = 400 * x1 * (x1**2 - x2) + 2 * (x1 - 1)
    df_dx2 = -200 * (x1**2 - x2) + 20.2 * (x2 - 1) + 19.8 * (x4 - 1)
    df_dx3 = 2 * (x3 - 1) + 360 * x3 * (x3**2 - x4)
    df_dx4 = -180 * (x3**2 - x4) + 20.2 * (x4 - 1) + 19.8 * (x2 - 1)
    return np.array([df_dx1, df_dx2, df_dx3, df_dx4])

def hessian(x1, x2, x3, x4):
    H = np.zeros((4, 4))
    H[0, 0] = 1200 * x1**2 - 400 * x2 + 2
    H[0, 1] = -400 * x1
    H[1, 0] = -400 * x1
    H[1, 1] = 220.2
    H[1, 3] = 19.8
    H[3, 1] = 19.8
    H[2, 2] = 1080 * x3**2 - 360 * x4 + 2
    H[2, 3] = -360 * x3
    H[3, 2] = -360 * x3
    H[3, 3] = 200.2
    return H

def hessianInverse(x1,x2,x3,x4):
    return np.linalg.inv(hessian(x1, x2, x3, x4))

def calcAlpha(x):
    beta = 1e-4
    rho = 0.9
    alpha = 1.0
    grad = gradient(x[0], x[1], x[2], x[3])
    hessInv = hessianInverse(x[0], x[1], x[2], x[3])
    direction = -hessInv @ grad
    while f(*(x + alpha * direction)) > f(x[0], x[1], x[2], x[3]) + beta * alpha * np.dot(grad, direction):
        alpha *= rho
        if alpha < 1e-8:
            print(f"Warning: Alpha became very small ({alpha}), stopping backtracking")
            break
    return alpha

def newton(starting_point, tolerance = 1e-8):
  x0 = starting_point
  count = 0
  
  # Store the optimization path
  path_x1 = [x0[0]]  # x1 coordinates
  path_x2 = [x0[1]]  # x2 coordinates
  path_x3 = [x0[2]]  # x3 coordinates
  path_x4 = [x0[3]]  # x4 coordinates
  path_f = [f(x0[0], x0[1], x0[2], x0[3])]  # function values
  
  while(True):
     count += 1
     grad = gradient(x0[0],x0[1],x0[2],x0[3])
     alpha = calcAlpha(x0)
     hessInv = hessianInverse(x0[0], x0[1], x0[2], x0[3])

     x1 = x0 - alpha*hessInv@grad

     # Store the new point in the path
     path_x1.append(x1[0])
     path_x2.append(x1[1])
     path_x3.append(x1[2])
     path_x4.append(x1[3])
     path_f.append(f(x1[0], x1[1], x1[2], x1[3]))
     
     # Calculate actual values for better display
     f_x0 = f(x0[0], x0[1], x0[2], x0[3])
     f_x1 = f(x1[0], x1[1], x1[2], x1[3])
     deviation = abs(f_x1 - f_x0)
     
     if deviation <= tolerance:
         print(f"Converged after {count} iterations!")
         print(f"Final point: x = [{x1[0]:.15f}, {x1[1]:.15f}, {x1[2]:.15f}, {x1[3]:.15f}]")
         print(f"Final function value: f(x) = {f_x1:.15f}")
         print(f"Final deviation: {deviation:.15f}")
         return x1, path_x1, path_x2, path_x3, path_x4, path_f

     print(f"Iteration {count}: x0 = [{x0[0]:.15f}, {x0[1]:.15f}], f(x0) = {f_x0:.12f}, alpha = {alpha:.15f}, deviation: {deviation:.15f}")
     x0 = x1
     
     # Add safety check to prevent infinite loops
     if count >= 100000:
         print("Maximum iterations (100000) reached!")
         print(f"Current point: x = [{x1[0]:.15f}, {x1[1]:.15f}]")  # Better formatting
         print(f"Current function value: f(x) = {f_x1:.15f}")
         print(f"Final deviation: {deviation:.15f}")  # Show final deviation
         return x1, path_x1, path_x2, path_x3, path_x4, path_f

# Run the optimization
result, path_x1, path_x2, path_x3, path_x4, path_f = newton(starting_point)
print(f"\nOptimization complete!")
print(f"Optimal point: {result}")
print(f"Optimal value: {f(result[0], result[1], result[2], result[3])}")
print(f"Total path length: {len(path_x1)} points")


# Step 1: Create a grid of points
print("Step 1: Creating a grid of (x1, x2) points...")
x1_range = np.linspace(-5, 5, 50)  # 50 points from -5 to 5
x2_range = np.linspace(-5, 5, 50)  # 50 points from -5 to 5

# Create meshgrid - this creates all combinations of x1 and x2
X1, X2 = np.meshgrid(x1_range, x2_range)
print(f"Grid shape: {X1.shape} (50x50 = 2500 points)")

# Step 2: Calculate function values at all grid points
print("Step 2: Calculating function values...")
Z = f(X1, X2, 1.0, 1.0)  # Fix x3 and x4 for visualization
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
ax1.set_zlabel('f(x1, x2, 1, 1)')
ax1.set_title('3D Surface: Wood\'s function (x3=1, x4=1)')

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
ax2.set_title('Contour Plot (x3=1, x4=1) with Optimization Path')
ax2.grid(True, alpha=0.3)

# Add optimization path to contour plot
ax2.plot(path_x1, path_x2, 'r.-', linewidth=2, markersize=6, label='Optimization Path')
ax2.scatter(path_x1[0], path_x2[0], color='green', s=100, label='Start', zorder=5)
ax2.scatter(path_x1[-1], path_x2[-1], color='red', s=100, label='End', zorder=5)
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

