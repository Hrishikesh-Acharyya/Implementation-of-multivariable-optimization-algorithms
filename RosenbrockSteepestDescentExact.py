import numpy as np
import matplotlib.pyplot as plt
import math

# Note: For exact line search, we don't need user input for alpha
# The algorithm calculates the optimal alpha at each iteration
starting_point = np.zeros(2)
starting_point[0] = float(input("Enter the starting point (x1): "))
starting_point[1] = float(input("Enter the starting point (x2): "))

def f(x1, x2):
    return 100*(x2 - x1**2)**2 + (1 - x1)**2

def gradient(x1,x2):
    return np.array([-400*x1*(x2 - x1**2) - 2*(1 - x1), 200*(x2 - x1**2)])

def hessian(x1,x2):
    return np.array([[-400*(x2 - x1**2) + 800*x1**2 + 2, -400*x1],
                     [-400*x1, 200]])

def checkAlpha(x1,x2):
    grad = gradient(x1,x2)
    
    # Check if gradient is zero (we're at the optimum)
    grad_norm = np.linalg.norm(grad)
    if grad_norm < 1e-15:
        return 0.0  # No step needed when at optimum
    
    hess = hessian(x1,x2)
    # Exact line search formula: alpha = (g^T * g) / (g^T * H * g)
    numerator = np.dot(grad, grad)
    denominator = np.dot(grad, np.dot(hess, grad))
    alpha = numerator / denominator
    return alpha

def steepest_descent(starting_point, tolerance = 1e-8):
  x0 = starting_point
  count = 0

  path_x1 = [x0[0]]  # x1 coordinates
  path_x2 = [x0[1]]  # x2 coordinates
  path_f = [f(x0[0], x0[1])]  # function values
  
  while(True):
     count += 1
     grad = gradient(x0[0],x0[1])

     
     # Check if gradient is essentially zero (convergence)
     grad_norm = np.linalg.norm(grad)
     if grad_norm < 1e-15:
         print(f"Converged after {count-1} iterations! (Gradient norm: {grad_norm:.2e})")
         print(f"Final point: x = [{x0[0]:.15f}, {x0[1]:.15f}]")
         print(f"Final function value: f(x) = {f(x0[0], x0[1]):.15f}")
         return x0,path_x1,path_x2,path_f
     
     alpha = checkAlpha(x0[0], x0[1])
     x1 = x0 - alpha*grad

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
         return x1,path_x1,path_x2,path_f

     print(f"Iteration {count}: x0 = [{x0[0]:.15f}, {x0[1]:.15f}], f(x0) = {f_x0:.12f}, alpha = {alpha:.15f}, deviation: {deviation:.15f}")
     x0 = x1
     
     # Add safety check to prevent infinite loops
     if count >= 100000:
         print("Maximum iterations (100000) reached!")
         print(f"Current point: x5 = [{x1[0]:.15f}, {x1[1]:.15f}]")  # Better formatting
         print(f"Current function value: f(x) = {f_x1:.15f}")
         print(f"Final deviation: {deviation:.15f}")  # Show final deviation
         return x1,path_x1,path_x2,path_f

# Run the optimization
result,path_x1,path_x2,path_f  = steepest_descent(starting_point)
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

fig = plt.figure(figsize=(18, 5))

# Plot 1: 3D Surface
print("Step 3a: Creating 3D surface plot...")
ax1 = fig.add_subplot(1, 2, 1, projection='3d') # adds a subplot(individual plot area = 1 row of plots 3 columns this is 1st plot)
surface = ax1.plot_surface(X1, X2, Z, cmap='viridis', alpha=1.0) # cmap is colormap, alpha is transparency
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('f(x1, x2)')
ax1.set_title('3D Surface: f(x1,x2) = 100x1² + x2²')

#add optimization path

ax1.plot(path_x1, path_x2, path_f,'r-', linewidth=3, alpha=0.8, label='Optimization Path')
# Plot each iteration point with numbers
for i in range(len(path_x1)):
    ax1.scatter(path_x1[i], path_x2[i], path_f[i], s=80, c='red', alpha=0.8)
    ax1.text(path_x1[i], path_x2[i], path_f[i], f'{i}', fontsize=10, ha='center')

ax1.scatter(path_x1[0],path_x2[0],path_f[0],color = "green",s = 150,label = "Start", marker='s')
ax1.scatter(path_x1[-1],path_x2[-1],path_f[-1],s = 150,color = "red",label= "End", marker='*')
ax1.legend()

# Plot 2: Contour plot (2D view from above)
print("Step 3b: Creating contour plot...")
ax2 = fig.add_subplot(1, 2, 2)
contour = ax2.contour(X1, X2, Z, levels=15)
ax2.clabel(contour, inline=True, fontsize=8)
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_title('Contour Plot (Top View)')
ax2.grid(True, alpha=0.3)

ax2.plot(path_x1, path_x2,'r-', linewidth=3, alpha=0.8, label='Optimization Path')
# Plot each iteration point with numbers
for i in range(len(path_x1)):
    ax2.scatter(path_x1[i], path_x2[i], s=80, c='red', alpha=0.8)
    ax2.text(path_x1[i], path_x2[i], f'{i}', fontsize=10, ha='center', va='bottom')

ax2.scatter(path_x1[0],path_x2[0],color = "green",s = 150,label = "Start", marker='s')
ax2.scatter(path_x1[-1],path_x2[-1],color = "red",s = 150,label= "End", marker='*')

# Mark the minimum point
ax2.plot(0, 0, 'r*', markersize=15, label='Global Minimum (0,0)')
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


