import numpy as np
from line_search import backtracking_line_search, exact_line_search, strong_wolfe_line_search


"""
The optimization.py file implements the three core optimization algorithms: Steepest Descent, Newton's Method, and BFGS (a Quasi-Newton method).
Each algorithm is designed to be universal, meaning it can work with any objective function that provides methods for evaluating the function value, gradient, and Hessian. 
The algorithms also include options for line search methods (backtracking and exact) to determine optimal step sizes during the optimization process.
"""

def steepest_descent(func_obj, starting_point, ls_type='backtracking', tolerance=1e-8, max_iter=100000):
    """
    Executes the classic  Steepest Descent algorithm, utilizing the negative 
    gradient as the primary search direction. 

    Parameters:
    -----------
    func_obj : object
        The objective function instance containing .evaluate() and .gradient() methods.
    starting_point : array-like
        The initial spatial coordinate to begin optimization.
    ls_type : str, optional
        The line search router. Accepts 'backtracking' (default) or 'exact'. 
    tolerance : float, optional
        The absolute deviation threshold between consecutive function evaluations. 
        If the change drops below this value, the algorithm declares convergence.
    max_iter : int, optional
        The hard limit on loop executions to prevent infinite hanging on pathological 
        topologies.

    Returns:
    --------
    tuple (x0, path_x, path_f)
        x0 : ndarray
            The final optimal coordinate found.
        path_x : ndarray
            A 2D array tracking the spatial trajectory of the algorithm for 3D plotting.
        path_f : ndarray
            A 1D array of objective function values per iteration for convergence curves.

    Algorithmic Mechanics:
    ----------------------
    * Search Direction: Strictly follows the vector of steepest local decline: 
      d_k = -nabla f(x_k).
    * Numerical Hardening: Integrates an active monitoring system for NaN/Inf 
      singularities. 
    * Dual-Convergence Triggers: Exits gracefully either when the gradient norm 
      vanishes (||nabla f(x)|| < 10^-15) or when function deviation falls 
      below the operational tolerance.

    Limitations & Architectural Shortcomings:
    -----------------------------------------
    * Hemstitching (Zig-Zagging): Because the search direction is strictly orthogonal 
      to the local contour lines, the algorithm ignores the global curvature. In 
      ill-conditioned topologies (like the Rosenbrock valley), this causes severe, 
      inefficient zig-zagging across the valley floor .
    * Convergence Rate: Mathematically restricted to a Linear convergence rate. As 
      it approaches the minimum, the gradient approaches zero, causing the algorithm 
      to take microscopic, creeping steps requiring massive iteration counts.
    """
    x0 = np.array(starting_point, dtype=float)
    count = -1
    
    # Store the optimization path for visualization/benchmarking
    path_x = [x0.copy()]
    path_f = [func_obj.evaluate(x0)]
    
    while True:
        count += 1
        grad = func_obj.gradient(x0)
        
        # Check if gradient is essentially zero (convergence)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            print(f"Steepest Descent converged after {count} iterations! (Gradient norm near zero)")
            break
            
        direction = -grad # The core steepest descent step
        
        # Select line search method
        if ls_type == 'exact':
            alpha = exact_line_search(func_obj, x0, grad, direction)
        else:
            alpha = backtracking_line_search(func_obj, x0, grad, direction)
            
        x1 = x0 + alpha * direction

        # Safety Check: NaN/Inf encountered?
        if np.any(np.isnan(x1)) or np.any(np.isinf(x1)):
            print(f"\nCRITICAL: Numerical instability (NaN/Inf) detected at iteration {count}!")
            print("The algorithm took a step into a singularity or flat region. Optimization diverged.")
            # Revert to the last safe point and break
            x1 = x0 
            break
        
        # Calculate actual values for deviation and tracking
        f_x0 = func_obj.evaluate(x0)
        f_x1 = func_obj.evaluate(x1)
        deviation = abs(f_x1 - f_x0)
        
        # Store the new point in the path
        path_x.append(x1.copy())
        path_f.append(f_x1)
        
        if deviation <= tolerance:
            print(f"Steepest Descent converged after {count + 1} iterations! (Deviation <= tolerance)")
            x0 = x1
            break
            
        x0 = x1
        
        if count >= max_iter - 1:
            print(f"Maximum iterations ({max_iter}) reached!")
            break
            
    return x0, np.array(path_x), np.array(path_f)


def newton(func_obj, starting_point, ls_type='backtracking', tolerance=1e-8, max_iter=100000):
    """
    Executes a Modified Newton's Method utilizing the exact analytical Hessian.
    Incorporates a Levenberg-Marquardt eigenvalue shift to guarantee 
    global convergence across non-convex topologies and saddle points.

    Parameters:
    -----------
    func_obj : object
        The objective function instance containing .evaluate(), .gradient(), 
        and .hessian() methods.
    starting_point : array-like
        The initial spatial coordinate to begin optimization.
    ls_type : str, optional
        The line search router. Accepts 'backtracking' (default) or 'exact'. 
    tolerance : float, optional
        The absolute deviation threshold between consecutive function evaluations.
    max_iter : int, optional
        The hard limit on loop executions.

    Returns:
    --------
    tuple (x0, path_x, path_f)
        x0 : ndarray
            The final optimal coordinate found.
        path_x : ndarray
            A 2D array tracking the spatial trajectory for 3D visualization.
        path_f : ndarray
            A 1D array of objective function values per iteration.

    Algorithmic Mechanics & The Levenberg-Marquardt Shift:
    ------------------------------------------------------
    * Pure Newton Step: In a perfectly convex bowl, the algorithm calculates the 
      search direction using d_k = -H^-1 nabla (f(x_k)), yielding a strictly 
      Quadratic convergence rate.
    * The Saddle Point Trap: In non-convex topologies (like the Wood's function), 
      the Hessian will frequently develop negative eigenvalues (representing 
      downward-curving slopes). A pure Newton method would follow this negative 
      curvature uphill, diverging away from the minimum.
    * The Shift Modification: At every step, the algorithm extracts the eigenvalues 
      of the exact Hessian. If the minimum eigenvalue (lambda_{min}) falls below 
      a safe positive threshold (1e^-5), the algorithm computes a scalar shift:
      tau = -lambda_{min} + 1e-3
      It then applies this shift to the diagonal of the Hessian:
      H_{mod} = H + tau*I where I is the identity matrix. This operation guarantees that all eigenvalues of H_{mod} are positive, ensuring a safe descent direction.
    * The Morphing Effect:  When tau is large, H_{mod} 
      becomes heavily diagonally dominant, causing the algorithm to smoothly morph 
      into a Steepest Descent trajectory to safely escape the saddle point. As 
      curvature improves, tau returns to 0, reverting to pure Newton quadratic 
      convergence.

    Limitations & Computational Cost:
    ---------------------------------
    * O(N^3) Complexity: Calculating exact eigenvalues and inverting the Hessian 
      at every single iteration is computationally devastating for high-dimensional 
      problems even though mathematically robust. 
    """
    x0 = np.array(starting_point, dtype=float)
    count = -1
    
    path_x = [x0.copy()]
    path_f = [func_obj.evaluate(x0)]
    
    while True:
        count += 1
        # Get gradient and raw Hessian
        grad = func_obj.gradient(x0)
        hess = func_obj.hessian(x0)
        
        # --- THE MODIFIED NEWTON FIX (Levenberg-Marquardt Shift) ---
        # Check for negative curvature (saddle points)
        eigenvalues = np.linalg.eigvals(hess)
        min_eigenvalue = np.min(eigenvalues)
        
        if min_eigenvalue < 1e-5:
            # Calculate shift (tau) to force positive definiteness
            tau = -min_eigenvalue + 1e-3 
            # Apply the shift: H_mod = H + tau * I
            hess = hess + tau * np.eye(len(x0))
        
        # 4. Safely invert the now positive-definite Hessian
        try:
            hess_inv = np.linalg.inv(hess)
            direction = -np.dot(hess_inv, grad)
        except np.linalg.LinAlgError:
            # Fallback if matrix is completely singular
            direction = -grad
        
        if ls_type == 'exact':
            alpha = exact_line_search(func_obj, x0, grad, direction)
        else:
            alpha = backtracking_line_search(func_obj, x0, grad, direction)
            
        x1 = x0 + alpha * direction

        # Safety Check:
        if np.any(np.isnan(x1)) or np.any(np.isinf(x1)):
            print(f"\nCRITICAL: Numerical instability (NaN/Inf) detected at iteration {count}!")
            print("The algorithm took a step into a singularity or flat region. Optimization diverged.")
            # Revert to the last safe point and break
            x1 = x0 
            break
        
        f_x0 = func_obj.evaluate(x0)
        f_x1 = func_obj.evaluate(x1)
        deviation = abs(f_x1 - f_x0)
        
        path_x.append(x1.copy())
        path_f.append(f_x1)
        
        if deviation <= tolerance:
            print(f"Newton's Method converged after {count + 1} iterations! (Deviation <= tolerance)")
            x0 = x1
            break
            
        x0 = x1
        
        if count >= max_iter - 1:
            print(f"Maximum iterations ({max_iter}) reached!")
            break
            
    return x0, np.array(path_x), np.array(path_f)



def bfgs(func_obj, starting_point, ls_type='backtracking', tolerance=1e-8, max_iter=100000):
    """
    Executes the Damped BFGS (Quasi-Newton) algorithm. 
    This method approximates the Inverse Hessian matrix iteratively, providing 
    near-Newton convergence speeds without the O(N^3) cost of exact Hessian 
    inversion or the memory overhead of analytical second derivatives.

    Parameters:
    -----------
    func_obj : object
        The objective function instance containing .evaluate() and .gradient() methods.
    starting_point : array-like
        The initial spatial coordinate to begin optimization.
    ls_type : str, optional
        The line search router. 
    tolerance : float, optional
        The absolute deviation threshold for convergence.
    max_iter : int, optional
        The hard limit on iterations.

    Returns:
    --------
    tuple (x0, path_x, path_f)
        x0 : ndarray
            The final optimal coordinate found.
        path_x : ndarray
            A 2D array tracking the spatial trajectory of the algorithm.
        path_f : ndarray
            A 1D array of objective function values per iteration.

    Algorithmic Mechanics & Powell's Damping:
    -----------------------------------------
    * Secant Equation: BFGS approximates the Hessian by solving the secant equation 
      B_{k+1}*s_k = y_k, where s_k is the change in position and y_k is the 
      change in gradient.
    * The Curvature Risk: For the BFGS update to remain "Positive Definite" (meaning 
      the algorithm always points downhill), the condition s_k^T y_k > 0 must hold. 
      In non-convex regions, this condition often fails, causing the approximated 
      Hessian to collapse or become singular.
    * Powell's Damping Modification: To prevent matrix collapse, the algorithm 
      implements a damping factor (theta). If the curvature is too low or negative, 
      the algorithm creates a synthetic gradient difference vector r_k:
      r_k = theta*y_k + (1 - theta)*B_k*s_k
      This intervention "dampens" the update, forcefully maintaining the 
      Positive Definite property of the Hessian approximation even when the 
      topology is working against it.
    * Efficiency: By updating the Inverse Hessian (H_{inv}) directly, the algorithm avoids matrix inversion entirely, 
      reducing the cost per iteration significantly compared to Newton's Method.

      Limitations & Computational Cost:
    ---------------------------------

      While this implementation utilizes np.linalg.inv to facilitate Powell's Damping 
      (an O(N^3) operation), it maintains the theoretical Superlinear Convergence
      characteristic of Quasi-Newton methods. The inversion is a numerical choice for low-dimensional
      robustness, though a production-scale version would utilize a Cholesky-based update to maintain 
      O(N^2) efficiency
    """
    x0 = np.array(starting_point, dtype=float)
    n = len(x0)
    
    # Initialize the inverse Hessian approximation as the Identity matrix
    H_inv = np.eye(n)
    
    count = 0 
    path_x = [x0.copy()]
    path_f = [func_obj.evaluate(x0)]
    
    # Calculate initial gradient
    grad0 = func_obj.gradient(x0)
    
    # Enforce max_iter to prevent infinite loops
    while count < max_iter: 
        # Check convergence (Gradient norm)
        grad_norm = np.linalg.norm(grad0)
        if grad_norm < 1e-15:
            print(f"BFGS converged after {count} iterations! (Gradient norm near zero)")
            break
            
        # The Quasi-Newton direction uses the approximated inverse Hessian
        direction = -H_inv @ grad0  
        
        # Line search options
        if ls_type == 'exact':
            alpha = exact_line_search(func_obj, x0, grad0, direction)
        elif ls_type == 'strong_wolfe':
            alpha = strong_wolfe_line_search(func_obj, x0, grad0, direction)
        else:
            alpha = backtracking_line_search(func_obj, x0, grad0, direction)
            
        # Update point
        x1 = x0 + alpha * direction

        # Safety Check: 
        if np.any(np.isnan(x1)) or np.any(np.isinf(x1)):
            print(f"\nCRITICAL: Numerical instability (NaN/Inf) detected at iteration {count}!")
            print("The algorithm took a step into a singularity or flat region. Optimization diverged.")
            x1 = x0 
            break
        
        # Calculate deviation
        f_x0 = func_obj.evaluate(x0)
        f_x1 = func_obj.evaluate(x1)
        deviation = abs(f_x1 - f_x0)
        
        # Store paths
        path_x.append(x1.copy())
        path_f.append(f_x1)
        
        # Check convergence (Step deviation)
        if deviation <= tolerance:
            print(f"BFGS converged after {count + 1} iterations! (Deviation <= tolerance)")
            x0 = x1
            break
            
        grad1 = func_obj.gradient(x1)
        
        # Calculate position and gradient differences
        s_k = x1 - x0
        y_k = grad1 - grad0

        # --- DAMPED BFGS FIX (Powell's Modification) ---
        try:
            B_k = np.linalg.inv(H_inv)
        except np.linalg.LinAlgError:
            B_k = np.eye(n) # Fallback if perfectly singular

        # Calculate dot products
        s_y = np.dot(s_k, y_k)
        Bs = B_k @ s_k
        sBs = np.dot(s_k, Bs)

        # The Damping Condition
        if s_y < 0.2 * sBs:
            theta = (0.8 * sBs) / (sBs - s_y)
        else:
            theta = 1.0

        # Create the synthetic, safe gradient difference vector (r_k)
        r_k = theta * y_k + (1 - theta) * Bs
        
        # Standard BFGS Update (using r_k instead of y_k)
        rho_den = np.dot(s_k, r_k)
        
        if rho_den > 1e-10: 
            rho = 1.0 / rho_den
            I = np.eye(n)
            term1 = I - rho * np.outer(s_k, r_k)
            term2 = I - rho * np.outer(r_k, s_k)
            H_inv = term1 @ H_inv @ term2 + rho * np.outer(s_k, s_k)

        # Prepare for next iteration
        x0 = x1
        grad0 = grad1
        count += 1
        
    # Final check if loop exhausted
    if count == max_iter:
        print(f"Maximum iterations ({max_iter}) reached!")
        
    return x0, np.array(path_x), np.array(path_f)