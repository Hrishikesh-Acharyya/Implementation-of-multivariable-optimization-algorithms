import numpy as np
from line_search import backtracking_line_search, exact_line_search, strong_wolfe_line_search

def steepest_descent(func_obj, starting_point, ls_type='backtracking', tolerance=1e-8, max_iter=100000):
    """
    Universal Steepest Descent algorithm.
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

        # Safety Check: Did the math explode?
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
    Universal Newton's Method algorithm.
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
        # 1. Check for negative curvature (saddle points)
        eigenvalues = np.linalg.eigvals(hess)
        min_eigenvalue = np.min(eigenvalues)
        
        if min_eigenvalue < 1e-5:
            # 2. Calculate shift (tau) to force positive definiteness
            tau = -min_eigenvalue + 1e-3 
            # 3. Apply the shift: H_mod = H + tau * I
            hess = hess + tau * np.eye(len(x0))
            # Optional: Print to terminal so you can see it working!
            # print(f"  [Iter {count}] Hessian modified (tau={tau:.2e}) to escape saddle point.")
        # -----------------------------------------------------------
        
        # 4. Safely invert the now-positive-definite Hessian
        try:
            hess_inv = np.linalg.inv(hess)
            direction = -np.dot(hess_inv, grad)
        except np.linalg.LinAlgError:
            # Ultimate fallback if matrix is completely singular
            direction = -grad
        
        if ls_type == 'exact':
            alpha = exact_line_search(func_obj, x0, grad, direction)
        else:
            alpha = backtracking_line_search(func_obj, x0, grad, direction)
            
        x1 = x0 + alpha * direction

        # Safety Check: Did the math explode?
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
    Universal Quasi-Newton (BFGS) algorithm with Powell's Damping.
    Approximates the inverse Hessian to avoid costly matrix inversions.
    """
    x0 = np.array(starting_point, dtype=float)
    n = len(x0)
    
    # Initialize the inverse Hessian approximation as the Identity matrix
    H_inv = np.eye(n)
    
    count = 0  # FIX 1: Start at 0
    path_x = [x0.copy()]
    path_f = [func_obj.evaluate(x0)]
    
    # Calculate initial gradient
    grad0 = func_obj.gradient(x0)
    
    # FIX 2: Enforce max_iter to prevent infinite loops
    while count < max_iter: 
        # Check convergence (Gradient norm)
        grad_norm = np.linalg.norm(grad0)
        if grad_norm < 1e-15:
            print(f"BFGS converged after {count} iterations! (Gradient norm near zero)")
            break
            
        # The Quasi-Newton direction uses the approximated inverse Hessian
        direction = -H_inv @ grad0  # FIX 3: Using the cleaner '@' operator
        
        # Line search options
        if ls_type == 'exact':
            alpha = exact_line_search(func_obj, x0, grad0, direction)
        elif ls_type == 'strong_wolfe':
            alpha = strong_wolfe_line_search(func_obj, x0, grad0, direction)
        else:
            alpha = backtracking_line_search(func_obj, x0, grad0, direction)
            
        # Update point
        x1 = x0 + alpha * direction

        # Safety Check: Did the math explode?
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
            
        # --- The Core BFGS Update ---
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
            # FIX 3: Replaced nested np.dot with the cleaner '@' operator
            H_inv = term1 @ H_inv @ term2 + rho * np.outer(s_k, s_k)
        # -----------------------------------------------

        # Prepare for next iteration
        x0 = x1
        grad0 = grad1
        count += 1
        
    # Final check if loop exhausted
    if count == max_iter:
        print(f"Maximum iterations ({max_iter}) reached!")
        
    return x0, np.array(path_x), np.array(path_f)