import numpy as np
from line_search import backtracking_line_search, exact_line_search

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
        grad = func_obj.gradient(x0)
        hess = func_obj.hessian(x0)
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            print(f"Newton's Method converged after {count} iterations! (Gradient norm near zero)")
            break
            
        # Add a safety net for singular Hessian matrices
        try:
            hess_inv = np.linalg.inv(hess)
        except np.linalg.LinAlgError:
            print("Singular Hessian matrix encountered. Stopping.")
            break
            
        direction = -hess_inv @ grad # The core Newton step
        
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
    Universal Quasi-Newton (BFGS) algorithm.
    Approximates the inverse Hessian to avoid costly matrix inversions.
    """
    x0 = np.array(starting_point, dtype=float)
    n = len(x0)
    
    # Initialize the inverse Hessian approximation as the Identity matrix
    H_inv = np.eye(n)
    
    count = -1
    path_x = [x0.copy()]
    path_f = [func_obj.evaluate(x0)]
    
    # Calculate initial gradient
    grad0 = func_obj.gradient(x0)
    
    while True:
        count += 1
        
        # Check convergence
        grad_norm = np.linalg.norm(grad0)
        if grad_norm < 1e-15:
            print(f"BFGS converged after {count} iterations! (Gradient norm near zero)")
            break
            
        # The Quasi-Newton direction uses the approximated inverse Hessian
        direction = -H_inv @ grad0
        
        # Line search
        if ls_type == 'exact':
            alpha = exact_line_search(func_obj, x0, grad0, direction)
        else:
            alpha = backtracking_line_search(func_obj, x0, grad0, direction)
            
        # Update point
        x1 = x0 + alpha * direction

        # Safety Check: Did the math explode?
        if np.any(np.isnan(x1)) or np.any(np.isinf(x1)):
            print(f"\nCRITICAL: Numerical instability (NaN/Inf) detected at iteration {count}!")
            print("The algorithm took a step into a singularity or flat region. Optimization diverged.")
            # Revert to the last safe point and break
            x1 = x0 
            break
        
        # Calculate deviation
        f_x0 = func_obj.evaluate(x0)
        f_x1 = func_obj.evaluate(x1)
        deviation = abs(f_x1 - f_x0)
        
        # Store paths
        path_x.append(x1.copy())
        path_f.append(f_x1)
        
        if deviation <= tolerance:
            print(f"BFGS converged after {count + 1} iterations! (Deviation <= tolerance)")
            x0 = x1
            break
            
        # --- The Core BFGS Update ---
        grad1 = func_obj.gradient(x1)
        
        s_k = x1 - x0
        y_k = grad1 - grad0
        
        rho_den = np.dot(y_k, s_k)
        
        # Safety check: Update H_inv only if the denominator is sufficiently large
        if rho_den > 1e-10:
            rho_k = 1.0 / rho_den
            I = np.eye(n)
            
            # BFGS Inverse Hessian Update Formula
            A = I - rho_k * np.outer(s_k, y_k)
            B = I - rho_k * np.outer(y_k, s_k)
            H_inv = A @ H_inv @ B + rho_k * np.outer(s_k, s_k)
            
        # Setup for next iteration
        x0 = x1
        grad0 = grad1
        
        if count >= max_iter - 1:
            print(f"Maximum iterations ({max_iter}) reached!")
            break
            
    return x0, np.array(path_x), np.array(path_f)