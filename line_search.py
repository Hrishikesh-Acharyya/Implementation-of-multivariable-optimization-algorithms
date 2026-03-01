import numpy as np

def backtracking_line_search(func_obj, x, grad, direction, alpha_init=1.0, rho=0.9, beta=1e-4):
    """
    Calculates the step size alpha using the Armijo condition (Backtracking Line Search).
    """
    alpha = alpha_init
    f_current = func_obj.evaluate(x)
    
    # Armijo condition loop
    while func_obj.evaluate(x + alpha * direction) > f_current + beta * alpha * np.dot(grad, direction):
        alpha *= rho
        
        # Safety check to prevent infinite loops (exactly as defined in your files)
        if alpha < 1e-8:
            print(f"Warning: Alpha became very small ({alpha}), stopping backtracking")
            return 0.0
            
            
    return alpha

def strong_wolfe_line_search(func_obj, x, grad, direction, alpha_init=1.0, c1=1e-4, c2=0.9, max_iter=20):
    """
    Attempts to find a step size alpha that satisfies the Strong Wolfe Conditions:
    1. Armijo (Sufficient Decrease)
    2. Curvature (Slope reduction)
    """
    alpha = alpha_init
    f_val = func_obj.evaluate(x)
    dir_deriv = np.dot(grad, direction)
    
    for _ in range(max_iter):
        x_new = x + alpha * direction
        f_new = func_obj.evaluate(x_new)
        
        # 1. Check Armijo Condition (Is the step too big?)
        if f_new > f_val + c1 * alpha * dir_deriv:
            alpha *= 0.5  # Shrink alpha and try again
            continue
            
        # 2. Check Curvature Condition (Is the step too small?)
        grad_new = func_obj.gradient(x_new)
        new_dir_deriv = np.dot(grad_new, direction)
        
        # We use absolute values for the "Strong" Wolfe condition
        if abs(new_dir_deriv) > c2 * abs(dir_deriv):
            # In a basic implementation, if we fail curvature, we expand or adjust.
            # To prevent infinite loops in non-convex space, we will accept the 
            # safe Armijo step, but in a production solver, we would 'zoom' here.
            break 
            
        # If we pass both Armijo AND Curvature, we have the perfect step!
        return alpha
        
    # Fallback to the safest calculated alpha if max iterations hit
    if alpha < 1e-8:
        print(f"Warning: Alpha became very small ({alpha}), stopping line search.")
        return 0.0
        
    return alpha

def exact_line_search(func_obj, x, grad, direction):
    """
    Calculates the optimal step size alpha using the Exact Line Search formula.
    Universal implementation that works for Steepest Descent, Newton's, and Quasi-Newton.
    """
    hess = func_obj.hessian(x)
    
    # Exact line search formula: alpha = -(grad^T * direction) / (direction^T * H * direction)
    numerator = -np.dot(grad, direction)
    denominator = np.dot(direction, hess @ direction)
    
    if denominator == 0:
        return 0.0
        
    alpha = numerator / denominator
    return alpha