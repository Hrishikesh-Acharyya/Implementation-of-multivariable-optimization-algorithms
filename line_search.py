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