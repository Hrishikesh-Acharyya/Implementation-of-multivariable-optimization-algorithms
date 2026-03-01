import numpy as np

"""
This file implements the line search algorithms used in the optimization methods.
The line search methods implemented are:
1. Backtracking Line Search (Armijo condition)
2. Strong Wolfe Line Search
3. Exact Line Search (using the Hessian for a closed-form solution)
"""
def backtracking_line_search(func_obj, x, grad, direction, alpha_init=1.0, rho=0.9, beta=1e-4):
    """
    Calculates the step size (alpha) using strictly the Armijo Condition 
    (Sufficient Decrease), commonly known as a Backtracking Line Search.

    Parameters:
    -----------
    func_obj : object
        The objective function instance containing the .evaluate() method.
    x : ndarray
        The current spatial coordinate array.
    grad : ndarray
        The gradient vector at the current coordinate.
    direction : ndarray
        The search direction vector (e.g., descent direction).
    alpha_init : float, optional
        Starting step size. Defaults to 1.0 
    rho : float, optional
        The contraction factor. Set to 0.7 for balanced decrease of alpha. A smaller rho (e.g., 0.5) would shrink alpha more aggressively, while a larger rho (e.g., 0.9) would be more conservative in shrinking.
    beta : float, optional
        The Armijo tolerance. A small value (1e-4) ensuring the function value drops 
        by at least a tiny fraction of the expected linear decrease.

    Algorithmic Mechanics:
    ----------------------
    * The algorithm assumes the initial step (alpha_init) is optimal. 
    * It evaluates the objective function at the new proposed coordinate. 
    * If the new height is greater than the acceptable Armijo threshold, it multiplies 
      alpha by the contraction factor (rho = 0.7) and checks again.
    * This loops until a safe, downward step is found or the safety floor (1e-8) is hit.
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
    Attempts to find a step size (alpha) that satisfies the Strong Wolfe Conditions:
    1. Armijo (Sufficient Decrease)
    2. Curvature (Slope reduction)

    Parameters:
    -----------
    func_obj : object
        The objective function instance containing .evaluate() and .gradient() methods.
    x : ndarray
        The current spatial coordinate array.
    grad : ndarray
        The gradient vector at the current coordinate.
    direction : ndarray
        The search direction vector (e.g., descent direction).
    alpha_init : float, optional
        Starting step size. Defaults to 1.0 (crucial for preserving superlinear/quadratic 
        convergence in Newton and Quasi-Newton methods).
    c1 : float, optional
        Armijo tolerance. A small value (e.g., 1e-4) requires only a tiny drop in function value.
    c2 : float, optional
        Curvature tolerance. A high value (e.g., 0.9) accepts relatively steep slopes 
        as long as they are flatter than the initial slope.
    max_iter : int, optional
        Maximum backtracking halving steps before a forced exit.

    Algorithmic Mechanics:
    ----------------------
    * Contraction Factor (0.7): If the Armijo condition fails, alpha is multiplied by 0.5. 
    * Sequential Gating: Conditions are checked sequentially. 
      If the Armijo check fails, the loop continues (restarts), bypassing the computationally 
      expensive gradient calculation required for the curvature check until a safe step is found.

    Limitations & Architectural Shortcomings:
    -----------------------------------------
    * Pseudo-Wolfe Implementation: This is a simplified backtracking search, not a mathematically 
      rigorous Strong Wolfe search. It lacks a "Zoom Phase" (cubic or quadratic interpolation) 
      to actively bracket and hunt down the exact minimum.
    * The 'Break' Trap: If the Armijo condition passes but the curvature condition fails (meaning 
      the slope is still too steep), the loop breaks and returns the current alpha anyway. It does 
      not force the algorithm to satisfy the curvature rule, making the extra gradient calculation 
      a wasted CPU cycle in failure scenarios.
    """
    alpha = alpha_init
    f_val = func_obj.evaluate(x)
    dir_deriv = np.dot(grad, direction)
    
    for _ in range(max_iter):
        x_new = x + alpha * direction
        f_new = func_obj.evaluate(x_new)
        
        # 1. Check Armijo Condition (Sufficient Decrease)
        if f_new > f_val + c1 * alpha * dir_deriv:
            alpha *= 0.7  # Shrink alpha and try again
            continue
            
        # 2. Check Curvature Condition
        grad_new = func_obj.gradient(x_new)
        new_dir_deriv = np.dot(grad_new, direction)
        
        if abs(new_dir_deriv) > c2 * abs(dir_deriv):
            break 
            
        #Both armijo and curvature conditions satisfied
        return alpha
        
    # Fallback 
    if alpha < 1e-8:
        print(f"Warning: Alpha became very small ({alpha}), stopping line search.")
        return 0.0
        
    return alpha

def exact_line_search(func_obj, x, grad, direction):
    """
    Calculates the analytical step size (alpha) by minimizing a quadratic 
    approximation of the objective function along the search direction.

    Parameters:
    -----------
    func_obj : object
        The objective function instance containing the .hessian() method.
    x : ndarray
        The current spatial coordinate array.
    grad : ndarray
        The gradient vector at the current coordinate.
    direction : ndarray
        The search direction vector (e.g., descent direction).

    Algorithmic Mechanics:
    ----------------------
    * Quadratic Minimization: The formula assumes the objective function can be 
      perfectly modeled as a uniform quadratic paraboloid. 
    * Closed-Form Solution: It takes the derivative of the 1D objective function 
      ray alpha = f(x) + alpha*nabla (f^T d) + 1/2 (alpha^2 d^T H d) 
      with respect to alpha, sets it to zero, and solves algebraically:
      alpha = -(nabla f^T d)/(d^T H d)}

    Limitations & Architectural Shortcomings:
    -----------------------------------------
    * Topological Danger: This closed-form matrix formula is only mathematically 
      "exact" for pure Quadratic functions (where the Hessian never changes). On 
      highly non-convex topologies (Rosenbrock, Woods, Eason-Fenton), the local 
      Hessian does not accurately represent the global valley. Using this formula 
      there will cause severe overshoots or numerical explosions (`NaN`).
    * The Quasi-Newton Contradiction: This function requires calculating the exact, 
      analytical Hessian matrix. Calling this inside a Quasi-Newton method (BFGS) 
      completely defeats the architectural purpose of BFGS, which was specifically 
      engineered to avoid expensive O(N^2) Hessian computations.
    """
    hess = func_obj.hessian(x)
    
   
    numerator = -np.dot(grad, direction)
    denominator = np.dot(direction, hess @ direction)
    
    if denominator == 0:
        return 0.0
        
    alpha = numerator / denominator
    return alpha