"""
Script to empirically find the minimum value of EasonFenton function
"""
import numpy as np
from objective_functions import EasonFenton
from scipy.optimize import minimize

def find_eason_fenton_minimum():
    """Find global minimum of EasonFenton through multiple random restarts"""
    func = EasonFenton()
    
    best_value = np.inf
    best_point = None
    
    # Try multiple random starting points
    for trial in range(50):
        x0 = np.random.uniform(-5, 5, 2)  # Random start in [-5, 5]
        
        # Use SciPy's global optimizer (not our algorithms)
        result = minimize(
            lambda x: func.evaluate(x),
            x0,
            method='BFGS',  # Use our implementation's reference
            jac=lambda x: func.gradient(x)
        )
        
        if result.fun < best_value:
            best_value = result.fun
            best_point = result.x
    
    print(f"EasonFenton Global Minimum (empirically found):")
    print(f"  Point: x1={best_point[0]:.6f}, x2={best_point[1]:.6f}")
    print(f"  Value: f(x) = {best_value:.6f}")
    return best_value, best_point

if __name__ == "__main__":
    min_val, min_point = find_eason_fenton_minimum()
