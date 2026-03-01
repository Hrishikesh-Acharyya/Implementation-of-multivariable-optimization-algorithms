import numpy as np

"""
This file creates the objective functions and their gradients and hessians for
use in the optimization algorithms. Each function is implemented as a class with methods to evaluate the function, compute its gradient, and compute its Hessian.
Functions used are: Rosenbrock, EasonFenton, Woods, and Quadratic. Each function has a global minimum that can be used to verify the correctness of the optimization algorithms.

"""

class Rosenbrock:
    """
    Generalized Rosenbrock function: f(x1,x2) = b*(x2 - x1^2)^2 + (a - x1)^2
    Global minimum at (a, a^2) where f(x) = 0.
    """
    def __init__(self, a=1.0, b=100.0): 
        #class constructor, initializes the parameters a and b for the Rosenbrock function
        # self stores these variables inside the specific object instance
        self.a = a
        self.b = b

    def evaluate(self, x):
        x1, x2 = x[0], x[1]
        return self.b * (x2 - x1**2)**2 + (self.a - x1)**2

    def gradient(self, x):
        x1, x2 = x[0], x[1]
        return np.array([
            -4.0 * self.b * x1 * (x2 - x1**2) - 2.0 * (self.a - x1),
            2.0 * self.b * (x2 - x1**2)
        ])

    def hessian(self, x):
        x1, x2 = x[0], x[1]
        return np.array([
            [12.0 * self.b * x1**2 - 4.0 * self.b * x2 + 2.0, -4.0 * self.b * x1],
            [-4.0 * self.b * x1, 2.0 * self.b]
        ])


class EasonFenton:
    """
    Eason and Fenton function.
    """
    def evaluate(self, x):
        x1, x2 = x[0], x[1]
        return (12 + x1**2 + (1 + x2**2)/x1**2 + (x1**2 * x2**2 + 150)/(x1 * x2)**4) / 10

    def gradient(self, x):
        x1, x2 = x[0], x[1]
        df_dx1 = ( 2*x1 - 2*(1 + x2**2)/x1**3 - 2/(x1**3 * x2**2) - 600/(x1**5 * x2**4) ) / 10
        df_dx2 = ( 2*x2 / x1**2 - 2 / (x1**2 * x2**3) - 600 / (x1**4 * x2**5) ) / 10
        return np.array([df_dx1, df_dx2])

    def hessian(self, x):
        x1, x2 = x[0], x[1]
        h11 = (2 + 6*(1 + x2**2 + 1/x2**2)/x1**4 + 3000/(x1**6 * x2**4)) / 10
        h22 = (2/x1**2 + 6/(x1**2 * x2**4) + 3000/(x1**4 * x2**6)) / 10
        h12 = (-4*x2/x1**3 + 4/(x1**3 * x2**3) + 2400/(x1**5 * x2**5)) / 10 
        return np.array([[h11, h12], [h12, h22]])


class Woods:
    """
    Wood's function (4D).
    Standard benchmark for assessing convergence rates in higher dimensions.
    Global minimum at (1, 1, 1, 1) where f(x) = 0
    """
    def evaluate(self, x):
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
        return (100 * (x2 - x1**2)**2 + 
                (1 - x1)**2 + 
                90 * (x4 - x3**2)**2 + 
                (1 - x3)**2 + 
                10.1 * ((x2 - 1)**2 + (x4 - 1)**2) + 
                19.8 * (x2 - 1) * (x4 - 1))

    def gradient(self, x):
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
        df_dx1 = -400 * (x2 - x1**2) * x1 - 2 * (1 - x1)
        df_dx2 = 200 * (x2 - x1**2) + 20.2 * (x2 - 1) + 19.8 * (x4 - 1)
        df_dx3 = -360 * (x4 - x3**2) * x3 - 2 * (1 - x3)
        df_dx4 = 180 * (x4 - x3**2) + 20.2 * (x4 - 1) + 19.8 * (x2 - 1)
        return np.array([df_dx1, df_dx2, df_dx3, df_dx4])

    def hessian(self, x):
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
        H = np.zeros((4,4))
        
        H[0,0] = -400*(x2 - x1**2) + 800*x1**2 + 2
        H[0,1] = H[1,0] = -400*x1
        H[1,1] = 200 + 20.2
        H[1,3] = H[3,1] = 19.8
        
        H[2,2] = -360*(x4 - x3**2) + 720*x3**2 + 2
        H[2,3] = H[3,2] = -360*x3
        H[3,3] = 180 + 20.2
        return H


class Quadratic:
    """
    Simple Quadratic function: f(x1,x2) = 100*x1^2 + x2^2
    Global minimum at (0, 0)
    """
    def __init__(self,a = 1.0,b = 1.0):

        self.a = a
        self.b = b
    def evaluate(self, x):
        x1, x2 = x[0], x[1]
        return self.a * x1**2 + self.b * x2**2

    def gradient(self, x):
        x1, x2 = x[0], x[1]
        return np.array([2 * self.a * x1, 2 * self.b * x2])

    def hessian(self, x):
        # The Hessian of a simple quadratic is constant
        return np.array([[2 * self.a, 0], [0, 2 * self.b]])