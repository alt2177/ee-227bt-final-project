import numpy as np
import cvxpy as cp
from scipy.stats import gamma
from typing import Callable


# Set seed 
np.random.seed(42)

# Parameters (# of samples, # of features, beta params, alpha params)
n = 100
p = 5
lambda_ = 10
a0 = 10.0  
b0 = 15.0  

# Bayesian regression model 
X = np.random.randn(n, p)  
alpha = gamma.rvs(1, loc = a0, scale = 1 / b0)
beta = np.random.normal(0, np.sqrt(1 / lambda_), p)
y = np.random.normal(X @ beta, 1 / np.sqrt(alpha))



def digamma_approx(a: float) -> float:
    """
    Approximate the digamma function using second order Taylor expansion
    """
    return cp.log(a) -(1 / 2) * cp.inv_pos(a) -(1 / 12) * (cp.inv_pos(a) ** 2)


def f_0(a: cp.Expression, c: cp.Expression, d: float, mu: cp.Expression, sigma: cp.Expression,
        X: np.ndarray = X, y: np.ndarray = y, lambda_: float = lambda_, b0: float = b0) -> cp.Expression:
    """
    Computes the f_0 component of the objective function.

    NOTE: THIS CURRENTLY CAUSES DCP ERRORS
    
    Args:
        a: A CVXPY expression related to a shape parameter.
        c: A CVXPY expression related to a scale parameter.
        d: A float representing a scaling factor.
        mu: A CVXPY expression for the mean vector.
        sigma: A CVXPY expression for the covariance matrix.
        X: A numpy array representing the data matrix.
        y: A numpy array representing the response vector.
        lambda_: Regularization parameter (default is 10.0).
        b0: Constant parameter (default is 15.0).

    Returns:
        A CVXPY expression representing the computed f_0 value.
    """
    # Define new variational parameter u
    u = d * mu

    # Number of data points
    n = X.shape[0]

    # Define the components that make up f_0
    t1 = (1 / 2) * cp.sum([d * y[i] ** 2 - 2 * (X[i].T @ u) + (X[i].T @ u) * (mu.T @ X[i])
        + d * (X[i].T @ sigma) @ X[i] for i in range(n)], axis=0)
    t2 = (lambda_ / 2) * (cp.vdot(mu.T, mu) + cp.trace(sigma))
    t3 = b0 * d

    return t1 + t2 + t3


def f_1(a: cp.Expression, c: cp.Expression, sigma: cp.Expression, a0: float = a0) -> cp.Expression:
    """
    Computes the f_1 component of the objective function without the log-gamma term.
    
    Args:
        a: A CVXPY expression related to a shape parameter.
        c: A CVXPY expression related to a scale parameter.
        sigma: A CVXPY expression for the covariance matrix.
        a0: Prior shape parameter (default is 10.0).

    Returns:
        A CVXPY expression representing the computed f_1 value.
    """
    # Number of data points (assuming globally defined n)
    n = sigma.shape[0]

    # Define components that make up f_1
    t6 = -(n / 2) * (digamma_approx(a) + cp.log(c))
    t7 = -(1 / 2) * cp.log_det(sigma)
    t8 = -a - cp.log(c)
    t9 = -digamma_approx(a) - cp.entr(a) + (1 / 2) + (1 / 12) * cp.inv_pos(a)
    t10 = -(a0 - 1) * cp.log(c)
    t11 = -(a0 - 1) * digamma_approx(a)

    return t6 + t7 + t8 + t9 + t10 + t11


def elbo(a: cp.Expression, c: cp.Expression, d: float, mu: cp.Expression, sigma: cp.Expression,
         X: np.ndarray = X, y: np.ndarray = y, lambda_: float = lambda_, a0: float = a0, b0: float = b0,
         n: int = n, p: int = p) -> cp.Expression:
    """
    Computes the Evidence Lower Bound (ELBO) for variational inference.
    Note: This is eq 10 in CRVI

    Args:
        a: A CVXPY expression related to a shape parameter.
        c: A CVXPY expression related to a scale parameter.
        d: A float representing a scaling factor.
        mu: A CVXPY expression for the mean vector.
        sigma: A CVXPY expression for the covariance matrix.
        X: A numpy array representing the data matrix.
        y: A numpy array representing the response vector.
        lambda_: Regularization parameter (default is 10.0).
        a0: Prior shape parameter (default is 10.0).
        b0: Constant parameter (default is 15.0).
        n: Number of data points (default is 100).
        p: Number of features (default is 5).

    Returns:
        A CVXPY expression representing the negative ELBO.
    """
    return -(f_0(a, c, d, mu, sigma, X, y, lambda_=lambda_, b0=b0) 
        + f_1(a, c, sigma, a0=a0))

