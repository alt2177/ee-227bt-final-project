import numpy as np
import cvxpy as cp
from typing import Callable, List, Tuple


def cavi(initial_params: np.ndarray, objective_fn: Callable[[List[cp.Variable]], cp.Expression],
         maximize: bool = False, max_iter: int = 100, tol: float = 1e-6) -> Tuple[np.ndarray, List[float]]:
    """
    Coordinate Ascent Variational Inference (CAVI) with optional maximization or minimization.
    
    Parameters
    ----------
    initial_params : np.ndarray
        Initial values of the variational parameters.
    objective_fn : Callable[[List[cp.Variable]], cp.Expression]
        A function that, given a list of cvxpy Variables (corresponding to 
        the variational parameters), returns a cvxpy Expression for the objective.
    maximize : bool, optional
        If True, the problem is solved as a maximization. If False, minimization is used.
    max_iter : int, optional
        Maximum number of full passes (cycles) over all parameters.
    tol : float, optional
        Stopping tolerance for improvements in the objective.
    
    Returns
    -------
    params : np.ndarray
        The optimized variational parameters.
    history : List[float]
        Objective values after each full iteration.
    """
    # initialize variables and history of parameters
    n_params: int = len(initial_params)
    history: List[float] = []

    
    # Evaluate initial parameters 
    initial_obj_expr = objective_fn(*initial_params)
    initial_problem = cp.Problem(cp.Maximize(initial_obj_expr) if maximize else cp.Minimize(initial_obj_expr))
    initial_obj_val = initial_problem.value

    # append initial objective fn value
    history.append(initial_obj_val if initial_obj_val is not None else (-np.inf if maximize else np.inf))

    # set initial parameters before starting optimization
    params = initial_params
    
    # main optimization loop
    for i in range(max_iter):
        # Store old params for comparison or debugging if needed
        old_params = params.copy()
        
        # cycle of coordinate updates
        for i in range(n_params):
            # get the one param that changes
            current_q_vars: List[cp.Variable] = []

            # fix the others
            for j in range(n_params):
                # let the i-th parameter be variable
                if j == i:
                    q_i_var = cp.Variable(old_params[j], name=f"q_{i}", nonneg=True)
                    current_q_vars.append(q_i_var)
                # all others are treated as constants
                else:
                    current_q_vars.append(old_params[j])
            
            # solve for this one param 
            obj_expr = objective_fn(*current_q_vars)
            problem = cp.Problem(cp.Minimize(obj_expr))
            problem.solve(solver=cp.SCS, verbose=False)
            
            # Update parameter
            if q_i_var.value is not None:
                params[i] = q_i_var.value
            else:
                # If no solution, keep old value
                params[i] = old_params[i]
        
        # Evaluate objective after this full cycle
        obj_expr = objective_fn(*params)
        current_obj_val = obj_expr.value
        
        # append current objective
        history.append(current_obj_val if current_obj_val is not None else (-np.inf if maximize else np.inf))
        
        # check for convergence
        improvement = history[-1] - history[-2]
        if maximize:
            # if maximizing, stop if improvement < tol
            if improvement < tol:
                break
        else:
            # if minimizing, improvement should be negative (history[-2] - history[-1]) > tol for a decrease
            if abs(improvement) < tol:
                break

    return params, history

