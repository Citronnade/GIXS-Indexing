from scipy import optimize
from fit_d import dSpaceGenerator
import numpy as np


def gen_f(generator, known_params, is_index=False):
    """
    Returns a function that can be minimized with scipy.optimize
    :param generator: dSpaceGenerator to be used
    :param known_params: the known y values to be fit to
    :param is_index: true if used for index operation, false for simulate/evaluate
    :return: function to be optimized
    """
    if is_index:
        target = known_params # if we are indexing then known_params is the other way around
    else:
        target = generator(known_params).reshape(1,-1) # if we are evaluating then known_params is a,b,gamma
    def f(guess):
        guess_ds = generator(guess.reshape(1,-1))
        norm = np.linalg.norm(guess_ds - target)# L2 distance between observed and calculated qs
        return norm
    return f

def fine_optimize(f, guess):
    """
    Helper function that calls scipy minimize with specific parameters.
    :param f: function to minimize
    :param guess: initial guess
    :return: optimization result
    """
    return optimize.minimize(f, guess, options={'disp': True}, tol=1e-9, method='Nelder-Mead')