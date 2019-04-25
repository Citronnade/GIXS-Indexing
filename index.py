import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy import optimize

from deep_d import SimpleNet
from fit_d import dSpaceGenerator

def gen_f(generator, ds):
    """
    Returns a function that can be optimized by scipy.optimize.
    """
    def f(guess):
        guess_ds = generator(guess.reshape(1,-1))
        return np.linalg.norm(guess_ds - ds)
    return f

def index(d_inputs, model, scaler=None):
    """

    :param d_inputs: array of input d spacings, sorted in ascending order
    :param model: callable model to provide first prediction
    :param scaler: a sklearn scaler object that matches the given model
    :return: An array of shape (,3) containing [a,b,gamma]

    Provides a,b,gamma that best indices the provided inputs.
    """
    generator = dSpaceGenerator(num_spacings=len(d_inputs))
    f = gen_f(generator, d_inputs)
    if scaler:
        input_d = scaler.transform(d_inputs).reshape(-1)
    model.eval()
    guess = model(torch.Tensor(input_d).unsqueeze(0)).detach().numpy()
    result = optimize.minimize(f, guess, options={'disp': True, 'gtol': 1e-8}) #optimize.basinhopping(f, guess)
    percent_error = 100 * np.sum((generator(result.x.reshape(1,-1)) - d_inputs) / d_inputs)
    print(percent_error)
    return result.x, percent_error


def main(model_path="model.pth", scaler=None, num_spacings=8):
    model = SimpleNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    while True:
        ds = input("Enter observed d spacings as a comma separated list (nothing to cancel):\n")
        if not ds:
            break
        ds = np.array([float(x) for x in ds.split(",")]).reshape(1,-1)
        params, percent_error = index(ds, model, scaler)
        print("Estimated parameters are ", params)



