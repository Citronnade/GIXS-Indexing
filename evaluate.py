import numpy as np
import matplotlib.pyplot as plt

import torch
from scipy import optimize

from deep_d import SimpleNet
from fit_d import dSpaceGenerator


def gen_f(generator, known_params):

    def f(guess):
        guess_ds = generator(guess.reshape(1,-1))
        return np.linalg.norm(guess_ds - generator(known_params.reshape(1,-1)))

    return f


def plot_results(a, b, g, scaler, model_path="model.pth"):
    """

    :param a: True value of a to test
    :param b: True value of b to test
    :param g: True value of gamma (in degrees) to test
    :param model_path: Path to saved model state dict
    :return: None
    """
    model = SimpleNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    generator = dSpaceGenerator()
    # generate empty arrays to store plotting points
    xs = np.array([])
    ys = np.array([])
    xs2 = np.array([])
    ys2 = np.array([])

    known_y = np.array([a, b, np.radians(g)]).reshape(1,-1)
    known_x = generator(known_y)#np.array(fit_d.gen_d_vector(*known_y[0]))

    known_x = scaler.transform(known_x.reshape(1,-1))
    known_x = torch.Tensor(known_x)
    print(known_x)

    # a = 0.65
    # b = 1.2
    # g = np.radians(111)
    # known_x = known_x.unsqueeze(-1)
    predicted_params = model(known_x).detach().numpy()[0]
    print("params:", predicted_params)
    a2 = predicted_params[0]
    b2 = predicted_params[1]
    g2 = predicted_params[2]
    actual_qs = 1 / generator(np.array([a, b, g]).reshape(1, -1))
    pred_qs = 1 / generator(predicted_params.reshape(1, -1))
    for M in range(-3, 3):
        for N in range(-3, 3):
            # if M == 0 and N == 0:
            #    continue
            xs = np.append(xs, M * a + N * b * np.cos(g))
            ys = np.append(ys, N * b * np.sin(g))
    for M in range(-3, 3):
        for N in range(-3, 3):
            # if M == 0 and N == 0:
            #    continue
            xs2 = np.append(xs2, M * a2 + N * b2 * np.cos(g2))
            ys2 = np.append(ys2, N * b2 * np.sin(g2))

    plt.scatter(xs, ys)
    plt.scatter(xs2, ys2)
    plt.show()

    plt.clf()
    print(actual_qs.shape)
    plt.scatter(actual_qs.reshape(-1), [0.5] * len(actual_qs.reshape(-1)))
    for q in pred_qs.reshape(-1):
        plt.axvline(x=q)
    plt.show()


def evaluate(model_path="model.pth"):
    """
    Tests model by evaluating it, then running BFGS, on a known input.
    :param model_path: Path to model to load from
    :return: None
    """
    model = SimpleNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    known_params = np.array([0.65, 1.2, np.radians(111)]) # Only used for comparison later
    generator = dSpaceGenerator()
    f = gen_f(generator, known_params)
    input_d = generator(known_params.reshape(1,-1)).reshape(-1)
    guess = model(torch.Tensor(input_d).unsqueeze(0)).detach().numpy()
    result = optimize.minimize(f, guess, options={'disp': True})
    print(result.x)
    print(known_params)

    actual_qs = 1 / generator(known_params.reshape(1,-1))
    pred_qs = 1 / generator(result.x.reshape(1,-1))
    plt.scatter(actual_qs.reshape(-1), [0.5] * len(actual_qs.reshape(-1)))
    for q in pred_qs.reshape(-1):
        plt.axvline(x=q)
    plt.show()