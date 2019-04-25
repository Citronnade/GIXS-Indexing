import numpy as np
import matplotlib.pyplot as plt


import torch
from scipy import optimize

from deep_d import SimpleNet
from fit_d import dSpaceGenerator


def gen_f(generator, known_params):
    # returns a function that can be minimized with scipy.optimize
    def f(guess):
        guess_ds = generator(guess.reshape(1,-1))
        return np.linalg.norm(guess_ds - generator(known_params.reshape(1,-1))) # L2 distance between observed and calculated qs

    return f


def plot_results(a, b, g, scaler, model_path="model.pth"):
    """

    :param a: True value of a to test
    :param b: True value of b to test
    :param g: True value of gamma (in degrees) to test
    :param model_path: Path to saved model state dict
    :return: None

    Plots the prediced and actual lattice structures for a given true indexing using only the neural network's predictions.
    """
    model = SimpleNet()
    model.load_state_dict(torch.load(model_path)) # load model
    model.eval() # set model to eval for batchnorm
    generator = dSpaceGenerator()
    # generate empty arrays to store plotting points
    xs = np.array([])
    ys = np.array([])
    xs2 = np.array([])
    ys2 = np.array([])

    known_y = np.array([a, b, np.radians(g)]).reshape(1,-1) # the true value to plot
    known_x = generator(known_y)

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
    # plotting the lattices
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
    # plot q values
    print(actual_qs.shape)
    plt.scatter(actual_qs.reshape(-1), [0.5] * len(actual_qs.reshape(-1)))
    for q in pred_qs.reshape(-1):
        plt.axvline(x=q)
    plt.show()


def evaluate(model_path="model.pth", a=0.65, b=1.2, gamma=111, use_qs=False, scaler=None, num_spacings=8, **kwargs):
    """

    :param model_path: Path to model to load from
    :param a: Value of a to be tested:
    :param b: Value of b to be tested:
    :param gamma: Value of gamma to be tested:
    :param use_qs: Whether the model was trained and should be evaluated using q values as inputs rather than ds:
    :scaler: An object implementing transform() that normalizes inputs for the neural network.
    :return: None

    Tests performance of a trained model by evaluating it and then running BFGS using a known input.
    """
    model = SimpleNet(num_spacings=num_spacings)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    known_params = np.array([a, b, np.radians(gamma)]).reshape(1,-1) # Only used for comparison later
    generator = dSpaceGenerator(gen_q=use_qs, num_spacings=num_spacings) # use for mapping
    f = gen_f(generator, known_params) #function to be optimized
    input_d = generator(known_params.reshape(1,-1)) # 1 data point
    print("generated inputs are: ", ",".join(map(str, input_d[0])))
    old_inputs = input_d
    if scaler:
        input_d = scaler.transform(input_d) # rescale input to be same as in training
    input_d = input_d.reshape(-1)
    guess = model(torch.Tensor(input_d).unsqueeze(0)).detach().numpy()
    print("guess:", guess)
    result_regular = optimize.minimize(f, guess, options={'disp': True, 'gtol': 1e-8}) # regular BFGS optimization
    #result = optimize.basinhopping(f, guess) # BFGS with basinhopping to find global minimum

    #print(result.x)
    print(result_regular.x)
    print("actual: ", known_params)
    return result_regular, old_inputs[0]

    # plot q's as before                                                                                                                                                                     
    #actual_qs = 1 / generator(known_params.reshape(1,-1))
    #pred_qs = 1 / generator(result_regular.x.reshape(1,-1))
    #plt.scatter(actual_qs.reshape(-1), [0.5] * len(actual_qs.reshape(-1)))
    #for q in pred_qs.reshape(-1):
    #    plt.axvline(x=q)
    #plt.show()
