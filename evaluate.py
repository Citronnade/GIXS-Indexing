import numpy as np
import matplotlib.pyplot as plt


import torch
from scipy import optimize

from deep_d import SimpleNet
from fit_d import dSpaceGenerator

from fine_optimize import gen_f, fine_optimize
import utils

def plot_results(a, b, g, model_path="model.pth", num_d=8):
    """

    :param a: True value of a to test
    :param b: True value of b to test
    :param g: True value of gamma (in degrees) to test
    :param model_path: Path to saved model state dict
    :return: None

    Plots the predicted and actual lattice structures for a given true indexing using only the neural network's predictions.
    """
    model = SimpleNet()
    m = utils.load_model_scaler(model_path, num_d)
    model.load_state_dict(m['model'])
    scaler = m['scaler']
    model.eval() # set model to eval for batchnorm
    generator = dSpaceGenerator()
    # generate empty arrays to store plotting points
    xs = np.array([])
    ys = np.array([])
    xs2 = np.array([])
    ys2 = np.array([])
    xs3 = np.array([])
    ys3 = np.array([])
    g = np.radians(g)
    known_y = np.array([a, b, g]).reshape(1,-1) # the true value to plot
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
    g2 = np.radians(predicted_params[2])
    #g2 = np.degrees(predicted_params[2])
    #predicted_params[2] = np.degrees(predicted_params[2])

    a3, b3, g3 = evaluate(model, a, b, np.degrees(g))[0].x
    print("after fitting:", a3, b3, g3)

    # plotting the lattices
    print("into actual_qs:", np.array([a,b,g]).reshape(1,-1))
    actual_qs = 1 / generator(np.array([a, b, g]).reshape(1, -1))
    pred_qs = 1 / generator(predicted_params.reshape(1, -1))
    fit_qs = 1 / generator(np.array([a3, b3, g3]).reshape(1, -1))
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
    for M in range(-3, 3):
        for N in range(-3, 3):
            # if M == 0 and N == 0:
            #    continue
            xs3 = np.append(xs3, M * a3 + N * b3 * np.cos(g3))
            ys3 = np.append(ys3, N * b3 * np.sin(g3))
    #"""

    fig, axs = plt.subplots(2)

    axs[0].scatter(xs, ys)
    axs[0].scatter(xs2, ys2, c='red')
    axs[0].scatter(xs3, ys3, s=4, c='orange', marker='*')
    #axs[0].plt.show()
    #axs[0].plt.waitforbuttonpress()
    #axs[0].plt.clf()
    #"""
    print("a,b,g:", a, b, g)
    print("predicted params:", predicted_params)
    print("actual_qs:", actual_qs)
    print("pred_qs:", pred_qs)

    # plot q values
    print(actual_qs.shape)
    axs[1].scatter(actual_qs.reshape(-1), [0.5] * len(actual_qs.reshape(-1)))
    for q in pred_qs.reshape(-1):
        axs[1].axvline(x=q)
    for q in fit_qs.reshape(-1):
        axs[1].axvline(x=q, color='orange')
    plt.show()


def evaluate(model, a=0.65, b=1.2, gamma=111, use_qs=False, scaler=None, **kwargs):
    """

    :param model: Model to use to evaluate
    :param a: Value of a to be tested:
    :param b: Value of b to be tested:
    :param gamma: Value of gamma to be tested:
    :param use_qs: Whether the model was trained and should be evaluated using q values as inputs rather than ds:
    :scaler: An object implementing transform() that normalizes inputs for the neural network.
    :return: None

    Tests performance of a trained model by evaluating it and then running BFGS using a known input.
    """
    model.eval()

    known_params = np.array([a, b, np.radians(gamma)]).reshape(1,-1) # Only used for comparison later
    generator = dSpaceGenerator(gen_q=use_qs, num_spacings=model.num_spacings) # use for mapping
    f = gen_f(generator, known_params) #function to be optimized
    input_d = generator(known_params.reshape(1,-1)) # 1 data point
    print("generated inputs are: ", ",".join(map(str, input_d[0])))
    old_inputs = input_d
    if scaler:
        input_d = scaler.transform(input_d) # rescale input to be same as in training
    input_d = input_d.reshape(-1)
    guess = model(torch.Tensor(input_d).unsqueeze(0)).detach().numpy()
    print("guess:", guess)
    result_regular = fine_optimize(f, guess)
    #result_regulara = optimize.basinhopping(f, guess) # BFGS with basinhopping to find global minimum

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
