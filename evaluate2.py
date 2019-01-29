from scipy import optimize
import numpy as np
import torch.nn as nn
from fit_d import dSpaceGenerator
import torch
from deep_d import SimpleNet
import numpy as np
import matplotlib.pyplot as plt

def gen_f(generator):

    def f(guess):
        guess_ds = generator(guess.reshape(1,-1))
        print("guess", guess_ds)
        #print("actual", generator(known_params.reshape(1,-1)))
        return np.linalg.norm(guess_ds - generator(known_params.reshape(1,-1)))

    return f
if __name__ == "__main__":

    model = SimpleNet()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    known_params = np.array([0.65, 1.2, np.radians(111)]) # Only used for comparison later
    generator = dSpaceGenerator()
    f = gen_f(generator)
    input_d = generator(known_params.reshape(1,-1)).reshape(-1)
    guess = model(torch.Tensor(input_d).unsqueeze(0)).detach().numpy()
    result = optimize.minimize(f, guess, options={'disp': True})
    print(result.x)
    print(known_params)

    actual_qs = 1 / generator(known_params.reshape(1,-1))
    pred_qs = 1 / generator(result.x.reshape(1,-1))
    plt.scatter(actual_qs, [0.5] * len(actual_qs))
    for q in pred_qs:
        plt.axvline(x=q)
    plt.show()