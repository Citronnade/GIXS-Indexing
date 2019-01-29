import numpy as np


def gen_d(a,b,gamma,H,K):
    """
    Helper function to take in a,b,gamma values or vectors and produce a d-spacing corresponding to a specific H or K.
    :param a:
    :param b:
    :param gamma:
    :param H:
    :param K:
    :return:
    """
    return np.sin(gamma) / np.sqrt(H**2/a**2 + K**2/b**2 - 2*H*K*np.cos(gamma) / (a*b))

class dSpaceGenerator:
    def __init__(self, H_max=10, K_max=10, num_spacings=8):
        """
        Returns a function that generates a Torch tensor of the vectors.
        Pydoc will create pretty documentation based off these docstrings.
        :param H_max: The maximum |H| to search over when generating d-spacings
        :param K_max: The maximum |K| to search over when generating d-spacings
        :param noise: Range of truncated normal, which will have range [-noise, noise].  Set to zero to disable addition of random noise.  #TODO: make this normally distributed with SD noise
        :param dropout: If True, then length of output vector will be truncated with 50% probability
        :return: A function f(a,b,gamma) that returns an array of the 5 highest d-spacings over the given H_max, K_max range for the given parameters.
        """
        self.H_max = H_max
        self.K_max = K_max
        self.num_spacings = num_spacings


    # __call__ is a builtin function that's called when you try to call an object
    def __call__(self, lattice_params):
        """
        :param self:
        :param lattice_params: n x 3 vector of [a,b, gamma], with n observations.  Must be at least 2D!
        :return: The num_spacings highest d spacings corresponding to the given a,b, gamma values, as a n x num_spacings vector.
        """
        n, _ = lattice_params.shape
        t1 = lattice_params[:,0] ** 2 # a**2 vector
        t2 = lattice_params[:,1] ** 2 # b**2 bector
        t3 = 2 * np.cos(lattice_params[:, 2]) / (lattice_params[:,0] * lattice_params[:, 1]) # part of the 2hk cos(gamma) vector
        top = np.sin(lattice_params[:,2]) # part of the sin(gamma) vector

        temp = np.zeros((2 * self.H_max * self.K_max, n)) # temporary vector to store all d-spacings for each a,b,gamma observation
        i = 0
        for H in range(-self.H_max, self.H_max):
            for K in range(0, H + 1):
                if H == 0 and K == 0: #skip the H==K==0 state
                    continue
                d = top / np.sqrt(H ** 2 / t1 + K ** 2 / t2 - H *  K * t3)
                temp[i] = d
                i += 1
        temp = temp.T
        temp[temp==0] = 2000 #sometimes we get a d-spacing of 0, ignore these when sorting

        indices = np.argsort(temp)[:, :self.num_spacings] #get the indices of the num_spacings lowest d-spacings for each observation
        return np.take_along_axis(temp, indices, axis=1) # get the actual lowest d-spacings and return

def gen_input(n):
    #a: 0.3-1.5
    #b: 0.5-2.5
    #gamma: 85-130
    """
    Generates random vectors of a,b,gamma for use in network training.
    :param n: Number of observations to generate
    :return: n x 3 vector of a,b, gamma observations
    """
    temp = np.random.random((n,3)) * np.array([1.2, 2, (130-85)]) # produces the range of each value
    temp += np.array([0.3, 0.5, 85]) # adds the base of each value
    temp[:,2] = np.radians(temp[:,2]) # Turns gamma into radians
    return temp
