import numpy as np


class dSpaceGenerator:
    """
    :param H_max: The maximum \|H\| to search over when generating d-spacings
    :param K_max: The maximum \|K\| to search over when generating d-spacings
    :param noise: Range of truncated normal, which will have range [-noise, noise].  Currently has no effect.
    :param dropout: If True, then length of output vector will be truncated with 50% probability.  Currently has no effect.
    :param num_spacings: The number of d-spacings to be generated for each input.
    :param gen_q: Whether or not to generate q values instead of d values.

    This class is callable to produce a Torch tensor of d-space inputs.
    """
    
    def __init__(self, H_max=10, K_max=10, num_spacings=8, gen_q=False):
        self.H_max = H_max
        self.K_max = K_max
        self.num_spacings = num_spacings
        self.gen_q = gen_q


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
        temp = temp.T # we need to transpose to sort in the right axis later
        temp[temp==0] = 2000 #sometimes we get a d-spacing of 0, ignore these when sorting

        indices = np.argsort(temp)[:, :self.num_spacings] #get the indices of the num_spacings lowest d-spacings for each observation
        if self.gen_q:
            return 1 / np.take_along_axis(temp, indices, axis=1)  # get the actual lowest d-spacings and return
        else:
            return np.take_along_axis(temp, indices, axis=1)  # get the actual lowest d-spacings and return

def gen_input(n):
    """
    :param n: Number of observations to generate
    :return: n x 3 vector of a,b, gamma observations

    Generates random vectors of a,b,gamma for use in network training.\n
    a range: 0.3-1.5\n
    b range: 0.5-2.5\n
    gamma range: 85-130 (degrees)
    """
    temp = np.random.random((n,3)) * np.array([1.2, 2, (130-85)]) # produces the range of each value
    temp += np.array([0.3, 0.5, 85]) # adds the base of each value
    temp[:,2] = np.radians(temp[:,2]) # Turns gamma into radians
    # random a,b between 0.3-2.5, gamma 85-130 degrees
    temp2 = np.tile(np.random.random((n, 1)) * 2.2, 2) + np.random.random((n,2)) * 0.08
    temp3 = np.radians(np.random.random((n, 1)) * (130-85))
    temp4 = np.column_stack((temp2, temp3))



    return np.concatenate((temp, temp4))
