import numpy as np
from numba import njit


class IsingMC:
    r"""
    Class for 2D square magnetic system simulations using Ising model.
    The model is solved with Monte-Carlo method.

    Parameters
    ----------
    size : int
        size of square system
    T : float
        temperature
    Jh : float
        horizontal exchange interaction
    Jv : float
        vertical exchange interaction
    Hmag : float
        magnetic field


    Attributes
    ----------
    E : float
        energy of the system
    state : np.ndarray
        spin state map of the system, shape=(size, size)

    """
    def __init__(self, size, T, Jh, Jv, Hmag):

        self.size = size
        self.Jh = Jh
        self.Jv = Jv
        self.Hmag = Hmag
        self.T = T

        self._gen_initial_state()
        self.E = self.calc_energy()

    def _gen_initial_state(self):
        r"""
        Generates an initial random state. Creates an array with randomly distributed -1 and 1.
        """
        self.state = np.random.uniform(low=-1, high=1, size=self.size * self.size).reshape(self.size, self.size)
        self.state = np.sign(self.state).astype(np.int64)
    
    def calc_energy(self):
        r"""
        Calculates energy of the system:

        Returns
        -------
            energy of the system
        """
        return _calc_full_energy(self.state, self.Jv, self.Jh, self.Hmag)

    def make_mc_step(self):
        r"""
        Performs one Monte-Carlo step. Changes self.state inplace.

        Note
        ----
            This method is fo square lattice only
        """
        x, y = np.random.randint(low=0, high=self.size, size=2)

        E_old = _calc_node_energy(self.state, x, y, self.Jv, self.Jh, self.Hmag)
        self.state[x, y] *= -1
        E_new = _calc_node_energy(self.state, x, y, self.Jv, self.Jh, self.Hmag)

        if(E_new < E_old):
            self.E = self.E - E_old + E_new
            return
        else:
            randf = np.random.rand()
            if(randf < np.exp(-(E_new - E_old) / self.T)):
                self.E = self.E - E_old + E_new
            else:
                self.state[x, y] *= -1

    def run(self, n_iter):
        r"""
        Run Monte-Carlo algorithm for a specified number of iterations

        Parameters
        ----------
            n_iter : int
                numer of Monte-Carlo steps to perform

        """
        for i in range(n_iter):
            self.make_mc_step()
            # if(i % 100 == 0):
            #     print(f'{i}: E = {self.E}')

@njit
def _calc_full_energy(state, Jv, Jh, Hmag):
    E = 0
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            if(j + 1 != state.shape[1]):
                E += -Jv * state[i][j] * state[i][j + 1]
            if(i + 1 != state.shape[0]):
                E += -Jh * state[i][j] * state[i + 1][j]
    return E - Hmag * state[i][j]

@njit
def _calc_node_energy(state, i, j, Jv, Jh, Hmag):
    E = 0
    if(i - 1 != -1):
        E += -Jh * state[i][j] * state[i - 1][j]
    if(i + 1 != state.shape[0]):
        E += -Jh * state[i][j] * state[i + 1][j]
    if(j - 1 != -1):
        E += -Jv * state[i][j] * state[i][j - 1]
    if(j + 1 != state.shape[1]):
        E += -Jv * state[i][j] * state[i][j + 1]
    return E - Hmag * state[i][j]