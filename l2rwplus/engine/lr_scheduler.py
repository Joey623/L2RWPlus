"""Learning-rate schedule used by local training."""

import numpy as np


def get_cyclic_lr(epoch, lr, epochs, peak):
    return np.interp([epoch], [0, peak, epochs], [1e-4 * lr, lr, 0])[0]

