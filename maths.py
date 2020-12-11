from math import sqrt, pi

import numpy as np
from numpy import exp

root_2_pi = sqrt(2. * pi) # square root is expensive, define as constant here

def gaussian(x: float, mu: float, sd: float):
    mu_pert = x - mu # mean pertubation

    coefficient = 1. / (sd * root_2_pi)

    return coefficient * exp( - (mu_pert**2)
                                        /
                                    (2.*sd**2))