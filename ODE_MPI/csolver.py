import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

from coarsesolver import CoarseSolver

np.random.seed(1234)
tf.set_random_seed(1234)

def csolver(t_range, NT, u0):
    
    csolver_ = CoarseSolver(t_range, NT, u0)
    u_coarse = csolver_.fdm()

    return u_coarse
