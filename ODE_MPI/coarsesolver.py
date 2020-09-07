import numpy as np
import matplotlib.pyplot as plt

'''
def f_u(t):
    f = t + np.sin(0.5*np.pi*t)
    return f
'''
class CoarseSolver:
    def __init__(self, t_range, Nc, u0):
        self.t0 = t_range[0]
        self.t1 = t_range[1]
        self.u0 = u0
        self.Nc = Nc

    def fdm(self):
        dt = (self.t1 - self.t0)/(self.Nc - 1)
        t = np.linspace(self.t1, self.t0, self.Nc).reshape((-1, 1))
        u = np.zeros((self.Nc, 1))
        u[0] = self.u0

        for i in range(1, self.Nc):
            u[i] = dt + u[i-1]

        return u
