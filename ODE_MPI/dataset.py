import tensorflow as tf
import numpy as np

class Dataset:
    def __init__(self, t_range, NT_train, N_train, N_bc):
        self.t_range = t_range
        self.NT = NT_train
        self.N_train = N_train
        self.N_bc = N_bc

    def bc(self, t_in):
        u_bc = np.zeros((t_in.shape[0], 1))

        return u_bc

    def build_data(self):
        t0 = self.t_range[0]
        t1 = self.t_range[1]
        t_ = np.linspace(t0, t1, self.NT).reshape((-1, 1))
        x_id = np.random.choice(self.NT, self.N_train, replace=False)
        X = t_
        X_input = X[x_id]

        #initial/bcs
        t_0 = t_.min(0)
        t_0 = np.reshape(t_0, (-1, 1))
        t_1 = t_.max(0)
        t_1 = np.reshape(t_1, (-1, 1))

        Xmin = t_0
        Xmax = t_1 
        
        t_bc = t_0
        u_bc = self.bc(t_bc)

        X_bc_0_input = t_bc
        u_bc_0_input = u_bc

        return X_input, \
               X_bc_0_input, u_bc_0_input, \
               Xmin, Xmax
