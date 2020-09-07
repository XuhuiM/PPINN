'''
PPINN solver for ODE (parallel):
Coarse solver: FDM
Fine solver: PINN
@Xuhui Meng 9/2020
'''
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from mpi4py import MPI

from dataset import Dataset
from net import DNN
from modeltrain import Train
from saveplot import SavePlot
from csolver import csolver
from coarsesolver import CoarseSolver

np.random.seed(1234)
tf.set_random_seed(1234)

def main():
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    #thread 0 --> coarse solver/remaining threads --> fine solver
    NT = 200001
    t_range = [0, 10]
    num = comm_size - 1
    Ne = int((NT-1)/num + 1)
    chunks = np.linspace(t_range[0], t_range[1], num+1)
    if comm_rank == 0:
        t_range_c = t_range

        #dataset for coarse solver for the first loop
        NT_c = 1001
        uc_0 = 0
        u_coarse_0 = csolver(t_range_c, NT_c, uc_0)
        np.savetxt('./Output/u_coarse_0', u_coarse_0, fmt='%e')

        #traing data for coarse solver
        Ne_c = int((NT_c-1)/num + 1)
    else:
        #dataset for fine solver
        #number of elements in each subdomain
        #number of training data
        N_train = 10000
        N_bc = 1
    
    if comm_rank == 0:
        #training dataset for coarse solver in each subdomain
        X_c_list = []
        for i in range(num):
            t_range_e = [chunks[i], chunks[i+1]] 
            X_c_list.append(t_range_e)
    
    #fine solver dataset for each thread/subdomain
    else:
        t_range_e = [chunks[comm_rank-1], chunks[comm_rank]] 
        data = Dataset(t_range_e, Ne, N_train, N_bc)
        X, X_bc_0, u_bc_0, Xmin, Xmax = data.build_data()
        
        #size of the DNN
        layers = [1] + 2*[20] + [1]

        t_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        t_bc_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        u_0_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    
        pinn = DNN(layers, Xmin, Xmax)
        W, b = pinn.hyper_initial()
        #physics-infromed neural networks for fine solver in each subdomain
        pinn = DNN(layers, Xmin, Xmax)
        u_pred = pinn.fnn(t_train, W, b)
        u_0_pred = pinn.fnn(t_bc_train, W, b)
        flag = 1
        f_pred = pinn.odenn(t_train, W, b, flag)

        loss = tf.reduce_mean(tf.square(f_pred)) + \
               tf.reduce_mean(tf.square(u_0_train - u_0_pred))

        train_adam = tf.train.AdamOptimizer().minimize(loss)
        train_lbfgs = tf.contrib.opt.ScipyOptimizerInterface(loss,
                                                             method = "L-BFGS-B",
                                                             options = {'maxiter': 50000,
                                                                        'ftol': 1.0*np.finfo(float).eps
                                                                       }
                                                            )
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
    
    if comm_rank == 0:
        t_init = np.zeros((1, 1))
        u0 = np.zeros((1, 1))
        U = [u0] + [np.zeros((t_init.shape[0], 1)) for i in range(num)]
        G = [np.zeros((t_init.shape[0], 1)) for i in range(num+1)]
        F = [np.zeros((t_init.shape[0], 1)) for i in range(num+1)]
        bc_id = np.linspace(0, NT_c-1, num+1)
        bc_id = bc_id.astype(int)
        for i in range(1, num):
            U[i] = u_coarse_0[bc_id[i]:(bc_id[i]+1)]
            G[i] = u_coarse_0[bc_id[i]:(bc_id[i]+1)]

        #passing interfacial information to slave threads, tag = 20
        for i in range(num):
            comm.send(U[i], dest=i+1, tag=20+i+1)
    else:
        u_initial = comm.recv(source=0, tag=20+comm_rank)

    comm.Barrier()
    if comm_rank == 0:
        start_time = time.perf_counter()
    #iteration
    niter = 1
    Error = 1
    while niter <= num and Error > 1.0e-2:
        #fine solver
        if comm_rank >= niter:
            print('Fine solver for chunk#: %d'%(comm_rank))
            t_f_train = t_train
            t_f_bc_train = t_bc_train
            u_f_0_train = u_0_train
            train_dict = {t_f_train: X, \
                          t_f_bc_train: X_bc_0, u_f_0_train: u_initial
                         }
    
            Model = Train(train_dict)
            Model.nntrain(sess, u_pred, loss, train_adam, train_lbfgs)
            NT_test = Ne
            t_range = [chunks[comm_rank-1], chunks[comm_rank]] 
            datasave = SavePlot(sess, t_range, NT_test)
            u_pred_f = datasave.saveplt(u_pred, t_train)
            filename = './Output/u_f_loop_' + str(niter) + '_chunk_' + str(comm_rank)
            np.savetxt(filename, u_pred_f, fmt='%e')
            F_bc = u_pred_f[-1:]
            #passing fine solver solution to coarse solver, tag = 50
            comm.send(F_bc, dest = 0, tag = 50+comm_rank)
        
        if comm_rank == 0:
            #rcv, tag = 30
            for i in range(niter, num+1):
                F[i] = comm.recv(source=i, tag=i+50)

        comm.Barrier()
        if comm_rank == 0:
            #coarse solver
            error = 0.0
            G_new = G[niter]
            for n in range(niter, num):
                print('Coarse solver for chunk#: %d'%(n))
                error += np.linalg.norm(G_new - G[n])/(np.linalg.norm(G_new) + 1.0e-9)
                U[n] = G_new + F[n] - G[n]
                G[n] = G_new
                X = X_c_list[n]
                u_bc_0 = U[n]
                csolver_ = CoarseSolver(X, Ne_c, u_bc_0)
                u_pred_c = csolver_.fdm()
                filename = './Output/u_c_loop_' + str(niter+1) + '_chunk_' + str(n)
                np.savetxt(filename, u_pred_c, fmt='%e')
                G_new = u_pred_c[-1:]
            
            Error = error/(num - niter + 1.0e-9)
            print('Iteration: %d, Error: %.3e'%(niter, Error))
            #send boundary values to fine solvers, tag = 80
            for i in range(niter, num):
                comm.send(U[i], dest=i+1, tag=80+i+1)
        elif comm_rank > niter:
            u_initial = comm.recv(source=0, tag=80+comm_rank)
        
        Error = comm.bcast(Error, root = 0)
        niter += 1

    if comm_rank == 0:
        if niter < num:
            print('Converges!')
        else:
            print('Exceed the maximum iteration number!')
        stop_time = time.perf_counter()
        print('Duration time is %.3f seconds'%(stop_time - start_time))

    '''
    start_time = time.perf_counter()
    stop_time = time.perf_counter()
    print('Duration time is %.3f seconds'%(stop_time - start_time))
    '''

if __name__ == '__main__':
    main()
