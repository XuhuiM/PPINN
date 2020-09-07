import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
'''
savepath='./Output'
if not os.path.exists(savepath):
    os.makedirs(savepath)
'''
class SavePlot:
    def __init__(self, sess, t_range, NT):
        self.t_range = t_range
        self.NT = NT
        self.sess = sess

    def saveplt(self, u_pred, t_train):
        t_test = np.linspace(self.t_range[0], self.t_range[1], self.NT).reshape((-1, 1))
        test_dict = {t_train: t_test}
        u_test = self.sess.run(u_pred, feed_dict=test_dict)
        u_test = np.reshape(u_test, (-1, 1))
#        np.savetxt('./Output/u_pred', u_test, fmt='%e')
        ''' 
        plt.imshow(u_test, cmap='rainbow', aspect='auto')
        plt.colorbar()
        plt.savefig('./Output/ucontour.png')
        plt.show()
        '''
        return u_test
