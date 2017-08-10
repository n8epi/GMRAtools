"""
GMRA

This is a minimalistic implementation of GMRA that depends upon

    1. a partition function
    2. a local model for the data (e.g. PCA)
    3. a metric for the ambient space (default is l2
    4. an optional function for displaying information

author: Nate Strawn
email: nate.strawn@georgetown.edu
website: http://natestrawn.com
license: see LICENSE.txt
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np

def l2(x, y):
    '''
    Simple Euclidean distance computation
    :param x:
    :param y:
    :return:
    '''
    d = x - y
    return np.sqrt(d @ d)

class GMRA:
    '''
    This is the GMRA class for
    '''
    def __init__(self, data, partition, local, metric=l2, info_fcn=None):
        '''
        Initialize GMRA
        :param data: data in rows
        :param partition: returns partition index lambda from data and specified metric
        :param local: returns local encode and decode functions
        :param metric: the base metric for comparison
        :param info_fcn: this returns visualizations coming from the local models
        '''
        self.partition = partition(data, metric)
        self.info_fcn = info_fcn
        indices = np.zeros(data.shape[0])

        for i in range(data.shape[0]):
            #print(self.partition(data[i,:]))
            indices[i] = int(self.partition(data[i, :])) # What if a cell index never comes up?
        self.max_index = int(np.max(indices))

        self.local = {}
        for i in range(self.max_index+1):
            self.local[i] = local(data[indices==i,:])

    def encode(self, p):
        idx = int(self.partition(p))
        #print(idx)
        #print(self.local.keys())
        #print(self.local[idx])
        code = self.local[idx]['encode'](p)
        return idx, code

    def decode(self, idx, code):
        return self.local[int(idx)]['decode'](code)

    def info(self):
        #print(self.local)
        if self.info != None:
            self.info_fcn(self.local)
        else:
            print('Warning: Info function undeclared.')


    #TODO: Add new local models for subspace clustering

    #TODO: Make examples for things

if __name__ == '__main__':
    # Simple test using points on the circle
    import numpy.random as rd
    from partition import uniform_covertree_partition
    from local import pca_coder
    from local_info import local_pca_info

    N = 10000
    th = 2*np.pi*rd.rand(N)
    x = np.zeros((N, 2))
    x[:, 0] = np.cos(th)
    x[:, 1] = np.sin(th)

    scale = -3

    g = GMRA(x,
             lambda d, m: uniform_covertree_partition(d, m, scale=scale),
             lambda p: pca_coder(p, rank=1),
             info_fcn=local_pca_info)
    idx, c = g.encode([1/np.sqrt(2), 1/np.sqrt(2)])
    print(g.decode(idx, c))
    g.info()

