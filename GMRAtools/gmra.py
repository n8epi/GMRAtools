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
    def __init__(self, data, partition, local, metric=l2):
        '''
        Initialize GMRA
        :param data: data in rows
        :param partition: returns partition index lambda from data and specified metric
        :param local: returns local encode and decode functions
        :param metric: the base metric for comparison
        '''
        self.partition = partition(data, metric)
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
