"""
Partition

This contains partition functions to feed to the GMRA

    uniform_covertree_partition: computes a partition induced by cutting off a covertree at a particular scale

author: Nate Strawn
email: nate.strawn@georgetown.edu
website: http://natestrawn.com
license: see LICENSE.txt
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
from GMRAtools.covertree import CoverTree

def uniform_covertree_partition(data, metric, scale=-1):
    '''
    Return partition index function induced by cover trees
    :param data: matrix of data in rows
    :param metric: metric to be applied between rows
    :param scale: scale where to extract cover tree members
    :return:
    '''
    ct0 = CoverTree(metric)
    for i in range(data.shape[0]):
        if i % 100 == 0:
            print('Processing data index %d of %d' % (i, data.shape[0]))
        ct0.insert(data[i, :])

    print('Extracting the cover at scale %d' % scale)
    # Extract the cover tree up to the desired scale
    ct = CoverTree(metric)
    lvl = ct0.maxlevel
    nodes = [ct0.root]
    ct.insert(np.array(ct0.root.data))
    num_nodes = 0
    while nodes:
        n0 = nodes.pop()
        for k in n0.children.keys():
            if k > scale:
                for n in n0.children[k]:
                    nodes.append(n)
                    ct.insert(np.array(n.data))
                    num_nodes += 1

    return (lambda z: ct.knn(z, 1)[0][0])
