"""
Partition

This contains partition functions to feed to the GMRA

    uniform_covertree_partition: computes a partition induced by cutting off a covertree at a particular scale
    randomized_binary_tree_partition: computes a binary partition of the data of a particular depth using affine planes

author: Nate Strawn
email: nate.strawn@georgetown.edu
website: http://natestrawn.com
license: see LICENSE.txt
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
import numpy.random as rd

from covertree.covertree import CoverTree


def uniform_covertree_partition(data, metric, scale=-1, ct_pre=None, verbose=False):
    '''
    Return partition index function induced by cover trees
    :param data: matrix of data in rows
    :param metric: metric to be applied between rows
    :param scale: scale where to extract cover tree members
    :param ct: a pre-computed cover tree for the construction
    :return:
    '''
    ct0 = CoverTree(metric)

    if ct_pre == None:
        for i in range(data.shape[0]):
            if verbose and i % 100 == 0:
                print('Processing data index %d of %d' % (i, data.shape[0]))
            ct0.insert(data[i, :])

    else:
        ct0 = ct_pre

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


class Bnode:
    def __init__(self, v, b):
        self.l = None
        self.r = None
        self.v = v
        self.b = b

def split(node, data, curr_depth, final_depth, iter=3):
    mu = np.mean(data, axis=0)
    A = data.T @ data - data.shape[0] * np.outer(mu, mu)

    # Perform a few steps of power iteration with random init
    v = rd.randn(data.shape[1])
    v = v / np.sqrt(np.sum(v**2))

    for i in range(iter):
        v = A @ v
        v = v / np.sqrt(np.sum(v**2))

    # Split using the median value
    ips = data @ v
    b = np.median(ips)

    node.v = v
    node.b = b

    # Recursive call...
    if curr_depth+1 < final_depth:
        node.l = Bnode(None, None)
        node.r = Bnode(None, None)
        split(node.l, data[ips > b, :], curr_depth+1, final_depth, iter=iter)
        split(node.r, data[ips <= b, :], curr_depth+1, final_depth, iter=iter)


def traverse(bnode, x, curr_depth, final_depth):
    if curr_depth < final_depth:
        if (bnode.v @ x > bnode.b):
            return 2**curr_depth + traverse(bnode.l, x, curr_depth+1, final_depth)
        else:
            return traverse(bnode.r, x, curr_depth+1, final_depth)
    else:
        return 0

def randomized_binary_tree_partition(data, metric, depth=8):
    btree_root = Bnode(None, None)
    split(btree_root, data, 0, depth)
    p = 2 ** np.array(list(range(depth)))
    print('Tree partition data:')
    f = lambda z: traverse(btree_root, z, 0, depth)
    return f

