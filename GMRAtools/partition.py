import numpy as np
from covertree import CoverTree


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
        ct0.insert(data[i, :])

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
