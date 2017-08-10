"""
Local

This contains functions for producing local encoders/decoders to feed to the GMRA

    pca_coder: computes a pca approximation with prespecified rank

author: Nate Strawn
email: nate.strawn@georgetown.edu
website: http://natestrawn.com
license: see LICENSE.txt
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np

#TODO: Add new local models for subspace clustering

def pca_coder(data, rank=2):
    m = np.mean(data, axis=0)
    u, s, v = np.linalg.svd(np.apply_along_axis(lambda p: p-m, 1, data))
    P = v[:rank, :]
    e = lambda q: (q - m) @ P.T
    d = lambda c: (c @ P) + m

    return {'encode': e, 'decode': d, 'info': s}


