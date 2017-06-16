import numpy as np

def pca_coder(data, rank=2):
    m = np.mean(data, axis=0)
    u, s, v = np.linalg.svd(np.apply_along_axis(lambda p: p-m, 1, data))
    P = v[:rank, :]
    e = lambda q: (q - m) @ P.T
    d = lambda c: (c @ P) + m

    return {'encode': e, 'decode': d }

