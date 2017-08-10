"""
Local information

This contains functions for displaying local information produced by the GMRA

    local_pca_info: displays the spectrum obtained from the local PCA approximations

author: Nate Strawn
email: nate.strawn@georgetown.edu
website: http://natestrawn.com
license: see LICENSE.txt
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
import matplotlib.pyplot as plt

def local_pca_info(local):
    '''
    Plot singular values of local approximations
    :param local: a dictionary of local elements containing local singular values
    :return: no return value
    '''

    k = len(local.keys())

    if k <= 25: # If number of local approximations is small, do a simultaneous line plots
        n = int(np.ceil(np.sqrt(k))) # size of the square for plotting
        plt.figure(0)
        for i in range(k):
            plt.subplot(n, n, i+1)
            plt.plot(local[i]['info'])
        plt.show()
    else: # If there are many, plot a matrix of values
        d = len(local[0]['info'])
        im = np.zeros((k, d))

        for i in range(k):
            im[i, :] = local[i]['info']

        # Set color limits
        a = np.min(im)
        b = np.max(im)

        plt.figure(0)
        plt.imshow(im, interpolation='nearest', clim=(a, b))
        plt.colorbar()
        plt.show()









