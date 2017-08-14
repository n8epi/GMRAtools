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
import time

def local_pca_info(local):
    '''
    Plot singular values of local approximations
    :param local: a dictionary of local elements containing local singular values
    :return: no return value
    '''

    k = len(local.keys())

    if k <= 25: # If number of local approximations is small, do a simultaneous line plots
        n = int(np.ceil(np.sqrt(k))) # size of the square for plotting
        title = 'Local Spectra %d' % int(time.time())
        plt.figure(title)
        for i in range(k):
            plt.subplot(n, n, i+1)
            plt.plot(local[i]['info'])

        plt.savefig(title + '.png')
        #plt.show()
    else: # If there are many, plot a matrix of values

        # Determine the largest dimension (constrained by number of points in a cell)
        d = 0
        for i in range(k):
            if local[i]['info'].shape[0] > d:
                d = local[i]['info'].shape[0]
        im = np.zeros((k, d))

        for i in range(k):
            im[i, :local[i]['info'].shape[0]] = local[i]['info'] # Neighborhoods may be degenerate

        # Set color limits
        a = np.min(im)
        #b = np.max(np.percentile(im, 0.95, axis=1))
        b = np.max(im)

        # Plot the full set of eigenvalues and then a small subset
        title = 'Local Spectra %d' % int(time.time())
        plt.figure(title)
        plt.subplot(1, 2, 1)
        plt.imshow(im, interpolation='nearest', clim=(a, b), aspect='auto')
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.imshow(im[:100, :50], interpolation='nearest', clim=(a, b))
        plt.colorbar()
        plt.savefig(title + '.png')
        #plt.show()









