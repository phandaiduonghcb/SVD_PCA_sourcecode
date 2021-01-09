# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 11:16:48 2021

@author: Duong
"""

from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams['figure.figsize'] = [16, 8]


A = imread(os.path.join('img_compress','test.jpg'))
X = np.mean(A, -1); # Convert RGB to grayscale

img = plt.imshow(X)
plt.imsave('img_compress/converted_to_gray.jpg',X,cmap = 'gray')
img.set_cmap('gray')
plt.axis('off')
plt.show()

U, S, VT = np.linalg.svd(X,full_matrices=False) #economy SVD
S = np.diag(S)

j = 0
for r in (5, 20, 100):
    # Construct approximate image
    Xapprox = U[:,:r] @ S[0:r,:r] @ VT[:r,:]
    plt.figure(j+1)
    j += 1
    img = plt.imshow(Xapprox)
    plt.imsave('img_compress/compressed_{}.jpg'.format(j),Xapprox,cmap = 'gray')
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('r = ' + str(r))
    plt.show()
    
plt.figure(1)
plt.semilogy(np.diag(S))
plt.title('Singular Values')
plt.show()

plt.figure(2)
plt.plot(np.cumsum(np.diag(S))/np.sum(np.diag(S)))
plt.title('Singular Values: Cumulative Sum')
plt.show()