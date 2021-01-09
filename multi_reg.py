# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 14:22:57 2021

@author: Duong
"""


import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams.update({'font.size': 18})

# Load dataset
A = np.loadtxt(os.path.join('DATA','hald_ingredients.csv'),delimiter=',')
b = np.loadtxt(os.path.join('DATA','hald_heat.csv'),delimiter=',')

# Solve Ax=b using SVD
U, S, VT = np.linalg.svd(A,full_matrices=0)
x = VT.T @ np.linalg.inv(np.diag(S)) @ U.T @ b

plt.plot(b, Color='k', LineWidth=2, label='Heat Data') # True relationship
plt.plot(A@x, '-o', Color='r', LineWidth=1.5, MarkerSize=6, label='Regression')
plt.legend()
plt.show()


# Alternative Methods:

# The first alternative is specific to Matlab:
# x = regress(b,A)

# Alternative 2:
x = np.linalg.pinv(A)@b