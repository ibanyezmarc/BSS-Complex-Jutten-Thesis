
# @author: ibany

# Script containing the JH algorithm
# STRUCTURE OF SECTIONS:
#     - Imports
#     - JH algorithm adapted for complex signals


#%% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import rfcutils
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sigmf import SigMFFile
from numba import jit
import time
import warnings
warnings.filterwarnings("ignore")


get_sinr = lambda s, i: 10*np.log10(np.mean(np.abs(s)**2)/np.mean(np.abs(i)**2))
get_pow = lambda s: np.mean(np.abs(s)**2)
get_mse = lambda s: np.mean(np.abs(s)**2)



# %% JH algorithm adapted for complex signals

@jit(nopython=True)
def separate_signals(s, mu, beta, nin, num_iter, c12, c21):
    
    ymean1 = 0.0 + 0.0j; 
    ymean2 = 0.0 + 0.0j; 
    y1 = np.zeros(num_iter, dtype=np.complex_)
    y2 = np.zeros(num_iter, dtype=np.complex_)
    gradient_term_1 = np.zeros(num_iter, dtype=np.complex_) 
    gradient_term_2 = np.zeros(num_iter, dtype=np.complex_) 
    for n in range(num_iter):
        
        C = np.array([[1.0, c12], [c21, 1.0]], dtype=np.complex_)
        zuso = np.linalg.solve(C, s[:, n])
        
        ymean1 = beta * ymean1 + (1 - beta) * zuso[0]
        y1[n] = zuso[0] - ymean1
        ymean2 = beta * ymean2 + (1 - beta) * zuso[1]
        y2[n] = zuso[1] - ymean2
        
        # Fourth power: mu = 0.0001; beta = 0.9997
        term_1 = ((y1[n]*y1[n]*np.conj(y1[n]))) * np.sinh(np.conj(y2[n]))
        term_2 = ((y2[n]*y2[n]*np.conj(y2[n]))) * np.sinh(np.conj(y1[n]))

        if n > 1:
            gradient_term_1[n] = beta * gradient_term_1[n-1] + (1 - beta) * term_1
            gradient_term_2[n] = beta * gradient_term_2[n-1] + (1 - beta) * term_2
        if n > nin:
            c12 += mu * term_1
            c21 += mu * term_2

    return np.array([[1.0, c12], [c21, 1.0]], dtype=np.complex_), gradient_term_1, gradient_term_2