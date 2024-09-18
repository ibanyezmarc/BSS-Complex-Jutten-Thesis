# @author: ibany


# STRUCTURE OF SECTIONS:
#     - Python adaptation of CFASTICA by Alex Bujan
#     - Create mixture and apply CFASTICA
#     - Plot all sinr levels CFASTICA


#%% Python adaptation of CFASTICA by Alex Bujan
from __future__ import division
import pdb,os,time,warnings
import numpy as np
from math import log
from numpy.linalg import *
from numpy.random import rand

"""
Author: Alex Bujan
Adapted from: Ella Bingham, 1999

Original article citation:
Ella Bingham and Aapo Hyvaerinen, "A fast fixed-point algorithm for 
independent component analysis of complex valued signals", 
International Journal of Neural Systems, Vol. 10, No. 1 (February, 2000) 1-8

Original code url:
http://users.ics.aalto.fi/ella/publications/cfastica_public.m

Date: 12/11/2015

TODO: include arbitrary contrast functions
"""

def abs_sqr(W,X):
    return abs(W.conj().T.dot(X))**2

def complex_FastICA(X,epsilon=.1,algorithm='parallel',\
                    max_iter=100,tol=1e-4,whiten=True,\
                    w_init=None,n_components=None):
    """Performs Fast Independent Component Analysis of complex-valued 
        signals

    Parameters
    ----------

    X : array, shape (n_features,n_samples)
        Input signal X = A S, where A is the mixing 
        matrix and S the latent sources.

    epsilon : float, optional
        Arbitrary constant in the contrast G function 
        used in the approximation to neg-entropy.

    algorithm : {'parallel', 'deflation'}, optional
        Apply a parallel or deflational FASTICA algorithm.

    w_init : (n_components, n_components) array, optional
        Initial un-mixing array.If None (default) then an 
        array of normally distributed r.v.s is used.

    tol: float, optional
        A positive scalar giving the tolerance at which the
        un-mixing matrix is considered to have converged.

    max_iter : int, optional
        Maximum number of iterations.
    
    whiten : boolean, optional
        If True, perform an initial whitening of the data.
        If False, the data is assumed to be already white.

    n_components : int, optional
        Number of components to extract. If None, 
        n_components = n_features.

    Returns
    -------

    W : array, shape (n_components, n_components)
        Estimated un-mixing matrix.

    K : array, shape (n_components, n_features)
        If whiten is 'True', K is the pre-whitening matrix 
        projecting the data onto the principal components. 
        If whiten is 'False', K is 'None'.

    EG : array, shape(n_components,max_iter)
        Expectation of the contrast function E[G(|W'*X|^2)]. 
        This array may be padded with NaNs at the end.

    S : array, shape (n_samples, n_components)
        Estimated sources (S = W K X).
    """

    n,m  = X.shape
    
    if n_components!=None:
        n = n_components

    if whiten:
        X-=X.mean(1,keepdims=True)
        Ux,Sx = eig(np.cov(X))
        K     = np.sqrt(inv(np.diag(Ux))).dot(Sx.conj().T)[:n]
        X     = K.dot(X)
        del Ux,Sx
    else:
        K = None

    EG = np.ones((n,max_iter))*np.nan

    if algorithm == 'deflation':
        W = np.zeros((n, n), dtype=np.complex128)

        for k in range(n):
            if w_init!=None:
                w = w_init[:,k]
            else:
                w = np.random.normal(size=(n,1))+\
                    1j*np.random.normal(size=(n,1))

            w/=norm(w)

            n_iter  = 0

            for i in range(max_iter):

                wold = np.copy(w)

                #derivative of the contrast function
                g  =  1/(epsilon+abs_sqr(w,X))
                #derivative of g
                dg = -1/(epsilon+abs_sqr(w,X))**2

                w  = (X * (w.conj().T.dot(X)).conj() * g).mean(1).reshape((n,1))-\
                     (g + abs_sqr(w,X) * dg).mean() * w

                del g,dg

                w/=norm(w)

                # Decorrelation
                w-=W.dot(W.conj().T).dot(w)
                w/=norm(w)

                EG[k,n_iter] = (np.log(epsilon+abs_sqr(w,X))).mean()

                n_iter+=1

                lim = (abs(abs(wold)-abs(w))).sum()
                if lim<tol:
                    break

            if n_iter==max_iter and lim>tol:
                warnings.warn('FastICA did not converge. Consider increasing '
                              'tolerance or the maximum number of iterations.')

            W[:,k] = w.ravel()

    elif algorithm=='parallel':

        if w_init!=None:
            W = w_init
        else:
            W = np.random.normal(size=(n,n))+\
                1j*np.random.normal(size=(n,n))

        n_iter = 0

        #cache the covariance matrix
        C = np.cov(X)

        for i in range(max_iter):

            Wold = np.copy(W)

            for j in range(n):

                #derivative of the contrast function
                g  =  (1/(epsilon+abs_sqr(W[:,j],X))).reshape((1,m))
                #derivative of g
                dg = -(1/(epsilon+abs_sqr(W[:,j],X))**2).reshape((1,m))

                W[:,j]  = (X * (W[:,j].conj().T.dot(X)).conj() * g).mean(1)-\
                          (g + abs_sqr(W[:,j],X) * dg).mean() * W[:,j]
                del g,dg

            # Symmetric decorrelation
            Uw,Sw = eig(W.conj().T.dot(C.dot(W)))
            W     = W.dot(Sw.dot(inv(np.sqrt(np.diag(Uw))).dot(Sw.conj().T)))
            del Uw,Sw

            EG[:,n_iter] = (np.log(epsilon+abs_sqr(W,X))).mean(1)

            n_iter+=1

            lim = (abs(abs(Wold)-abs(W))).sum()
            if lim < tol:
                break

        if n_iter==max_iter and lim>tol:
            warnings.warn('FastICA did not converge. Consider increasing '
                          'tolerance or the maximum number of iterations.')

    S = W.conj().T.dot(X)

    return K,W,S,EG

#%% CREATE MIXTURE AND APPLY CFASTICA
import numpy as np
from numpy.linalg import eig, inv
from numpy.random import randn, rand
from libs_ch1 import separate_signals
import rfcutils
import time
from sigmf import SigMFFile

# Define functions for SINR, power, and MSE calculations
get_sinr = lambda s, i: 10 * np.log10(np.mean(np.abs(s) ** 2) / np.mean(np.abs(i) ** 2))
get_pow = lambda s: np.mean(np.abs(s) ** 2)
get_mse = lambda s: np.mean(np.abs(s) ** 2)


# Set parameters
mu = 0.0001
beta = 0.9997
sig_type = 'EMISignal1'
target_sinr_db = -30
sinr_db_2 = 15
dataset_type = 'train'


# Create mixture and reference signals
mixture1, mixture2, data1, data2, meta1, meta2, a12, a21 = rfcutils.create_sep_mixture_HJ(
    sig_type, target_sinr_db, sinr_db_2, seed=None, dataset_type=dataset_type)

s = np.array([mixture1, mixture2_pos]).astype('complex128')

x1 = data1
x2 = data2

X = s

# Run the signal separation
start_time = time.time()

# Apply FastICA
K, W, Shat, EG = complex_FastICA(X, max_iter=100, algorithm='parallel', n_components=2)

end_time = time.time()
loop_time = end_time - start_time

# Output the execution time
np.set_printoptions(suppress=True, precision=4)
print("Execution time of the algorithm: {:.4f} seconds".format(loop_time))

separated_signals = Shat


# Results and Plots
# Calculate correlation coefficients
corr_coef_x1_z1 = np.abs(np.corrcoef(x1, separated_signals[0, :])[0, 1])
corr_coef_x1_z2 = np.abs(np.corrcoef(x1, separated_signals[1, :])[0, 1])
corr_coef_x2_z1 = np.abs(np.corrcoef(x2, separated_signals[0, :])[0, 1])
corr_coef_x2_z2 = np.abs(np.corrcoef(x2, separated_signals[1, :])[0, 1])

# In case we get the separated signals in opposite outputs
if corr_coef_x1_z2 + corr_coef_x2_z1 > corr_coef_x1_z1 + corr_coef_x2_z2:
    # Swap separated signals
    separated_signals = np.array([separated_signals[1, :], separated_signals[0, :]])
    # Also swap the correlation coefficients
    corr_coef_x1_z1, corr_coef_x1_z2 = corr_coef_x1_z2, corr_coef_x1_z1
    corr_coef_x2_z1, corr_coef_x2_z2 = corr_coef_x2_z2, corr_coef_x2_z1
    print("\033[1;33m" + "Salidas cruzadas intercambiadas" + "\033[0m")

# Detect and correct signal inversion
if np.sum((x1.real * separated_signals[0, :].real) < 0) > (0.5 * len(x1)):
    separated_signals[0, :] *= -1
    print("\033[1;33m" + "Señal 1 invertida y corregida" + "\033[0m")
if np.sum((x2.real * separated_signals[1, :].real) < 0) > (0.5 * len(x2)):
    separated_signals[1, :] *= -1
    print("\033[1;33m" + "Señal 2 invertida y corregida" + "\033[0m")

# Calculate the optimal scaling factor using correlation method
alpha1 = np.sum(x1 * separated_signals[0, :]) / np.sum(separated_signals[0, :] ** 2)
alpha2 = np.sum(x2 * separated_signals[1, :]) / np.sum(separated_signals[1, :] ** 2)

separated_signals[0, :] *= alpha1
separated_signals[1, :] *= alpha2

print("Scaling factor for Signal 1:", alpha1)
print("Scaling factor for Signal 2:", alpha2)

# Calculate MSE
mse1 = get_mse(x1 - separated_signals[0, :])
mse2 = get_mse(x2 - separated_signals[1, :])

# Convert MSE to dB
mse1_db = 10 * np.log10(mse1)
mse2_db = 10 * np.log10(mse2)

# MSE as a percentage of the reference signal power
mse1_percent = 100 * mse1 / get_pow(x1)
mse2_percent = 100 * mse2 / get_pow(x2)

print(f"MSE for Signal 1: {mse1} (Linear), {mse1_db:.2f} dB, {mse1_percent:.4f}% of reference power")
print(f"MSE for Signal 2: {mse2} (Linear), {mse2_db:.2f} dB, {mse2_percent:.4f}% of reference power")

# Print correlation coefficients
print("\033[1;35m" + "Coeficiente correlacion x1 y z1: " + str(np.round(corr_coef_x1_z1, 8)) + "\033[0m")
print("Coeficiente correlacion x1 y z2:", np.round(corr_coef_x1_z2, 8))
print("Coeficiente correlacion x2 y z1:", np.round(corr_coef_x2_z1, 8))
print("\033[1;35m" + "Coeficiente correlacion x2 y z2: " + str(np.round(corr_coef_x2_z2, 8)) + "\033[0m")


#### PLOTS ####
span = 100
start = np.random.randint(m - span)

fig = plt.figure('fastICA_demo')
fig.clf()

ax1 = fig.add_subplot(121)
for j in range(2):
    ax1.plot(np.ma.masked_invalid(EG[j]), '.-', label='c_%i' % (j + 1))
ax1.set_ylabel('E[G(|W.T*X|^2)]')
ax1.set_xlabel('iteration #')
plt.legend(loc='best')
plt.show()


# 2X2 PLOTS TO VERIFY SEPARATION
fs1 = meta1.get_global_field(SigMFFile.SAMPLE_RATE_KEY)  # sampling rate from the meta data file
desc1 = meta1.get_global_field(SigMFFile.DESCRIPTION_KEY).split('--')  # description from meta data

# Change 1D numpy array to 1xN
data1_col = data1.reshape(1, -1)

fs2 = meta2.get_global_field(SigMFFile.SAMPLE_RATE_KEY)
desc2 = meta2.get_global_field(SigMFFile.DESCRIPTION_KEY).split('--')  # description from meta data
# Change 1D numpy array to 1xN
data2_col = data2.reshape(1, -1)

t_ax1 = np.arange(len(data1)) / fs1
t_ax2 = np.arange(len(data2)) / fs2

#### Final 2x2 time-domain plot to compare
plt.figure(figsize=(15, 10))
# Plot x1
plt.subplot(2, 2, 1)
plt.plot(t_ax1, data1.real)
plt.xlabel('Time (s)')
plt.title("Random sample from " + desc1[0])

# Plot x2
t_ax2 = np.arange(len(data2)) / fs2
plt.subplot(2, 2, 2)
plt.plot(t_ax2, data2.real, color='tab:green')
plt.xlabel('Time (s)')
plt.title("Original " + desc2[0])

# Plot separated_signals[0,:]
t_ax_sep1 = np.arange(len(separated_signals[0, :])) / fs1
plt.subplot(2, 2, 3)
plt.plot(t_ax_sep1, separated_signals[0, :].real)
plt.xlabel('Time (s)')
plt.title("Separated " + desc1[0])

# Plot separated_signals[1,:]
t_ax_sep2 = np.arange(len(separated_signals[1, :])) / fs2
plt.subplot(2, 2, 4)
plt.plot(t_ax_sep2, separated_signals[1, :].real, color='tab:green')
plt.xlabel('Time (s)')
plt.title("Separated " + desc2[0])
plt.show()


# %% PLOT ALL SINR LEVELS CFASTICA
from libs_ch1 import separate_signals
import rfcutils
import json


get_sinr = lambda s, i: 10*np.log10(np.mean(np.abs(s)**2)/np.mean(np.abs(i)**2))
get_pow = lambda s: np.mean(np.abs(s)**2)
get_mse = lambda s: np.mean(np.abs(s)**2)

get_sinr = lambda s, i: 10*np.log10(np.mean(np.abs(s)**2)/np.mean(np.abs(i)**2))
get_pow = lambda s: np.mean(np.abs(s)**2)
get_mse = lambda s: np.mean(np.abs(s)**2)


total_mse = 0 
total_mse_db = 0

sig_type = 'EMISignal1'  
dataset_type = 'train' 

# Specify ranges
sinr1_range = np.arange(-30, -28, 1)
sinr2_range = np.arange(0, 50, 1)



num_iterations = 50

mse_results = np.zeros((len(sinr1_range), len(sinr2_range)))

for i, sinr1 in enumerate(sinr1_range):
    sum_mse = 0
    for j, sinr2 in enumerate(sinr2_range):
        total_mse_db = 0
        for iteration in range(num_iterations):
            mixture1, mixture2, data1, data2, meta1, meta2, a12, a21 = rfcutils.create_sep_mixture_HJ(sig_type, sinr1, sinr2, seed=None, dataset_type=dataset_type)

           
            x1 = data1
            x2 = data2          

            s = np.array([mixture1, mixture2]).astype('complex128')
            

            X = s
            try:
            # Apply FastICA
                K, W, Shat, EG = complex_FastICA(X, max_iter=100, algorithm='parallel', n_components=2)
            
            except Exception as e:
                print("error")
                continue
            separated_signals = Shat
            
            # Calculate correlation coefficients
            corr_coef_x1_z1 = np.abs(np.corrcoef(x1, separated_signals[0, :])[0, 1])
            corr_coef_x1_z2 = np.abs(np.corrcoef(x1, separated_signals[1, :])[0, 1])
            corr_coef_x2_z1 = np.abs(np.corrcoef(x2, separated_signals[0, :])[0, 1])
            corr_coef_x2_z2 = np.abs(np.corrcoef(x2, separated_signals[1, :])[0, 1])

            # In case we get the separated signals in opposite outputs
            if corr_coef_x1_z2 + corr_coef_x2_z1 > corr_coef_x1_z1 + corr_coef_x2_z2:
                # Swap separated signals
                separated_signals = np.array([separated_signals[1, :], separated_signals[0, :]])
                # Also swap the correlation coefficients
                corr_coef_x1_z1, corr_coef_x1_z2 = corr_coef_x1_z2, corr_coef_x1_z1
                corr_coef_x2_z1, corr_coef_x2_z2 = corr_coef_x2_z2, corr_coef_x2_z1
                # print("\033[1;33m" + "Salidas cruzadas intercambiadas" + "\033[0m")

            # Detect and correct signal inversion
            if np.sum((x1.real * separated_signals[0, :].real) < 0) > (0.5 * len(x1)):
                separated_signals[0, :] *= -1
                # print("\033[1;33m" + "Señal 1 invertida y corregida" + "\033[0m")
            if np.sum((x2.real * separated_signals[1, :].real) < 0) > (0.5 * len(x2)):
                separated_signals[1, :] *= -1
                # print("\033[1;33m" + "Señal 2 invertida y corregida" + "\033[0m")

            # Calculate the optimal scaling factor using correlation method
            alpha1 = np.sum(x1 * separated_signals[0, :]) / np.sum(separated_signals[0, :] ** 2)
            alpha2 = np.sum(x2 * separated_signals[1, :]) / np.sum(separated_signals[1, :] ** 2)

            separated_signals[0, :] *= alpha1
            separated_signals[1, :] *= alpha2


            # Calculate MSE
            mse = get_mse(x1 - separated_signals[0, :])
            mse_db =  10 * np.log10(mse)
            mse_db = max(mse_db, -50)
            total_mse_db += mse_db
            # print("mse: ", mse)
            # print("mse_db: ", mse_db)

        
        average_mse_db = total_mse_db / num_iterations
        print(f"SINR1: {sinr1}, SINR2: {sinr2}, MSE (dB): {mse_db}")
        mse_results[i, j] = average_mse_db
        sum_mse += average_mse_db
        

sum_mses_for_each_sinr1 = {}
average_mses_for_each_sinr1 = {}
for i, sinr1 in enumerate(sinr1_range):
    # Sum MSE of all SINR2 levels
    sum_mse_for_sinr1 = sum(mse_results[i])
    # Average of MSE for all SINR2 levels
    average_mse_for_sinr1 = sum_mse_for_sinr1 / len(mse_results[i])
    
    sum_mses_for_each_sinr1[sinr1] = sum_mse_for_sinr1
    average_mses_for_each_sinr1[sinr1] = average_mse_for_sinr1
    
print("Sum and Average of MSEs for each SINR1:")    
for sinr1, total_mse in sum_mses_for_each_sinr1.items():
    print(f"SINR1 = {sinr1}: Sum of MSEs = {sum_mses_for_each_sinr1[sinr1]}, Average of MSEs = {average_mses_for_each_sinr1[sinr1]}")   

#####################
# Save results to a txt file
sinr1_start, sinr1_end = sinr1_range[0], sinr1_range[-1]
sinr2_start, sinr2_end = sinr2_range[0], sinr2_range[-1]
base_filename = f"FASTICA_pos_SINR1_[{sinr1_start},{sinr1_end}]_SINR2_[{sinr2_start},{sinr2_end}, alldb]"
txt_filepath = os.path.join("C:\\Users\\ibany\\Desktop\\MET\\Q4\\TFM\\Code\\2 MIT Challenge\\Challenge 1\\resultados\\plot_results\\txt\\fastica", f"{base_filename}.txt")
json_filepath = os.path.join("C:\\Users\\ibany\\Desktop\\MET\\Q4\\TFM\\Code\\2 MIT Challenge\\Challenge 1\\resultados\\plot_results\\txt\\fastica", f"{base_filename}.json")

# Save summary results to text file
with open(txt_filepath, 'w') as f:
    f.write("Sum and Average of MSEs for each SINR1:\n")
    for sinr1, total_mse in sum_mses_for_each_sinr1.items():
        f.write(f"SINR1 = {sinr1}: Sum of MSEs = {sum_mses_for_each_sinr1[sinr1]}, Average of MSEs = {average_mses_for_each_sinr1[sinr1]}\n")

# Save detailed results to JSON file
results = {
    "sinr1_range": sinr1_range.tolist(),
    "sinr2_range": sinr2_range.tolist(),
    "mse_results": mse_results.tolist()
}

with open(json_filepath, 'w') as f:
    json.dump(results, f)



#### PLOT ####
plt.figure(figsize=(10, 6))
for i, sinr1 in enumerate(sinr1_range):
    plt.plot(sinr2_range, mse_results[i, :], '-', label=f'SINR1 = {sinr1} dB')

plt.xlabel('SINR2 (dB)')
plt.ylabel('Average MSE (dB)')
plt.title('Average MSE at different SINR1 levels varying SINR2 ')
plt.text(0.5, 0.01, f'Parameters: mu = {mu}, beta = {beta}', fontsize=10, ha='center', va='top', transform=plt.gcf().transFigure)
plt.legend()
plt.grid(True)


# Save plot to plot_results folder
sinr1_start, sinr1_end = sinr1_range[0], sinr1_range[-1]
sinr2_start, sinr2_end = sinr2_range[0], sinr2_range[-1]
filename = f"FASTICA_pos_SINR1_[{sinr1_start},{sinr1_end}]_SINR2_[{sinr2_start},{sinr2_end}, alldb].png"
filepath = os.path.join("C:\\Users\\ibany\\Desktop\\MET\\Q4\\TFM\\Code\\2 MIT Challenge\\Challenge 1\\resultados\\plot_results\\txt\\fastica", filename)
plt.savefig(filepath, bbox_inches='tight')
plt.show()
