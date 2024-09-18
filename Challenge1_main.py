
# @author: ibany


# STRUCTURE OF SECTIONS:
#     - Imports
#     - See 2 input Signals
#     - Main
#     - Plot all sinr levels


#%% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir(r'C:\Users\ibany\Desktop\MET\Q4\TFM\Code\2 MIT Challenge\Challenge 1')
import rfcutils
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sigmf import SigMFFile
from numba import jit
import time
import warnings
warnings.filterwarnings("ignore")
from libs_ch1 import separate_signals
import json

get_sinr = lambda s, i: 10*np.log10(np.mean(np.abs(s)**2)/np.mean(np.abs(i)**2))
get_pow = lambda s: np.mean(np.abs(s)**2)
get_mse = lambda s: np.mean(np.abs(s)**2)



# %% See 2 input signals
sig_type = 'EMISignal1'  
target_sinr_db = 4
sinr_db_2 = -4
dataset_type = 'train'  

mixture1, mixture2, data1, data2, meta1, meta2, a12, a21 = rfcutils.create_sep_mixture_HJ(
    sig_type, target_sinr_db, sinr_db_2, seed=None, dataset_type=dataset_type)

fs1 = meta1.get_global_field(SigMFFile.SAMPLE_RATE_KEY) # sampling rate from the meta data file
desc1 = meta1.get_global_field(SigMFFile.DESCRIPTION_KEY).split('--') # description from meta data

# Change 1D numpy array to 1xN
data1_col = data1.reshape(1, -1)
print("data1 size:", len(data1))


fs2 = meta2.get_global_field(SigMFFile.SAMPLE_RATE_KEY)
desc2 = meta2.get_global_field(SigMFFile.DESCRIPTION_KEY).split('--') # description from meta data
# Change 1D numpy array to 1xN
data2_col = data2.reshape(1, -1)
print("data2 size:", len(data2))

t_ax1 = np.arange(len(data1))/fs1
t_ax2 = np.arange(len(data2))/fs2

# Plot original signal segments
plt.figure(figsize=(15, 5 ))
plt.subplot(1, 2, 1)
plt.plot(t_ax1, data1.real)
plt.xlabel('Time (s)')
plt.title(desc1[0])
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(t_ax2, data2.real)
plt.xlabel('Time (s)')
plt.title(desc2[0])
plt.grid()


print("fs1:", fs1)
print("fs2:", fs2)



# %% MAIN
import numpy as np
import matplotlib.pyplot as plt
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
target_sinr_db = -4
sinr_db_2 = 4
dataset_type = 'train'


# Create mixture and reference signals
mixture1, mixture2, data1, data2, meta1, meta2, a12, a21 = rfcutils.create_sep_mixture_HJ(
    sig_type, target_sinr_db, sinr_db_2, seed=None, dataset_type=dataset_type)


num_iter = len(mixture1)
nin = num_iter / 7
x1 = data1
x2 = data2

A = np.array([[1, a12],
              [a21, 1]])

s = np.array([mixture1, mixture2]).astype('complex128')


c12 = 0.0 + 0.0j
c21 = 0.0 + 0.0j

# Run the signal separation
start_time = time.time()

C, gradient_term_1, gradient_term_2 = separate_signals(s, mu, beta, nin, num_iter, c12, c21)

# Calculate the final product for verification
final_product = np.linalg.inv(C).dot(A)

# Save separated signals
separated_signals = np.linalg.solve(C, s)

end_time = time.time()
loop_time = end_time - start_time

# Output of execution time
np.set_printoptions(suppress=True, precision=4)
print("Execution time of the algorithm: {:.4f} seconds".format(loop_time))


# Output of results
print('Final estimated matrix C:')
print(np.matrix.round(C, 4))
print('Final inv(C)*A (desired result I):')
print(np.matrix.round(final_product, 4))


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

# Calculate the optimal scaling factor
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

# Plot original signal segments
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(t_ax1, data1.real)
plt.xlabel('Time (s)')
plt.title(desc1[0])
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(t_ax2, data2.real, color='tab:green')
plt.xlabel('Time (s)')
plt.title(desc2[0])
plt.grid()

# Change 1D numpy array to 1xN
t_ax1 = np.arange(len(x1)) / fs1

plt.figure(3)
plt.plot(t_ax1, x1.real)
plt.xlabel('Time (s)')
plt.title("Sample taken from " + desc2[0])
plt.show()

# Plot histograms of x1 and x2
plt.figure(figsize=(8, 5))
plt.subplot(1, 2, 1)
plt.hist(x1.real, 50)
plt.title('Histogram of x1')
plt.grid()

plt.subplot(1, 2, 2)
plt.hist(x2.real, 100, color='tab:green')
plt.title('Histogram of x2')
plt.grid()

# Plot histograms of the mixed signals
plt.figure(figsize=(8, 5))
plt.subplot(1, 2, 1)
plt.hist(s[0, :].real, 100, color='brown')
plt.title('Histogram of Mixed Signal 1')
plt.grid()

plt.subplot(1, 2, 2)
plt.hist(s[1, :].real, 100, color='tab:orange')
plt.title('Histogram of Mixed Signal 2')
plt.grid()

# Histograms of the separated signals
plt.figure(figsize=(8, 5))
plt.subplot(1, 2, 1)
plt.hist(separated_signals[0, :].real, 100)
plt.title('Histogram of Separated Signal 1')
plt.grid()

plt.subplot(1, 2, 2)
plt.hist(separated_signals[1, :].real, 100, color='tab:green')
plt.title('Histogram of Separated Signal 2')
plt.grid()

# Two gradient evolutions
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_iter + 1), gradient_term_1)
plt.xlabel('Iteration')
plt.ylabel('Gradient of term_1')
plt.title(r'Evolution of $(y_1)^2 \cdot y_1^* \cdot y_2^*$')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_iter + 1), gradient_term_2, color='tab:green')
plt.grid()
plt.xlabel('Iteration')
plt.ylabel('Gradient of term_1')
plt.title(r'Evolution of $(y_2)^2 \cdot y_2^* \cdot y_1^*$')

# Separated signals over time
t_ax1 = np.arange(len(separated_signals[0, :])) / fs1
plt.figure(8)
plt.plot(t_ax1, separated_signals[0, :].real)
plt.xlabel('Time (s)')
plt.title("Separated " + desc2[0])
plt.show()

t_ax2 = np.arange(len(separated_signals[1, :])) / fs2
plt.figure(9)
plt.plot(t_ax2, separated_signals[1, :].real, color='tab:green')
plt.xlabel('Time (s)')
plt.title("Separated " + desc2[0])
plt.show()

#### Final 2x2 time-domain plot to compare
plt.figure(figsize=(15, 10))

# Plot x1
t_ax1 = np.arange(len(x1)) / fs1
plt.subplot(2, 2, 1)
plt.plot(t_ax1, x1.real)
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

#### Final 2x2 spectrograms plot to compare
plt.figure(figsize=(16, 10))

# Spectrogram of data1
plt.subplot(221)
plt.specgram(data1, Fs=fs1, NFFT=256, scale='dB', cmap='ocean')
plt.title("Spectrogram " + desc1[0])

# Spectrogram of data2
plt.subplot(222)
plt.specgram(data2, Fs=fs2, NFFT=256, scale='dB')
plt.title("Spectrogram " + desc2[0])

# Spectrogram of separated signal 1
plt.subplot(223)
plt.specgram(separated_signals[0, :], Fs=fs1, NFFT=256, scale='dB', cmap='ocean')
plt.title("Spectrogram of Separated " + desc1[0])

# Spectrogram of separated signal 2
plt.subplot(224)
plt.specgram(separated_signals[1, :], Fs=fs2, NFFT=256, scale='dB')
plt.title("Spectrogram of Separated " + desc2[0])

plt.tight_layout()
plt.show()

#### Time-Domain Plot with Custom Window
t_axis = np.arange(40960) / fs1
xlim_window = [4.0e-4, 4.1e-4]

plt.figure(figsize=(16, 7))

# Plot for CommSignal2
plt.subplot(1, 2, 1)
plt.plot(t_axis, x1.real, '--', color='tab:green')
plt.plot(t_axis, separated_signals[0, :].real, color='tab:red', alpha=0.6)
plt.ticklabel_format(scilimits=[-2, 3])
plt.xlabel('Time (s)')
plt.xlim(xlim_window)
plt.legend(['True $s$', 'Estimated $\hat s$'])
plt.title('Estimated Component 1 -- '+ desc1[0])

# Plot for EMISignal
plt.subplot(1, 2, 2)
plt.plot(t_axis, x2.real, '--', color='tab:brown')
plt.plot(t_axis, separated_signals[1, :].real, color='tab:cyan', alpha=0.6)
plt.ticklabel_format(scilimits=[-2, 3])
plt.xlabel('Time (s)')
plt.xlim(xlim_window)
plt.legend(['True $b$', 'Estimated $\hat b$'])
plt.title(f'Estimated Component 2 -- {sig_type}')
plt.show()



# %% PLOT ALL SINR LEVELS
from libs_ch1 import separate_signals
import rfcutils
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
sinr1_range = np.arange(-20, -12, 1)
sinr2_range = np.arange(0, 40, 1)



num_iterations = 200

mse_results = np.zeros((len(sinr1_range), len(sinr2_range)))

for i, sinr1 in enumerate(sinr1_range):
    sum_mse = 0
    for j, sinr2 in enumerate(sinr2_range):
        total_mse_db = 0
        for iteration in range(num_iterations):
            mixture1, mixture2, data1, data2, meta1, meta2, a12, a21 = rfcutils.create_sep_mixture_HJ(sig_type, sinr1, sinr2, seed=None, dataset_type=dataset_type)

            mu = 0.0001
            beta = 0.9997
           
            x1 = data1
            x2 = data2          
            num_iter = len(mixture1)
            nin = num_iter/7

            c12 = 0.0 + 0.0j; 
            c21 = 0.0 + 0.0j; 
            s = np.array([mixture1, mixture2]).astype('complex128')
            
            # Signal separation
            C, gradient_term_1, gradient_term_2 = separate_signals(s, mu, beta, nin, num_iter, c12, c21)
            separated_signals = np.linalg.solve(C, s)
            
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
base_filename = f"pos_mu_{mu}_beta_{beta}_SINR1_[{sinr1_start},{sinr1_end}]_SINR2_[{sinr2_start},{sinr2_end}, alldb]"
txt_filepath = os.path.join("C:\\Users\\ibany\\Desktop\\MET\\Q4\\TFM\\Code\\2 MIT Challenge\\Challenge 1\\resultados\\plot_results\\txt", f"{base_filename}.txt")
json_filepath = os.path.join("C:\\Users\\ibany\\Desktop\\MET\\Q4\\TFM\\Code\\2 MIT Challenge\\Challenge 1\\resultados\\plot_results\\txt", f"{base_filename}.json")

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
filename = f"pos_mu_{mu}_beta_{beta}_SINR1_[{sinr1_start},{sinr1_end}]_SINR2_[{sinr2_start},{sinr2_end}].png"
filepath = os.path.join("C:\\Users\\ibany\\Desktop\\MET\\Q4\\TFM\\Code\\2 MIT Challenge\\Challenge 1\\resultados", filename)
plt.savefig(filepath, bbox_inches='tight')
plt.show()