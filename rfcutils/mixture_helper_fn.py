# @author: ibany


# STRUCTURE OF SECTIONS:
#     - Imports
#     - Sep mixture function


#%% IMPORTS
import numpy as np
import random

from .sigmf_helper_fn import write_sigmf_file, read_sigmf_file
from .dataset_helper_fn import load_dataset_sample

window_len = 40960
get_rand_start_idx = lambda sig_len: np.random.randint(sig_len-window_len)
get_sinr = lambda s, i: 10*np.log10(np.mean(np.abs(s)**2)/np.mean(np.abs(i)**2))
get_pow = lambda s: np.mean(np.abs(s)**2)



#%% Sep mixture
num_train_frame = {"EMISignal1": 530, "CommSignal2": 100, "CommSignal3": 139}
num_trainval_frame = {"EMISignal1": 580, "CommSignal2": 150, "CommSignal3": 189}


def create_sep_mixture_HJ(sig_type, target_sinr_db, sinr_db_2, seed=None, dataset_type="train"):   
    #### FIRST PART SAME AS ORIGINAL FUNCTION FROM THE MIT RF CHALLENGE ####
    # Function first gets CommSignal2 and then the specified sig_type
    np.random.seed(seed)
    random.seed(seed)
    
    if dataset_type == "all" or dataset_type == "val":
        comm2_chosen_idx = np.random.randint(num_trainval_frame["CommSignal2"])
    elif dataset_type == "train":
        comm2_chosen_idx = np.random.randint(num_train_frame["CommSignal2"])
    
    comm2_data, comm2_meta = load_dataset_sample(comm2_chosen_idx, 'train_frame', 'CommSignal2')
    
    
    if dataset_type == "all":
        chosen_idx = np.random.randint(num_trainval_frame[sig_type])
    elif dataset_type == "train":
        chosen_idx = np.random.randint(num_train_frame[sig_type])
    elif dataset_type == "val":
        if comm2_chosen_idx < num_train_frame["CommSignal2"]:
            # mixture should be from an unseen pool:
            chosen_idx = np.random.randint(num_train_frame[sig_type], num_trainval_frame[sig_type])
    
    data, meta = load_dataset_sample(chosen_idx, 'train_frame', sig_type)
    
    
    
    #### FIRST MIXTURE ####
    # CommSignal2 window_len (40960) segment
    comm2_start_idx = get_rand_start_idx(len(comm2_data))
    comm2_segment = comm2_data[comm2_start_idx:comm2_start_idx+window_len]
    
    # 2nd signal segment
    start_idx = get_rand_start_idx(len(data))
    sig_type_segment = data[start_idx:start_idx+window_len]
    
    # Power of each signal segment
    power_comm2 = get_pow(comm2_segment)
    power_sig = get_pow(sig_type_segment)
    
    # First mixture with CommSignal2 at original level and sig_type adjusted for target SINR
    coeff_1_sinr = np.sqrt(np.mean(np.abs(comm2_segment)**2)/(np.mean(np.abs(sig_type_segment)**2)*(10**(target_sinr_db/10))))
    mixture1 = comm2_segment +  coeff_1_sinr * sig_type_segment
    
    #### SECOND MIXTURE ####
    # Adjust sig_type to match the target SINR for the 2nd mixture
    coeff_2_sinr = np.sqrt(np.mean(np.abs(sig_type_segment)**2/((np.mean(np.abs(comm2_segment)**2))*(10**(sinr_db_2/10)))))
    mixture2 =   coeff_2_sinr * comm2_segment + sig_type_segment    
        
    # Prints for verification
    # print(f"Power of 1st signal: {power_comm2:.2f}")
    # print(f"Power of 2nd signal: {power_sig:.2f}") 
    # print(f"Mixture 1: x1 + {coeff_1_sinr:.2f}*x2")
    # print(f"Mixture 2: {coeff_2_sinr:.2f}*x1 + x2")
    # print(f"|c1*c2|= {np.abs(coeff_1_sinr * coeff_2_sinr):.2f}")
    # print(f"SINR for Mixture 1: {get_sinr(comm2_segment, coeff_1_sinr * sig_type_segment):.2f}")
    # print(f"SINR for Mixture 2: {get_sinr(sig_type_segment, coeff_2_sinr * comm2_segment):.2f}")
    return mixture1, mixture2, comm2_segment, sig_type_segment, comm2_meta, meta, coeff_1_sinr, coeff_2_sinr
