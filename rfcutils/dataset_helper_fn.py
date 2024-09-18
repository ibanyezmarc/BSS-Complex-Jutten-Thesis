### FILE FROM MIT RF CHALLENGE FUNCTIONS LIBRARY
### Changed paths

import os
import warnings
import numpy as np
import pickle

from .sigmf_helper_fn import write_sigmf_file, read_sigmf_file

def load_dataset_sample(idx, dataset_type, sig_type):
    #foldername = os.path.join('dataset',dataset_type,sig_type)
    #filename = f'{sig_type}_{dataset_type}_{idx:04d}'
    base_dir = r'C:\Users\ibany\Desktop\MET\Q4\TFM\Code\2 MIT Challenge\Challenge 1\rfc_dataset'
    foldername = os.path.join(base_dir, 'dataset', dataset_type, sig_type)
    filename = f'{sig_type}_{dataset_type}_{idx:04d}'
    # Special handling for "Separation" validation and test set; only using Comm2 vs [sig_type] for this iteration
    if 'sep_' in dataset_type:
        filename = f'CommSignal2_vs_{sig_type}_{dataset_type}_{idx:04d}'
    data, meta = read_sigmf_file(filename=filename, folderpath=foldername)
    return data, meta

def load_dataset_sample_components(idx, dataset_type, sig_type):
    assert 'train' in dataset_type or 'val' in dataset_type or 'test' in dataset_type, f'Invalid dataset type requested for obtaining components: {dataset_type}'
    
    soi_name = 'Comm2' if 'sep_' in dataset_type else 'QPSK'
    foldername1 = os.path.join('dataset',dataset_type,'Components', sig_type, soi_name)
    filename1 = f'{sig_type}_{dataset_type}_{idx:04d}'
    # Special handling for "Separation" validation and test set; only using Comm2 vs [sig_type] for this iteration
    if 'sep_' in dataset_type:
        filename1 = f'CommSignal2_vs_{sig_type}_{dataset_type}_{idx:04d}'
    data1, meta1 = read_sigmf_file(filename=filename1, folderpath=foldername1)
    
    foldername2 = os.path.join('dataset',dataset_type,'Components', sig_type, 'Interference')
    filename2 = f'{sig_type}_{dataset_type}_{idx:04d}'
    # Special handling for "Separation" validation and test set; only using Comm2 vs [sig_type] for this iteration
    if 'sep_' in dataset_type:
        filename2 = f'CommSignal2_vs_{sig_type}_{dataset_type}_{idx:04d}'
    data2, meta2 = read_sigmf_file(filename=filename1, folderpath=foldername2)
    
    return data1, meta1, data2, meta2



