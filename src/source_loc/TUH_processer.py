# source: https://github.com/gjoelbye/TCAV-BENDR/blob/development/data_process/TUH_processer.py

import os, sys, pickle, random, datetime, argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.cm as cm
from tqdm import tqdm
from pathlib import Path
import mne
if os.getcwd().split("/")[-1] != 'BENDR-XAI': os.chdir("../")
sys.path.append(os.getcwd())
from utils import * # if problem - delete source_loc
from functools import partial
import multiprocessing

def compute_power(window_dict, labels, fwd, compute_inverse):
    power_dict = {} # for storing the power of each label for each window
    total_count = 0
    sum_of_means = np.zeros(fwd['nsource'])

    # Iterate through all annotations
    for annotation in window_dict.keys():
        power_dict[annotation] = np.empty((len(window_dict[annotation]), sum(len(hemi) for hemi in labels)))

        # Iterate through all chopped data segments
        for idx, window in enumerate(window_dict[annotation]):        
            src_estimate = compute_inverse(window) # shape: 20484, 1024 - ie voxels, time points
            sum_of_means += np.mean(src_estimate.data, axis=1) # 20484 voxel  values for each window # add mean window voxel values for every window.
            # ie 20484 mean voxel values x windows = 20484 summed voxel values 
            total_count += 1

            src_power_label = [np.empty(len(labels[0])), np.empty(len(labels[1]))] # 2 empty lists for each hemisphere

            # Calculate power for each label
            for hemi in range(2):
                for i in range(len(labels[hemi])):
                    src_estimate_label = src_estimate.in_label(labels[hemi][i]) # only the voxels in the label/ area; shape: 
                    src_power_label[hemi][i] = np.mean(src_estimate_label.data**2) # **2 to get power; mean over time ?;  shape:

            power_dict[annotation][idx] = np.concatenate(src_power_label) # shape
            
    # Compute the true mean 
    true_mean = (sum_of_means / total_count).reshape(-1, 1) # real mean of the voxel values over all windows ie whole recording

    return power_dict, true_mean

def compute_variance(window_dict, labels, compute_inverse, true_mean):
    # Initialize the variance dictionary
    variance_dict = {}

    # Iterate through all annotations
    for annotation in window_dict.keys():
        variance_dict[annotation] = np.empty((len(window_dict[annotation]), sum(len(hemi) for hemi in labels)))

        # Iterate through all chopped data segments
        for idx, window in enumerate(window_dict[annotation]):
            src_estimate = compute_inverse(window)
            src_variance_label = [np.empty(len(labels[0])), np.empty(len(labels[1]))]

            # Calculate variance for each label
            for hemi in range(2):
                for i in range(len(labels[hemi])):
                    src_estimate_label = src_estimate.in_label(labels[hemi][i])
                    true_mean_label = true_mean[labels[hemi][i].get_vertices_used()]
                    src_variance_label[hemi][i] = np.mean((src_estimate_label.data - true_mean_label)**2)

            variance_dict[annotation][idx] = np.concatenate(src_variance_label)

    return variance_dict

def process_file(file_path, labels, fwd, high_pass, low_pass, window_length, end_crop, snr, proj):
    raw = read_TUH_edf(file_path, high_pass=high_pass, low_pass=low_pass, proj=proj)
    
    # Length of the recording in seconds
    length = (raw.n_times / raw.info['sfreq'])

    # Diregard the first 5 seconds and the last 5 seconds of the recording
    onset = np.arange(end_crop, length-end_crop, window_length)

    # Duration of each chopped segment
    duration = np.repeat(window_length, len(onset)) - 1/raw.info['sfreq']

    # Description of each chopped segment
    description = np.repeat('T0', len(onset))
    
    # Create the annotations object
    annotations = mne.Annotations(onset=onset, duration=duration, description=description) # useless for TUH

    # annotations = mne.Annotations(onset=onset, duration=duration, description=description)
    window_dict, annotations_dict = get_window_dict_extra(raw, annotations)
    
    cov = get_cov(raw) # Get the covariance matrix
    compute_inverse = make_fast_inverse_operator(raw.info, fwd, cov, snr=snr)

    # Process annotations to get power_dict, sum_of_means, and total_count
    power_dict, true_mean = compute_power(window_dict, labels, fwd, compute_inverse)

    # Compute the variance dictionary
    variance_dict = compute_variance(window_dict, labels, compute_inverse, true_mean)    
    
    return (file_path.name, power_dict, variance_dict, annotations_dict)

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--high_pass", type=float, default=0.1, help="High pass frequency in Hz")
    parser.add_argument("--low_pass", type=float, default=100.0, help="Low pass frequency in Hz")
    parser.add_argument("--end_crop", type=float, default=5.0, help="Length of the recording to disregard at the beginning and end in seconds")
    parser.add_argument("--window_length", type=float, default=4.0, help="Length of each chopped segment in seconds")
    parser.add_argument("--n_processes", type=int, default=-1, help="Number of processes to use for multiprocessing")
    parser.add_argument("--snr", type=float, default=100.0, help="Signal to noise ratio for computing the inverse operator")
    parser.add_argument("--edf_dir", type=str, default="/scratch/s194260/tuh_eeg", help="Path to the directory containing the EDF files")
    parser.add_argument("--parcellation_name", type=str, default="HCPMMP1_combined", help="Name of the parcellation to use")
    parser.add_argument("--save_dir", type=str, default="", help="Path to the directory to save the processed data")
    parser.add_argument("--name", type=str, default="", help="Name of the experiment")
    parser.add_argument('--proj', type=str, default=False, help="Whether to apply SSP projection vectors")
    args = parser.parse_args()
    
    end_crop = args.end_crop # Length of the recording to disregard at the beginning and end in seconds
    window_length = args.window_length # Length of each chopped segment in seconds
    n_processes = args.n_processes # Number of processes to use for multiprocessing
    low_pass = args.low_pass # Low pass frequency in Hz
    high_pass = args.high_pass # High pass frequency in Hz
    snr = args.snr # Signal to noise ratio for computing the inverse operator
    edf_dir = Path(args.edf_dir) # Path to the directory containing the EDF files
    parcellation_name = args.parcellation_name # Name of the parcellation to use
    save_dir = Path(args.save_dir) # Path to the directory to save the processed data
    name = args.name
    
    assert args.proj in ['True', 'False'], "proj must be either True or False"
    
    proj = True if args.proj == 'True' else False
    
    now = datetime.datetime.now()
    now_str = now.strftime("%H%M%S_%d%m%y")
    tqdm.write(f"[INFO] Starting at {now_str}")
    
    tqdm.write(f"[INFO] High pass: {high_pass} Hz")
    tqdm.write(f"[INFO] Low pass: {low_pass} Hz")
    tqdm.write(f"[INFO] End crop: {end_crop} s")
    tqdm.write(f"[INFO] Window length: {window_length} s")
    tqdm.write(f"[INFO] Number of processes: {n_processes}")
    tqdm.write(f"[INFO] Signal to noise ratio: {snr}")
    tqdm.write(f"[INFO] EDF directory: {edf_dir}")
    tqdm.write(f"[INFO] Parcellation name: {parcellation_name}")
    tqdm.write(f"[INFO] Save directory: {save_dir}")
    tqdm.write(f"[INFO] Name: {name}")
    tqdm.write(f"[INFO] Projection: {proj}")
    
    # Get paths
    subjects_dir, subject, trans, src_path, bem_path = get_fsaverage() # import data from freesurfer
    
    # Get the labels for the parcellation
    #labels = get_labels(subjects_dir, parcellation_name = parcellation_name) # get the labels for the parcellation from freesurfer ############################################# temp removed
    
    tqdm.write(f"[INFO] Loaded {parcellation_name} parcellation")
        # Get EDF file paths    
    edf_files = [edf_dir / file for file in os.listdir(edf_dir)] #[:2] # !!!!!!!!!!!!!!! change back to all files
   
    
    tqdm.write(f"[INFO] Found {len(edf_files)} EDF files")
    
    # Get forward model
    info = read_TUH_edf(edf_files[0]).info # read the first edf file to get the info for the forward model
    
    fwd = get_fwd(info, trans, src_path, bem_path) # get the forward model 

    src = fwd['src'] ############################################# temp added
    print('custom parcellation - 1 area per hemisphere')
    labels = [[mne.Label(src[0]['vertno'], hemi='lh')], [mne.Label(src[1]['vertno'], hemi='rh')]] ############################################# temp added

    
    tqdm.write("[INFO] Forward model loaded")
    
    custom_functions = partial(process_file, labels=labels, fwd=fwd, high_pass=high_pass, # file processing, afterwards code irrelevant until results and saving
                               low_pass=low_pass, window_length=window_length,
                               end_crop=end_crop, snr=snr, proj=proj)
    
    tqdm.write(f"[INFO] Custom functions defined")
    
    if n_processes == -1:
        pool = multiprocessing.Pool()
    else:
        pool = multiprocessing.Pool(processes=n_processes)
        
    tqdm.write(f"[INFO] Starting multiprocessing with {pool._processes} processes with {multiprocessing.cpu_count()} CPUs available")
    
    results = []
    with tqdm(total = len(edf_files)) as pbar:
        for result in pool.imap_unordered(custom_functions, edf_files):
            results.append(result)
            pbar.update(1)
            pbar.set_description(f"Running... {result[0]}")

        pool.close()
        pool.join()
    
    tqdm.write(f"[INFO] Finished multiprocessing")    
    
    dataset = dict()

    for result in results:
        dataset[result[0]] = {
            "power": result[1],
            "variance": result[2],
            "annotations": result[3]
        }
    
    tqdm.write(f"[INFO] Created dataset")
    
    proj_applied = "proj" if proj else "no_proj"
    
    output_name = f"{parcellation_name}_{high_pass}_{low_pass}_{snr}_{now_str}_{proj_applied}_{window_length}.npy" # changed name -> window_length
    
    np.save(save_dir / output_name, dataset, allow_pickle=True)
    
    tqdm.write(f"[INFO] Saved dataset to {output_name}")
    tqdm.write(f"[INFO] Total time: {str(datetime.datetime.now() - now)}")

# python src/source_loc/source_loc.py $(cat src/source_loc/temp_config.txt)
# change to normal - not multiprocessing