import os
from torch.utils.data import Dataset
import mne
import pandas as pd
from mne.io import read_raw_edf
from pathlib import Path


# class EDFDataset(Dataset):
#     def __init__(self, edf_dir: str, label_extension: str = ".rec"):
#         self.edf_paths = []
#         self.label_extension = label_extension
        
#         for root, dirs, files in os.walk(edf_dir):
#             for file in files:
#                 if file.endswith(".edf"):
#                     self.edf_paths.append(os.path.join(root, file))
        
#         self.edf_paths.sort()
        
#     def __len__(self):
#         return len(self.edf_paths)
    
#     def _get_label_path(self, edf_path):
#         return edf_path.replace(".edf", self.label_extension)

#     def _load_edf_data(self, edf_path):
#         raw = mne.io.read_raw_edf(edf_path, preload=True)
#         return raw
    
#     def _load_labels(self, label_path):
#         if os.path.exists(label_path):
#             labels = pd.read_csv(label_path, sep=",", header=None, names=['channel_number', 'start_seconds', 'stop_seconds', 'label'])
#             return labels
#         else:
#             raise FileNotFoundError(f"Label file {label_path} not found")

#     def __getitem__(self, index):
#         edf_path = self.edf_paths[index]
#         label_path = self._get_label_path(edf_path)
        
#         raw = self._load_edf_data(edf_path)
#         labels = self._load_labels(label_path)
        
#         return {
#             'raw': raw,  # Return the raw MNE object
#             'labels': labels  # Labels DataFrame
#         }


# def collate_fn(batch):
#     raws = [item['raw'] for item in batch]
#     labels = [item['labels'] for item in batch]
    
#     return {
#         'raw': raws,  # List of MNE Raw objects
#         'labels': labels  # List of labels DataFrames
#     }



class EDFDataset(Dataset):
    """Loads EDF files from a directory."""
    def __init__(self, edf_dir: str):
        self.edf_paths = sorted([os.path.join(root, file)
                                 for root, _, files in os.walk(edf_dir)
                                 for file in files if file.endswith(".edf")])

    def __len__(self):
        return len(self.edf_paths)

    def _load_edf_data(self, edf_path):
        """Reads EDF file and returns raw MNE object."""
        return mne.io.read_raw_edf(edf_path, preload=True)

    def __getitem__(self, index):
        return {'raw': self._load_edf_data(self.edf_paths[index])}




def load_mmidb_files(base_path, task_numbers = None):
    """
    Recursively finds all files for specific task numbers in a given base directory.

    Parameters:
        base_path (str or Path): The root directory to search.
        task_numbers (list[int]): A list of task numbers (e.g., [3, 4, 7, 8, 11, 12]).

    Returns:
        list[Path]: A list of file paths matching the task numbers.
    """
    base_path = Path(base_path)
    path_mmidb_files = [] 
    
    if task_numbers:
        valid_suffixes = tuple(f"R{str(task).zfill(2)}.edf" for task in task_numbers)
    else:
        valid_suffixes = ".edf"
    
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(valid_suffixes):
                path_mmidb_files.append(Path(root) / file)
    
    raws = [read_raw_edf(x) for x in path_mmidb_files]
    return raws