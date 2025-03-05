import mne
mne.set_log_level('ERROR')
import os
from dataloaders import load_mmidb_files

# load data:
print('loading data')
mmidb_dir = 'paths*'
output_dir = 'paths*'
raws = load_mmidb_files(mmidb_dir, task_numbers=[3, 4, 7, 8, 11, 12])
print('data loaded')

for raw in raws:
    for i, annotation in enumerate(raw.annotations):
        # Get annotation details
        start = annotation['onset']  # Start time of the annotation
        duration = annotation['duration']  # Duration of the annotation
        description = annotation['description']  # Annotation label (e.g., T0, T1, T2)
        
        # Create a cropped version of the raw object
        raw_window = raw.copy().crop(tmin=start, tmax=min(start + duration, raw.times[-1]))
        
        file_name = raw.filenames[0].split('/')[-1].replace('.edf', f'_w{i}_{description}.edf')
        file_path = os.path.join(output_dir, file_name)
        raw_window.export(file_path, fmt='edf', overwrite=True)

# add tqdm     
print('done')
