import os
import shutil
import numpy as np
import logging
from pathlib import Path
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))
from SPEED.src.pipeline import Pipeline, DynamicPipeline

import mne
mne.set_log_level("ERROR")

def setup_logging():
    """Configures logging for the script."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def copy_selected_files(src_list_path, src_root, dest_dir):
    """Copies selected files from source to destination directory, skipping existing ones."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    with open(src_list_path, 'r') as f:
        selected_files = {Path(line.strip()).name for line in f}
    
    copied_files = 0
    for root, _, files in os.walk(src_root):
        for file in files:
            if file in selected_files:
                dest_path = dest_dir / file
                if not dest_path.exists():
                    shutil.copy(os.path.join(root, file), dest_path)
                    copied_files += 1
    logging.info(f"Copied {copied_files} new files to {dest_dir}")

def preprocess_files(src_dir, dest_dir, do_ica=False, line_freqs=[60], batch_size=10, n_jobs=6, shuffle_files=True):
    """Preprocesses EEG files using SPEED pipeline and saves them as EDF files, skipping existing ones."""
    src_dir = Path(src_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    src_paths = list(src_dir.glob("*.edf")) # !!! For testing! beware to change this for preprocessing all files!!!!!!!!
    if shuffle_files:
        np.random.shuffle(src_paths)
      # Process all available files
    pipeline = DynamicPipeline(do_ica=do_ica, line_freqs=line_freqs)
    
    src_batches = [src_paths[i:i + batch_size] for i in range(0, len(src_paths), batch_size)]
    
    logging.info(f"Total files: {len(src_paths)} | Batch size: {batch_size} | Total batches: {len(src_batches)}")
    
    from joblib import Parallel, delayed
    Parallel(n_jobs=n_jobs)(delayed(process_batch)(pipeline, batch, dest_dir) for batch in src_batches)

def process_batch(pipeline, batch, dest_dir):
    for src_path in batch:
        output_filename = dest_dir / f"{src_path.stem}_w0.edf"  # check only the first window
        if output_filename.exists():
            logging.info(f"Skipping already processed file: {output_filename}")
            continue
        
        raws, times, indices = pipeline([src_path])
        for i, (raw, idx) in enumerate(zip(raws, indices)):
            output_filename = dest_dir / f"{src_path.stem}_w{i}.edf"
            raw.export(str(output_filename), fmt='edf', overwrite=True) 
            logging.info(f"Saved {output_filename}")

def main():
    setup_logging()
    copy_data = False
    #src_list_path = "tuh_final_selected.txt"
    src_list_path = Path(__file__).parent / "tuh_final_selected.txt"
    src_root = "paths*"
    dest_dir = "paths*"
    preprocessed_dir = "paths*"
    
    if copy_data:
        copy_selected_files(src_list_path, src_root, dest_dir)
    
    preprocess_files(dest_dir, preprocessed_dir, batch_size=10, n_jobs=6, shuffle_files=True)
    
if __name__ == "__main__":
    main()
