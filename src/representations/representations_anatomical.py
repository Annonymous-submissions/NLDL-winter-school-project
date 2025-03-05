import os
import logging
import numpy as np
import torch
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import pickle
from torch.utils.data import DataLoader
from eegatscale.models.fullbendr import FullBENDR

# Import utility functions
from src.utils.dataloaders import EDFDataset
from src.utils.model_loaders import HookManager

def load_model(checkpoint_path, device):
    """
    Load the FullBENDR model from checkpoint.
    """
    full_bendr = FullBENDR(checkpoint_path, encoder_h=512, in_features=19, out_features=2)
    state_dict = torch.load(checkpoint_path, map_location=device)['state_dict']
    full_bendr.load_state_dict(state_dict, strict=False)
    full_bendr.to(device)
    full_bendr.eval()
    return full_bendr

def get_layers_of_interest():
    """
    Defines the layers of interest for hooking.
    """
    layers = [
        "encoder.encoder.Encoder_0", "encoder.encoder.Encoder_1", "encoder.encoder.Encoder_2",
        "encoder.encoder.Encoder_3", "encoder.encoder.Encoder_4", "encoder.encoder.Encoder_5",
        "contextualizer.relative_position", "contextualizer.input_conditioning",
        "contextualizer.transformer_layers", "contextualizer.output_layer"
    ]
    layers += [f"contextualizer.transformer_layers.{i}" for i in range(8)]
    return layers

def extract_representations(model, data_loader, hook_manager, device):
    """
    Extracts representations from the model using hooks and associates them with filenames.
    """
    dataset_representations = {}
    
    for batch in tqdm(data_loader, desc="Extracting representations"):
        for raw in batch['raw']:
            file_name = os.path.basename(raw.filenames[0])
            
            hook_manager.layer_outputs.clear()
            data = torch.from_numpy(raw.get_data()).float().unsqueeze(0).to(device)
            model(data)
            
            detached_outputs = {k: v[0].detach().clone().cpu() for k, v in hook_manager.layer_outputs.items()}
            dataset_representations[file_name] = detached_outputs
    
    return dataset_representations

def save_representations(dataset_representations, filename):
    """
    Save extracted representations as a dictionary where keys are filenames.
    """
    with open(filename, 'wb') as f:
        pickle.dump(dataset_representations, f)

config_path = os.path.abspath(os.path.join(os.getcwd(), "configs/representations/"))
@hydra.main(config_path=config_path, config_name="config", version_base=None)
def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    
    logger.info("Loading model.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg.model_path, device)
    
    logger.info("Registering hooks.")
    layers_of_interest = get_layers_of_interest()
    hook_manager = HookManager(layers_of_interest)
    hook_manager.register_hooks(model)
    
    logger.info("Loading data.")
    edf_dataset = EDFDataset(cfg.edf_data_path)
    data_loader = DataLoader(edf_dataset, batch_size=cfg.batch_size, shuffle=True, 
                             collate_fn=lambda x: {"raw": [item['raw'] for item in x]})
    
    logger.info("Extracting representations.")
    dataset_representations = extract_representations(model, data_loader, hook_manager, device)
    
    logger.info("Saving representations.")
    
    window_length = 60.0 # temp, later based on input file path
    name = f"representations_{window_length}.pkl"
    save_representations(dataset_representations, filename=os.path.join(cfg.save_results_path, name))
    
    logger.info("Done!")

if __name__ == "__main__":
    main()
