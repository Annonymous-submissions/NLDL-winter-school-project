import numpy as np
import pickle
import hydra
import os
import logging
import multiprocessing
from tqdm import tqdm
from omegaconf import DictConfig
from scipy.sparse.linalg import svds

def svd_worker(args):
    """ Compute SVD for a single layer. """
    key, matrix, reduced_dim = args
    U, S, _ = svds(matrix, k=reduced_dim)
    return key, U @ np.diag(S)  # Equivalent to PCA scores

def economy_svd_sequential(data, reduced_dim):
    """ Perform SVD sequentially. """
    results = {}
    for key in tqdm(data.keys(), desc="Computing SVD sequentially"):
        U, S, _ = svds(data[key], k=reduced_dim)
        results[key] = U @ np.diag(S)
    return results

def economy_svd_multiprocessing(data, reduced_dim, n_processes=-1):
    """ Perform SVD in parallel using multiprocessing. """
    if n_processes == -1:
        n_processes = multiprocessing.cpu_count()

    tqdm.write(f"[INFO] Starting multiprocessing with {n_processes} processes.")

    with multiprocessing.Pool(processes=n_processes) as pool:
        results = pool.map(svd_worker, [(key, data[key], reduced_dim) for key in data.keys()])

    return dict(results)  # Convert list of tuples back to a dictionary

config_path = os.path.abspath(os.path.join(os.getcwd(), "configs/representations/svd/"))
@hydra.main(config_path=config_path, config_name="config.yaml", version_base=None)
def main(cfg: DictConfig):
    log = logging.getLogger(__name__)

    log.info("Loading data.")
    with open(cfg.paths.file_in, 'rb') as f:
        representations_all = pickle.load(f)

    # log.info("Parsing data.")
    # representation_tensors = {
    #     layer: np.array([rep[1][layer] for rep in representations_all])
    #     for layer in representations_all[0][1].keys()
    # }
    log.info("Parsing data.")
    representation_tensors = {
        layer: np.reshape(
            np.stack([rep[layer].reshape(-1) for rep in representations_all.values()], axis=0),
            (len(representations_all), -1)
        )
        for layer in next(iter(representations_all.values())).keys()
    }

    log.info("Performing SVD.")
    if cfg.svd.n_processes and cfg.svd.n_processes > 1:
        representation_tensors_reduced = economy_svd_multiprocessing(
            representation_tensors, 
            cfg.svd.num_reduced_features, 
            cfg.svd.n_processes
        )
    else:
        representation_tensors_reduced = economy_svd_sequential(
            representation_tensors, 
            cfg.svd.num_reduced_features
        )

    # log.info("Parsing back to original format.")
    # labels = [entry[0] for entry in representations_all]
    # representations_reduced_all = [
    #     (labels[i], {layer: representation_tensors_reduced[layer][i] for layer in representation_tensors_reduced})
    #     for i in range(len(labels))
    # ]

    log.info("Parsing back to original format.")
    file_names = list(representations_all.keys())
    representations_reduced_all = {
        file_names[i]: {layer: representation_tensors_reduced[layer][i] for layer in representation_tensors_reduced}
        for i in range(len(file_names))
    }

    log.info("Saving results.")
    filename_out = os.path.join(
        cfg.paths.folder_out,
        os.path.basename(cfg.paths.file_in).replace('.pkl', f'_svd_{cfg.svd.num_reduced_features}.pkl')
    )
    with open(filename_out, 'wb') as f:
        pickle.dump(representations_reduced_all, f)

    log.info("Done!")

if __name__ == "__main__":
    main()
