import numpy as np
import pickle
import os
import logging
import gc
import hydra
from hydra.utils import instantiate
from cca_zoo.linear import rCCA
from cca_zoo.model_selection import GridSearchCV
from omegaconf import DictConfig
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.utils import resample

def load_data(file_path):
    logging.info("Loading data.")
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_source_labels(label_path):
    return np.load(label_path, allow_pickle=True).item() 

def parse_data(data, labels):
    label_files = labels.keys()
    
    labeled_representations = list(
        (labels[file]['power']['T0'][0], data[file]) 
        for file in data if file in label_files
    )
    for layer in labeled_representations[0][1]:
        representation_tensor = np.array([rep[1][layer] for rep in labeled_representations])
        label_tensor = np.array([rep[0] for rep in labeled_representations])
        # np.random.shuffle(label_tensor) ######################################################################## only to get baseline pwcca score!
        yield layer, representation_tensor, label_tensor

def cca_scorer(estimator, X):
    dim_corrs = estimator.score(X)
    return dim_corrs.mean()

def normalize_data(X, Y):
    logging.info("Normalizing data.")
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_normalized = scaler_X.fit_transform(X)
    Y_normalized = scaler_Y.fit_transform(Y)
    return X_normalized, Y_normalized


def compute_cca(representations, label_data, cca, param_grid, cv):
    logging.info("Running CCA.")
    global max_latent_dim        
    if cca.latent_dimensions > label_data.shape[1]:
        max_latent_dim = label_data.shape[1] 
        logging.warning(f"Reducing latent_dimensions from {cca.latent_dimensions} to {max_latent_dim}")
        cca.latent_dimensions = max_latent_dim
    else:
        max_latent_dim = cca.latent_dimensions 

    if cv is None or cv == 1:
        best_cca = cca.fit((representations, label_data))
        # print cca directions #########################################################################
        # for i in range(max_latent_dim):
        # print(f"CCA direction: {best_cca.weights_[1]}")
        print(f"CCA direction: {np.array2string(best_cca.weights_[1], separator=', ')}")
    else:
        cca = GridSearchCV(cca, param_grid=param_grid, cv=cv, verbose=True, scoring=cca_scorer)
        cca.fit((representations, label_data))
        best_cca = cca.best_estimator_
    logging.info(f"Best CCA params: {best_cca.get_params()}")
    return best_cca

def compute_pwcca(X_raw, Y_raw, X_transformed, Y_transformed, cca_corrs):
    """Computes projection weighting for weighting CCA coefficients

    Args:
        proj_mat_x: square projection matrix of size valid indices in x_idxs <= d1
        proj_mat_y: square projection matrix of size valid indices in y_idxs <= d2
        x_idxs: boolean array for view 1 indices corresponding to "valid" dimensions;
                size (d1,)
        y_idxs: boolean array for view 2 indices corresponding to "valid" dimensions;
                size (d2,)

    Returns:
        Projection weighted mean of cca coefficients
    """
    def compute_weighted_sum(acts_real, dirns_transf, coefs): # acts_means
        """Computes weights for projection weighing"""
        P, _ = np.linalg.qr(dirns_transf) # not .T becuase org dirns is num_directions x N
        weights = np.sum(np.abs(np.dot(P.T, acts_real)), axis=1) # not  .T because "acts1: (num_neurons1, data_points) a 2d numpy array of neurons by datapoints where entry (i,j) is the output of neuron i on datapoint j."
        weights = weights / np.sum(weights)
        return np.sum(weights * coefs)

    score_x = compute_weighted_sum(X_raw, X_transformed, cca_corrs)
    score_y = compute_weighted_sum(Y_raw, Y_transformed, cca_corrs)
    return (score_x + score_y) / 2

def process_cca(cfg):
    if cfg.bootstrap_iterations is None:
        bootstrap_iterations = 1
    else:
        bootstrap_iterations = cfg.bootstrap_iterations
    data = load_data(cfg.paths.file_in)
    labels = load_source_labels(cfg.paths.labels_in)
    cca = instantiate(cfg.cca)
    param_grid = {'c': [cfg.cca_param_grid.c1, cfg.cca_param_grid.c2]}
    results = {}
    for layer, representation_tensors, labels in parse_data(data, labels):
        bootstrap_pwcca_scores_train = []
        bootstrap_pwcca_scores_test = []
        bootstrap_cca_corrs_train = []
        bootstrap_cca_corrs_test = []
        for ii in tqdm(range(bootstrap_iterations), desc=f"Bootstrapping {layer}"):
            all_indices = np.arange(len(representation_tensors))
            train_indices = resample(all_indices)
            test_indices = np.setdiff1d(all_indices, train_indices)
            print(f"Train: {len(train_indices)}, Test: {len(test_indices)}")

            X_train, Y_train = representation_tensors[train_indices], labels[train_indices]
            X_test, Y_test = representation_tensors[test_indices], labels[test_indices]

            X_train, Y_train = normalize_data(X_train, Y_train)
            X_test, Y_test = normalize_data(X_test, Y_test)


            best_cca = compute_cca(X_train, Y_train, cca, param_grid, cfg.cv)

            X_transformed_train, Y_transformed_train = best_cca.transform((X_train, Y_train))
            X_transformed_test, Y_transformed_test = best_cca.transform((X_test, Y_test)) 

            corrs_train = [np.corrcoef(X_transformed_train[:, i], Y_transformed_train[:, i])[0, 1] for i in range(best_cca.latent_dimensions)]   
            corrs_test = [np.corrcoef(X_transformed_test[:, i], Y_transformed_test[:, i])[0, 1] for i in range(best_cca.latent_dimensions)]                          
            
            pwcca_score_train = compute_pwcca(X_train, Y_train, X_transformed_train, Y_transformed_train, corrs_train)
            pwcca_score_test = compute_pwcca(X_test, Y_test, X_transformed_test, Y_transformed_test, corrs_test)          

            bootstrap_pwcca_scores_train.append(pwcca_score_train)
            bootstrap_pwcca_scores_test.append(pwcca_score_test)
            bootstrap_cca_corrs_train.append(corrs_train)
            bootstrap_cca_corrs_test.append(corrs_test)
        
        results[layer] = {
            'pwcca_train': bootstrap_pwcca_scores_train,
            'pwcca_test': bootstrap_pwcca_scores_test,
            'cca_corrs_train': bootstrap_cca_corrs_train,
            'cca_corrs_test': bootstrap_cca_corrs_test
        }
        del representation_tensors, labels
        gc.collect()
    return results            

def save_results(results, file_path_in, labels_in, folder_path_out, cca_type, cca_latent_dim, bootstrap=None):
    logging.info("Saving results.")
    
    file_name = os.path.basename(file_path_in)
    file_core = "_".join(file_name.split("_")[1:4])
    print(file_core)

    labels_name = os.path.basename(labels_in)  
    labels_parts = labels_name.split("_")  
    labels_core = "_".join(labels_parts[:-6])
    print(labels_core)

    filename_out = os.path.join(
        folder_path_out,
        f"{labels_core}_{file_core}_{cca_type}_comp_{cca_latent_dim}_bootstrap_{bootstrap}_v3_only_for_cc_directions.pkl"
    )

    print(filename_out)
    with open(filename_out, 'wb') as f:
        pickle.dump(results, f)
    
    logging.info(f"Results saved to {filename_out}")

config_path = os.path.abspath(os.path.join(os.getcwd(), "configs/cca/"))
@hydra.main(config_path=config_path, config_name="config", version_base=None)
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Labels: {cfg.paths.labels_in}")
    max_latent_dim = None
    results = process_cca(cfg)
    print(results)
    save_results(results, cfg.paths.file_in, cfg.paths.labels_in, cfg.paths.folder_out, cfg.cca._target_.split('.')[-1], max_latent_dim, cfg.bootstrap_iterations)

if __name__ == "__main__":
    main()
