import numpy as np
from cca_zoo.linear import rCCA
from cca_zoo.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from tqdm import tqdm
from src.synthetic import SyntheticSignalGenerator

def compute_pwcca(X_raw, Y_raw, X_transformed, Y_transformed, cca_corrs):
    def compute_weighted_sum(acts_real, dirns_transf, coefs):
        P, _ = np.linalg.qr(dirns_transf)
        weights = np.sum(np.abs(np.dot(P.T, acts_real)), axis=1)
        weights = weights / np.sum(weights)
        # for weight, cca_corr in zip(weights, cca_corrs):
        #     print(f'Weight: {weight}, CCA corr: {cca_corr}')
        return np.sum(weights * coefs)

    score_x = compute_weighted_sum(X_raw, X_transformed, cca_corrs)
    score_y = compute_weighted_sum(Y_raw, Y_transformed, cca_corrs)
    return (score_x + score_y) / 2

# Parameters
n_timepoints = 1000
n_features_a = 320
n_features_b = 220
n_shared_components = 2
shared_freq = [5, 12]
shared_phase = [np.pi, np.pi / 2]
n_independent_components_a = 0
n_independent_components_b = 0
noise_level = 0.1
cca_components = 10
param_grid = {'c': [[0.3], [0.3]]}
cv = 3
use_kfold = False  # Set to False for Bootstrap, True for K-Fold CV

all_pwcca_scores = {}

for i in range(6):  # Increase complexity for 5 iterations
    
    signal_generator_a = SyntheticSignalGenerator(n_timepoints, n_features_a, n_shared_components, 
                                                  n_independent_components_a, shared_freq, shared_phase, noise_level, seed=69)
    signal_generator_b = SyntheticSignalGenerator(n_timepoints, n_features_b, n_shared_components, 
                                                  n_independent_components_b, shared_freq, shared_phase, noise_level, seed=1)
    signal_a = signal_generator_a.make_signal()
    signal_b = signal_generator_b.make_signal()
    
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    signal_a = scaler_X.fit_transform(signal_a)
    signal_b = scaler_Y.fit_transform(signal_b)

    def scorer(estimator, X):
        dim_corrs = estimator.score(X)
        return dim_corrs.mean()

    cca = rCCA(latent_dimensions=cca_components)
    cca = GridSearchCV(cca, param_grid=param_grid, cv=cv, verbose=True, scoring=scorer).fit((signal_a, signal_b))
    best_cca = cca.best_estimator_

    if use_kfold:
        # K-Fold Cross-Validation
        kf = KFold(n_splits=5, shuffle=False, random_state=42)
        fold_corrs = []
        pwcca_scores = []

        for train_idx, test_idx in kf.split(signal_a):
            best_cca.fit((signal_a[train_idx], signal_b[train_idx]))
            X_transformed, Y_transformed = best_cca.transform((signal_a[test_idx], signal_b[test_idx]))

            fold_corrs.append([np.corrcoef(X_transformed[:, i], Y_transformed[:, i])[0, 1] for i in range(cca_components)])
            pwcca_scores.append(compute_pwcca(signal_a[test_idx], signal_b[test_idx], X_transformed, Y_transformed, fold_corrs[-1]))

        fold_corrs = np.array(fold_corrs)
        mean_fold_corrs = fold_corrs.mean(axis=0)
        std_fold_corrs = fold_corrs.std(axis=0)
        mean_pwcca = np.mean(pwcca_scores)
        ci_lower, ci_upper = np.percentile(pwcca_scores, [2.5, 97.5])

        results = {
            f"Canonical correlation {i+1}": {"Mean": mean_corr, "Std Dev": std_corr}
            for i, (mean_corr, std_corr) in enumerate(zip(mean_fold_corrs, std_fold_corrs))
        }
        #print(results)
        print(f"KFold PWCCA: Mean PWCCA: {mean_pwcca:.3f}, CI: ({ci_lower:.3f}, {ci_upper:.3f})")


    else:
        # Bootstrap Resampling
        bootstrap_pwcca_scores = []
        for _ in tqdm(range(6), desc=f"Bootstrapping Complexity {i+1}"):
            all_indices = np.arange(len(signal_a))
            train_indices = resample(all_indices)
            test_indices = np.setdiff1d(all_indices, train_indices)

            X_train, Y_train = signal_a[train_indices], signal_b[train_indices]
            X_test, Y_test = signal_a[test_indices], signal_b[test_indices]

            best_cca.fit((X_train, Y_train))
            X_transformed, Y_transformed = best_cca.transform((X_test, Y_test))

            cca_corrs = [np.corrcoef(X_transformed[:, i], Y_transformed[:, i])[0, 1] for i in range(cca_components)]
            pwcca_score = compute_pwcca(X_test, Y_test, X_transformed, Y_transformed, cca_corrs)
            bootstrap_pwcca_scores.append(pwcca_score)

        mean_pwcca = np.mean(bootstrap_pwcca_scores)
        ci_lower, ci_upper = np.percentile(bootstrap_pwcca_scores, [2.5, 97.5])
        print(f"Complexity {i+1}: Mean PWCCA: {mean_pwcca:.3f}, CI: ({ci_lower:.3f}, {ci_upper:.3f})")


    all_pwcca_scores[i] = {'pwcca':mean_pwcca, 'ci_lower':ci_lower, 'ci_upper':ci_upper, 'cca_corrs':cca_corrs}

    # update on uneven:
    # if i % 2 == 0:
        
    # else:
    noise_level += 0.25
    n_independent_components_a += 1
    n_independent_components_b += 1

print(all_pwcca_scores)