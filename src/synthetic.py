import numpy as np
from cca_zoo.linear import rCCA
from cca_zoo.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from numpy.random import default_rng

###############################################################

n_timepoints = 1000
n_features_a = 3200
n_features_b = 2200
n_shared_components = 2
shared_freq = [5, 12]
shared_phase = [np.pi, np.pi/2]
n_independent_components_a = 3
n_independent_components_b = 3
noise_level = 1

cca_components = 10 # are latent dimentions tunable? 
cca = rCCA(latent_dimensions=cca_components)
c1 = [0.1, 0.3, 0.7, 0.9]
c2 = [0.1, 0.3, 0.7, 0.9]
param_grid = {'c': [c1, c2]} 
cv = 3

verbosity = True

plot_reconstructed_signal = True
if plot_reconstructed_signal:
    components_to_restore = 2
    comment = f""
    name = f'SignalA_shared{n_shared_components}_independent{n_independent_components_a}_{n_independent_components_b}_noise{noise_level}_timepoints{n_timepoints}_{comment}'

###############################################################
used_freq = []
for f in shared_freq:
    used_freq.append(f)

used_phase = []
for p in shared_phase:
    used_phase.append(p)
###############################################################
class SyntheticSignalGenerator:
    def __init__(
        self,
        n_timepoints=10000,
        n_features=100,
        n_shared_components=None,
        n_independent_components=None,
        frequencies_shared=None,
        phase_shared=None,
        noise_level=0.1,
        seed=43,
        verbose=False
    ):
        """
        Initialize the SyntheticSignalGenerator.

        Args:
            n_timepoints (int): Number of time points in the signal.
            n_features (int): Number of features/channels in the signal.
            n_shared_components (int or None): Number of shared components between signals.
                Defaults to 1 if None.
            n_independent_components (int or None): Number of independent components for each signal.
                Defaults to half of n_features if None.
            frequencies_shared (list): Frequencies for the shared sinusoidal components.
            phase_shared (list): Phases for the shared sinusoidal components.
            noise_level (float): Amount of noise to add to the signal.
            seed (int, optional): Random seed for reproducibility.
            verbose (bool): Whether to print additional information during generation.
        """
        self.n_timepoints = n_timepoints
        self.n_features = n_features
        self.n_shared_components = n_shared_components
        self.n_independent_components = n_independent_components
        self.frequencies_shared = frequencies_shared
        self.phase_shared = phase_shared
        self.noise_level = noise_level
        self.seed = seed
        self.verbose = verbose
        self.shared_components = []
        self.independent_components = []

        if self.seed is not None:
        #     np.random.seed(self.seed)
            self.rng = default_rng(seed)  # Use a local random generator for isolation


    def generate_shared_components(self, timepoints):
        if self.n_shared_components is not None:
            assert len(self.frequencies_shared) == self.n_shared_components, \
                "Number of shared frequencies must match n_shared_components"
            assert len(self.phase_shared) == self.n_shared_components, \
                "Number of shared phases must match n_shared_components"
            
            for i in range(self.n_shared_components):
                self.shared_components.append(np.sin(2 * np.pi * self.frequencies_shared[i] * timepoints + self.phase_shared[i]))

            if self.verbose:
                for i in range(self.n_shared_components):
                    print(f"Shared component {i}: freq={self.frequencies_shared[i]}, phase={self.phase_shared[i]}")

        else:
            print("No shared components")
        #     freq = self.rng.integers(1, 50)
        #     phase = self.rng.random() * np.pi
        #     self.shared_components.append(np.sin(2 * np.pi * freq * timepoints + phase))

        #     if self.verbose:
        #         print(f"Shared component: freq={freq}, phase={phase}")

    def generate_independent_components(self, timepoints):
        for i in range(self.n_independent_components):
            np.random.seed(self.seed + i)  # Different seed for each independent component
            freq = self.rng.integers(1, 50)
            # while freq in used_freq:
            #     freq = self.rng.integers(1, 50)  # Regenerate frequency if already used
            used_freq.append(freq)
            phase = self.rng.random() * np.pi
            # while phase in used_phase:
            #     phase = self.rng.random() * np.pi ############################################################## temp disabled 
            used_phase.append(phase)
            self.independent_components.append(np.sin(2 * np.pi * freq * timepoints + phase))
            if self.verbose:
                print(f"Independent component {i}: freq={freq}, phase={phase}")

    def make_signal(self):
        """
        Generate synthetic signal data with shared and independent components.

        Returns:
            np.ndarray: Generated synthetic signal of shape (n_timepoints, n_features).
        """
        timepoints = np.linspace(0, 1, self.n_timepoints)

        self.generate_shared_components(timepoints)   

        if self.n_independent_components is not None:
            self.generate_independent_components(timepoints)
            self.num_components = self.n_shared_components + self.n_independent_components
            self.sin_stacked = np.vstack(self.shared_components + self.independent_components).T 
        else:
            self.num_components = self.n_shared_components
            self.sin_stacked = np.vstack(self.shared_components).T

        if self.verbose:
            print(f"num_components: {self.num_components}")

        # Create scaling matrix
        self.scaling_matrix = self.rng.random((self.num_components, self.n_features)) # random scaling matrix
        #self.scaling_matrix = np.ones((self.num_components, self.n_features)) # scaling matrix of ones
        self.signal = np.dot(self.sin_stacked, self.scaling_matrix)
        
        # Add noise
        noise = self.noise_level * self.rng.random((self.n_timepoints, self.n_features))
        self.signal += noise

        if self.verbose:
            print(f"Signal shape: {self.signal.shape}")

        return self.signal 

###############################################################

# if verbosity:
#     print("\nSignal A")
# signal_generator_a = SyntheticSignalGenerator(
#     n_timepoints=n_timepoints,
#     n_features=n_features_a,
#     n_shared_components=n_shared_components,
#     frequencies_shared=shared_freq,
#     phase_shared=shared_phase,
#     n_independent_components=n_independent_components_a,
#     noise_level=noise_level,
#     verbose=verbosity,
#     seed=69
# )

# signal_generator_b = SyntheticSignalGenerator(
#     n_timepoints=n_timepoints,
#     n_features=n_features_b,
#     n_shared_components=n_shared_components,
#     frequencies_shared=shared_freq,
#     phase_shared=shared_phase,
#     n_independent_components=n_independent_components_b,
#     noise_level=noise_level,
#     verbose=verbosity,
#     seed=1
# )

# signal_a = signal_generator_a.make_signal() # shape: (n_timepoints, n_features_a)
# signal_b = signal_generator_b.make_signal()

# ##############################################################

# def scorer(estimator, X):
#    dim_corrs = estimator.score(X)
#    return dim_corrs.mean()

# #zero mean signal:
# signal_a = signal_a - np.mean(signal_a, axis=0)
# signal_b = signal_b - np.mean(signal_b, axis=0)

# if cv:
#     cca = GridSearchCV(cca, param_grid=param_grid, cv=cv, verbose=True, scoring=scorer).fit((signal_a, signal_b)) # fit_tarnsform()
#     best_cca = cca.best_estimator_
# else:
#     cca = cca.fit((signal_a, signal_b))
#     best_cca = cca
# print(best_cca.get_params())

# fold_corrs = []
# kf = KFold(n_splits=5)
# for train_idx, test_idx in kf.split(signal_a):
#     best_cca.fit((signal_a[train_idx], signal_b[train_idx]))
#     X_transformed, Y_transformed = best_cca.transform((signal_a[test_idx], signal_b[test_idx])) # shape: (n_timepoints, cca_components)
#     fold_corrs.append([np.corrcoef(X_transformed[:, i], Y_transformed[:, i])[0, 1] for i in range(cca_components)])

# fold_corrs = np.array(fold_corrs)
# mean_fold_corrs = fold_corrs.mean(axis=0)
# std_fold_corrs = fold_corrs.std(axis=0)

# results = {f"Canonical correlation {i}": {"Mean": mean_corr, "Std Dev": std_corr} for i, (mean_corr, std_corr) in enumerate(zip(mean_fold_corrs, std_fold_corrs), start=1)}
# print("")
# print(results)

# ##############################################################

# if plot_reconstructed_signal:

#     import matplotlib.pyplot as plt
#     weights_a = best_cca.weights_[0] # shape: (n_features_a, cca_components)
#     weights_b = best_cca.weights_[1]
#     # print('weights A:')
#     # print(weights_a)
#     X_transformed, Y_transformed = best_cca.transform((signal_a, signal_b)) # Project transformed data back to the original space. shape: (n_timepoints, cca_components)
#     reconstructed_signal_a = np.dot(X_transformed[:, :components_to_restore], np.linalg.pinv(weights_a[:, :components_to_restore]))
#     # reconstructed_signal_a1 = np.dot(X_transformed[:, 0:1], np.linalg.pinv(weights_a[:, 0:1]))
#     # reconstructed_signal_a2 = np.dot(X_transformed[:, 1:2], np.linalg.pinv(weights_a[:, 1:2]))
#     # reconstructed_signal_a3 = np.dot(X_transformed[:, 2:3], np.linalg.pinv(weights_a[:, 2:3]))



#     feature1 = 3
#     feature2 = 1

#     plt.figure(figsize=(14, 5))

#     # Original signal (feature1)
#     plt.plot(signal_a[:, feature1], label=f'Original Signal (feature {feature1})', color='darkgrey') 
#     # Reconstructed signal (feature1)
#     plt.plot(reconstructed_signal_a[:, feature1], label=f'Reconstructed Signal (comp 1+2)', color='tomato')
#     # plt.plot(reconstructed_signal_a3[:, feature1], label=f'Reconstructed Signal (comp 3)', color='lightblue') 
#     # plt.plot(reconstructed_signal_a1[:, feature1], label=f'Reconstructed Signal (comp 1)', color='tomato') 
#     # plt.plot(reconstructed_signal_a2[:, feature1], label=f'Reconstructed Signal (comp 2)', color='hotpink') 
#     # plt.plot(reconstructed_signal_a1[:, feature1] + reconstructed_signal_a2[:, feature1], label=f'Reconstructed Signal (comp 1+2)', color='purple', linestyle=':') # want dotted line do : 

#     # # feature2 original + reconstructed
#     # plt.plot(signal_a[:, feature2], label=f'Original Signal (feature {feature2})', color='darkgrey')
#     # plt.plot(reconstructed_signal_a[:, feature2], label=f'Reconstructed Signal (feature {feature2})', color='hotpink') 
    
#     # Shared components
#     shared_signal = np.sum(signal_generator_a.shared_components, axis=0)
#     plt.plot(shared_signal, label='Shared components sum', color='dimgrey')

#     plt.legend(loc = 'lower right')
#     plt.savefig(f'plots/{name}.png')

############################################
def main():
    pass

if __name__ == "__main__":
    main()


