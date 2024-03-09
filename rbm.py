import numpy as np
from numba import njit
from utils import *
import multiprocessing as mp

def visibile_bias_init(data_binary):
    '''Initialize the visible bias with the log of the ratio of the empirical frequencies of the binary data.'''
    frequencies = np.mean(data_binary, axis=0)
    visible_bias_t0 = np.log(frequencies / (1 - frequencies))
    return visible_bias_t0

def initialize_rbm(data, num_visible, num_hidden, batch_size):
    '''Initialize the weights and biases of the RBM. The weights and the hidden bias are 
    initialized with a normal distribution with mean 0 and standard deviation 0.01. 
    The visible bias is initialized with the log of the ratio of the empirical frequencies 
    of the binary data.
    
    Parameters
    ----------
    data : np.array
        The binary data used for the RBM.
    num_visible : int
        The number of visible units.
    num_hidden : int
        The number of hidden units.
    
    Returns
    -------
    weights : np.array
        The weights of the RBM.
    hidden_bias : np.array
        The hidden bias of the RBM.
    visible_bias : np.array
        The visible bias of the RBM.'''
    np.random.seed(666)
    weights = np.random.normal(0, 0.01, (num_visible, num_hidden))
    # hidden_bias = -np.ones(num_hidden) * 4
    hidden_bias = np.random.normal(0, 0.01, num_hidden)
    visible_bias = visibile_bias_init(data)
    h_persistent = np.random.binomial(n=1, p=0.5, size=[batch_size, hidden_bias.shape[0]])
    
    return weights, hidden_bias, visible_bias, h_persistent

@njit
def sample_hidden(visible, weights, hidden_bias):
    '''Sample the hidden units given the visible units (positive phase). 
    During the positive phase, the network learns from the data.
    This is thefore the data-driven phase.
    
    Parameters
    ----------
    visible : np.array
        The visible units.
    weights : np.array
        The weights of the RBM.
    hidden_bias : np.array
        The hidden bias of the RBM.
    
    Returns
    -------
    hidden_probabilities : np.array
        The probabilities of the hidden units computed as the sigmoid of the total imput.
    hidden_states : np.array
        The states of the hidden units computed picking a 1 or 0 with a probability determined
         by hidden_probabilities .'''
    hidden_activations = dot_product(visible, weights) + hidden_bias
    if np.isnan(hidden_activations).any():
        print(f"There NaNs in the hidden activations")
    hidden_probabilities = sigmoid(hidden_activations)
    hidden_states = boolean_to_int(hidden_probabilities > np.random.random(hidden_probabilities.shape))
    return hidden_probabilities, hidden_states

@njit
def sample_visible(hidden, weights, visible_bias):
    '''Sample the visible units given the hidden units (negative phase).
    In this phase the network reconstructs the data. This is the recontruction-driven phase.'''
    visible_activations = dot_product(hidden, weights.T) + visible_bias
    if np.isnan(visible_activations).any():
        print(f"There NaNs in the visible activations")
    visible_probabilities = sigmoid(visible_activations)
    visible_states = boolean_to_int(visible_probabilities > np.random.random(visible_probabilities.shape))
    return visible_probabilities, visible_states

# @njit
def train(data, val, weights, hidden_bias, visible_bias, h_persistent, num_epochs, batch_size, learning_rate, k, monitoring, id, additional_quantities):
    '''Train the RBM using the contrastive divergence algorithm.
    
    Parameters
    ----------
    data : np.array
        The training binary data.
    val : np.array
        The validation binary data.
    weights : np.array
        The weight matrix of the RBM before training.
    hidden_bias : np.array
        The hidden bias of the RBM before training.
    visible_bias : np.array
        The visible bias of the RBM before training.
    num_epochs : int
        The number of epochs for training.
    batch_size : int
        The size of the batches used for training.
    learning_rate : float
        The learning rate used for training.
    k : int
        The number of Gibbs steps used for sampling.
    monitoring : bool
        Whether to monitor the training process with plots.
    id : str
        The id of the training process. Used for tracking the plots across runs.
    
    Returns
    -------
    reconstruction_error : list
        The reconstruction error at each epoch.
    f_energy_overfitting : list
        The free energy of the training and validation data at each epoch.
    f_energy_diff : list
        The difference in free energy before and after the sampling at each epoch.
    diff_fenergy : list
        The difference in free energy between the training and validation data at each epoch.
    weights : np.array
        The weights of the RBM after training.
    hidden_bias : np.array
        The hidden bias of the RBM after training.
    visible_bias : np.array
        The visible bias of the RBM after training.
    '''
    X_min_train, X_max_train, indexes_vol_indicators, vol_indicators,  currencies = additional_quantities
    num_samples = data.shape[0]

    best_params = []
    reconstruction_error = []
    f_energy_overfitting = []
    f_energy_diff = []
    diff_fenergy = []
    counter = 0
    j = 0

    # Initialize velocities
    velocity_w = np.zeros_like(weights)
    velocity_hidden_bias = np.zeros_like(hidden_bias)
    velocity_visible_bias = np.zeros_like(visible_bias)

    # Set the training data that I will use for monitoring the free energy
    start_idx = np.random.randint(low=0, high=num_samples-200)
    data_subset = data[start_idx: start_idx+200]

    for epoch in range(num_epochs):
        if epoch % 100 == 0: print(f"Epoch: {epoch}/{num_epochs}")
        for i in range(0, num_samples, batch_size):
            # Initiliaze the visible units with the training vectors
            v0 = data[i:i+batch_size]

            # Positive phase
            pos_hidden_prob, pos_hidden_states = sample_hidden(v0, weights, hidden_bias)

            # Copy the probabilities to use them in the positive associations
            pos_hidden_prob0 = pos_hidden_prob.copy()

            # Gibbs sampling
            for _ in range(k):
                if _ == 0:
                    neg_visible_prob, neg_visible_states = sample_visible(h_persistent, weights, visible_bias)
                pos_hidden_prob , pos_hidden_states = sample_hidden(neg_visible_states, weights, hidden_bias)
                neg_visible_prob, neg_visible_states = sample_visible(pos_hidden_states, weights, visible_bias)
            
            # Update the persistent chain
            h_persistent = pos_hidden_states

            # Update weights and biases
            """In the positive statistics collection, using pos_hidden_states is closer to the mathematical 
            model of an RBM, but using pos_hidden_prob usually has less sampling noise which allows slightly 
            faster learning. Hence, I can swap pos_hidden_states with pos_hidden_prob."""
            positive_associations = dot_product(pos_hidden_prob0.T, v0)
            negative_associations = dot_product(pos_hidden_prob.T, neg_visible_states)

            # update the learning rate according to initial_step_size * (1.0 - (1.0*self.global_step)/(1.0*iterations*epochs))
            # learning_rate = learning_rate * (1.0 - (epoch*i)/(num_epochs*num_samples//batch_size))
            lr = learning_rate / batch_size

            penalty = np.sum(weights.ravel()) * 0.001

            delta_w = lr * ((positive_associations - negative_associations) - penalty)
            delta_hidden_bias = lr * mean_axis_0(pos_hidden_prob0 - pos_hidden_prob)
            delta_visible_bias = lr * mean_axis_0(v0 - neg_visible_states)

            deltas = [delta_w, delta_hidden_bias, delta_visible_bias]

            # Update weights and biases with velocity
            weights += delta_w.T
            hidden_bias += delta_hidden_bias
            visible_bias += delta_visible_bias

        if monitoring:
            if epoch % 100 == 0 and epoch != 0:
                monitoring_plots(weights, hidden_bias, visible_bias, deltas, pos_hidden_prob, epoch, id)
            # Calculate free energy for overfitting monitoring
            start_idx = np.random.randint(low=0, high=val.shape[0]-200)
            val_subset = val[start_idx: start_idx+200]
            f_e_data = free_energy(data_subset, weights, visible_bias, hidden_bias)
            f_e_val = free_energy(val_subset, weights, visible_bias, hidden_bias)
            diff = f_e_data - f_e_val

            diff_fenergy.append(np.abs(diff))
            f_energy_overfitting.append([f_e_data, f_e_val])
        
            # Calculate free energy for monitoring
            f_e_before = free_energy(v0, weights, visible_bias, hidden_bias)
            f_e_after = free_energy(neg_visible_states, weights, visible_bias, hidden_bias)
            f_energy_diff.append(f_e_before - f_e_after)

            # Calculate reconstruction error
            error = np.sum((v0 - neg_visible_states) ** 2) / batch_size
            reconstruction_error.append(error)

            if epoch % 500 == 0 and epoch != 0:
                id_epoch = f"{id}_epoch_{epoch}"
                print(f"{id} - Sampling from the RBM...")
                samples = parallel_sample(weights, hidden_bias, visible_bias, 1000, indexes_vol_indicators, vol_indicators, data.shape[0])
                print(f"Done\n")

                # Convert to real values
                print("Converting the samples from binary to real values...")
                # Remove the volatility indicators
                samples = np.delete(samples, indexes_vol_indicators, axis=1)
                data_plot = np.delete(data, indexes_vol_indicators, axis=1)
                samples = from_binary_to_real(samples, X_min_train, X_max_train).to_numpy()
                data_plot = from_binary_to_real(data_plot, X_min_train, X_max_train).to_numpy()
                print(f"Done\n")

                # Compute correlations
                print("Computing correlations...")  
                gen_correlations = calculate_correlations(pd.DataFrame(samples, columns=currencies))
                original_correlations = calculate_correlations(pd.DataFrame(data_plot, columns=currencies))
                plot_correlation_heatmap(original_correlations, gen_correlations, id_epoch)

                print("Plotting results...")
                # Plot the samples and the recontructed error
                plot_distributions(samples, data_plot, currencies, id_epoch)
                qq_plots(samples, data_plot, currencies, id_epoch)
                plot_objectives(reconstruction_error, f_energy_overfitting, f_energy_diff, diff_fenergy, id_epoch)
                plot_tail_concentration_functions(data_plot, samples, currencies, id_epoch)
                print(f"Done\n")

    return reconstruction_error, f_energy_overfitting, f_energy_diff, diff_fenergy, weights, hidden_bias, visible_bias

@njit
def sample_from_state(state, weights, hidden_bias, visible_bias, indexes_vol_indicators, vol_indicators, k):
    '''Sample from the RBM given a particular state.
    
    Parameters
    ----------
    state : np.array
        The state from which to sample. This can be a previously generated sample.
    weights : np.array
        The weights of the trained RBM.
    hidden_bias : np.array
        The hidden bias of the trained RBM.
    visible_bias : np.array
        The visible bias of the trained RBM.
    indexes_vol_indicators : list
        The indexes of the volatility indicators in the state.
    vol_indicators : np.array
        The volatility indicators.
    k : int
        The number of Gibbs steps used for sampling.
    
    Returns
    -------
    samples : np.array
        The samples generated from the RBM.'''
    # np.random.seed(666)
    samples = state
    samples[:, indexes_vol_indicators] = vol_indicators
    for _ in range(k):  # number of Gibbs steps
        _, samples = sample_hidden(samples, weights, hidden_bias)
        _, samples = sample_visible(samples, weights, visible_bias)
        samples[:, indexes_vol_indicators] = vol_indicators
    return samples

# @njit
def sample_batch(weights, hidden_bias, visible_bias, k, indexes_vol_indicators, vol_indicators, batch_size):
    '''Sample from the RBM.'''
    samples = np.random.randint(low=0, high=2, size=(batch_size, weights.shape[0]))
    samples[:, indexes_vol_indicators] = vol_indicators
    for _ in range(k):  # number of Gibbs steps
        _, samples = sample_hidden(samples, weights, hidden_bias)
        _, samples = sample_visible(samples, weights, visible_bias)
        samples[:, indexes_vol_indicators] = vol_indicators
    return samples

def parallel_sample(weights, hidden_bias, visible_bias, k, indexes_vol_indicators, vol_indicators, n_samples, n_processors=8):
    '''Sample from the RBM in parallel.
    
    Parameters
    ----------
    weights : np.array
        The weights of the trained RBM.
    hidden_bias : np.array
        The hidden bias of the trained RBM.
    visible_bias : np.array
        The visible bias of the trained RBM.
    k : int
        The number of Gibbs steps used for sampling.
    indexes_vol_indicators : list
        The indexes of the volatility indicators in the state.
    vol_indicators : np.array
        The volatility indicators.
    n_samples : int
        The number of samples to generate.
    n_processors : int
        The number of processors to use for parallel sampling.
    
    Returns
    -------
    samples : np.array
        The samples generated from the RBM.'''
    # np.random.seed(666)
    pool = mp.Pool(processes=n_processors)
    
    # Calculate the number of samples per processor
    batch_sizes = [n_samples // n_processors for _ in range(n_processors)]
    batch_sizes[-1] += n_samples % n_processors

    vol_indicators_batch = []
    for i in range(n_processors):
        if i < n_processors - 1:
            vol_indicators_batch.append(vol_indicators[i*batch_sizes[0]: (i+1)*batch_sizes[0]])
        else:
            vol_indicators_batch.append(vol_indicators[i*batch_sizes[0]:])
    
    # Create a list of arguments for each batch
    args = [(weights, hidden_bias, visible_bias, k, indexes_vol_indicators, vol_ind, batch_size) for _, batch_size, vol_ind in zip(range(n_processors), batch_sizes, vol_indicators_batch)]
    
    # Execute the function in parallel
    results = pool.starmap(sample_batch, args)
    
    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()
    
    # Combine the results
    samples = np.vstack(results)
    return samples

# TO DO
# 1. Evaluate low and high volatility regimes

