import numpy as np
import numba as nb
from utils import *
from scipy.stats import wasserstein_distance
from scipy.special import kl_div
np.random.seed(666)

def visibile_bias_init(data_binary):
    frequencies = np.mean(data_binary, axis=0)
    visible_bias_t0 = np.log(frequencies / (1 - frequencies))
    return visible_bias_t0

def initialize_rbm(data, num_visible, num_hidden):
    np.random.seed(666) # I set the seed in the main.py file
    weights = np.random.normal(0, 0.01, (num_visible, num_hidden))
    hidden_bias = -np.ones(num_hidden) * 4
    visible_bias = visibile_bias_init(data)
    
    return weights, hidden_bias, visible_bias

@nb.njit
def sample_hidden(visible, weights, hidden_bias):
        '''Sample the hidden units given the visible units (positive phase). 
        During the positive phase, the network learns from the data.
        This is thefore the data-driven phase.'''
        hidden_activations = dot_product(visible, weights) + hidden_bias
        hidden_probabilities = sigmoid(hidden_activations)
        hidden_states = boolean_to_int(hidden_probabilities > np.random.random(hidden_probabilities.shape))
        return hidden_probabilities, hidden_states

@nb.njit
def sample_visible(hidden, weights, visible_bias):
    '''Sample the visible units given the hidden units (negative phase).
    In this phase the network reconstructs the data. This is the recontruction-driven phase.'''
    visible_activations = dot_product(hidden, weights.T) + visible_bias
    if np.isnan(visible_activations).any():
        print(f'ecco\n', visible_activations, hidden, weights.T, visible_bias)
    visible_probabilities = sigmoid(visible_activations)
    visible_states = boolean_to_int(visible_probabilities > np.random.random(visible_probabilities.shape))
    return visible_probabilities, visible_states

# @nb.njit
def train(data, val, weights, hidden_bias, visible_bias, num_epochs, batch_size, learning_rate, k, monitoring, id, var_mon):
    X_min, X_max, currencies = var_mon
    num_samples = data.shape[0]

    best_params = []
    reconstructed_error = []
    f_energy = []
    KL_div = []
    counter = 0

    # Initialize velocities
    velocity_w = np.zeros_like(weights)
    velocity_hidden_bias = np.zeros_like(hidden_bias)
    velocity_visible_bias = np.zeros_like(visible_bias)

    # Set the training data that I will use for monitoring the free energy
    start_idx = np.random.randint(low=0, high=num_samples-200)
    data_subset = data[start_idx: start_idx+200]

    for epoch in range(num_epochs):
        for i in range(0, num_samples, batch_size):
            v0 = data[i:i+batch_size]
            # Positive phase
            pos_hidden_prob, pos_hidden_states = sample_hidden(v0, weights, hidden_bias)

            # neg_visible_states = v0.astype(np.int64)
            # Gibbs sampling
            for _ in range(k):
                neg_visible_prob, neg_visible_states = sample_visible(pos_hidden_states, weights, visible_bias)
                neg_hidden_prob , neg_hidden_states = sample_hidden(neg_visible_states, weights, hidden_bias)

            # neg_hidden_prob,  neg_hidden_states  = sample_hidden(neg_visible_states, weights, hidden_bias)

            # Update weights and biases
            """In the positive statistics collection, using pos_hidden_states is closer to the mathematical 
            model of an RBM, but using pos_hidden_prob usually has less sampling noise which allows slightly 
            faster learning. Hence, I can swap pos_hidden_states with pos_hidden_prob."""
            positive_associations = dot_product(v0.T, pos_hidden_prob)
            negative_associations = dot_product(neg_visible_states.T, neg_hidden_prob)

            lr = learning_rate / batch_size

            penalty = np.sum(weights.ravel()) * 0.001

            delta_w = lr * ((positive_associations - negative_associations) - penalty)
            delta_hidden_bias = lr * mean_axis_0(pos_hidden_prob - neg_hidden_prob)
            delta_visible_bias = lr * mean_axis_0(v0 - neg_visible_states)

            deltas = [delta_w, delta_hidden_bias, delta_visible_bias]

            # Apply momentum
            momentum = 0.5
            velocity_w = momentum * velocity_w + delta_w
            velocity_hidden_bias = momentum * velocity_hidden_bias + delta_hidden_bias
            velocity_visible_bias = momentum * velocity_visible_bias + delta_visible_bias

            # Update weights and biases with velocity
            weights += velocity_w
            hidden_bias += velocity_hidden_bias
            visible_bias += velocity_visible_bias

        if monitoring:
            if epoch % 50  == 0 and epoch !=0:
                print(f"Epoch: {epoch}/{num_epochs}")

                # Calculate free energy
                start_idx = np.random.randint(low=0, high=val.shape[0]-100)
                val_subset = val[start_idx: start_idx+100]
                f_e_data = free_energy(data_subset, weights, visible_bias, hidden_bias)
                f_e_val = free_energy(val_subset, weights, visible_bias, hidden_bias)
                f_energy.append([f_e_data, f_e_val])

                # Calculate reconstruction error
                error = np.sum((v0 - neg_visible_states) ** 2) / batch_size
                reconstructed_error.append(error)

                # Plot for monitoring
                # monitoring_plots(weights, hidden_bias, visible_bias, deltas, pos_hidden_prob, epoch, id)

            if epoch % 100 == 0 and epoch != 0:
                id_epoch = f"{id}_epoch_{epoch}"
                print(f"{id} - Sampling from the RBM...")
                samples = sample(data.shape[1], weights, hidden_bias, visible_bias, k=1000, n_samples=data.shape[0])
                print(f"Done\n")

                # Convert to real values
                print("Converting the samples from binary to real values...")
                samples = from_binary_to_real(samples, X_min, X_max).to_numpy()
                data_plot = from_binary_to_real(data, X_min, X_max).to_numpy()
                print(f"Done\n")

                # Calculate Wasserstein distance
                print("Calculating the KL divergence...")
                kldiv = kl_div(data.ravel(), samples.ravel())
                KL_div.append(kldiv)
                print(f"Done\n")

                if epoch >= 1000:
                    if epoch == 1000: print(f"Early stopping monitoring on the KL divergence activated")
                    # Check the early stopping criteria on the difference in Wasserstein distances
                    min_w = np.min(KL_div[10:])
                    if kldiv - min_w > 0:
                        counter += 1
                        best_params.append([weights, hidden_bias, visible_bias])
                        print(f"Counter early stopping: {counter}")
                        if counter == 25:
                            print(f"Early stopping at epoch {epoch}")
                            print(f"Best epoch: {np.argmin(KL_div[10:]) * 100}")
                            weights, hidden_bias, visible_bias = best_params[0]
                            break
                    else:
                        best_params = []
                        counter = 0
                        print(f"Counter early stopping: {counter}")

                print("Plotting results...")
                # Plot the samples and the recontructed error
                plot_distributions(samples, data_plot, currencies, id_epoch)
                qq_plots(samples, data_plot, currencies, id_epoch)
                plot_objectives(reconstructed_error, f_energy, KL_div, id_epoch)

                print(f"Done\n")
    return reconstructed_error, f_energy, KL_div, weights, hidden_bias, visible_bias

@nb.njit
def sample(num_visible, weights, hidden_bias, visible_bias, k, n_samples):
    np.random.seed(666)
    samples = np.random.randint(low=0, high=2, size=(n_samples, num_visible))
    for _ in range(k):  # number of Gibbs steps
        _, samples = sample_hidden(samples, weights, hidden_bias)
        _, samples = sample_visible(samples, weights, visible_bias)
    return samples



