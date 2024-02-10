import numpy as np
import numba as nb
import time
from utils import *
from scipy.stats import wasserstein_distance
from scipy.special import kl_div
import multiprocessing as mp
import tensorflow as tf
np.random.seed(666)

def visible_bias_init(data_binary):
    # Assuming data_binary is a TensorFlow tensor
    frequencies = tf.reduce_mean(data_binary, axis=0)
    visible_bias_t0 = tf.math.log(frequencies / (1 - frequencies))
    return visible_bias_t0

def initialize_rbm(data_binary, num_visible, num_hidden):
    # Ensure random seed is set in a TensorFlow-compatible manner if needed
    tf.random.set_seed(666)
    weights = tf.Variable(tf.random.normal(shape=(num_visible, num_hidden), mean=0.0, stddev=0.01), dtype=tf.float32)
    hidden_bias = tf.Variable(tf.fill([num_hidden], 1.0), dtype=tf.float32)  # Creates a tensor filled with -4.0 of shape [num_hidden]
    visible_bias = tf.Variable(tf.fill([num_visible], 1.0), dtype=tf.float32)
    
    return weights, hidden_bias, visible_bias

def sample_hidden(visible, weights, hidden_bias):
    """Sample the hidden units given the visible units (positive phase)."""
    hidden_activations = tf.matmul(visible, weights) + hidden_bias
    hidden_probabilities = tf.sigmoid(hidden_activations)
    hidden_states = tf.cast(hidden_probabilities > tf.random.uniform(tf.shape(hidden_probabilities)), tf.float32)
    return hidden_probabilities, hidden_states

def sample_visible(hidden, weights, visible_bias):
    """Sample the visible units given the hidden units (negative phase)."""
    visible_activations = tf.matmul(hidden, tf.transpose(weights)) + visible_bias
    visible_probabilities = tf.sigmoid(visible_activations)
    visible_states = tf.cast(visible_probabilities > tf.random.uniform(tf.shape(visible_probabilities)), tf.float32)
    return visible_probabilities, visible_states

def sample(num_visible, weights, hidden_bias, visible_bias, k, n_samples):
    """Perform Gibbs sampling"""
    # Set the seed for reproducibility
    tf.random.set_seed(666)
    # Initialize samples with random binary values
    samples = tf.cast(tf.random.uniform((n_samples, num_visible), minval=0, maxval=2, dtype=tf.int32), tf.float32)
    for _ in range(k):  # Perform k Gibbs sampling steps
        _, samples = sample_hidden(samples, weights, hidden_bias)
        _, samples = sample_visible(samples, weights, visible_bias)
    return samples

# def sample(num_visible, weights, hidden_bias, visible_bias, k, n_samples):
#     np.random.seed(666)
#     samples = np.random.randint(low=0, high=2, size=(n_samples, num_visible))
#     for _ in range(k):  # number of Gibbs steps
#         _, samples = sample_hidden(samples, weights, hidden_bias)
#         _, samples = sample_visible(samples, weights, visible_bias)
#     return samples

def train(data, val, weights, hidden_bias, visible_bias, num_epochs, batch_size, learning_rate, k, monitoring, id, var_mon):
    X_min, X_max, currencies = var_mon
    num_samples = data.shape[0]

    best_params = []
    reconstructed_error = []
    f_energy_overfitting = []
    f_energy_diff = []
    losses = []
    wasserstein_dist = []
    counter = 0

    # Set the training data that I will use for monitoring the free energy
    start_idx = np.random.randint(low=0, high=num_samples-200)
    data_subset = data[start_idx: start_idx+200]

    lr = learning_rate / batch_size
    optimizer = tf.optimizers.Adam(learning_rate=lr)
    for epoch in range(num_epochs):
        for i in range(0, num_samples, batch_size):
            v0 = data[i:i+batch_size]
            v0 = tf.cast(v0, tf.float32)
            
            with tf.GradientTape() as tape:
                tape.watch([weights, hidden_bias, visible_bias])
                
                # Positive phase
                pos_hidden_prob, pos_hidden_states = sample_hidden(v0, weights, hidden_bias)

                # Gibbs sampling
                for _ in range(k):
                    neg_visible_prob, neg_visible_states = sample_visible(pos_hidden_states, weights, visible_bias)
                    neg_hidden_prob, neg_hidden_states = sample_hidden(neg_visible_states, weights, hidden_bias)
                
                # Loss calculation
                penalty = tf.reduce_sum(tf.reshape(weights, [-1])) * 0.001
                loss = rbm_loss(v0, neg_visible_states, visible_bias, hidden_bias, weights, penalty)

            # Compute gradients
            grads = tape.gradient(loss, [weights, hidden_bias, visible_bias])
            
             # Filter out None gradients if necessary
            grads_and_vars = [(grad, var) for grad, var in zip(grads, [weights, hidden_bias, visible_bias]) if grad is not None]
            
            # Apply gradients to update weights, hidden_bias, and visible_bias
            if grads_and_vars:
                optimizer.apply_gradients(grads_and_vars)
            else:
                print("No gradients to apply.")

        if monitoring:
            print(f"Epoch: {epoch}/{num_epochs}")
            print(f"\tLoss: {loss}")
            losses.append(loss)
            min_loss = np.min(losses)
            if loss > min_loss:
                counter += 1
                best_params.append([weights, hidden_bias, visible_bias])
                print(f"\tCounter: {counter}")
                if counter == 100:
                    print(f"\tEarly stopping at epoch {epoch}")
                    print(f"\tBest epoch: {epoch-100}")
                    weights, hidden_bias, visible_bias = best_params[0]
                    break
            else:
                counter = 0
                best_params = []
                print(f"\tCounter reset")

            # Calculate free energy for overfitting monitoring
            start_idx = np.random.randint(low=0, high=val.shape[0]-200)
            val_subset = val[start_idx: start_idx+200]
            f_e_data = free_energy(data_subset, weights, visible_bias, hidden_bias)
            f_e_val = free_energy(val_subset, weights, visible_bias, hidden_bias)
            diff = f_e_data - f_e_val
            print(f"\tFree energy difference for overfitting: {diff}")
            f_energy_overfitting.append([f_e_data, f_e_val])

            # Calculate reconstruction error
            error = np.sum((v0 - neg_visible_states) ** 2) / batch_size
            reconstructed_error.append(error)
            print(f"\tReconstruction error: {error}\n")

            # Plot for monitoring
            # monitoring_plots(weights, hidden_bias, visible_bias, deltas, pos_hidden_prob, epoch, id)

            if epoch % 10 == 0 and epoch != 0:
                id_epoch = f"{id}_epoch_{epoch}"

                print("Plotting results...")
                plot_objectives(reconstructed_error, f_energy_overfitting, losses, wasserstein_dist, id_epoch)
                print(f"Done\n")

            if epoch % 50 == 0 and epoch != 0:
                print(f"{id} - Sampling from the RBM...")
                id_epoch = f"{id}_epoch_{epoch}"
                samples = sample(data.shape[1], weights, hidden_bias, visible_bias, k=1000, n_samples=data.shape[0])
                print(f"Done\n")

                # Convert to real values
                print("Converting the samples from binary to real values...")
                samples = from_binary_to_real(samples, X_min, X_max).numpy()
                data_plot = from_binary_to_real(data, X_min, X_max).numpy()
                print(f"Done\n")

                # Calculate Wasserstein distance
                print("Calculating the Wasserstein dstance...")
                w_dist = wasserstein_distance(tf.reshape(data, [-1]), tf.reshape(samples, [-1]))
                wasserstein_dist.append(w_dist)
                print(f"Done")

                print("Plotting distributions and qq plots...")
                plot_distributions(samples, data_plot, currencies, id_epoch)
                qq_plots(samples, data_plot, currencies, id_epoch)
                print(f"Done\n")

    return reconstructed_error, f_energy_overfitting, f_energy_diff, wasserstein_dist, weights, hidden_bias, visible_bias




