import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import time
import logging
import argparse
import os
import numpy as np
from numba import njit
from utils import *
import multiprocessing as mp

def visibile_bias_init(data_binary):
    frequencies = np.mean(data_binary, axis=0)
    visible_bias_t0 = np.log(frequencies / (1 - frequencies))
    return visible_bias_t0

def initialize_rbm(data, num_visible, num_hidden):
    np.random.seed(666)
    weights = np.random.normal(0, 0.01, (num_visible, num_hidden))
    # hidden_bias = -np.ones(num_hidden) * 4
    hidden_bias = np.random.normal(0, 0.01, num_hidden)
    visible_bias = visibile_bias_init(data)
    
    return weights, hidden_bias, visible_bias

@njit
def sample_hidden(visible, weights, hidden_bias):
    hidden_activations = dot_product(visible, weights) + hidden_bias
    if np.isnan(hidden_activations).any():
        print(f"There NaNs in the hidden activations")
    hidden_probabilities = sigmoid(hidden_activations)
    hidden_states = boolean_to_int(hidden_probabilities > np.random.random(hidden_probabilities.shape))
    return hidden_probabilities, hidden_states

@njit
def sample_visible(hidden, weights, visible_bias):
    visible_activations = dot_product(hidden, weights.T) + visible_bias
    if np.isnan(visible_activations).any():
        print(f"There NaNs in the visible activations")
    visible_probabilities = sigmoid(visible_activations)
    visible_states = boolean_to_int(visible_probabilities > np.random.random(visible_probabilities.shape))
    return visible_probabilities, visible_states

# @njit
def train(data, val, weights, hidden_bias, visible_bias, num_epochs, batch_size, learning_rate, k, monitoring, id, additional_quantities):
    X_min_train, X_max_train, currencies = additional_quantities
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
        if epoch % 100 == 0: print_(f"Epoch: {epoch}/{num_epochs}")
        for i in range(0, num_samples, batch_size):
            v0 = data[i:i+batch_size]

            # Positive phase
            pos_hidden_prob, pos_hidden_states = sample_hidden(v0, weights, hidden_bias)
            pos_hidden_prob0 = pos_hidden_prob.copy()
            # neg_visible_states = v0.astype(np.int64)
            # Gibbs sampling
            for _ in range(k):
                neg_visible_prob, neg_visible_states = sample_visible(pos_hidden_states, weights, visible_bias)
                pos_hidden_prob , pos_hidden_states = sample_hidden(neg_visible_states, weights, hidden_bias)

            # Update weights and biases
            positive_associations = dot_product(pos_hidden_prob0.T, v0)
            negative_associations = dot_product(pos_hidden_prob.T, neg_visible_states)

            lr = learning_rate / batch_size

            penalty = np.sum(weights.ravel()) * 0.001

            delta_w = lr * ((positive_associations - negative_associations) - penalty)
            delta_hidden_bias = lr * mean_axis_0(pos_hidden_prob0 - pos_hidden_prob)
            delta_visible_bias = lr * mean_axis_0(v0 - neg_visible_states)

            # Apply momentum
            if epoch < 100:
                momentum = 0.5
            else:
                momentum = 0.9

            velocity_w = momentum * velocity_w + delta_w.T
            velocity_hidden_bias = momentum * velocity_hidden_bias + delta_hidden_bias
            velocity_visible_bias = momentum * velocity_visible_bias + delta_visible_bias

            deltas = [velocity_w, velocity_hidden_bias, velocity_visible_bias]

            # Update weights and biases with velocity
            weights += velocity_w
            hidden_bias += velocity_hidden_bias
            visible_bias += velocity_visible_bias

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

            if epoch % 100 == 0 and epoch != 0:
                id_epoch = f"{id}_epoch_{epoch}"
                print_(f"{id} - Sampling from the RBM...")
                samples = parallel_sample(weights, hidden_bias, visible_bias, 1000, data.shape[0])
                print_(f"Done\n")

                # Convert to real values
                print_("Converting the samples from binary to real values...")
                # Remove the volatility indicators
                samples = from_binary_to_real(samples, X_min_train, X_max_train).to_numpy()
                data_plot = from_binary_to_real(data, X_min_train, X_max_train).to_numpy()
                print_(f"Done\n")

                print_("Plotting results...")
                # Plot the samples and the recontructed error
                plot_distributions(samples, data_plot, currencies, id_epoch)
                qq_plots(samples, data_plot, currencies, id_epoch)
                plot_objectives(reconstruction_error, f_energy_overfitting, f_energy_diff, diff_fenergy, id_epoch)
                plot_tail_concentration_functions(data_plot, samples, currencies, id_epoch)
                print_(f"Done\n")

    return reconstruction_error, f_energy_overfitting, f_energy_diff, diff_fenergy, weights, hidden_bias, visible_bias

# @njit
def sample_batch(weights, hidden_bias, visible_bias, k, batch_size):
    '''Sample from the RBM.'''
    samples = np.random.randint(low=0, high=2, size=(batch_size, weights.shape[0]))
    for _ in range(k):  # number of Gibbs steps
        _, samples = sample_hidden(samples, weights, hidden_bias)
        _, samples = sample_visible(samples, weights, visible_bias)
    return samples

def parallel_sample(weights, hidden_bias, visible_bias, k, n_samples, n_processors=8):
    # np.random.seed(666)
    pool = mp.Pool(processes=n_processors)
    
    # Calculate the number of samples per processor
    batch_sizes = [n_samples // n_processors for _ in range(n_processors)]
    batch_sizes[-1] += n_samples % n_processors
    
    # Create a list of arguments for each batch
    args = [(weights, hidden_bias, visible_bias, k, batch_size) for _, batch_size in zip(range(n_processors), batch_sizes)]
    
    # Execute the function in parallel
    results = pool.starmap(sample_batch, args)
    
    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()
    
    # Combine the results
    samples = np.vstack(results)
    return samples


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log", default="info",
                            help=("Provide logging level. Example --log debug', default='info'"))
    parser.add_argument("--train_rbm", "-t", action="store_true", help="Train the RBM")
    parser.add_argument("--dataset", "-d", type=str, default="normal", help="Dataset to use. Options: normal, bi_normal, poisson, AR3, mixed")
    parser.add_argument("--num_features", "-f", type=int, default=1, help="Number of features. Uselees if you chose mixed dataset. Default: 1")
    parser.add_argument("--hidden_units", "-hu", type=int, default=12, help="Number of hidden units. Default: 12")
    parser.add_argument("--epochs", "-e", type=int, default=1500, help="Number of epochs. Default: 1500")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.001, help="Learning rate, Default: 0.001")
    parser.add_argument("--batch_size", "-b", type=int, default=10, help="Batch size for training. Default: 10")
    parser.add_argument("--k_step", "-k", type=int, default=1, help="Number of Gibbs sampling steps in the training process. Default: 1")
    levels = {'critical': logging.CRITICAL,
                'error': logging.ERROR,
                'warning': logging.WARNING,
                'info': logging.INFO,
                'debug': logging.DEBUG}
    args = parser.parse_args()

    # Track the process id
    id = f'S{os.getpid()}'

    #Check if the output folder exists
    if not os.path.exists("logs"):
        os.makedirs("logs")

    start = time.time()
    print_(f"{id} - TRAINING RBM ON SYNTHETIC DATA\n")
    np.random.seed(666)

    # Create a synthetic normal dataset
    if args.dataset == "normal":
        print_(f"Dataset: \n\tNormal distribution")
        data = np.random.normal(-2, 1, (10000)).reshape(-1, args.num_features)
    if args.dataset == "bi_normal":
        print_(f"Dataset: \n\tBi-normal distribution")
        normal_1 = np.random.normal(-2, 1, (5000))
        normal_2 = np.random.normal(2, 2, (5000))
        data = np.concatenate((normal_1, normal_2)).reshape(-1, args.num_features)
    if args.dataset == "poisson":
        print_(f"Dataset: \n\tPoisson distribution")
        data = np.random.poisson(5, (10000)).reshape(-1, args.num_features)
    if args.dataset == "AR3":
        print_(f"Dataset: \n\tAR(3)")
        data = np.random.normal(0, 1, (10000))
        for i in range(3, data.shape[0]):
            data[i] = 0.5 * data[i-1] + 0.3 * data[i-2] + 0.2 * data[i-3] + np.random.normal(0, 1)
        data = data.reshape(-1, args.num_features)
    if args.dataset == "mixed":
        data = mixed_dataset(n_samples=10000)

    names = [f'Dataset {i}' for i in range(data.shape[1])]

    # Convert the data to binary
    data_binary, (X_min, X_max) = from_real_to_binary(data)
    # Split the data into train and test sets
    train_data, val = train_test_split(data_binary, test_size=0.1)
    print_(f"Data entries type:\n\t{data[np.random.randint(0, data.shape[0])].dtype}")
    print_(f"Data binary entries type:\n\t{data_binary[np.random.randint(0, data_binary.shape[0])].dtype}\n")
    print_(f"Data binary shape:\n\t{data_binary.shape}")
    print_(f"Training data shape:\n\t{train_data.shape}")
    print_(f"Validation data shape:\n\t{val.shape}\n")

    if args.train_rbm:
        # Define the RBM
        num_visible = train_data.shape[1]
        num_hidden = args.hidden_units
        print_(f"Number of visible units:\n\t{num_visible}")
        print_(f"Number of hidden units:\n\t{num_hidden}")
        weights, hidden_bias, visible_bias = initialize_rbm(train_data, num_visible, num_hidden)
        print_(f"Weights shape:\n\t{weights.shape}")
        print_(f"Initial hidden bias:\n\t{hidden_bias}")
        print_(f"Initial visible bias:\n\t{visible_bias}\n")

        # Train the RBM
        variables_for_monitoring = [X_min, X_max, names]
        reconstruction_error, f_energy_overfitting, f_energy_diff, diff_fenergy, weights, hidden_bias, visible_bias = train(
            train_data, val, weights, hidden_bias, visible_bias, num_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, k=args.k_step, monitoring=True, id=id, additional_quantities=variables_for_monitoring)
        np.save("output/weights.npy", weights)
        np.save("output/hidden_bias.npy", hidden_bias)
        np.save("output/visible_bias.npy", visible_bias)
        np.save("output/reconstruction_error.npy", reconstruction_error)
        np.save("output/f_energy_overfitting.npy", f_energy_overfitting)
        np.save("output/f_energy_diff.npy", f_energy_diff)
        np.save("output/diff_fenergy.npy", diff_fenergy)
        print_(f"Final weights:\n\t{weights}")
        print_(f"Final hidden bias:\n\t{hidden_bias}")
        print_(f"Final visible bias:\n\t{visible_bias}")
    else:
        print_("Loading weights...")
        weights = np.load("output/weights.npy")
        hidden_bias = np.load("output/hidden_bias.npy")
        visible_bias = np.load("output/visible_bias.npy")
        reconstruction_error = np.load("output/reconstruction_error.npy")
        f_energy_overfitting = np.load("output/f_energy_overfitting.npy")
        f_energy_diff = np.load("output/f_energy_diff.npy")
        diff_fenergy = np.load("output/diff_fenergy.npy")
        print_(f"Done\n")

    # Plot the objectives
    plot_objectives(reconstruction_error, f_energy_overfitting, f_energy_diff, diff_fenergy, id)

    # Sample from the RBM
    print_("Sampling from the RBM...")
    samples = parallel_sample(weights, hidden_bias, visible_bias, k=1000, n_samples=train_data.shape[0])
    print_(f"Done\n")

    # Convert to real values
    print_("Converting the samples from binary to real values...")
    samples = from_binary_to_real(samples, X_min, X_max).to_numpy()
    print_(f"Done\n")

    total_time = time.time() - start
    print_(f"Total time: {total_time} seconds")
   
    print_("Plotting results...")
    train_data = data[:train_data.shape[0]].reshape(samples.shape)

    # Plot the original and generated distributions
    plot_distributions(samples, train_data, names, id)

    # Generate QQ plot data
    qq_plots(samples, train_data, names, id)

    # Plot the concentration functions
    plot_tail_concentration_functions(train_data, samples, names, id)

    if samples.shape[1] > 2:
        # Plot PCA components with marginals
        plot_pca_with_marginals(samples, train_data, id)

    # Create the animated gifs
    print_("Creating animated gifs...")
    create_animated_gif('output/historgrams', id, output_filename=f'{id}_histograms.gif')
    create_animated_gif('output/weights_receptive_field', id, output_filename=f'{id}_weights_receptive_field.gif')
    print_(f"Done\n")
    print_(f'Finished id {id}!')






