import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from rbm import *
from utils import *
import time
import logging
import argparse
import os


# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info'"))
parser.add_argument("--train_rbm", "-t", action="store_true", help="Train the RBM")
parser.add_argument("--dataset", "-d", type=str, default="normal", help="Dataset to use. Options: normal, bi_normal, poisson, AR3, mixed")
parser.add_argument("--num_features", "-f", type=int, default=1, help="Number of features. Uselees if you chose mixed dataset. Default: 1")
parser.add_argument("--hidden_units", "-h", type=int, default=12, help="Number of hidden units. Default: 12")
parser.add_argument("--epochs", "-e", type=int, default=1500, help="Number of epochs. Default: 1500")
parser.add_argument("--learning_rate", "-lr", type=float, default=0.01, help="Learning rate, Default: 0.01")
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
logging.basicConfig(filename=f'logs/main_sythetic_{id}.log', format='%(message)s', level=levels[args.log])

start = time.time()
print(f"{id} - TRAINING RBM ON SYNTHETIC DATA\n")
np.random.seed(666)

# Create a synthetic normal dataset
if args.dataset == "normal":
    print(f"Dataset: \n\tNormal distribution")
    data = np.random.normal(-2, 1, (10000)).reshape(-1, args.num_features)
if args.dataset == "bi_normal":
    print(f"Dataset: \n\tBi-normal distribution")
    normal_1 = np.random.normal(-2, 1, (5000))
    normal_2 = np.random.normal(2, 2, (5000))
    data = np.concatenate((normal_1, normal_2)).reshape(-1, args.num_features)
if args.dataset == "poisson":
    print(f"Dataset: \n\tPoisson distribution")
    data = np.random.poisson(5, (10000)).reshape(-1, args.num_features)
if args.dataset == "AR3":
    print(f"Dataset: \n\tAR(3)")
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
print(f"Data entries type:\n\t{data[np.random.randint(0, data.shape[0])].dtype}")
print(f"Data binary entries type:\n\t{data_binary[np.random.randint(0, data_binary.shape[0])].dtype}\n")
print(f"Data binary shape:\n\t{data_binary.shape}")
print(f"Training data shape:\n\t{train_data.shape}")
print(f"Validation data shape:\n\t{val.shape}\n")

if args.train_rbm:
    # Define the RBM
    num_visible = train_data.shape[1]
    num_hidden = args.hidden_units
    print(f"Number of visible units:\n\t{num_visible}")
    print(f"Number of hidden units:\n\t{num_hidden}")
    weights, hidden_bias, visible_bias = initialize_rbm(train_data, num_visible, num_hidden)
    print(f"Weights shape:\n\t{weights.shape}")
    print(f"Initial hidden bias:\n\t{hidden_bias}")
    print(f"Initial visible bias:\n\t{visible_bias}\n")

    # Train the RBM
    variables_for_monitoring = [X_min, X_max, names]
    reconstruction_error, f_energy, wasserstein_dist, weights, hidden_bias, visible_bias = train(
        train_data, val, weights, hidden_bias, visible_bias, num_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, k=args.k_step, monitoring=True,id=id, var_mon=variables_for_monitoring)
    np.save("output/weights.npy", weights)
    np.save("output/hidden_bias.npy", hidden_bias)
    np.save("output/visible_bias.npy", visible_bias)
    print(f"Final weights:\n\t{weights}")
    print(f"Final hidden bias:\n\t{hidden_bias}")
    print(f"Final visible bias:\n\t{visible_bias}")
else:
    print("Loading weights...")
    weights = np.load("output/weights.npy")
    hidden_bias = np.load("output/hidden_bias.npy")
    visible_bias = np.load("output/visible_bias.npy")

# Plot the objectives
plot_objectives(reconstruction_error, f_energy, wasserstein_dist, id)

# Sample from the RBM
print("Sampling from the RBM...")
samples = parallel_sample(train_data.shape[1], weights, hidden_bias, visible_bias, k=1000, n_samples=train_data.shape[0])
print(f"Done\n")

# Convert to real values
print("Converting the samples from binary to real values...")
samples = from_binary_to_real(samples, X_min, X_max).to_numpy()
print(f"Done\n")

total_time = time.time() - start
print(f"Total time: {total_time} seconds")

# Compute correlations on generated data
print("Computing correlations...")
gen_correlations = calculate_correlations(pd.DataFrame(samples))
# Compute correlations on original data
original_correlations = calculate_correlations(pd.DataFrame(train_data))

print(f"Original correlations:\n{original_correlations}")
print(f"Generated correlations:\n{gen_correlations}")

print("Plotting results...")
train_data = data[:train_data.shape[0]].reshape(samples.shape)

# Plot the original and generated distributions
plot_distributions(samples, train_data, names, id)

# Generate QQ plot data
qq_plots(samples, train_data, names, id)

# Plot the concentration functions
plot_tail_concentration_functions(train_data, samples, names, id)

# Plot upper and lower tail distribution functions
# plot_tail_distributions(samples, train_data, names, id)

# Plot the tail concentration functions
dict_f_conc = plot_tail_concentration_functions(pd.DataFrame(samples), pd.DataFrame(train_data), id)

if samples.shape[1] > 2:
    # Plot PCA components with marginals
    plot_pca_with_marginals(samples, train_data, id)

# Create the animated gifs
print("Creating animated gifs...")
create_animated_gif('output/historgrams', id, output_filename=f'{id}_histograms.gif')
create_animated_gif('output/weights_receptive_field', id, output_filename=f'{id}_weights_receptive_field.gif')
print(f"Done\n")
print(f'Finished id {id}!')






