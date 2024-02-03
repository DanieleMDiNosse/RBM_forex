import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
# plt.style.use('seaborn')
from sklearn.model_selection import train_test_split
import pandas as pd
from rbm import *
from utils import *
import time
from scipy import stats
import argparse
import os


# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train_rbm", "-t", action="store_true", help="Train the RBM")
parser.add_argument("--dataset", "-d", type=str, default="normal", help="Dataset to use. Options: normal, bi_normal, poisson, AR3")
parser.add_argument("--num_features", "-f", type=int, default=1, help="Number of features. Default: 1")
parser.add_argument("--epochs", "-e", type=int, default=1500, help="Number of epochs. Default: 1500")
parser.add_argument("--learning_rate", "-lr", type=float, default=0.01, help="Learning rate, Default: 0.01")
parser.add_argument("--batch_size", "-b", type=int, default=10, help="Batch size for training. Default: 10")
parser.add_argument("--k_step", "-k", type=int, default=1, help="Number of Gibbs sampling steps in the training process. Default: 1")
args = parser.parse_args()

start = time.time()
id = f'S{os.getpid()}'
print(f"{id} - TRAINING RBM ON SYNTHETIC DATA\n")
# retrive the id of the process
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
    num_hidden = 12
    print(f"Number of visible units:\n\t{num_visible}")
    print(f"Number of hidden units:\n\t{num_hidden}")
    weights, hidden_bias, visible_bias = initialize_rbm(train_data, num_visible, num_hidden)
    print(f"Weights shape:\n\t{weights.shape}")
    print(f"Initial hidden bias:\n\t{hidden_bias}")
    print(f"Initial visible bias:\n\t{visible_bias}\n")

    # Train the RBM
    reconstruction_error, f_energy, weights, hidden_bias, visible_bias = train(
        train_data, val,  weights, hidden_bias, visible_bias, num_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, k=args.k_step, monitoring=True,id=id)
    np.save("output/weights.npy", weights)
    np.save("output/hidden_bias.npy", hidden_bias)
    np.save("output/visible_bias.npy", visible_bias)
    print(f"Final weights:\n\t{weights}")
    print(f"Final hidden bias:\n\t{hidden_bias}")
    print(f"Final visible bias:\n\t{visible_bias}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 5), tight_layout=True)
    ax[0].plot(reconstruction_error)
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Reconstruction error")
    ax[0].set_title("Reconstruction error")
    ax[1].plot(np.array(f_energy)[:,0], 'green', label="Training data", alpha=0.7)
    ax[1].plot(np.array(f_energy)[:,1], 'blue', label="Validation data", alpha=0.7)
    ax[1].legend()
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Free energy")
    ax[1].set_title("Free energy")
    plt.savefig("output/rec_fenergy.png")
else:
    print("Loading weights...")
    weights = np.load("output/weights.npy")
    hidden_bias = np.load("output/hidden_bias.npy")
    visible_bias = np.load("output/visible_bias.npy")

# Sample from the RBM
print("Sampling from the RBM...")
samples = sample(train_data.shape[1], weights, hidden_bias, visible_bias, k=1000, n_samples=train_data.shape[0])
print(f"Done\n")

# Convert to real values
print("Converting the samples from binary to real values...")
samples = from_binary_to_real(samples, X_min, X_max).to_numpy()
print(f"Done\n")

total_time = time.time() - start
print(f"Total time: {total_time} seconds")

# Compute correlations
print("Computing correlations...")
correlations = calculate_correlations(pd.DataFrame(samples))
for key, value in correlations.items():
    print(f"Pais: {key}, Value: {value}")
print(f"Done\n")

train_data = data[:train_data.shape[0]].reshape(samples.shape)
currencies = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD']

# Plot the samples and the recontructed error
plot_distributions(samples, train_data, currencies, id)

# Generate QQ plot data
qq_plots(samples, train_data, currencies, id)

# Plot upper and lower tail distribution functions
plot_tail_distributions(samples, train_data, currencies, id)

if args.num_features > 2:
    # Plot PCA components with marginals
    plot_pca_with_marginals(samples, train_data, id)





