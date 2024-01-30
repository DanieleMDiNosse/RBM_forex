import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from sklearn.model_selection import train_test_split
import pandas as pd
from rbm import *
from utils import *
import time
import argparse
import os
np.random.seed(666)

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train_rbm", "-t", action="store_true", help="Train the RBM. Default: False")
parser.add_argument("--epochs", "-e", type=int, default=1500, help="Number of epochs. Default: 1500")
parser.add_argument("--learning_rate", "-lr", type=float, default=0.01, help="Learning rate, Default: 0.01")
parser.add_argument("--continue_train", "-c", action="store_true", help="Load the weights and continue training for the specified number of epochs. Default: False")
args = parser.parse_args()


start = time.time()
print(f"TRAINING RBM ON CURRENCY DATA\n")
# Define the currency pairs
currency_pairs = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCAD=X']
start_date = "1999-01-01"
end_date = "2023-01-01"
# Check if the data is already downloaded
try:
    print("Loading data...")
    data = pd.read_pickle(f'data/currencies_data_{start_date}_{end_date}.pkl')
    data = data.values
except FileNotFoundError:
    print("Downloading data...")
    data = data_download(currency_pairs, start_date=start_date, end_date=end_date)
    data.to_pickle(f'data/currencies_data_{start_date}_{end_date}.pkl')
    data = data.values
    print(f"Done\n")

# Remove missing values
data = remove_missing_values(data)

# Convert the data to binary
data_binary, (X_min, X_max) = from_real_to_binary(data)

# Split the data into train and test sets
train_data, val = train_test_split(data_binary, test_size=0.1)
print(f"Data entries type:\n\t{data[np.random.randint(0, data.shape[0])].dtype}")
print(f"Data binary entries type:\n\t{data_binary[np.random.randint(0, data_binary.shape[0])].dtype}")
print(f"Data binary shape:\n\t{data_binary.shape}")
print(f"Training data shape:\n\t{train_data.shape}")
print(f"Validation data shape:\n\t{val.shape}")

if args.train_rbm:
    # Define the RBM
    num_visible = train_data.shape[1]
    num_hidden = 30
    print(f"Number of visible units:\n\t{num_visible}")
    print(f"Number of hidden units:\n\t{num_hidden}\n")
    if args.continue_train:
        print("Continue training from past learned RBM parameters...")
        weights = np.load("output/weights.npy")
        hidden_bias = np.load("output/hidden_bias.npy")
        visible_bias = np.load("output/visible_bias.npy")
    else:
        weights, hidden_bias, visible_bias = initialize_rbm(train_data, num_visible, num_hidden)
    print(f"Initial weights shape:\n\t{weights.shape}")
    print(f"Initial hidden bias:\n\t{hidden_bias}")
    print(f"Initial visible bias:\n\t{visible_bias}\n")

    # Train the RBM
    reconstruction_error, f_energy, weights, hidden_bias, visible_bias = train(
        train_data, val,  weights, hidden_bias, visible_bias, num_epochs=args.epochs, batch_size=10, learning_rate=args.learning_rate, k=10, monitoring=True)
    np.save("output/weights.npy", weights)
    np.save("output/hidden_bias.npy", hidden_bias)
    np.save("output/visible_bias.npy", visible_bias)
    np.save("output/reconstruction_error.npy", reconstruction_error)
    np.save("output/f_energy.npy", f_energy)

    print(f"Final weights:\n\t{weights}")
    print(f"Final hidden bias:\n\t{hidden_bias}")
    print(f"Final visible bias:\n\t{visible_bias}\n")

else:
    print("Loading weights, recontruction error and mean free energy...")
    weights = np.load("output/weights.npy")
    hidden_bias = np.load("output/hidden_bias.npy")
    visible_bias = np.load("output/visible_bias.npy")
    reconstruction_error = np.load("output/reconstruction_error.npy")
    f_energy = np.load("output/f_energy.npy")
    print(f"Done\n")

fig, ax = plt.subplots(1, 2, figsize=(11, 5), tight_layout=True)
ax[0].plot(reconstruction_error)
ax[0].set_xlabel("Epoch x 100")
ax[0].set_ylabel("Reconstruction error")
ax[0].set_title("Reconstruction error")
ax[1].plot(np.array(f_energy)[:,0], 'green', label="Training data", alpha=0.7)
ax[1].plot(np.array(f_energy)[:,1], 'blue', label="Validation data", alpha=0.7)
ax[1].legend()
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Free energy")
ax[1].set_title("Free energy")
c = 0
while os.path.exists(f"output/reconstruction_error_{c}.png"):
    c += 1
plt.savefig(f"output/reconstruction_error_{c}.png")

# try:
#     samples = np.load(f"output/samples_{start_date}_{end_date}_{args.epochs}_{args.learning_rate}.npy")
#     print(f"Samples were already been generated. Loaded from the output folder.\n")
# except FileNotFoundError:
print("Sampling from the RBM...")
samples = sample(train_data.shape[1], weights, hidden_bias, visible_bias, k=1000, n_samples=train_data.shape[0])
np.save(f"output/samples_{start_date}_{end_date}_{args.epochs}_{args.learning_rate}.npy", samples)
print(f"Done\n")

# Convert to real values
print("Converting the samples from binary to real values...")
samples = from_binary_to_real(samples, X_min, X_max).to_numpy()
print(f"Done\n")

total_time = time.time() - start
print(f"Total time: {total_time} seconds")

# Compute correlations
print("Computing correlations...")
currencies_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD']
currencies_pairs = list(itertools.combinations(currencies_pairs, 2))

gen_correlations = calculate_correlations(pd.DataFrame(samples))
original_correlations = calculate_correlations(pd.DataFrame(data[:train_data.shape[0]]))

print(f"Original correlations:")
for pair, value in zip(currencies_pairs, original_correlations.values()):
    print(f"Pairs: {pair}, Value: {value}")

print(f"Generated correlations:")
for pair, value in zip(currencies_pairs, gen_correlations.values()):
    print(f"Pairs: {pair}, Value: {value}")
print(f"Done\n")

data = data[:train_data.shape[0]].reshape(samples.shape)
# Plot the samples and the recontructed error
plot_distributions(samples, data, currencies_pairs)

# Generate QQ plot data
qq_plots(samples, data, currencies_pairs)

# Plot upper and lower tail distribution functions
plot_tail_distributions(samples, data, currencies_pairs)

#Plot PCA with 2 components
plot_pca_with_marginals(samples, data)

plt.show()

'''Things to do:
- t-SNE of generated and original data to check if the RBM generates enough variability
- Check hidden activation patterns. For example I can see if there is some common pattern in stree periods'''
