import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from rbm import *
from utils import *
import time
from scipy import stats
import argparse
np.random.seed(666)

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train_rbm", "-t", action="store_true", help="Train the RBM")
args = parser.parse_args()


start = time.time()
# print(f"TRAINING RBM ON CURRENCY DATA\n")
# # Define the currency pairs
# currency_pairs = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCAD=X']

# # Check if the data is already downloaded
# try:
#     data = pd.read_pickle('currencies_data.pkl')
# except FileNotFoundError:
#     print("Downloading data...")
#     data = data_download(currency_pairs, start_date="2022-01-01", end_date="2023-01-01")
#     data.to_pickle('data/currencies_data.pkl')
#     print(f"Done\n")


# # Check for missing values. If there is a missing value, drop the corresponding row
# if data.isnull().values.any():
#     print("Missing values detected. Dropping rows with missing values...")
#     # collect the number of rows before dropping
#     num_rows = data.shape[0]
#     # drop rows with missing values
#     data = data.dropna(axis=0)
#     # collect the number of rows after dropping
#     num_rows_dropped = data.shape[0]
#     print(f"Done. Number of rows dropped: {num_rows - num_rows_dropped}\n")

# Create a synthetic normal dataset
normal_1 = np.random.normal(-2, 1, (5000))
normal_2 = np.random.normal(2, 2, (5000))
data = np.concatenate((normal_1, normal_2)).reshape(-1, 1)
data = pd.DataFrame(data)
# Convert the data to binary
data_binary, (X_min, X_max) = from_real_to_binary(data)

if args.train_rbm:
    # Define the RBM
    num_visible = data_binary.shape[1]
    num_hidden = 12
    print(f"Number of visible units:\n\t{num_visible}")
    print(f"Number of hidden units:\n\t{num_hidden}")
    weights, hidden_bias, visible_bias = initialize_rbm(num_visible, num_hidden)
    print(f"Initial weights:\n\t{weights}")
    print(f"Initial hidden bias:\n\t{hidden_bias}")
    print(f"Initial visible bias:\n\t{visible_bias}")

    # Train the RBM
    reconstruction_error, weights, hidden_bias, visible_bias = train(
        data_binary, weights, hidden_bias, visible_bias, num_epochs=1000, batch_size=10, learning_rate=0.1, k=1, monitoring=True)
    np.save("output/weights.npy", weights)
    np.save("output/hidden_bias.npy", hidden_bias)
    np.save("output/visible_bias.npy", visible_bias)
    fig, ax = plt.subplots(1, 1, figsize=(13, 5), tight_layout=True)
    ax.plot(reconstruction_error)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction error")
    ax.set_title("Reconstruction error")
else:
    weights = np.load("output/weights.npy")
    hidden_bias = np.load("output/hidden_bias.npy")
    visible_bias = np.load("output/visible_bias.npy")

# Sample from the RBM
print("Sampling from the RBM...")
samples = parallel_sample(weights, hidden_bias, visible_bias, num_samples=10000, num_visible=data_binary.shape[1], K=10e3)
print(f"Done\n")
# Convert to real values
print("Converting the samples from binary to real values...")
samples = from_binary_to_real(samples, X_min, X_max).to_numpy().reshape(-1)
print(f"Done\n")

data = data.to_numpy().reshape(-1)

total_time = time.time() - start
print(f"Total time: {total_time} seconds")

# Plot the samples and the recontructed error
fig, ax = plt.subplots(1, 2, figsize=(13, 5), tight_layout=True)
ax[0].hist(samples, bins=50)
ax[0].hist(data, bins=50, alpha=0.5)
ax[0].set_xlabel("Value")
ax[0].set_ylabel("Frequency")
ax[0].set_title("Histogram of RBM samples and original data")
# Generate QQ plot data
quantiles1, quantiles2 = stats.probplot(samples, dist="norm")[0][0], stats.probplot(data, dist="norm")[0][0]
(osm, osr), (slope, intercept, r) = stats.probplot(samples, dist="norm", sparams=(data.mean(), data.std()))
# Create QQ plot
ax[1].scatter(quantiles1, quantiles2)
ax[1].plot(osm, slope * osm + intercept, color='r')
ax[1].set_title('QQ-plot: Generated RBM vs. Data')
plt.show()





