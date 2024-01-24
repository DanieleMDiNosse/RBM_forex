import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from rbm import *
from utils import *
import time
np.random.seed(666)


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
data = np.random.normal(0, 1, (1000, 1))
data = pd.DataFrame(data)

# Convert the data to binary
data_binary, (X_min, X_max) = from_real_to_binary(data)

# Define the RBM
num_visible = data_binary.shape[1]  # Number of binary features in your data
num_hidden = 32   # Number of hidden units
# rbm = RBM(num_visible, num_hidden)
weights, hidden_bias, visible_bias = initialize_rbm(num_visible, num_hidden)

# Train the RBM
# reconstruction_error = rbm.train(data_binary, batch_size=10, num_epochs=100, learning_rate=0.1, k=1, monitoring=True)
reconstruction_error, weights, hidden_bias, visible_bias = train(
    data_binary, weights, hidden_bias, visible_bias, num_epochs=10000, batch_size=10, learning_rate=0.1, k=10, monitoring=True)

# Sample from the RBM
num_samples = 1000
# samples = rbm.sample(num_samples, k=10)
samples = sample(weights, hidden_bias, visible_bias, num_samples, num_visible, k=10)

# Convert the samples back to real values
samples = from_binary_to_real(samples, X_min, X_max)

total_time = time.time() - start
print(f"Total time: {total_time} seconds")

# Plot the samples and the recontructed error
fig, ax = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
ax[0].plot(reconstruction_error)
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Reconstruction error")
ax[1].hist(samples, bins=50)
ax[1].hist(data, bins=50, alpha=0.5)
ax[1].set_xlabel("Value")
ax[1].set_ylabel("Frequency")
plt.show()




