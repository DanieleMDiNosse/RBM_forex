import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit
import itertools
from sklearn.decomposition import PCA
import seaborn as sns
import os


def data_download(currency_pairs, start_date, end_date, type='Close'):
    data = yf.download(currency_pairs, start=start_date, end=end_date)
    return data[type]

def binarize(number):
    # Convert number to binary, remove the '0b' prefix, and pad with zeros to 16 bits
    return bin(number)[2:].zfill(16)

def from_real_to_binary(data):
    '''Conversion of real-valued data set into binary features'''
    if isinstance(data, pd.DataFrame):
        data = data.values
    data_binary = np.array([[0] * (16 * data.shape[1]) for _ in range(data.shape[0])])
    x_min, x_max = np.min(data, axis=0), np.max(data, axis=0)
    for n in range(data.shape[1]):
        for l in range(data.shape[0]):
            x_integer = int(65535 * (data[l, n] - x_min[n]) / (x_max[n] - x_min[n]))
            binary_string = binarize(x_integer)
            for bit_index, bit in enumerate(binary_string):
                data_binary[l, n * 16 + bit_index] = int(bit)
    # convert to int32
    data_binary = data_binary.astype(np.int32)
    return data_binary, [x_min, x_max]

def from_binary_to_real(X_binary, X_min, X_max):
    """
    Converts a set of binary features back into real-valued data.
    """
    N_samples = len(X_binary)
    if isinstance(X_min, float):
        N_variables = 1
    else:
        N_variables = len(X_min)
    X_real = [[0] * N_variables for _ in range(N_samples)]

    for n in range(N_variables):
        for l in range(N_samples):
            X_integer = sum(X_binary[l][n * 16 + m] * (2 ** (15 - m)) for m in range(16))
            X_real[l][n] = X_min[n] + (X_integer * (X_max[n] - X_min[n]) / 65535)
    
    X_real = pd.DataFrame(X_real)

    return X_real

def remove_missing_values(data):
    """Check for missing values. If there is a missing value, drop the corresponding row."""
    if np.isnan(data).any():
        print("Missing values detected. Dropping rows with missing values...")
        # collect the number of rows before dropping
        num_rows = data.shape[0]
        # drop rows with missing values
        data = data[~np.isnan(data).any(axis=1)]
        # collect the number of rows after dropping
        num_rows_dropped = data.shape[0]
        print(f"Done. Number of rows dropped: {num_rows - num_rows_dropped}\n")
    else:
        pass
    return data

# def estimate_number_hidden_units(dataset):
#     """Estimate the number of hidden units based on the average informational
#     contents of a datavector."""
#     # Convert dataset to tuples for counting
#     tuple_dataset = map(tuple, dataset)

#     # Calculate relative frequencies (empirical distribution)
#     vector_counts = Counter(tuple_dataset)
#     total_vectors = len(dataset)
#     relative_frequencies = {vector: count / total_vectors for vector, count in vector_counts.items()}
#     values = list(relative_frequencies.values())
#     values.sort(reverse=True)
#     import matplotlib.pyplot as plt
#     plt.scatter(range(len(values)), values)
#     plt.show()

#     # Compute entropy of the empirical distribution
#     entropy_empirical = entropy(list(relative_frequencies.values()), base=2)
#     print(entropy_empirical)

#     # Total information content
#     total_info_content = entropy_empirical * total_vectors

#     # Estimate number of parameters
#     estimated_parameters = total_info_content / 10

#     return int(estimated_parameters)

def calculate_correlations(dataset):
    """
    Calculate Pearson, Spearman, and Kendall correlation coefficients for all 
    possible pairs of features in the dataset.

    Parameters:
    dataset (DataFrame): A pandas DataFrame with the multidimensional dataset.

    Returns:
    dict: A dictionary with tuple keys representing feature pairs and values 
          being another dictionary with the correlation coefficients.
    """
    # Initialize a dictionary to store the results
    correlations = {}

    # Generate all unique pairs of features
    features = dataset.columns
    pairs = itertools.combinations(features, 2)

    for pair in pairs:
        x, y = pair

        # Calculate correlation coefficients
        pearson_corr = dataset[x].corr(dataset[y], method='pearson')
        spearman_corr = dataset[x].corr(dataset[y], method='spearman')
        kendall_corr = dataset[x].corr(dataset[y], method='kendall')

        # Store in the dictionary
        correlations[pair] = {
            'Pearson': pearson_corr,
            'Spearman': spearman_corr,
            'Kendall': kendall_corr
        }

    return correlations

def compute_tail_distributions(time_series):
    """ Compute the upper and lower tail distribution functions for a time series """
    sorted_series = np.sort(time_series)
    n = len(sorted_series)
    lower_tail = np.array([np.mean(sorted_series <= x) for x in sorted_series])
    upper_tail = np.array([np.mean(sorted_series >= x) for x in sorted_series])
    return sorted_series, lower_tail, upper_tail

def plot_tail_distributions(generated_samples, train_data, currencies_names):
    """ Plot the upper and lower tail distribution functions for two time series """
    fig, ax = plt.subplots(2, 2, figsize=(11, 5), tight_layout=True)
    for i, title in zip(range(train_data.shape[1]), currencies_names):
        x1, lower_tail1, upper_tail1 = compute_tail_distributions(generated_samples[:,i])
        x2, lower_tail2, upper_tail2 = compute_tail_distributions(train_data[:,i])
        row = i // 2
        col = i % 2
        ax[row, col].plot(x1, lower_tail1, 'blue', label='Lower Tail Generated Samples', alpha=0.8)
        ax[row, col].plot(x2, lower_tail2, 'green', label='Lower Tail Training Data', alpha=0.8)
        ax[row, col].plot(x1, upper_tail1, 'blue', linestyle='--', label='Upper Tail Generated Samples', alpha=0.8)
        ax[row, col].plot(x2, upper_tail2, 'green', linestyle='--', label='Upper Tail Training Data', alpha=0.8)
        ax[row, col].set_title(f"{title}")
        ax[row, col].legend()
    c = 0
    while os.path.exists(f"output/tail_distributions_{c}.png"):
        c += 1
    plt.savefig(f"output/tail_distributions_{c}.png")

    return None

def plot_distributions(generated_samples, train_data, currencies_names):
    """Plot the distributions of the generated samples and the training data"""
    fig, ax = plt.subplots(2, 2, figsize=(11, 5), tight_layout=True)
    for i, title in zip(range(train_data.shape[1]), currencies_names):
        row = i // 2
        col = i % 2
        ax[row, col].hist(generated_samples[:,i], bins=50, label="RBM samples", alpha=0.8)
        ax[row, col].hist(train_data[:,i], bins=50, alpha=0.5, label="Original data")
        ax[row, col].set_xlabel("Value")
        ax[row, col].set_ylabel("Frequency")
        ax[row, col].set_title(f"{title}")
        ax[row, col].legend()
    c = 0
    while os.path.exists(f"output/distributions_{c}.png"):
        c += 1
    plt.savefig(f"output/distributions_{c}.png")
    return None

# def plot_qqplots(generated_samples, train_data, currencies_names):
#     fig, ax = plt.subplots(2, 2, figsize=(11, 5), tight_layout=True)
#     for i, title in zip(range(train_data.shape[1]), currencies_names):
#         row = i // 2
#         col = i % 2
#         # Calculate quantiles
#         quantiles1, quantiles2 = stats.probplot(generated_samples[:,i], dist="norm")[0][0], stats.probplot(train_data[:,i], dist="norm")[0][0]
#         (osm, osr), (slope, intercept, r) = stats.probplot(generated_samples[:,i], dist="norm", sparams=(train_data[:,i].mean(), train_data[:,i].std()))
#         # Create QQ plot
#         ax[row, col].scatter(quantiles1, quantiles2)
#         ax[row, col].plot(osm, slope * osm + intercept, color='r', alpha=0.5)
#         ax[row, col].set_title(f'{title}')
#         # ax[row, col].legend()

def qq_plots(generated_samples, train_data, currencies_names):
    """Plot the QQ plots for the generated samples and the training data"""
    fig, ax = plt.subplots(2, 2, figsize=(11, 5), tight_layout=True)
    for i, title in zip(range(train_data.shape[1]), currencies_names):
        gen_samples = np.sort(generated_samples[:, i])
        original_data = np.sort(train_data[:, i])
        gen_quantiles = np.percentile(gen_samples, q=np.linspace(0, 100, len(gen_samples)))
        train_quantiles = np.percentile(original_data, q=np.linspace(0, 100, len(original_data)))
        row = i // 2
        col = i % 2
        ax[row, col].scatter(gen_quantiles, train_quantiles, s=4, alpha=0.8)
        ax[row, col].plot([0, 1], [0, 1], transform=ax[row, col].transAxes, ls="--", c=".3")
        ax[row, col].set_xlabel("Generated samples quantiles")
        ax[row, col].set_ylabel("Training data quantiles")
        ax[row, col].set_title(f"{title}")

    c = 0
    while os.path.exists(f"output/QQ_plots_{c}.png"):
        c += 1
    plt.savefig(f"output/QQ_plots_{c}.png")


@njit
def dot_product(A, B):

    try:
        # Get the shape of the matrices
        rows_A, cols_A = A.shape
        rows_B, cols_B = B.shape
    except:
        raise ValueError("Input matrices must be 2D or input are not numpy arrays")

    # Check if the matrices can be multiplied
    if cols_A != rows_B:
        raise ValueError("Number of columns in A must be equal to number of rows in B")

    # Initialize the result matrix with zeros
    result = np.zeros((rows_A, cols_B))

    # Perform the matrix multiplication
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A): # or rows_B, they are the same
                result[i, j] += A[i, k] * B[k, j]

    return result

@njit
def boolean_to_int(boolean_array):
    # Initialize an empty array of the same shape as the boolean array
    int_array = np.empty(boolean_array.shape, dtype=np.int64) # or np.int32, depending on your needs
    for i in range(boolean_array.shape[0]):
        for j in range(boolean_array.shape[1]):
            int_array[i, j] = int(boolean_array[i, j])
    return int_array

@njit
def mean_axis_0(arr):
    # Calculate the mean along axis 0
    mean_arr = np.zeros(arr.shape[1])
    for j in range(arr.shape[1]):
        sum = 0.0
        for i in range(arr.shape[0]):
            sum += arr[i, j]
        mean_arr[j] = sum / arr.shape[0]
    return mean_arr

@njit
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

@njit
def free_energy(v, weights, visible_bias, hidden_bias):
    v_float = v.astype(np.float64)
    return np.mean(-np.sum(np.log(1 + np.exp(v_float @ weights + hidden_bias)), axis=1) - v_float @ visible_bias)

def visibile_bias_init(data_binary):
    frequencies = np.mean(data_binary, axis=0)
    visible_bias_t0 = np.log(frequencies / (1 - frequencies))
    return visible_bias_t0

def plot_pca_with_marginals(dataset_gen, dataset_real):
    # Perform PCA on both datasets
    pca1 = PCA(n_components=2)
    pca2 = PCA(n_components=2)
    principalComponents_real = pca1.fit_transform(dataset_real)
    principalComponents_gen = pca2.fit_transform(dataset_gen)

    # Combine the PCA components into a DataFrame
    df_real = pd.DataFrame(data=principalComponents_real, columns=['PC1', 'PC2'])
    df_real['Type'] = 'Real'
    
    df_gen = pd.DataFrame(data=principalComponents_gen, columns=['PC1', 'PC2'])
    df_gen['Type'] = 'Generated'
    
    # Concatenate the two DataFrames
    df = pd.concat([df_real, df_gen])

    # Use Seaborn's jointplot to plot the scatterplot with marginal histograms
    g = sns.jointplot(data=df, x='PC1', y='PC2', hue='Type', height=10, alpha=0.5) # set alpha to 0.5 for 50% transparency

    # Add a main title
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("First two PCA Components of real and fake dataset with marginal distributions")

    c = 0
    while os.path.exists(f"output/PCA_{c}.png"):
        c += 1
    plt.savefig(f"output/PCA_{c}.png")

    return None

if __name__ == '__main__':
    # Define the currency pairs
    currency_pairs = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCAD=X']

    # Check if the data is already downloaded
    try:
        data = pd.read_pickle('currencies_data.pkl')
    except FileNotFoundError:
        print("Downloading data...")
        data = data_download(currency_pairs, start_date="2022-01-01", end_date="2023-01-01")
        data.to_pickle('currencies_data.pkl')
        print('Done')
    # Example usage
    # Define your real valued dataset here as a 2D list
    real_data_example = pd.DataFrame(np.random.normal(size=(5,2)))  # Replace with actual data
    print(real_data_example.shape)
    print(real_data_example)

    # Convert to binary
    binary_data, bounds = from_real_to_binary(real_data_example)
    print(binary_data.shape)
    print(binary_data)

    # Convert back to real values
    reconstructed_real_data = from_binary_to_real(binary_data, bounds[0], bounds[1])

    print(reconstructed_real_data.shape)
    print(reconstructed_real_data)  # Displaying the binary and reconstructed data for checking
