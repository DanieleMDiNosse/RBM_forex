import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit
import itertools
from sklearn.decomposition import PCA
import seaborn as sns
import os
import imageio
from natsort import natsorted
import requests
from scipy.stats import weibull_min, beta
from scipy.stats import cauchy

def print_(text):
    # Open the file in append mode, or 'w' for write mode (which overwrites the file)
    with open(f'logs/log_{os.getpid()}.txt', 'a') as file:
        # Print to terminal
        print(text)
        # Write the same text to the file
        file.write(text + '\n')  # Adding '\n' to move to the next line for each call

class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    RESET = '\033[0m'  # Reset the color to the default

def create_animated_gif(folder_path, id, output_filename='animated.gif'):
    """
    Creates an animated GIF from all the images in the specified folder.

    Args:
    - folder_path: The path to the folder containing images.
    - output_filename: The name of the output GIF file.

    Returns:
    None. An animated GIF is saved at the folder path.
    """
    images = []
    for file_name in natsorted(os.listdir(folder_path)):
        if file_name.endswith(('.png', '.jpg', '.jpeg')) and file_name.startswith(id):
            file_path = os.path.join(folder_path, file_name)
            images.append(imageio.imread(file_path))
            os.remove(os.path.join(folder_path, file_name))
    imageio.mimsave(os.path.join(folder_path, output_filename), images)

    return None

def data_download(currency_pairs, start_date, end_date, type='Close', provider='alpha_vantage'):
    if provider == 'yfinance':
        data = yf.download(currency_pairs, start=start_date, end=end_date)
    if provider == 'alpha_vantage':
        time_series_data = []
        api_key = 'Z2JPMSHJDV1V73TY'
        for fiat1, fiat2 in currency_pairs:
            print(fiat1, fiat2)
            url = f'https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={fiat1}&to_symbol={fiat2}&apikey={api_key}'
            r = requests.get(url)
            data = r.json()

            # Extracting the time series data
            time_series_data.append(data['Time Series FX (Daily)'])

        print(time_series_data)

    return data[type]

def binarize(number):
    '''Convert number to binary, remove the '0b' prefix, and pad with zeros to 16 bits.
    
    Parameters
    -----------
    number: int
        The number to convert to binary.
    
    Returns
    --------
    binary_representation: str
        The binary representation of the number, padded with zeros to 16 bits.'''
    binary_representation = bin(number)[2:].zfill(16)
    return binary_representation

def from_real_to_binary(data):
    '''Conversion of real-valued data set into binary features. Every real valued number is 
    converted into a 16-bit binary number.
    
    Parameters
    -----------
    data: DataFrame
        The real-valued data set of shape (n_samples, n_features).
    
    Returns
    --------
    data_binary: DataFrame
        The binary data set fo shape (n_samples, 16 * n_features).'''
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

    Parameters
    -----------
    X_binary: DataFrame
        The binary data set of shape (n_samples, 16 * n_features).
    X_min: float or array-like
        The minimum value for each feature (output of from_real_to_binary).
    X_max: float or array-like
        The maximum value for each feature (output of from_real_to_binary).
    
    Returns
    --------
    X_real: DataFrame
        The real-valued data set of shape (n_samples, n_features).
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

def mixed_dataset(n_samples):
    '''Create a mixed dataset with 4 columns. The first column is a mix of two normal distributions, the second column
    is a Cauchy distribution, the third column is a Weibull distribution, and the fourth column is a Beta distribution.
    
    Parameters
    -----------
    n_samples: int
        The number of samples to generate.
    
    Returns
    --------
    df: DataFrame
        The mixed dataset with 4 columns.'''
    # Column 1: Mix of two normal distributions
    mean1, var1 = 0, 1  # Mean and variance for the first normal distribution
    mean2, var2 = 5, 2  # Mean and variance for the second normal distribution
    samples1 = np.random.normal(mean1, np.sqrt(var1), n_samples // 2)
    samples2 = np.random.normal(mean2, np.sqrt(var2), n_samples // 2)
    column1 = np.concatenate([samples1, samples2])

    # Column 2: Uniform distribution mixed with a normal distribution
    column2 = np.concatenate([np.random.normal(0, 1, n_samples // 2), np.random.uniform(-10, 10, n_samples // 2)])

    # Column 3: Weibull distribution
    shape, scale = 1, 1.5
    column3 = weibull_min.rvs(shape, scale=scale, size=n_samples)

    # Column 4: Beta distribution
    alpha, beta_param = 2, 5
    column4 = beta.rvs(alpha, beta_param, size=n_samples)

    # Creating the DataFrame
    df = pd.DataFrame({
        "Mixed_Normal": column1,
        "Cauchy": column2,
        "Weibull": column3,
        "Beta": column4
    })

    # Shuffling all the rows
    df = df.sample(frac=1).reset_index(drop=True)

    return df.values

def remove_missing_values(data):
    """Check for missing values. If there is a missing value, the dataframe is cut from the start to the last
    missing values rows. This is done to prevent the RBM to learn from a sequence non continuous data in time.
    
    Parameters
    -----------
    data: DataFrame
        The dataset to check for missing values. 
    
    Returns
    --------
    data: DataFrame
        The dataset without missing values."""
    if np.isnan(data).any():
        last_nan_index = np.where(np.isnan(data))[0].max() + 1
        print(f"Missing values detected. The dataframe will start from rows {last_nan_index}")
        data = data[last_nan_index:]
    else:
        pass
    return data

def calculate_correlations(dataset):
    """
    Calculate Pearson, Spearman, and Kendall correlation coefficients for all 
    possible pairs of features in the dataset.

    Parameters
    -----------
    dataset: DataFrame
        The dataset to calculate the correlations for.

    Returns
    --------
    correlations: DataFrame
        The correlation coefficients for all possible pairs of features.
    """
    # Initialize a dictionary to store the results
    correlations = {}

    # Generate all unique pairs of features
    features = dataset.columns
    pairs = list(itertools.combinations(features, 2))

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

    correlations = pd.DataFrame(correlations, index=['Pearson', 'Spearman', 'Kendall']).T
    return correlations

def plot_correlation_heatmap(correlations_real, correlations_gen, id):
    """
    Plot a heatmap of the correlation coefficients for both the real and generated ones.

    Parameters
    -----------
    correlations_real: DataFrame
        The correlation coefficients for the real data.
    correlations_gen: DataFrame
        The correlation coefficients for the generated data.
    
    Returns
    --------
    None. The heatmap is saved as a .png file.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    sns.heatmap(correlations_real, annot=True, cmap='coolwarm', ax=ax[0])
    ax[0].set_title('Real data')
    sns.heatmap(correlations_gen, annot=True, cmap='coolwarm', ax=ax[1])
    ax[1].set_title('Generated data')

    # Check if the output folder exists
    if not os.path.exists("output/correlation_heatmaps"):
        os.makedirs("output/correlation_heatmaps")

    plt.savefig(f"output/correlation_heatmaps/{id}_correlation_heatmap.png")
    plt.close()

def plot_objectives(reconstruction_error, f_energy_overfitting, f_energy_diff, diff_fenergy, id):
    """Plot the objectives of the training process"""
    _, ax = plt.subplots(2, 2, figsize=(10, 5), tight_layout=True)
    ax[0, 0].plot(reconstruction_error)
    ax[0, 0].set_xlabel("Epoch")
    ax[0, 0].set_ylabel("Reconstruction error")
    ax[0, 0].set_title("Reconstruction error")
    ax[0, 1].plot(np.array(f_energy_overfitting)[:,0], 'green', label="Training data", alpha=0.7)
    ax[0, 1].plot(np.array(f_energy_overfitting)[:,1], 'blue', label="Validation data", alpha=0.7)
    ax[0, 1].legend()
    ax[0, 1].set_xlabel("Epoch")
    ax[0, 1].set_ylabel("Free energy")
    ax[0, 1].set_title("Free energy")
    ax[1, 0].plot(diff_fenergy)
    ax[1, 0].set_xlabel("Epoch")
    ax[1, 0].set_ylabel("|F(data) - F(val)|")
    ax[1, 0].set_title("|F(data) - F(val)|")
    ax[1, 1].plot(f_energy_diff)
    ax[1, 1].set_xlabel("Epoch")
    ax[1, 1].set_ylabel("F(v0) - F(vk)")
    ax[1, 1].set_title("F(v0) - F(vk)")
    # Check if the output folder exists
    if not os.path.exists("output/objectives"):
        os.makedirs("output/objectives")
    plt.savefig(f"output/objectives/{id}_objectives.png")
    plt.close()

def compute_tail_distributions(time_series):
    """ Compute the upper and lower tail distribution functions for a time series """
    sorted_series = np.sort(time_series)
    n = len(sorted_series)
    lower_tail = np.array([np.mean(sorted_series <= x) for x in sorted_series])
    upper_tail = np.array([np.mean(sorted_series >= x) for x in sorted_series])
    return sorted_series, lower_tail, upper_tail

def plot_tail_distributions(generated_samples, train_data, currencies_names, id):
    """ Plot the upper and lower tail distribution functions for two time series """
    n_features = generated_samples.shape[1]
    
    # Determine the layout based on the number of features
    if n_features > 1:
        fig, axs = plt.subplots(2, 2, figsize=(11, 5), tight_layout=True)
        axs = axs.flatten() # Flatten the axs array for easier indexing
    else:
        fig, axs = plt.subplots(1, 1, figsize=(11, 5), tight_layout=True)
        axs = [axs] # Wrap the single axs object in a list for consistent access

    for i, title in zip(range(train_data.shape[1]), currencies_names):
        x1, lower_tail1, upper_tail1 = compute_tail_distributions(generated_samples[:,i])
        x2, lower_tail2, upper_tail2 = compute_tail_distributions(train_data[:,i])
        ax = axs[i]
        ax.plot(x1, lower_tail1, 'blue', label='Lower Tail Generated Samples', alpha=0.8)
        ax.plot(x2, lower_tail2, 'green', label='Lower Tail Training Data', alpha=0.8)
        ax.plot(x1, upper_tail1, 'blue', linestyle='--', label='Upper Tail Generated Samples', alpha=0.8)
        ax.plot(x2, upper_tail2, 'green', linestyle='--', label='Upper Tail Training Data', alpha=0.8)
        ax.set_title(f"{title}")
        ax.legend()

    # Check if the output folder exists
    if not os.path.exists("output/tail_distributions"):
        os.makedirs("output/tail_distributions")

    plt.savefig(f"output/tail_distributions/{id}_tail_distributions.png")
    plt.close()

    return None

def plot_distributions(generated_samples, train_data, currencies_names, id):
    """Plot the distributions of the generated samples and the training data"""
    n_features = generated_samples.shape[1]
    
    # Determine the layout based on the number of features
    if n_features > 1:
        fig, axs = plt.subplots(2, 2, figsize=(11, 5), tight_layout=True)
        axs = axs.flatten() # Flatten the axs array for easier indexing
    else:
        fig, axs = plt.subplots(1, 1, figsize=(11, 5), tight_layout=True)
        axs = [axs] # Wrap the single axs object in a list for consistent access
    
    for i, title in zip(range(train_data.shape[1]), currencies_names):
        ax = axs[i]
        ax.hist(generated_samples[:,i], bins=50, label="RBM samples", alpha=0.8)
        ax.hist(train_data[:,i], bins=50, alpha=0.5, label="Original data")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{title}")
        ax.legend()

    # Check if the output folder exists
    if not os.path.exists("output/distributions"):
        os.makedirs("output/distributions")

    plt.savefig(f"output/distributions/{id}_distributions.png")
    plt.close()

def tail_conc(val1, val2):
    """
    Calculate the tail concentration of two arrays.
    This function calculates the tail concentration of two arrays, `val1` and `val2`.
    It computes the fraction of values in `val1` and `val2` that fall below or above
    certain quantiles, and returns a list of these fractions. In other words, tail 
    concentration is a measure of the degree to which extreme values in one variable
    are associated with extreme values in another variable.

    Parameters:
    ----------
    val1: list or array-like
        The first array of values.
    val2: list or array-like
        The second array of values.
    
    Returns:
    -------
    f: list
        The list of tail concentration values.

    Example:
    >>> val1 = [1, 2, 3, 4, 5]
    >>> val2 = [6, 7, 8, 9, 10]
    >>> tail_conc(val1, val2)
    [0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.2]
    """
    f = []
    n = len(val1)
    linspace = np.arange(0.01, 0.5, 0.01)
    quants1 = np.quantile(val1, linspace)
    quants2 = np.quantile(val2, linspace)
    for j in range(len(linspace)):
        tot = 0
        for i in range(len(val1)):
            if val1[i] <= quants1[j] and val2[i] <= quants2[j]:
                tot += 1
        f.append(tot/(n*linspace[j]))
    linspace = np.arange(0.5, 0.99, 0.01)
    quants1 = np.quantile(val1, linspace)
    quants2 = np.quantile(val2, linspace)
    for j in range(len(linspace)):
        tot = 0
        for i in range(len(val1)):
            if val1[i] >= quants1[j] and val2[i] >= quants2[j]:
                tot += 1
        f.append(tot/(n*(1-linspace[j])))
    return f


def plot_tail_concentration_functions(real_data, generated_samples, names, id):
    if not os.path.exists("output/tail_concentration"):
        os.makedirs("output/tail_concentration")
    dim = np.array(generated_samples).ndim
    
    real_df = pd.DataFrame(real_data, columns=names)
    pairs = itertools.combinations(names, 2)
    
    for col1, col2 in pairs:
        plt.figure(figsize=(11, 5), tight_layout=True)
        # Plot for real data
        plt.plot(tail_conc(real_df[col1], real_df[col2]), label='Real data', color='blue', alpha=0.8)
        
        if dim > 2:
            # Calculate and plot for generated data
            tail_concs = []
            for sample in generated_samples:
                gen_df = pd.DataFrame(sample, columns=names)
                tail_concs.append(tail_conc(gen_df[col1], gen_df[col2]))
            
            # Calculate mean and std of tail_conc across all generated samples
            tail_concs = np.array(tail_concs)
            mean_tail_conc = np.mean(tail_concs, axis=0)
            std_tail_conc = np.std(tail_concs, axis=0)
            alpha = 0.3
        else:
            gen_df = pd.DataFrame(generated_samples, columns=names)
            mean_tail_conc = tail_conc(gen_df[col1], gen_df[col2])
            std_tail_conc, alpha = np.zeros_like(mean_tail_conc), 0
        
        # Plotting
        plt.fill_between(range(len(mean_tail_conc)), mean_tail_conc - std_tail_conc, mean_tail_conc + std_tail_conc, color='red', alpha=alpha)
        plt.plot(mean_tail_conc, label='Generated data mean', color='red', alpha=0.8)
        
        plt.title(f'{col1}/{col2}')
        plt.legend()
        plt.savefig(f"output/tail_concentration/{col1}_{col2}_{id}.png")
        plt.close()

def qq_plots(generated_samples, train_data, currencies_names, id):
    """Plot the QQ plots for the generated samples and the training data"""
    n_features = train_data.shape[1]
    dim = np.array(generated_samples).ndim

    # Determine the layout based on the number of features
    if n_features > 1:
        _, axs = plt.subplots(2, 2, figsize=(11, 5), tight_layout=True)
        axs = axs.flatten() # Flatten the axs array for easier indexing
    else:
        _, axs = plt.subplots(1, 1, figsize=(11, 5), tight_layout=True)
        axs = [axs] # Wrap the single axs object in a list for consistent access

    if dim > 2:
        gen_quantiles_means = np.zeros((n_features, 100))
        gen_quantiles_stds = np.zeros((n_features, 100))
        gen_quantiles = np.zeros((len(generated_samples), n_features, 100))
        for i in range(n_features):
            for j, sample in enumerate(generated_samples):
                gen_quantiles[j, i] = np.quantile(sample[:, i], q=np.arange(0, 1, 0.01))
            gen_quantiles_means[i] = np.mean(gen_quantiles[:, i, :], axis=0)
            gen_quantiles_stds[i] = np.std(gen_quantiles[:, i, :], axis=0)
    else:
        gen_quantiles = np.zeros((n_features, 100))
        for i in range(n_features):
            gen_quantiles[i] = np.quantile(generated_samples[:, i], q=np.arange(0, 1, 0.01))

    for i, title in zip(range(n_features), currencies_names):
        train_quantiles = (np.quantile(train_data[:, i], q=np.arange(0, 1, 0.01)))
        ax = axs[i]
        if dim > 2: 
            ax.errorbar(gen_quantiles_means[i], train_quantiles, xerr=None, yerr=2*gen_quantiles_stds[i], fmt='o', alpha=0.8, markersize=3.5)
        else:
            ax.plot(gen_quantiles[i], train_quantiles, 'o', alpha=0.8, markersize=3.5)
        ax.plot([gen_quantiles_means[i][0], gen_quantiles_means[i][-1]], [gen_quantiles_means[i][0], gen_quantiles_means[i][-1]], ls="--", c=".3")
        ax.set_xlabel("Generated samples quantiles")
        ax.set_ylabel("Training data quantiles")
        ax.set_title(f"{title}")

    # Check if the output folder exists
    if not os.path.exists("output/qq_plots"):
        os.makedirs("output/qq_plots")

    plt.savefig(f"output/qq_plots/{id}_qq_plots.png")
    plt.close()


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

# @njit
def cross_entropy_error(v0, vk):
    """
    Calculate the cross-entropy error between the original and reconstructed data.

    Parameters:
    - v0: NumPy array of original data.
    - vk: NumPy array of reconstructed data after k-steps of Gibbs sampling.

    Returns:
    - Cross-entropy error as a float.
    """
    # Avoid division by zero and log(0) by adding a small value to vk and 1-vk
    epsilon = 1e-10
    vk = np.clip(vk, epsilon, 1 - epsilon)

    # Calculate cross-entropy error
    cross_entropy = -np.mean(np.sum(v0 * np.log(vk) + (1 - v0) * np.log(1 - vk), axis=1))
    return cross_entropy

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

def annualized_volatility(log_returns, period=252):
    return np.std(log_returns, axis=0) * np.sqrt(period)

@njit
def energy(v, h, weights, visible_bias, hidden_bias):
    return -np.sum(v * visible_bias) - np.sum(h * hidden_bias) - np.sum(v @ weights * h)

@njit
def free_energy(v, weights, visible_bias, hidden_bias):
    '''Average free energy over a batch of data represented by v. It is valid only if sites can be either 0 or 1. The general expression is:
    F(v) = <-log(sum_h exp (-E*(v,h)))>'''
    v_float = v.astype(np.float64)
    free_energy = np.mean(-np.sum(np.log(1 + np.exp(v_float @ weights + hidden_bias)), axis=1) - v_float @ visible_bias)
    return free_energy

def plot_pca_with_marginals(dataset_gen, dataset_real, id='C'):
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

    # Check if the output folder exists
    if not os.path.exists("output/PCA"):
        os.makedirs("output/PCA")

    plt.savefig(f"output/PCA/{id}_PCA.png")
    plt.close()

    return None

def monitoring_plots(weights, hidden_bias, visible_bias, deltas, pos_hidden_prob, epoch, id='C'):
    delta_w, delta_hidden_bias, delta_visible_bias = deltas

    # Check if the output folder exists
    if not os.path.exists("output/weights_receptive_field"):
        os.makedirs("output/weights_receptive_field")
    if not os.path.exists("output/historgrams"):
        os.makedirs("output/historgrams")

    # Display a grayscale image of the weights
    '''For domains in which the visible units have spatial or temporal structure (e.g. images or speech)
    it is very helpful to display, for each hidden unit, the weights connecting that hidden unit to the
    visible units. These “receptive” fields are a good way of visualizing what features the hidden units
    have learned'''
    fig, axes = plt.subplots(1, 2, figsize=(12, 12), tight_layout=True)
    img0 = axes[0].imshow(weights, cmap="gray")
    fig.colorbar(img0, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].set_xlabel("Hidden units")
    axes[0].set_ylabel("Visible units")
    axes[0].set_title(f"Weights at epoch {epoch}")
    # Display a grayscale image of the hidden probabilities
    '''This immediately allows you to see if some hidden units are never used or if some training cases
    activate an unusually large or small number of hidden units. It also shows how certain the hidden
    units are. When learning is working properly, this display should look thoroughly random without
    any obvious vertical or horizontal lines'''
    img1 = axes[1].imshow(pos_hidden_prob, cmap="gray")
    fig.colorbar(img1, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_ylabel("Training vectors")
    axes[1].set_xlabel("Hidden units")
    axes[1].set_title(f"Hidden units probabilities epoch {epoch}")
    plt.savefig(f"output/weights_receptive_field/{id}_weights_receptivefield_{epoch}.png")
    plt.close()

    # Display the histograms of the weights, hidden and visible biases
    '''A good rule of thumb for setting the learning rate (Max Welling, personal communication, 2002) is
    to look at a histogram of the weight updates and a histogram of the weights. The updates should
    be about 10e-3 times the weights (to within about an order of magnitude)'''
    x_w, x_h, x_v = np.arange(weights.ravel().size), np.arange(hidden_bias.size), np.arange(visible_bias.size)
    fig, axes = plt.subplots(2, 3, figsize=(15, 5), tight_layout=True)
    axes[1,0].bar(x_w, delta_w.ravel()/weights.ravel(), color='k', alpha=0.5)
    axes[1,0].set_yscale('log')
    axes[1,0].set_title(r"$\frac{\Delta W}{W}$")
    axes[0,0].bar(x_w, weights.ravel(), color='k', alpha=0.8)
    # ax = axes[0,0].twiny()
    # ax.hist(weights.ravel(), bins=40, density=True, color='purple', alpha=0.3)
    axes[0,0].set_title("Weights")
    axes[0,1].bar(x_h, hidden_bias, color='k', alpha=0.8)
    axes[0,1].set_title("Hidden bias")
    axes[1,1].bar(x_h, delta_hidden_bias, color='k', alpha=0.5)
    axes[1,1].set_title(r"$\Delta h$")
    
    axes[0,2].bar(x_v, visible_bias, color='k', alpha=0.8)
    axes[0,2].set_title("Visible bias")
    axes[1,2].bar(x_v, delta_visible_bias, color='k', alpha=0.5)
    axes[1,2].set_title(r"$\Delta v$")
    fig.suptitle(f"Epoch {epoch}")
    plt.savefig(f"output/historgrams/{id}_histograms_{epoch}.png")
    plt.close()


    return None

def plot_autocorr_wrt_K(weights, hidden_bias, visible_bias, k_max, n_samples, X_min, X_max, indexes_vol_indicators, vol_indicators):
    from rbm import parallel_sample, sample_from_state
    average_autocorrs = []
    # Set the number of autocorrelations to compute for each k
    N = 1000
    for k in range(1, k_max+100, 100):
        print_(f"K = {k}/{k_max}")
        autocorr = 0
        for i in range(N):
            if i % 100 == 0 and i != 0: print(f"\t Genereting sample #{i}")

            # Create the state at time 0
            samples_0 = parallel_sample(weights, hidden_bias, visible_bias, k, indexes_vol_indicators, vol_indicators, n_samples)
            samples_00 = np.delete(samples_0, indexes_vol_indicators, axis=1)
            # Use the state at time zero to initialise a new gibbs sampling
            samples_1 = sample_from_state(samples_0, weights, hidden_bias, visible_bias, indexes_vol_indicators, vol_indicators, k)
            samples_11 = np.delete(samples_1, indexes_vol_indicators, axis=1)
            
            # Convert back to real numbers
            samples_00 = from_binary_to_real(samples_00, X_min, X_max).values[:, 0]
            samples_11 = from_binary_to_real(samples_11, X_min, X_max).values[:, 0]

            autocorr += np.correlate(samples_00, samples_11)

        # Evaluate the mean correlation
        average_autocorrs.append(np.mean(autocorr/N))
    
    np.save("output/average_autocorrs.npy", average_autocorrs)
    plt.figure(figsize=(10,5), tight_layout=True)
    plt.plot(average_autocorrs, '-o', alpha=0.7)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("K")
    plt.ylabel("Average 1-day autocorrelation")
    plt.savefig("output/averge_corr.png")
    plt.close()

    return average_autocorrs


def calculate_historical_volatility(df, window=90):
    """
    Calculates the historical volatility over a specified window.
    df: DataFrame with columns representing different currency pairs and rows as daily returns.
    window: Integer representing the rolling window size for volatility calculation.
    """
    # Check if df is a dataframe. If not, convert it to a dataframe
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    return df.rolling(window=window).std() * np.sqrt(252)  # Annualizing volatility

def compute_binary_volatility_indicator(df, window=90):
    """
    Assigns a binary indicator for volatility above or below the median for each currency pair.
    df: DataFrame as described previously.
    window: Rolling window size for volatility calculation.
    """
    # Convert to dataframe if not already
    df = pd.DataFrame(df)

    # Calculate historical volatility
    historical_volatility = calculate_historical_volatility(df, window)
    
    # Calculate long-term median volatility for each currency pair
    median_volatility = historical_volatility.median()
    
    # Initialize a DataFrame to store binary indicators
    vol_indicators = pd.DataFrame(index=df.index, columns=df.columns)
    
    # Assign binary indicators based on condition
    for currency_pair in df.columns:
        vol_indicators[currency_pair] = (historical_volatility[currency_pair] > median_volatility[currency_pair]).astype(int)
        
    return vol_indicators.to_numpy(), median_volatility

def add_vol_indicators(data_binary, vol_indicators):
    '''Add the binary volatility indicator to the binary data set. The binary indicator is calculated using the historical
    volatility of the data. The binary indicator is added after each currency pair in the data set. The binary indicator
    is calculated using the median volatility as the threshold. If the volatility is greater than the median, the binary
    indicator is set to 1, otherwise it is set to 0. The window parameter is used to calculate the historical volatility.
    
    Parameters:
    - data_binary: DataFrame with the binary data set.
    - vol_indicators: DataFrame with the binary volatility indicators.
    
    Returns:
    - DataFrame with the binary volatility indicator added to the data set.'''

    data_binary = pd.DataFrame(data_binary)
    indexes = list(range(16, data_binary.shape[1]+1, 16))

    # Initialize a counter to keep track of the number of columns added
    i = 0
    for col_bin, col_vol in zip(indexes, range(vol_indicators.shape[1])):
        # Inserting the vol_indicators column at col_bin location
        data_binary.insert(loc=col_bin+i, column=f'binary_{col_vol}', value=vol_indicators[:, col_vol])
        i += 1
    indexes = [int(np.where(data_binary.columns == f'binary_{i}')[0]) for i in range(vol_indicators.shape[1])]

    return data_binary.values, np.array(indexes)

def mean_std_statistics(samples_list, train_data, currencies):
    # Evaluate the volatilities in the 'high' and 'low' volatility regimes. The results correspond
    # to Table 4 in the paper.
    # high_real_ret = train_data[np.where(vol_indicators[:train_data.shape[0]] == 1)]
    # low_real_ret = train_data[np.where(vol_indicators[:train_data.shape[0]] == 0)]
    # high_real_vol = calculate_historical_volatility(high_real_ret, window=high_real_ret.shape[0])
    # low_real_vol = calculate_historical_volatility(low_real_ret, window=low_real_ret.shape[0])

    # high_gen_ret = samples[np.where(vol_indicators[:train_data.shape[0]] == 1)]
    # low_gen_ret = samples[np.where(vol_indicators[:train_data.shape[0]] == 0)]
    # high_gen_vol = calculate_historical_volatility(high_gen_ret, window=high_gen_ret.shape[0])
    # low_gen_vol = calculate_historical_volatility(low_gen_ret, window=low_gen_ret.shape[0])
    # Remove the binary volatility indicators from the data set
    # Generate all unique pairs of features
    '''In Python, itertools.combinations returns an iterator that generates combinations of the input iterable. 
    Once you iterate over all the elements of an iterator, it becomes exhausted. This means if you try to iterate
    over it again, it won't yield any elements.
    To solve this, you can convert the iterator to a list, which will store all the combinations in memory.'''
    pairs = list(itertools.combinations(currencies, 2))

    gen_corr_list = []
    for idx, samples in enumerate(samples_list):
        gen_corr = calculate_correlations(pd.DataFrame(samples))
        gen_corr_list.append(gen_corr)
    
    # Collect all the standard deviations for all the pairs
    pearson_std = [np.std([gen_corr['Pearson'].iloc[i] for gen_corr in gen_corr_list]) for i in range(6)]
    spearman_std = [np.std([gen_corr['Spearman'].iloc[i] for gen_corr in gen_corr_list]) for i in range(6)]
    kendall_std = [np.std([gen_corr['Kendall'].iloc[i] for gen_corr in gen_corr_list]) for i in range(6)]

    # Collect all the means for all the pairs
    pearson_mean = [np.mean([gen_corr['Pearson'].iloc[i] for gen_corr in gen_corr_list]) for i in range(6)]
    spearman_mean = [np.mean([gen_corr['Spearman'].iloc[i] for gen_corr in gen_corr_list]) for i in range(6)]
    kendall_mean = [np.mean([gen_corr['Kendall'].iloc[i] for gen_corr in gen_corr_list]) for i in range(6)]

    # Create a dictionary that contains keys and values like pairs[0]: [mean, std]
    pearson_dict = {pair: [pearson_mean[i], pearson_std[i]] for i, pair in enumerate(pairs)}
    spearman_dict = {pair: [spearman_mean[i], spearman_std[i]] for i, pair in enumerate(pairs)}
    kendall_dict = {pair: [kendall_mean[i], kendall_std[i]] for i, pair in enumerate(pairs)}

    # Convert to pandas DataFrame
    pearson_df = pd.DataFrame(pearson_dict, index=['Mean', 'Std']).T
    spearman_df = pd.DataFrame(spearman_dict, index=['Mean', 'Std']).T
    kendall_df = pd.DataFrame(kendall_dict, index=['Mean', 'Std']).T

    first_gen_percentile = {}
    last_gen_percentile = {}
    first_real_percentile = {}
    last_real_percentile = {}
    for i in range(len(currencies)):
        gen_percentiles = [np.quantile(samples[:, i], q=np.arange(0, 1, 0.01)) for samples in samples_list]
        first_percentiles = [perc[0] for perc in gen_percentiles]
        last_percentiles = [perc[-1] for perc in gen_percentiles]
        first_gen_percentile[currencies[i]] = [np.mean(first_percentiles), np.std(first_percentiles)]
        last_gen_percentile[currencies[i]] = [np.mean(last_percentiles), np.std(last_percentiles)]
        real_percentiles = np.quantile(train_data[:, i], q=np.arange(0, 1, 0.01))
        first_real_percentile[currencies[i]] = real_percentiles[0]
        last_real_percentile[currencies[i]] = real_percentiles[-1]

    first_last_gen = pd.DataFrame({'First percentile': first_gen_percentile, 'Last percentile': last_gen_percentile})
    first_last_real = pd.DataFrame({'First percentile': first_real_percentile, 'Last percentile': last_real_percentile})

    # vol_indicators = 

    return pearson_df, spearman_df, kendall_df, first_last_gen, first_last_real
