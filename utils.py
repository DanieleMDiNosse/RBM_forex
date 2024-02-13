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

# from rbm import parallel_sample

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
        if file_name.endswith(('.png', '.jpg', '.jpeg', '.gif')) and file_name.startswith(id):
            file_path = os.path.join(folder_path, file_name)
            images.append(imageio.imread(file_path))
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
    '''Convert number to binary, remove the '0b' prefix, and pad with zeros to 16 bits'''
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

def mixed_dataset(n_samples):

    # Column 1: Mix of two normal distributions
    mean1, var1 = 0, 1  # Mean and variance for the first normal distribution
    mean2, var2 = 5, 2  # Mean and variance for the second normal distribution
    samples1 = np.random.normal(mean1, np.sqrt(var1), n_samples // 2)
    samples2 = np.random.normal(mean2, np.sqrt(var2), n_samples // 2)
    column1 = np.concatenate([samples1, samples2])

    # Column 2: Simple normal distribution
    location, scale = 0.5, 3  # Mean and variance
    column2 = cauchy.rvs(loc=location, scale=scale, size=n_samples)


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
    missing values rows. This is done to prevent the RBM to learn from a sequence non continuous data in time."""
    if np.isnan(data).any():
        last_nan_index = np.where(np.isnan(data))[0].max() + 1
        print(f"Missing values detected. The dataframe will start from rows {last_nan_index}")
        data = data[last_nan_index:]
        print(f"Done.\n")
    else:
        pass
    return data

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

    correlations = pd.DataFrame(correlations, index=['Pearson', 'Spearman', 'Kendall']).T
    return correlations

def plot_correlation_heatmap(correlations_real, correlations_gen, id):
    """
    Plot a heatmap of the correlation coefficients for both the real and generated ones.

    Parameters:
    correlations_real (DataFrame): A pandas DataFrame with the correlation coefficients for the real data.
    correlations_gen (DataFrame): A pandas DataFrame with the correlation coefficients for the generated data.
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
    fig, ax = plt.subplots(2, 2, figsize=(10, 5), tight_layout=True)
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

def plot_tail_concentration_functions(real_data, generated_data, names, id):
    """Calculate the tail concordance of two time series"""
    if not os.path.exists("output/tail_concentration"):
        os.makedirs("output/tail_concentration")
    gen_df = pd.DataFrame(generated_data, columns=names)
    real_df = pd.DataFrame(real_data, columns=names)
    pairs = itertools.combinations(names, 2)
    for col1, col2 in pairs:
        plt.figure(figsize=(11, 5), tight_layout=True)
        plt.plot(tail_conc(real_df[col1], real_df[col2]), label='Real data')
        plt.plot(tail_conc(gen_df[col1], gen_df[col2]), label='Generated data')
        plt.title(f'{col1}/{col2}')
        plt.legend()
        plt.savefig(f"output/tail_concentration/{col1}_{col2}_{id}.png")

def qq_plots(generated_samples, train_data, currencies_names, id):
    """Plot the QQ plots for the generated samples and the training data"""
    n_features = generated_samples.shape[1]
    
    # Determine the layout based on the number of features
    if n_features > 1:
        fig, axs = plt.subplots(2, 2, figsize=(11, 5), tight_layout=True)
        axs = axs.flatten() # Flatten the axs array for easier indexing
    else:
        fig, axs = plt.subplots(1, 1, figsize=(11, 5), tight_layout=True)
        axs = [axs] # Wrap the single axs object in a list for consistent access

    for i, title in zip(range(train_data.shape[1]), currencies_names):
        gen_quantiles = np.quantile(generated_samples[:, i], q=np.arange(0, 1, 0.01))
        train_quantiles = np.quantile(train_data[:, i], q=np.arange(0, 1, 0.01))
        ax = axs[i]
        ax.scatter(gen_quantiles, train_quantiles, s=4, alpha=0.8)
        ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
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
    plt.savefig(f"output/historgrams/{id}_histograms_{epoch}.png")
    fig.suptitle(f"Epoch {epoch}")
    plt.close()


    return None

def plot_autocorr_wrt_K(num_visible, weights, hidden_bias, visible_bias, k_max, n_samples, X_min, X_max):
    from rbm import parallel_sample, sample_from_state
    np.random.seed(666)
    average_autocorrs = []
    N = 100
    for k in range(1, k_max+1, 100):
        print(f"K = {k}/{k_max}")
        autocorr = 0
        for i in range(N):
            if i % 10 == 0 and i != 0: print(f"\t Genereting sample #{i}")

            # Create the state at time 0
            samples_0 = parallel_sample(num_visible, weights, hidden_bias, visible_bias, k, n_samples)
            # Use the state at time zero to initialise a new gibbs sampling
            samples_1 = sample_from_state(samples_0, weights, hidden_bias, visible_bias, k)
            
            # Convert bakc to real numbers
            samples_0 = from_binary_to_real(samples_0, X_min, X_max).values[:, 0]
            samples_1 = from_binary_to_real(samples_1, X_min, X_max).values[:, 0]

            autocorr += np.correlate(samples_0, samples_1)

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

