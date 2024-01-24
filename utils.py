import yfinance as yf
import numpy as np
import pandas as pd
from numba import njit, prange

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

@njit
def dot_product(A, B):
    rows_A, cols_A = A.shape
    rows_B, cols_B = B.shape

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
    return -np.sum(np.log(1 + np.exp(v_float @ weights + hidden_bias)), axis=1) - v_float @ visible_bias

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
