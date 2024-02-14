import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
# plt.style.use('seaborn')
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
parser.add_argument("--train_rbm", "-t", action="store_true", help="Train the RBM. Default: False")
parser.add_argument("--hidden_units", "-hu", type=int, help="Train the RBM. Default: 30")
parser.add_argument("--epochs", "-e", type=int, default=1500, help="Number of epochs. Default: 1500")
parser.add_argument("--learning_rate", "-lr", type=float, default=0.01, help="Learning rate. Default: 0.01")
parser.add_argument("--batch_size", "-b", type=int, default=10, help="Batch size for training. Default: 10")
parser.add_argument("--continue_train", "-c", action="store_true", help="Load the weights and continue training for the specified number of epochs. Default: False")
parser.add_argument("--k_step", "-k", type=int, default=1, help="Number of Gibbs sampling steps in the training process. Default: 1")
parser.add_argument("-std", "--std", action="store_true", help="Compute the means and standard deviations of the correlations and historical volatilties. Default: False")
levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}
args = parser.parse_args()
if not os.path.exists("logs"):
    os.makedirs("logs")
# logging.basicConfig(filename=f'logs/main_real_{id}.log', format='%(message)s', level=levels[args.log])


start = time.time()
currencies = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD']
np.random.seed(666)

start_date = "1999-01-01"
end_date = "2024-01-01"
# Check if the data is already downloaded
try:
    print("Loading data...")
    data = pd.read_pickle(f'data/currencies_data_{start_date}_{end_date}.pkl')
    data = data.values
except FileNotFoundError:
    print("Downloading data...")
    data = data_download(['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCAD=X'],
                         start_date=start_date, end_date=end_date, provider='yfinance')
    data.to_pickle(f'data/currencies_data_{start_date}_{end_date}.pkl')
    data = data.values

id = f'C{os.getpid()}_{args.learning_rate}_{args.k_step}'
print(f"{id} - TRAINING RBM ON CURRENCY DATA\n")

print(f"Pre processing data...")
# Apply log transformation
data = np.log(data)
# Compute returns
data = np.diff(data, axis=0)
# Remove missing values
data = remove_missing_values(data)
# Compute binary indicators
vol_indicators, median = compute_binary_volatility_indicator(data)
# Split the data into train and test sets
train_data, val = train_test_split(data, test_size=0.1)
original_shape = train_data.shape[0]
# Remove the last elements if it is not divisible by the batch size. This is done to avoid errors in the training process.
train_data, val = train_data[:int(train_data.shape[0]-train_data.shape[0]%args.batch_size)], val[:int(val.shape[0]-val.shape[0]%args.batch_size)]
# Convert train_data and val to binary
train_data, (X_min_train, X_max_train) = from_real_to_binary(train_data)
val_binary, (X_min_val, X_max_val) = from_real_to_binary(val)
# Add volatility indicators to the binary dataframes. Be careful to add the correct volatility indicators to the correct dataframes.
train_data, indexes_vol_indicators_train = add_vol_indicators(train_data, vol_indicators[:train_data.shape[0]])
val_binary, indexes_vol_indicators_val = add_vol_indicators(val_binary, vol_indicators[original_shape: original_shape+val.shape[0]])

# Check if train_data and val_binary have missing values
if np.isnan(train_data).any():
    print(f"\nTrain data has missing values after pre processing! Control the pre processing steps.")
    print(f"Indexes of missing values in train data:\n{np.argwhere(np.isnan(train_data))}")
    exit()
if np.isnan(val_binary).any():
    print(f"\nValidation data has missing values after pre processing! Control the pre processing steps.")
    print(f"Indexes of missing values in validation data:\n{np.argwhere(np.isnan(val_binary))}")
    exit()
print(f"Done\n")

print(f"Original data entries type:\n\t{data[np.random.randint(0, data.shape[0])].dtype}")
print(f"Data binary entries type:\n\t{train_data[np.random.randint(0, train_data.shape[0])].dtype}")
print(f"Training data binary shape:\n\t{train_data.shape}")
print(f"Validation data binary shape:\n\t{val_binary.shape}\n")

if args.train_rbm:
    # Define the RBM
    num_visible = train_data.shape[1]
    num_hidden = args.hidden_units
    print(f"Number of visible units:\n\t{num_visible}")
    print(f"Number of hidden units:\n\t{num_hidden}")
    print(f"Learning rate:\n\t{args.learning_rate}")
    print(f"Batch size:\n\t{args.batch_size}")
    print(f"Number of gibbs sampling in CD:\n\t{args.k_step}\n")
    if args.continue_train:
        print("Continue training from past learned RBM parameters...")
        input = input("Enter the id of the trained RBM: ")
        weights = np.load(f"output/weights_{input}.npy")
        hidden_bias = np.load(f"output/hidden_bias_{input}.npy")
        visible_bias = np.load(f"output/visible_bias_{input}.npy")
    else:
        weights, hidden_bias, visible_bias = initialize_rbm(train_data, num_visible, num_hidden)
    print(f"Initial weights shape:\n\t{weights.shape}")
    print(f"Initial hidden bias:\n\t{hidden_bias}")
    print(f"Initial visible bias:\n\t{visible_bias}\n")

    # Train the RBM
    print(f"Start training the RBM...")
    additional_quantities = [X_min_train, X_max_train, indexes_vol_indicators_train, currencies]
    reconstruction_error, f_energy_overfitting, f_energy_diff, diff_fenergy, weights, hidden_bias, visible_bias = train(
        train_data, val_binary, weights, hidden_bias, visible_bias, num_epochs=args.epochs, batch_size=args.batch_size, 
        learning_rate=args.learning_rate, k=args.k_step, monitoring=True, id=id, additional_quantities=additional_quantities)
    print(f"Done\n")

    print(f"Saving rbm parameters, reconstruction error and free energies quantites...")
    np.save(f"output/weights_{id}.npy", weights)
    np.save(f"output/hidden_bias_{id}.npy", hidden_bias)
    np.save(f"output/visible_bias_{id}.npy", visible_bias)
    np.save(f"output/reconstruction_error_{id}.npy", reconstruction_error)
    np.save(f"output/f_energy_overfitting_{id}.npy", f_energy_overfitting)
    np.save(f"output/f_energy_diff_{id}.npy", f_energy_diff)
    np.save(f"output/diff_fenergy_{id}.npy", diff_fenergy)
    print(f"Done\n")

    print(f"Final weights:\n\t{weights}")
    print(f"Final hidden bias:\n\t{hidden_bias}")
    print(f"Final visible bias:\n\t{visible_bias}\n")

else:
    print("Loading weights, reconstruction error and free energies quantites...")
    input = input("Enter the id of the trained RBM: ")
    weights = np.load(f"output/weights_{input}.npy")
    hidden_bias = np.load(f"output/hidden_bias_{input}.npy")
    visible_bias = np.load(f"output/visible_bias_{input}.npy")
    reconstruction_error = np.load(f"output/reconstruction_error_{input}.npy")
    f_energy_overfitting = np.load(f"output/f_energy_overfitting_{input}.npy")
    f_energy_diff = np.load(f"output/f_energy_diff_{input}.npy")
    diff_fenergy = np.load(f"output/diff_fenergy_{input}.npy")
    print(f"Done\n")

# Plot the objectives
plot_objectives(reconstruction_error, f_energy_overfitting, f_energy_diff, diff_fenergy, id)

print("Sampling from the RBM...")
samples = parallel_sample(weights, hidden_bias, visible_bias, k=1000, n_samples=train_data.shape[0])
print(f"Done\n")

print(f"Saving the samples")
np.save(f"output/samples_{start_date}_{end_date}_{args.epochs}_{args.learning_rate}.npy", samples)
print(f"Done\n")

# Remove the volatility indicators
samples = np.delete(samples, indexes_vol_indicators_train, axis=1)

# Convert to real values
print("Converting the samples from binary to real values...")
samples = from_binary_to_real(samples, X_min_train, X_max_train).to_numpy()
print(f"Done\n")

total_time = time.time() - start
print(f"Total time: {total_time} seconds\n")

print("Plotting results...")
data = data[:train_data.shape[0]].reshape(samples.shape)
# Plot the samples and the recontructed error
plot_distributions(samples, data, currencies, id)

# Generate QQ plot data
qq_plots(samples, data, currencies, id)

# Plot the concentration functions
plot_tail_concentration_functions(data, samples, currencies, id)

# Plot upper and lower tail distribution functions
plot_tail_distributions(samples, data, currencies, id)

#Plot PCA with 2 components
plot_pca_with_marginals(samples, data, id)
print(f"Done\n")


# Correlations
if args.std:
    print(f"Computing means and standard deviations of the correlations, historical volatilties and 1st and 99th percentiles...")
    print(f"This will take a while...")
    pearson_df, spearman_df, kendall_df, first_last_gen_perc, first_last_real_perc = mean_std_statistics(
        data[:train_data.shape[0]], weights, hidden_bias, visible_bias, currencies, X_min_train, X_max_train, 
        indexes_vol_indicators_train, times=50)
    original_correlations = calculate_correlations(pd.DataFrame(data[:train_data.shape[0]], columns=currencies))
    print(f"Original correlations:\n{original_correlations}\n")
    print(f"Generated correlations:\n\t")
    print(f"Pearson:\n{pearson_df}\n")
    print(f"Spearman:\n{spearman_df}\n")
    print(f"Kendall:\n{kendall_df}\n")
    
    samples_df = pd.DataFrame(samples, columns=currencies)
    data_train_df = pd.DataFrame(data[:train_data.shape[0]], columns=currencies)
    historical_volatilities_gen = calculate_historical_volatility(samples_df, window=samples.shape[0]).iloc[-1]
    historical_volatilities_real = calculate_historical_volatility(data_train_df, window=train_data.shape[0]).iloc[-1]
    print(f"Real historical volatilities:\n{historical_volatilities_real}\n")
    print(f"Generated historical volatilities:\n{historical_volatilities_gen}\n")

    print(f"1st and 99th percentiles of the generated and real data:")
    print(f"Generated:\n{first_last_gen_perc}\n")
    print(f"Real:\n{first_last_real_perc}\n")
    print(f"Done\n")
else:
    print("Computing correlations...")
    gen_correlations = calculate_correlations(pd.DataFrame(samples, columns=currencies))
    original_correlations = calculate_correlations(pd.DataFrame(data[:train_data.shape[0]], columns=currencies))
    plot_correlation_heatmap(original_correlations, gen_correlations, id)
    print(f"Original correlations:\n{original_correlations}")
    print(f"Generated correlations:\n{gen_correlations}")
    print(f"Done\n")

# Compute 1-day autocorrelation
print("Computing 1-day autocorrelation wrt to K...")
print("This will take a while. If you want you can stop the execution")
plot_autocorr_wrt_K(weights, hidden_bias, visible_bias, k_max=1000, n_samples=1000, X_min=X_min_train, X_max=X_max_train)

# Create the animated gifs
print("Creating animated gifs...")
try:
    create_animated_gif('output/historgrams', id, output_filename=f'{id}_histograms.gif')
    create_animated_gif('output/weights_receptive_field', id, output_filename=f'{id}_weights_receptive_field.gif')
except Exception as e:
    print(f"Error creating animated gifs: {e}")
print(f"Done\n")
print(f'Finished id {id}!')
