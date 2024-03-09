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
parser.add_argument("--train_rbm", "-t", action="store_true", help="Train the RBM. Default: False")
parser.add_argument("--hidden_units", "-hu", type=int, default=30, help="Train the RBM. Default: 30")
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
    print_("Loading data...")
    data = pd.read_pickle(f'data/currencies_data_{start_date}_{end_date}.pkl')
    data = data.values
except FileNotFoundError:
    print_("Downloading data...")
    data = data_download(['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCAD=X'],
                         start_date=start_date, end_date=end_date, provider='yfinance')
    data.to_pickle(f'data/currencies_data_{start_date}_{end_date}.pkl')
    data = data.values

id = f'C{os.getpid()}_{args.learning_rate}_{args.k_step}'
print_(f"{id} - TRAINING RBM ON CURRENCY DATA\n")

print_(f"Pre processing data...")
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
vol_indicators_train = vol_indicators[:train_data.shape[0]]
vol_indicators_val = vol_indicators[original_shape: original_shape+val.shape[0]]

train_data, indexes_vol_indicators_train = add_vol_indicators(train_data, vol_indicators_train)
val_binary, indexes_vol_indicators_val = add_vol_indicators(val_binary, vol_indicators_val)

# Check if train_data and val_binary have missing values
if np.isnan(train_data).any():
    print_(f"\nTrain data has missing values after pre processing! Control the pre processing steps.")
    print_(f"Indexes of missing values in train data:\n{np.argwhere(np.isnan(train_data))}")
    exit()
if np.isnan(val_binary).any():
    print_(f"\nValidation data has missing values after pre processing! Control the pre processing steps.")
    print_(f"Indexes of missing values in validation data:\n{np.argwhere(np.isnan(val_binary))}")
    exit()
print_(f"Done\n")

print_(f"Original data entries type:\n\t{data[np.random.randint(0, data.shape[0])].dtype}")
print_(f"Data binary entries type:\n\t{train_data[np.random.randint(0, train_data.shape[0])].dtype}")
print_(f"Training data binary shape:\n\t{train_data.shape}")
print_(f"Validation data binary shape:\n\t{val_binary.shape}\n")

if args.train_rbm:
    # Define the RBM
    num_visible = train_data.shape[1]
    num_hidden = args.hidden_units
    print_(f"Number of visible units:\n\t{num_visible}")
    print_(f"Number of hidden units:\n\t{num_hidden}")
    print_(f"Learning rate:\n\t{args.learning_rate}")
    print_(f"Batch size:\n\t{args.batch_size}")
    print_(f"Number of gibbs sampling in CD:\n\t{args.k_step}\n")
    if args.continue_train:
        print_("Continue training from past learned RBM parameters...")
        input = input("Enter the id of the trained RBM: ")
        weights = np.load(f"output/weights_{input}.npy")
        hidden_bias = np.load(f"output/hidden_bias_{input}.npy")
        visible_bias = np.load(f"output/visible_bias_{input}.npy")
    else:
        weights, hidden_bias, visible_bias = initialize_rbm(train_data, num_visible, num_hidden)

    print_(f"Initial weights shape:\n\t{weights.shape}")
    print_(f"Initial hidden bias:\n\t{hidden_bias}")
    print_(f"Initial visible bias:\n\t{visible_bias}\n")

    # Train the RBM
    print_(f"Start training the RBM...")
    additional_quantities = [X_min_train, X_max_train, indexes_vol_indicators_train, vol_indicators_train, currencies]

    reconstruction_error, f_energy_overfitting, f_energy_diff, diff_fenergy, weights, hidden_bias, visible_bias = train(
        train_data, val_binary, weights, hidden_bias, visible_bias, num_epochs=args.epochs, batch_size=args.batch_size, 
        learning_rate=args.learning_rate, k=args.k_step, monitoring=True, id=id, additional_quantities=additional_quantities)
    print_(f"Done\n")

    print_(f"Saving rbm parameters, reconstruction error and free energies quantites...")
    np.save(f"output/weights_{id}.npy", weights)
    np.save(f"output/hidden_bias_{id}.npy", hidden_bias)
    np.save(f"output/visible_bias_{id}.npy", visible_bias)
    np.save(f"output/reconstruction_error_{id}.npy", reconstruction_error)
    np.save(f"output/f_energy_overfitting_{id}.npy", f_energy_overfitting)
    np.save(f"output/f_energy_diff_{id}.npy", f_energy_diff)
    np.save(f"output/diff_fenergy_{id}.npy", diff_fenergy)
    print_(f"Done\n")

    print_(f"Final weights:\n\t{weights}")
    print_(f"Final hidden bias:\n\t{hidden_bias}")
    print_(f"Final visible bias:\n\t{visible_bias}\n")

else:
    print_("Loading weights, reconstruction error and free energies quantites...")
    input = input("Enter the id of the trained RBM (something like C<number>_<lr>_<ksteps>): ")
    weights = np.load(f"output/weights_{input}.npy")
    hidden_bias = np.load(f"output/hidden_bias_{input}.npy")
    visible_bias = np.load(f"output/visible_bias_{input}.npy")
    reconstruction_error = np.load(f"output/reconstruction_error_{input}.npy")
    f_energy_overfitting = np.load(f"output/f_energy_overfitting_{input}.npy")
    f_energy_diff = np.load(f"output/f_energy_diff_{input}.npy")
    diff_fenergy = np.load(f"output/diff_fenergy_{input}.npy")
    print_(f"Done\n")

# Plot the objectives
# plot_objectives(reconstruction_error, f_energy_overfitting, f_energy_diff, diff_fenergy, id)

num_drawing = 100
samples = np.zeros((num_drawing, train_data.shape[0], train_data.shape[1]))
print_(f"Sampling from the RBM for {num_drawing} times...")
for i in range(num_drawing):
    print_(f"Drawing {i+1}...")
    samples[i] = parallel_sample(weights, hidden_bias, visible_bias, 1000, indexes_vol_indicators_train, vol_indicators_train, train_data.shape[0], n_processors=8)
print_(f"Done\n")

# Collect indexes of low and high volatility regimes. These will be used to compute the historical volatilities
print_(f"Collecting indexes of low and high volatility regimes...")
samples_dfs = [pd.DataFrame(sample) for sample in samples]
condition_low_vol = [sample.iloc[:, indexes_vol_indicators_train] == 0 for sample in samples_dfs]
condition_high_vol = [sample.iloc[:, indexes_vol_indicators_train] == 1 for sample in samples_dfs]
indexes_low_vol = [np.unique(np.where(condition)) for condition in condition_low_vol]
indexes_high_vol = [np.unique(np.where(condition)) for condition in condition_high_vol]
print_(f"Done\n")


print_(f"Saving the samples")
np.save(f"output/samples_{start_date}_{end_date}_{args.epochs}_{args.learning_rate}.npy", samples)
print_(f"Done\n")

# Remove the volatility indicators
print_("Removing the volatility indicators from the samples...")
samples_no_vol_indicators = np.zeros((num_drawing, train_data.shape[0], train_data.shape[1]-len(indexes_vol_indicators_train)))
for i in range(len(samples)):
    s = np.delete(samples[i], indexes_vol_indicators_train, axis=1)
    samples_no_vol_indicators[i] = s
samples = samples_no_vol_indicators
print_(f"Done\n")

# Convert to real values
print_("Converting the samples from binary to real values...")
samples_real = np.zeros((num_drawing, train_data.shape[0], data.shape[1]))
for i in range(len(samples)):
    s = from_binary_to_real(samples[i], X_min_train, X_max_train).to_numpy()
    samples_real[i] = s
samples = samples_real
print_(f"Done\n")

# total_time = time.time() - start
# print_(f"Total time: {total_time} seconds\n")

print_("Plotting results...")
data = data[:train_data.shape[0]].reshape(samples[0].shape)
# Plot the samples and the recontructed error
plot_distributions(samples[np.random.randint(0, len(samples))], data, currencies, id)

# Generate QQ plot data
qq_plots(samples, data, currencies, id)

# Plot the concentration functions
plot_tail_concentration_functions(data, samples, currencies, id)

# # Plot upper and lower tail distribution functions
# plot_tail_distributions(samples, data, currencies, id)

#Plot PCA with 2 components
plot_pca_with_marginals(samples[np.random.randint(0, len(samples))], data, id)
print_(f"Done\n")

# Correlations
print_(f"Computing means and standard deviations of the correlations, historical volatilties and 1st and 99th percentiles...")
pearson_df, spearman_df, kendall_df, first_last_gen_perc, first_last_real_perc = mean_std_statistics(samples, data, currencies)
original_correlations = calculate_correlations(pd.DataFrame(data[:train_data.shape[0]], columns=currencies))
print_(f"{Colors.GREEN}Correlations{Colors.RESET}:")
print_(f"{Colors.MAGENTA}Real:{Colors.RESET}:\n{original_correlations}\n")
print_(f"{Colors.MAGENTA}Generated{Colors.RESET}:")
print_(f"Pearson:\n{pearson_df}\n")
print_(f"Spearman:\n{spearman_df}\n")
print_(f"Kendall:\n{kendall_df}\n")


print_(f"Computing historical volatilities...")
data_train_df = pd.DataFrame(data[:train_data.shape[0]], columns=currencies)
samples_dfs = [pd.DataFrame(sample, columns=currencies) for sample in samples]
data_train_low_vol = data[np.where(vol_indicators_train == 0)[0]]
data_train_high_vol = data[np.where(vol_indicators_train == 1)[0]]

historical_volatilities_gen_low = []
historical_volatilities_gen_high = []
for sample_df, idx_low, idx_high in zip(samples_dfs, indexes_low_vol, indexes_high_vol):
    df_low = sample_df.iloc[idx_low]
    df_high = sample_df.iloc[idx_high]
    historical_volatilities_gen_low.append(calculate_historical_volatility(df_low, window=df_low.shape[0]).iloc[-1])
    historical_volatilities_gen_high.append(calculate_historical_volatility(df_high, window=df_high.shape[0]).iloc[-1])

historical_volatilities_gen_low_mean = np.mean(np.array(historical_volatilities_gen_low), axis=0)
historical_volatilities_gen_low_std = np.std(np.array(historical_volatilities_gen_low), axis=0)
historical_volatilities_gen_high_mean = np.mean(np.array(historical_volatilities_gen_high), axis=0)
historical_volatilities_gen_high_std = np.std(np.array(historical_volatilities_gen_high), axis=0)

historical_volatilities_gen_low = pd.DataFrame({'Mean':historical_volatilities_gen_low_mean, 'Std':historical_volatilities_gen_low_std}, index=currencies)
historical_volatilities_gen_high = pd.DataFrame({'Mean':historical_volatilities_gen_high_mean, 'Std':historical_volatilities_gen_high_std}, index=currencies)

historical_volatilities_real_low = calculate_historical_volatility(data_train_low_vol, window=train_data.shape[0]).iloc[-1]
historical_volatilities_real_high = calculate_historical_volatility(data_train_high_vol, window=train_data.shape[0]).iloc[-1]

print_(f"{Colors.GREEN}Historical volatilities{Colors.RESET}:")
print_(f"{Colors.MAGENTA}Real (low vol regime):{Colors.RESET}\n{historical_volatilities_real_low}\n")
print_(f"{Colors.MAGENTA}Generated (low vol regime){Colors.RESET}:\n{historical_volatilities_gen_low}\n")
print_(f"{Colors.MAGENTA}Real (high vol regime) :{Colors.RESET}\n{historical_volatilities_real_high}\n")
print_(f"{Colors.MAGENTA}Generated (high vol regime) {Colors.RESET}:\n{historical_volatilities_gen_high}\n")

print_(f"{Colors.GREEN}1st and 99th percentiles of the generated and real data{Colors.RESET}:")
print_(f"{Colors.MAGENTA}Real:{Colors.RESET}:\n{first_last_real_perc}\n")
print_(f"{Colors.MAGENTA}Generated{Colors.RESET}:\n{first_last_gen_perc}\n")

# Compute 1-day autocorrelation
choice = input("Do you want to compute the 1-day autocorrelation wrt to K? (y/n): ")
if choice == 'y':
    print_("With the current implementation, this will take a really long time (1 day on my machine). If you want you can stop the execution or maybe optimize the code!.")
    plot_autocorr_wrt_K(
        weights, hidden_bias, visible_bias, k_max=1000, n_samples=1000, X_min=X_min_train, X_max=X_max_train, 
        indexes_vol_indicators=indexes_vol_indicators_train, vol_indicators=vol_indicators_train[:1000])

# Create the animated gifs
print_("Creating animated gifs...")
try:
    create_animated_gif('output/historgrams', id, output_filename=f'{id}_histograms.gif')
    create_animated_gif('output/weights_receptive_field', id, output_filename=f'{id}_weights_receptive_field.gif')
except Exception as e:
    print_(f"Error creating animated gifs (probably no files): {e}")
print_(f"Done\n")
print_(f'Finished id {id}!')
