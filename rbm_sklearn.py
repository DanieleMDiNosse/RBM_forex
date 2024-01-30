import numpy as np
import pandas as pd
from sklearn.neural_network import BernoulliRBM
import matplotlib.pyplot as plt
from utils import *
plt.style.use('seaborn')


def gibbs_sampling_step(rbm, v):
    '''Performs one step of Gibbs sampling.'''
    h = rbm._sample_hiddens(v, np.random.RandomState())
    v = rbm._sample_visibles(h, np.random.RandomState())
    return v


# Generate synthetic data
N = 5000  # Specify the number of data points
data = np.random.normal(loc=0.0, scale=1.0, size=(N, 1))  # Normal distribution
# data2 = np.random.normal(loc=2.0, scale=2.0, size=(N, 1))
# data = np.concatenate((data1, data2))

# Convert real data to binary
data_binary, scale_params = from_real_to_binary(data)

# Train the BernoulliRBM model
rbm = BernoulliRBM(n_components=12, learning_rate=0.01, n_iter=5000, random_state=0)
rbm.fit(data_binary)

# Initialize a random sample
random_samples = np.random.randint(2, size=(N, rbm.components_.shape[1]))

# Perform Gibbs sampling
for _ in range(1000):  # number of Gibbs steps; you can adjust this
    random_samples = gibbs_sampling_step(rbm, random_samples)

# Convert binary data back to real
sampled_data_real = from_binary_to_real(random_samples, scale_params[0], scale_params[1])

# Plot the original data distribution and the sampled data distribution
fig, ax = plt.subplots(1, 1, figsize=(8, 5), tight_layout=True)
ax.hist(data, bins=50, alpha=0.6, label="Original data")
ax.set_title("Original data vs Sampled data")
ax.hist(sampled_data_real, bins=50, alpha=0.6, label="Sampled data")
plt.legend()
plt.show()
