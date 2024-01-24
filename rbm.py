import numpy as np
import numba as nb
from utils import *
np.random.seed(666)

@nb.njit
def initialize_rbm(num_visible, num_hidden):
    # np.random.seed(666) # I set the seed in the main.py file
    weights = np.asarray(np.random.uniform(
        low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                    high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                    size=(num_visible, num_hidden)))
    hidden_bias = np.zeros(num_hidden)
    visible_bias = np.zeros(num_visible)
    
    return weights, hidden_bias, visible_bias

@nb.njit
def train(data, weights, hidden_bias, visible_bias, num_epochs, batch_size, learning_rate, k, monitoring=True):

    num_samples = data.shape[0]

    reconstructed_error = []
    for epoch in range(num_epochs):
        for i in range(num_samples):
            v0 = data[i:i+batch_size]
            # Positive phase
            h0_prob = sigmoid(dot_product(v0, weights) + hidden_bias)
            h0 = boolean_to_int(h0_prob > np.random.random(h0_prob.shape))

            # Gibbs sampling
            vk = np.copy(v0)
            for _ in range(k):
                hk_prob = sigmoid(dot_product(vk, weights) + hidden_bias)
                hk = boolean_to_int(hk_prob > np.random.random(hk_prob.shape))
                vk_prob = sigmoid(dot_product(hk, weights.T) + visible_bias)
                # I can avoid the conversion to int here
                vk = boolean_to_int(vk_prob > np.random.random(vk_prob.shape))

            # Negative phase
            hk_prob = sigmoid(dot_product(vk, weights) + hidden_bias)

            # Update weights and biases
            weights += learning_rate * (dot_product(v0.T, h0) - dot_product(vk_prob.T, hk_prob)) / batch_size
            visible_bias += learning_rate * mean_axis_0(v0 - vk_prob) / batch_size
            hidden_bias += learning_rate * mean_axis_0(h0 - hk_prob) / batch_size
    
        if monitoring:
            # Compute the reconstruction error
            # v0 = data
            # h0_prob = _sigmoid(dot_product(v0, weights) + hidden_bias)
            # h0 = (h0_prob > np.random.random(h0_prob.shape)).astype(int)
            # vk_prob = _sigmoid(dot_product(h0, weights.T) + visible_bias)
            # vk = (vk_prob > np.random.random(vk_prob.shape)).astype(int)
            if epoch % 100 == 0:
                # Compute the reconstruction error
                error = np.sum((v0 - vk) ** 2) / num_samples
                print(f"Epoch: {epoch}")
                print(error)
                reconstructed_error.append(error)

                # Compute the free energy of a subset of the data
                start_idx = np.random.randint(low=0, high=num_samples-100)
                v = data[start_idx: start_idx+100]
                f_energy = free_energy(v, weights, visible_bias, hidden_bias)
                print(np.mean(f_energy))

        
    return reconstructed_error, weights, hidden_bias, visible_bias

@nb.njit
def sample(weights, hidden_bias, visible_bias, num_samples, num_visible, k):
    samples = np.zeros((num_samples, num_visible))
    for i in range(num_samples):
        random_input = np.random.randint(low=0, high=2, size=(1,num_visible))
        if i % 100 == 0 and i != 0:
                print(f"Sampled the first {i} samples")
        for j in range(10e3):
            # random_input = random_input.reshape(1, -1)
            h = sigmoid(dot_product(random_input, weights) + hidden_bias)
            h = boolean_to_int(h > np.random.random(h.shape))
            v = sigmoid(dot_product(h, weights.T) + visible_bias)
            sample_k = boolean_to_int(v > np.random.random(v.shape))
        # h = _sigmoid(dot_product(sample, weights) + hidden_bias)
        # h = (h > np.random.random(h.shape)).astype(int)
        # v = _sigmoid(dot_product(h, weights.T) + visible_bias)
        # sample = (v > np.random.random(v.shape)).astype(int)
        samples[i] = sample_k

    return samples



