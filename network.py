import numpy as np
import matplotlib.pyplot as plt

class RestrictedBoltzmannMachine:
    def __init__(self, n_visible, n_hidden):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.weights = np.random.normal(0, 0.1, (n_visible, n_hidden))
        self.hidden_bias = np.random.normal(size=n_hidden)
        self.visible_bias = np.random.normal(size=n_visible)

    def sample_hidden(self, visible):
        """ Sample hidden units given visible units """
        activation = np.dot(visible, self.weights) + self.hidden_bias
        probabilities = self.sigmoid(activation)
        return probabilities, np.random.binomial(1, probabilities)

    def sample_visible(self, hidden):
        """ Sample visible units given hidden units """
        activation = np.dot(hidden, self.weights.T) + self.visible_bias
        # For a normal distribution, we return the activation itself
        return activation

    def train(self, data, lr=0.1, batch_size=10, epochs=1000):
        """ Train the RBM """
        for epoch in range(epochs):
            np.random.shuffle(data)
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]

                v0 = batch
                _, h0 = self.sample_hidden(v0)
                v1 = self.sample_visible(h0)
                _, h1 = self.sample_hidden(v1)

                # Calculate the gradients
                w_grad = np.dot(v0.T, h0) - np.dot(v1.T, h1)
                vb_grad = np.sum(v0 - v1, axis=0)
                hb_grad = np.sum(h0 - h1, axis=0)

                # Update weights and biases
                self.weights += lr * w_grad / batch_size
                self.visible_bias += lr * vb_grad / batch_size
                self.hidden_bias += lr * hb_grad / batch_size

            # Monitor the progress
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs} completed")

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

class RestrictedBoltzmannMachineGaussian(RestrictedBoltzmannMachine):
    def __init__(self, n_visible, n_hidden, sigma=1.0):
        super().__init__(n_visible, n_hidden)
        self.sigma = sigma  # Standard deviation for the Gaussian visible units
    
    def sample_visible(self, hidden):
        """ Sample visible units given hidden units using Gaussian distribution """
        mean_activation = np.dot(hidden, self.weights.T) + self.visible_bias
        # Sample from a Gaussian distribution with mean=mean_activation and standard deviation=self.sigma
        return np.random.normal(mean_activation, self.sigma)

# RBM parameters
n_visible = 1  # Since we're modeling a 1D normal distribution
n_hidden = 100  # Increase number of hidden units for more capacity

training_data = np.random.normal(0, 1, (1000, n_visible))

# Create the RBM with Gaussian visible units
rbm_gaussian = RestrictedBoltzmannMachineGaussian(n_visible, n_hidden)

# Train the RBM
rbm_gaussian.train(training_data, lr=0.01, epochs=5000)  # Adjust learning rate and increase epochs

# Let's test by sampling from the RBM
_, sample = rbm_gaussian.sample_hidden(np.random.normal(0, 1, (1000, n_visible)))
generated_data_gaussian = rbm_gaussian.sample_visible(sample)

# Generate a histogram of the training data and the generated data
plt.figure(figsize=(10, 5), tight_layout=True)
plt.hist(generated_data_gaussian, bins=50, density=True, color='green', alpha=0.65, label="RBM generated data")
plt.hist(training_data, bins=50, density=True, color='blue', alpha=0.65, label="Training data")
plt.legend()
plt.show()

# Return the generated data for further analysis if needed
generated_data_gaussian

