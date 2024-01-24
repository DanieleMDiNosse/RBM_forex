import numpy as np
from tqdm import tqdm
from scipy.special import kl_div


class RBM:
    def __init__(self, num_visible, num_hidden):
        np_rng = np.random.RandomState(1234)
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.weights = np.asarray(np_rng.uniform(
			low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                       	high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                       	size=(num_visible, num_hidden)))
        self.hidden_bias = np.zeros(num_hidden)
        self.visible_bias = np.zeros(num_visible)

    def train(self, data, num_epochs, batch_size, learning_rate, k, monitoring=True):

        num_samples = data.shape[0]

        reconstructed_error = []
        for epoch in range(num_epochs):
            for i in range(num_samples):
                v0 = data[i:i+batch_size]
                # Positive phase
                h0_prob = self._sigmoid(np.dot(v0, self.weights) + self.hidden_bias)
                h0 = (h0_prob > np.random.random(h0_prob.shape)).astype(int)

                # Gibbs sampling
                vk = np.copy(v0)
                for _ in range(k):
                    hk_prob = self._sigmoid(np.dot(vk, self.weights) + self.hidden_bias)
                    hk = (hk_prob > np.random.random(hk_prob.shape)).astype(int)
                    vk_prob = self._sigmoid(np.dot(hk, self.weights.T) + self.visible_bias)
                    vk = (vk_prob > np.random.random(vk_prob.shape)).astype(int)

                # Negative phase
                hk_prob = self._sigmoid(np.dot(vk, self.weights) + self.hidden_bias)

                # Update weights and biases
                self.weights += learning_rate * (np.dot(v0.T, h0) - np.dot(vk.T, hk_prob)) / batch_size
                self.visible_bias += learning_rate * np.mean(v0 - vk, axis=0) / batch_size
                self.hidden_bias += learning_rate * np.mean(h0 - hk_prob, axis=0) / batch_size
        
            if monitoring:
                # Compute the reconstruction error
                # v0 = data
                # h0_prob = self._sigmoid(np.dot(v0, self.weights) + self.hidden_bias)
                # h0 = (h0_prob > np.random.random(h0_prob.shape)).astype(int)
                # vk_prob = self._sigmoid(np.dot(h0, self.weights.T) + self.visible_bias)
                # vk = (vk_prob > np.random.random(vk_prob.shape)).astype(int)
                error = np.sum((v0 - vk) ** 2) / num_samples
                reconstructed_error.append(error)
                print(f"Epoch: {epoch} - Reconstruction error: {error}")
        return reconstructed_error


    def sample(self, num_samples, k):
        samples = np.zeros((num_samples, self.num_visible))
        sample = np.random.randint(2, size=(1, self.num_visible))

        for i in tqdm(range(num_samples), desc='Sampling'):
            for _ in range(k):
                h = self._sigmoid(np.dot(sample, self.weights) + self.hidden_bias)
                h = (h > np.random.random(h.shape)).astype(int)
                v = self._sigmoid(np.dot(h, self.weights.T) + self.visible_bias)
                sample = (v > np.random.random(v.shape)).astype(int)
            # h = self._sigmoid(np.dot(sample, self.weights) + self.hidden_bias)
            # h = (h > np.random.random(h.shape)).astype(int)
            # v = self._sigmoid(np.dot(h, self.weights.T) + self.visible_bias)
            # sample = (v > np.random.random(v.shape)).astype(int)
            samples[i] = sample

        return samples

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

if __name__ == '__main__':
    # Example usage
    num_visible = 32  # Number of binary features in your data
    num_hidden = 64   # Number of hidden units
    rbm = RBM(num_visible, num_hidden)

    # Assuming binary_data is your dataset in binary form
    # rbm.train(binary_data, num_epochs=1000, learning_rate=0.1, k=1)

    # Generating new samples
    # new_samples = rbm.sample(500)
