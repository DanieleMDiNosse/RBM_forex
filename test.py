import unittest
import numpy as np
from utils import *

class TestBinaryConversion(unittest.TestCase):
    
    def setUp(self):
        # Generate a random dataset
        np.random.seed(0)
        self.data = np.random.rand(10, 3)  # 10 samples, 3 features

        # Convert to binary
        self.data_binary, (self.X_min, self.X_max) = from_real_to_binary(self.data)

        # Convert back to real values
        self.data_reconstructed = from_binary_to_real(self.data_binary, self.X_min, self.X_max)

    def test_binary_length(self):
        # Check that each binary number has 16 digits
        for row in self.data_binary:
            for number in row.reshape(-1, 16):
                self.assertTrue(all(bit == 0 or bit == 1 for bit in number), "Binary number doesn't consist of 0s and 1s.")
                self.assertEqual(len(number), 16, "Binary number doesn't have 16 digits.")

    def test_binary_dataset_shape(self):
        # Check that the binary dataset has the correct shape
        self.assertEqual(self.data_binary.shape, (self.data.shape[0], 16 * self.data.shape[1]))

    def test_reconstruction_accuracy(self):
        # Check that the reconstructed data coincides with the original data
        np.testing.assert_allclose(self.data, self.data_reconstructed, atol=1e-4)

# Running the tests
unittest.main(argv=[''], exit=False)
