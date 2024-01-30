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

class TestDotProduct(unittest.TestCase):

    def test_dot_product_valid_input(self):
        # Test the dot product with valid inputs
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[2, 0], [1, 2]])
        expected_result = np.array([[4, 4], [10, 8]])
        np.testing.assert_array_equal(dot_product(A, B), expected_result, "Dot product is incorrect.")

    def test_dot_product_invalid_input(self):
        # Test the dot product with invalid (incompatible) inputs
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[2, 0]])
        with self.assertRaises(ValueError):
            dot_product(A, B)

    def test_dot_product_large_matrices(self):
        # Optional: Test the dot product with larger matrices for performance
        np.random.seed(0)
        A = np.random.rand(100, 100)
        B = np.random.rand(100, 100)
        # No specific assertion, just checking if the function handles large matrices without error
        result = dot_product(A, B)
        self.assertEqual(result.shape, (100, 100), "Dot product result shape is incorrect for large matrices.")

# Running the tests
unittest.main(argv=[''], exit=False)
