import copy
import unittest
import pytest
import numpy as np
import pandas as pd
import random

from numnet.nn import NN, cost_derivative, sigmoid, sigmoid_prime


class TestNN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_data = pd.read_csv("data/mnist_train.csv", nrows=10)
        cls.x_train = cls.test_data.iloc[:, 1:].values.astype("float32") / 255
        cls.y_train = cls.test_data.iloc[:, 0].values
        cls.y_train_one_hot = np.eye(10)[cls.y_train]
        cls.training_data = [
            (x.reshape(-1, 1), y.reshape(-1, 1))
            for x, y in zip(cls.x_train, cls.y_train_one_hot)
        ]
        cls.test_data = [
            (x.reshape(-1, 1), y) for x, y in zip(cls.x_train, cls.y_train)
        ]

    def setUp(self):
        layers = [784, 30, 10]
        self.net = NN(layers, sigmoid, sigmoid_prime, cost_derivative)
        self.net.seed = 1337
        self.net.w_and_b_initializer(layers)

    def test_nn_overfits(self):
        evaluation_before = self.net.evaluate(self.test_data)
        self.net.train(self.training_data, 100, 0.3, 1)
        evaluation_after = self.net.evaluate(self.test_data)
        assert evaluation_after - evaluation_before > 0

    def test_biases_change(self):
        bias_before = [copy.deepcopy(biases) for biases in self.net.b]
        self.net.train(self.training_data, 1, 0.3, 1)
        bias_after = self.net.b
        for layer_number, (before, after) in enumerate(zip(bias_before, bias_after)):
            with self.subTest(layer=layer_number):
                self.assertFalse(np.array_equal(before, after), f"Bias error in layer {layer_number}")

    # @pytest.mark.weights
    def test_weights_change(self):
        weights_before = [copy.deepcopy(weights) for weights in self.net.w]
        self.net.train(self.training_data, 1, 0.3, 1)
        weights_after = self.net.w
        
        for layer_number, (before, after) in enumerate(zip(weights_before, weights_after)):
            with self.subTest(layer=layer_number):
                self.assertFalse(np.array_equal(before, after), f"Weight error in layer {layer_number}")

    def test_predict_raises_error_if_input_wrong_size(self):
        a = [1, 2, 3]
        self.assertRaises(ValueError, self.net.predict, a)

    def test_predict_returns_something_with_right_input(self):
        a = self.training_data[0][0]
        self.assertEqual(10, len(self.net.predict(a)))

    def test_sample_order_doesnt_matter(self):
        x_batch = np.hstack([x for x, _ in self.training_data]) 
        n = x_batch.shape[1]

        x_all_as, _ = self.net.forward(x_batch)
        x_output = x_all_as[-1]
        random_index = np.random.permutation(n)
        shuffled_x_batch = x_batch[:, random_index]
        shuffled_x_all_as, _ = self.net.forward(shuffled_x_batch)
        shuffled_x_output = shuffled_x_all_as[-1]

        flag = True
        for i in range(n):
            index = random_index[i]
            if not np.allclose( x_output[:, index], shuffled_x_output[:, i]):
                flag = False
                break
        self.assertTrue(flag)

    def test_sigmoid(self):
        pass

    def test_sigmoid_prime(self):
        pass

    def test_mse_prime(self):
        pass
