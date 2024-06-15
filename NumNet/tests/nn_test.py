import copy
import unittest

import numpy as np
import pandas as pd

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
        self.net = NN([784, 30, 10], sigmoid, sigmoid_prime, cost_derivative)

    def test_hello_world(self):
        self.assertEqual("hello world", "hello world")

    def test_nn_overfits(self):
        evaluation_before = self.net.evaluate(self.test_data)
        self.net.train(self.training_data, 100, 0.3, 1)
        evaluation_after = self.net.evaluate(self.test_data)
        assert evaluation_after - evaluation_before > 0

    def test_biases_change(self):
        bias_before = str(copy.deepcopy(self.net.b))
        self.net.train(self.training_data, 1, 0.3, 1)
        bias_after = str(self.net.b)
        self.assertNotEqual(bias_before, bias_after)

    def test_weights_change(self):
        weight_before = str(copy.deepcopy(self.net.w))
        self.net.train(self.training_data, 1, 0.3, 1)
        weight_after = str(self.net.w)
        self.assertNotEqual(weight_before, weight_after)

    def test_predict_raises_error_if_input_wrong_size(self):
        a = [1, 2, 3]
        self.assertRaises(ValueError, self.net.predict, a)

    def test_predict_returns_something_with_right_input(self):
        a = self.training_data[0][0]
        self.assertEqual(10, len(self.net.predict(a)))

    def test_sigmoid(self):
        pass

    def test_sigmoid_prime(self):
        pass

    def test_mse_prime(self):
        pass
