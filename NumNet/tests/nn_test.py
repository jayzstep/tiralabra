import unittest
from numnet.nn import NN, sigmoid, sigmoid_prime, cost_derivative
import numpy as np
import pandas as pd
import copy


class TestNN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_data = pd.read_csv('data/mnist_train.csv', nrows=10)
        cls.x_train = cls.test_data.iloc[:, 1:].values.astype('float32') / 255
        cls.y_train = cls.test_data.iloc[:, 0].values
        cls.y_train_one_hot = np.eye(10)[cls.y_train]
        cls.training_data = [(x.reshape(-1, 1), y.reshape(-1, 1)) for x, y in zip(cls.x_train, cls.y_train_one_hot)]

        cls.net = NN([784, 30, 10], sigmoid, sigmoid_prime, cost_derivative)

    def setUp(self):
        pass
         
    def test_hello_world(self):
        self.assertEqual("hello world", "hello world")

    def test_biases_change(self):
        bias_before = str(copy.deepcopy(self.net.b))
        self.net.train(self.training_data, 50, 0.3)
        bias_after = str(self.net.b)
        self.assertNotEqual(bias_before, bias_after)

    def test_weights_change(self):
        weight_before = str(copy.deepcopy(self.net.w))
        self.net.train(self.training_data, 50, 0.3)
        weight_after = str(self.net.w)
        self.assertNotEqual(weight_before, weight_after)

    def test_sigmoid(self):
        pass
    def test_sigmoid_prime(self):
        pass
    def test_mse_prime(self):
        pass

