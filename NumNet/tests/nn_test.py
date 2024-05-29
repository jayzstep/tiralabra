import unittest
from numnet.nn import NN

class TestNN(unittest.TestCase):
    def setUp(self):
        print("Set up goes here")

    def test_hello_world(self):
        self.assertEqual("hello world", "hello world")

