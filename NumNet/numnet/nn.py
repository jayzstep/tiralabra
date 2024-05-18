import numpy as np
import pandas as pd

class Network(object):
    """Sizes määrittää kerrosten sekä kerroksella olevien neuronien määrän.
    Biases ja weights alustetaan listoiksi oikean kokoisia matriiseja."""

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        


if __name__ == "__main__":
    net = Network([2,3,1])
    print(net.biases) 
    print(net.weights)
    
