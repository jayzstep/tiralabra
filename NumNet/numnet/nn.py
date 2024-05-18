import pandas as pd
import numpy as np

train_data = pd.read_csv('data/mnist_train.csv')
test_data = pd.read_csv('data/mnist_test.csv')

print(test_data.head())
