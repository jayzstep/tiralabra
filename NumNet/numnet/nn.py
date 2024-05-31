import random
import numpy as np
import pandas as pd
import gradio as gr
from PIL import Image


def cost_derivative(output_activations, y):
    return (output_activations-y)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

class NN():
    def __init__(self, layers, activation, activation_derivative, cost_derivative):
        self.b = [np.random.randn(y,1) for y in layers[1:]]
        self.w = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.cost_derivative = cost_derivative

    def predict(self, a):
        for b, w in zip(self.b, self.w):
            a = self.activation(np.dot(w, a) + b)
        return a

    def train(self, training_data, epochs, learning_rate, test_data=None):
        training_data = list(training_data)
        for i in range(epochs):

            random.shuffle(training_data)
            for sample in training_data:
                x, y = sample
                a = x
                all_as = [x]
                zs = []
                w1 = self.w[0]
                w2 = self.w[1]
                b1 = self.b[0]
                b2 = self.b[1]

                # forward
                z1 = np.dot(w1, x) + b1
                zs.append(z1)
                a = self.activation(z1)
                all_as.append(a)
                z2 = np.dot(w2, a) +b2
                zs.append(z2)
                a = self.activation(z2)
                all_as.append(a)

                # backward
                # output layer:
                d2 = self.cost_derivative(all_as[-1], y) # 10x1
                ad3 = self.activation_derivative(zs[-1]) # 10x1
                d2 = d2 * ad3 # 10x1
                dz2 = all_as[-2] # 30x1
                nabla_w2 = np.dot(d2, dz2.T) # 30x10
                nabla_b2 = d2 # 10x1

                # hidden layer:
                d1 = np.dot(w2.T, d2) * self.activation_derivative(zs[-2])
                dz1 = x
                nabla_w1 = np.dot(d1, dz1.T)
                nabla_b1 = d1

                # update weights/biases
                self.w[0] -= learning_rate * nabla_w1
                self.w[1] -= learning_rate * nabla_w2

                self.b[0] -= learning_rate * nabla_b1
                self.b[1] -= learning_rate * nabla_b2
                
            if test_data:
                print(f"Epoch {i}: {self.evaluate(test_data)} / 10000")


    def evaluate(self, test_data):
        test_results = [(np.argmax(self.predict(x)), y) for (x, y) in test_data]
        return sum(int(x==y) for x, y in test_results)

def preprocess_image(image_dict):
    image_data = image_dict['composite']
    image = Image.fromarray(image_data).convert('L')
    image = np.array(image).astype('float32') / 255
    image = image.reshape(784, 1)
    return image

def predict_digit(image_dict):
    preprocessed_image = preprocess_image(image_dict)
    prediction = net.predict(preprocessed_image)
    return int(np.argmax(prediction))

if __name__ == "__main__":
    train_data = pd.read_csv('../data/mnist_train.csv')
    test_data = pd.read_csv('../data/mnist_test.csv')

    x_train = train_data.iloc[:, 1:].values.astype('float32') / 255
    y_train = train_data.iloc[:, 0].values
    x_test = test_data.iloc[:, 1:].values.astype('float32') / 255
    y_test = test_data.iloc[:, 0].values

    y_train_one_hot = np.eye(10)[y_train]

    training_data = [(x.reshape(-1, 1), y.reshape(-1, 1)) for x, y in zip(x_train, y_train_one_hot)]
    test_data = [(x.reshape(-1, 1), y) for x, y in zip(x_test, y_test)]

    net = NN([784,30,10], sigmoid, sigmoid_prime, cost_derivative)
    net.train(training_data, 30, 3, test_data)
    demo = gr.Interface(
        fn=predict_digit,
        # inputs= gr.ImageEditor(sources=(), image_mode='L', crop_size=(28,28)),
        inputs= gr.ImageEditor(sources=(), crop_size=(28,28)),
        outputs=gr.Label(num_top_classes=3)
    )
    demo.launch()
