import argparse
import pickle
import random

import gradio as gr
import numpy as np
import pandas as pd


def cost_derivative(output_activations, y):
    """
    Laskee cost-funktion derivaatan.

    Args:
        output_activations: output layerin aktivaatiot, "outputti"
        y: mitä pitäisi olla, eli ns. oikeat vastaukset.

    Returns:
        cost funktion derivaatta.
    """
    return output_activations - y

def softmax(z):
    """
    Softmax output-layerille

    Args:
        z: output np array

    Returns:
        aktivoitu output layer
    """
    exp = np.exp(z - np.max(z))
    return exp / exp.sum(axis=0)

def sigmoid(z):
    """
    Vanha kunnon sigmoid.

    Args:
        z: aktivaatiota kaipaava vektori/numpy array

    Returns:
        aktivoitu inputti seuraavalle layerille.
    """
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """
    Sigmoidin derivaatta vastavirran laskemiseen.

    Args:
        z: aktivoimaton neuronin sisältö.

    Returns:
        sigmoidin derivaatta.
    """
    return sigmoid(z) * (1 - sigmoid(z))


class NN:
    """
    Neuroverkolle oma luokka.

    Attributes:
        b: lista numpy arrayta, joka layerille alustetut biasit
        w: lista numpy arrayta, joka layerille alustetut painot
        + tarvittavat funktiot

    Args:
        layers: listana layereiden neuronien määrät.
        activation: käytettävä aktivaatiofunktio
        activation_derivative: aktivaatiofunktion derivaatta
        cost_derivative: cost-funktion derivaatta
    """

    def __init__(self, layers, activation, activation_derivative, cost_derivative):
        self.b = [np.random.randn(y, 1) for y in layers[1:]]
        self.w = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.cost_derivative = cost_derivative

    def predict(self, a):
        """
        Tulkitsee kuvasta mikä luku piirrettynä.

        Args:
            a: inputtina tuleva numpy array (kuva)

        Returns:
            output layerin tulos.

        """
        if len(a) != 784:
            raise ValueError("Wrong input size.")
        for b, w in zip(self.b, self.w):
            a = self.activation(np.dot(w, a) + b)
        return a

    def train(self, training_data, epochs, learning_rate, test_data=None):
        """
        Kouluttaa neuroverkon.

        Args:
            training_data: Lista tupleja (x,y) joissa x yksi sample ja y toivottu lopputulos.
            epochs: kuinka monta rundia muhotaan treenidata läpi.
            learning_rate: kerroin painojen ja biasin korjaamiselle kohti derivaatan osoittamaa suuntaa.
            test_data: Voi antaa testidatan (samanlainen kuin training_data) jos haluaa mitata koulutuksen sujumista.
        """
        training_data = list(training_data)
        for i in range(epochs):

            random.shuffle(training_data)
            for sample in training_data:
                x, y = sample
                a = x
                all_as = [x]
                zs = []
                w2 = self.w[1]

                # forward
                for w, b in zip(self.w, self.b):
                    z = np.dot(w, a) + b
                    zs.append(z)
                    a = self.activation(z)
                    all_as.append(a)

                # backward
                # output layer:
                d2 = self.cost_derivative(all_as[-1], y)  # 10x1
                ad3 = self.activation_derivative(zs[-1])  # 10x1
                d2 = d2 * ad3  # 10x1
                dz2 = all_as[-2]  # 30x1
                nabla_w2 = np.dot(d2, dz2.T)  # 30x10
                nabla_b2 = d2  # 10x1

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
        """
        Arvioi neuroverkon kykyä lajitella numeroita, joita se ei ole nähnyt.

        Args:
            test_data: lista tupleja (x,y) jossa x on sample ja y on toivottu tulos.

        Returns:
            kuinka monta meni oikein 10 000:sta.
        """
        test_results = [(np.argmax(self.predict(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for x, y in test_results)


def predict_digit(raw_image):
    """
    Tulkitsee kuvan. Tai ainakin yrittää.

    Args:
        raw_image: GUI:n lähettämä kuvadata

    Returns:
        Tulkittu numero
    """
    image = np.array(raw_image).astype("float32") / 255
    image = image.reshape(784, 1)
    prediction = net.predict(image)
    return int(np.argmax(prediction))


def save_weights_and_biases(model, filename):
    """
    Tallenna painot ja biasit.

    Args:
        model: neuroverkko
        filename: tiedoston nimi johon tallennettaan
    """
    with open(filename, "wb") as file:
        pickle.dump((model.w, model.b), file)


def load_weights_and_biases(filename):
    """
    Lataa tallennetut painot ja biasit, niin ei tarvitse joka kerta kouluttaa uudestaan.

    Args:
        filename: tiedosto jossa painot ja biasit sijaitsee

    Returns:
        Neuroverkko, jossa koulutetut painot ja biasit.
    """
    with open(filename, "rb") as file:
        w, b = pickle.load(file)
    model = NN([784, 30, 10], sigmoid, sigmoid_prime, cost_derivative)
    model.w = w
    model.b = b
    return model

def load_csv():
        train_data = pd.read_csv("../data/mnist_train.csv")
        test_data = pd.read_csv("../data/mnist_test.csv")

        x_train = train_data.iloc[:, 1:].values.astype("float32") / 255
        y_train = train_data.iloc[:, 0].values
        x_test = test_data.iloc[:, 1:].values.astype("float32") / 255
        y_test = test_data.iloc[:, 0].values

        y_train_one_hot = np.eye(10)[y_train]

        training_data = [
            (x.reshape(-1, 1), y.reshape(-1, 1))
            for x, y in zip(x_train, y_train_one_hot)
        ]
        test_data = [(x.reshape(-1, 1), y) for x, y in zip(x_test, y_test)]

        return training_data, test_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose train or run")
    parser.add_argument(
        "mode",
        choices=["train", "run"],
        help="'train' to train and save weights/biases to memory and start the app, 'run' to load weights and biases from memory and start the app.",
    )
    args = parser.parse_args()

    if args.mode == "train":
        training_data, test_data = load_csv()
        net = NN([784, 50, 10], sigmoid, sigmoid_prime, cost_derivative)
        net.train(training_data, 30, 0.3, test_data)
        save_weights_and_biases(net, "weights_and_biases.pkl")

    elif args.mode == "run":
        net = load_weights_and_biases("weights_and_biases.pkl")

    examples = [f"../data/testSample/img_{i}.jpg" for i in range(1, 350)]
    gr.Interface(
        fn=predict_digit,
        inputs=gr.Image(type="numpy", image_mode="L"),
        outputs=gr.Label(num_top_classes=1),
        examples=examples,
    ).launch()
