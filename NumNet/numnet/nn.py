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
    Softmax-funktio output-layerille, tätä ei tosin käytetä, mutta jatkossa -- kuka tietää!

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
        b: lista numpy arrayta, joka layerille alustetaan biasit
        w: lista numpy arrayta, joka layerille alustetaan painot
        + tarvittavat funktiot

    Args:
        layers: listana layereiden neuronien määrät.
        activation: käytettävä aktivaatiofunktio
        activation_derivative: aktivaatiofunktion derivaatta
        cost_derivative: cost-funktion derivaatta
    """

    def __init__(self, layers, activation, activation_derivative, cost_derivative):
        self.b = [np.random.randn(y, 1) * 0.1 for y in layers[1:]]
        self.w = [np.random.randn(y, x) * 0.1 for x, y in zip(layers[:-1], layers[1:])]
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.cost_derivative = cost_derivative

    def predict(self, a):
        """
        Tulkitsee kuvasta mikä luku piirrettynä.

        Args:
            a: samplekuva (784, 1) kokoisena nunpy arrayna 

        Returns:
            output layerin tulos vektorina.

        """
        if len(a) != 784:
            raise ValueError("Wrong input size.")
        for b, w in zip(self.b, self.w):
            a = self.activation(np.dot(w, a) + b)
        return a

    def train(self, training_data, epochs, learning_rate, batch_size, test_data=None):
        """
        Kouluttaa neuroverkon. Muodostaa sampleista mini batch -matriiseja, jotka raahataan ensin verkossa eteenpäin. 
        Tästä otetaan talteen vastavirran tarvitsemat arvot. Sen jälkeen eri kerrosten vaiheille lasketaan gradientit 
        lopusta alkuun (tämä on se vastavirta-algoritmi), joiden avulla saadaan laskettua jokaiselle painolle ja biasille 
        muutos nabla-muuttujiin. Tämä muutos lisätään nykyisiin arvoihin, näin verkko oppii.

        Args:
            training_data: Lista tupleja (x,y) joissa x yksi sample ja y toivottu lopputulos.
            epochs: kuinka monta rundia muhotaan treenidata läpi.
            learning_rate: kerroin painojen ja biasin korjaamiselle kohti derivaatan osoittamaa suuntaa.
            test_data: Voi antaa testidatan (samanlainen kuin training_data) jos haluaa mitata koulutuksen sujumista.
        """
        training_data = list(training_data)
        for i in range(epochs):

            random.shuffle(training_data) # sekoittaa training_datan järjestyksen joka eepokille
            mini_batches = [training_data[k:k + batch_size] for k in range(0, len(training_data), batch_size)] # pilkkoo training_datan mini_batcheiksi, operoidaan siis monta samplea kerrallaan
            for mini_batch in mini_batches:
                x_batch = np.hstack([x for x, _ in mini_batch]) # muunnetaan matriisiksi
                y_batch = np.hstack([y for _, y in mini_batch])

                a = x_batch 
                all_as = [x_batch] # kerätään layereiden aktivaatiot listaan
                zs = [] # lista aktivoimattomista outputeista
                w2 = self.w[1] # hidden layerin weightit talteen jatkoa varten

                # forward propagation / viedään samplet verkon läpi, otetaan talteen relevantit vaiheet z ja a
                for w, b in zip(self.w, self.b):
                    z = np.dot(w, a) + b
                    zs.append(z)
                    a = self.activation(z)
                    all_as.append(a)

                # backward propagation / vastavirta-algoritmi. Lasketaan gradientit painoille ja biaseille
                # output layerin:
                d2 = self.cost_derivative(all_as[-1], y_batch) 
                ad3 = self.activation_derivative(zs[-1])  
                d2 = d2 * ad3
                dz2 = all_as[-2]
                nabla_w2 = np.dot(d2, dz2.T) / batch_size # gradientit talteen tälle kerrokselle
                nabla_b2 = np.sum(d2, axis=1, keepdims=True) / batch_size # sama painojen gradienteille

                # hidden layer:
                d1 = np.dot(w2.T, d2) * self.activation_derivative(zs[-2]) # 
                dz1 = x_batch
                nabla_w1 = np.dot(d1, dz1.T) / batch_size
                nabla_b1 = np.sum(d1, axis=1, keepdims=True) / batch_size

                # päivitetään joka kerroksen painot ja biasit
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


def predict_digit(digit):
    """
    Tulkitsee kuvan. Tai ainakin yrittää. argmax poimii output-vektorin korkeimman arvon indexin, 
    joka MNISTin tapauksessa on sopivasti sama kuin korkeimman ennusteen saanut numero.

    Args:
        digit: GUI:n lähettämä kuvadata, nimetty digit, koska se näkyy input-kentän otsikkona.

    Returns:
        Tulkittu numero
    """
    image = np.array(digit).astype("float32") / 255
    image = image.reshape(784, 1)
    prediction = net.predict(image)
    return int(np.argmax(prediction))


def save_weights_and_biases(model, filename):
    """
    Tallentaa painot ja biasit tiedostoon.

    Args:
        model: neuroverkko
        filename: tiedoston nimi johon tallennettaan
    """
    with open(filename, "wb") as file:
        pickle.dump((model.w, model.b), file)


def load_weights_and_biases(filename):
    """
    Lataa tallennetut painot ja biasit, näin verkkoa ei tarvitse joka kerta kouluttaa uudestaan. 
    On myös fiksu siltä osin, että luo juuri oikean kokoisen verkon tsekkaamalla datasta hidden layerille oikean koon.

    Args:
        filename: tiedosto jossa painot ja biasit sijaitsee

    Returns:
        Neuroverkko, jossa koulutetut painot ja biasit.
    """
    with open(filename, "rb") as file:
        w, b = pickle.load(file)
    hidden_layer_size = w[0].shape[0]
    model = NN([784, hidden_layer_size, 10], sigmoid, sigmoid_prime, cost_derivative)
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
        # Debuggausta:
        # training_data = training_data[0:10]
        # net = NN([784, 50, 10], sigmoid, sigmoid_prime, cost_derivative)
        # net.train(training_data, 1, 0.3, 10)

        # For real
        net = NN([784, 30, 10], sigmoid, sigmoid_prime, cost_derivative)
        net.train(training_data, 30, 0.3, 10, test_data)
        save_weights_and_biases(net, "weights_and_biases.pkl")

    elif args.mode == "run":
        net = load_weights_and_biases("weights_and_biases.pkl")

    examples = [f"../data/testSample/img_{i}.jpg" for i in range(1, 350)]
    gr.Interface(
        fn=predict_digit,
        inputs=gr.Image(type="numpy", image_mode="L"),
        outputs=gr.Label(num_top_classes=1),
        examples=examples,
        examples_per_page=50,
        live=True,
        title="NumNet",
        description="A Neural Network trained to classify hand written digits. Click a number to present it to the net.",
        article="",
        allow_flagging="never",
    ).launch()
