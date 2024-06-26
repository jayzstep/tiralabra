"""
Neuroverkkoluokka ja sen koulutuksessa tarvitsemat funktiot.
Tallentaa verkon painot ja biasit tiedostoon ja hakee ne sieltä.
"""

import pickle
import random

import matplotlib.pyplot as plt
import numpy as np


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
        seed: käytetään testauksessa apuna jäädyttämään satunnaisarvot.
        w_and_b_initializer:
            b: lista numpy arrayta, joka layerille alustetaan biasit
            w: lista numpy arrayta, joka layerille alustetaan painot
        + vastavirtaan tarvittavat funktiot
        epochs/success_rates: plottaamista varten tallennettuna x ja y akselien data

    Args:
        layers: listana layereiden neuronien määrät.
        activation: käytettävä aktivaatiofunktio
        activation_derivative: aktivaatiofunktion derivaatta
        cost_derivative: cost-funktion derivaatta
    """

    def __init__(self, layers, activation, activation_derivative, cost_derivative):
        self.seed = None
        self.w_and_b_initializer(layers)
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.cost_derivative = cost_derivative
        self.epochs = []
        self.success_rates = []

    def w_and_b_initializer(self, layers):
        """
        Alustaa neuroverkon biasit ja weightit satunnaisarvoilla.

        Args:
            layers: listana layereiden neuronien määrät
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        self.b = [np.random.randn(y, 1) * 0.1 for y in layers[1:]]
        self.w = [np.random.randn(y, x) * 0.1 for x, y in zip(layers[:-1], layers[1:])]

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

    def forward(self, x_batch):
        """
        Vie input datan verkon läpi. Tallentaa joka kerrokselta aktivoimattomat (z) ja
        aktivoidut (a) neuronien outputit.

        Args:
            x_batch: training datan sampleja matriisina.

        Returns:
            Kaksi listaa, joissa toisessa aktivoidut neuronit ja toisessa aktivoimattomat per
            kerros.
        """

        a = x_batch
        all_as = [x_batch]  # kerätään layereiden aktivaatiot listaan
        zs = []  # lista aktivoimattomista outputeista

        # forward propagation / viedään samplet verkon läpi
        # otetaan talteen relevantit vaiheet z ja a
        for w, b in zip(self.w, self.b):
            z = np.dot(w, a) + b
            zs.append(z)
            a = self.activation(z)
            all_as.append(a)

        return all_as, zs

    def train(self, training_data, epochs, learning_rate, batch_size, test_data=None):
        """
        Kouluttaa neuroverkon. Muodostaa sampleista mini batch -matriiseja, jotka raahataan
        ensin verkossa eteenpäin. Tästä otetaan talteen vastavirran tarvitsemat arvot.
        Sen jälkeen eri kerrosten vaiheille lasketaan gradientit lopusta alkuun (tämä on se
        vastavirta-algoritmi), joiden avulla saadaan laskettua jokaiselle painolle ja biasille
        muutos nabla-muuttujiin. Tämä muutos lisätään (tai vähennetään) nykyisiin weight/bias
        arvoihin, näin verkko oppii.

        Args:
            training_data: Lista tupleja (x,y) joissa x yksi sample ja y toivottu lopputulos.
            epochs: kuinka monta kierrosta treenidata käydään läpi kouluttaessa.
            learning_rate: kerroin painojen ja biasin korjaamiselle kohti derivaatan osoittamaa
                suuntaa.
            batch_size: määrittelee kuinka monta samplea kerrallaan käytetään
                kouluttamiseen.
            test_data: Voi antaa testidatan (samanlainen kuin training_data)
                jos haluaa mitata koulutuksen sujumista.
        """
        # testausta varten jäädytetään satunnaisarvot:
        if self.seed is not None:
            np.random.seed(self.seed)

        training_data = list(training_data)

        # Koulutusvaihe
        for i in range(epochs):
            random.shuffle(training_data)

            mini_batches = [
                training_data[k : k + batch_size]
                for k in range(0, len(training_data), batch_size)
            ]

            for mini_batch in mini_batches:
                # muunnetaan x- ja y-batchit matriiseiksi:
                x_batch = np.hstack([x for x, _ in mini_batch])
                y_batch = np.hstack([y for _, y in mini_batch])

                # Otetaan talteen muuttujia vastavirtaa varten
                w2 = self.w[1]
                all_as, zs = self.forward(x_batch)

                # backward propagation / vastavirta-algoritmi. Lasketaan gradientit painoille
                # ja biaseille ja tallennetaan ne nabla-muuttujiin

                # output layer:
                d2 = self.cost_derivative(all_as[-1], y_batch)
                ad3 = self.activation_derivative(zs[-1])
                d2 = d2 * ad3
                dz2 = all_as[-2]
                nabla_w2 = np.dot(d2, dz2.T) / batch_size
                nabla_b2 = np.sum(d2, axis=1, keepdims=True) / batch_size

                # hidden layer:
                d1 = np.dot(w2.T, d2) * self.activation_derivative(zs[-2])  #
                dz1 = x_batch
                nabla_w1 = np.dot(d1, dz1.T) / batch_size
                nabla_b1 = np.sum(d1, axis=1, keepdims=True) / batch_size

                # päivitetään kerrosten painot ja biasit
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
        correct = sum(int(x == y) for x, y in test_results)
        success_rate = correct / len(test_data)

        self.epochs.append(len(self.epochs) + 1)
        self.success_rates.append(success_rate)

        return correct

    def plot(self):
        """
        Plottaa onnistumisprosentin eepokeittain.
        """

        plt.plot(self.epochs, self.success_rates, marker="o")
        plt.xlabel("Epochs")
        plt.ylabel("Success Rate")
        plt.title("Learning Process")
        plt.grid(True)
        plt.show()


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
    On myös fiksu siltä osin, että luo juuri oikean kokoisen verkon tsekkaamalla datasta hidden
    layerille oikean koon.

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
