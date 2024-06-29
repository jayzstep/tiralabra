import argparse

import gradio as gr
import numpy as np
import pandas as pd
from nn import (NN, cost_derivative, load_weights_and_biases,
                save_weights_and_biases, sigmoid, sigmoid_prime)


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
    try:
        image = image.reshape(784, 1)
    except:
        return ""
    prediction = net.predict(image)
    return int(np.argmax(prediction))


def load_csv():
    train_data = pd.read_csv("../data/mnist_train.csv")
    test_data = pd.read_csv("../data/mnist_test.csv")

    x_train = train_data.iloc[:, 1:].values.astype("float32") / 255
    y_train = train_data.iloc[:, 0].values
    x_test = test_data.iloc[:, 1:].values.astype("float32") / 255
    y_test = test_data.iloc[:, 0].values

    y_train_one_hot = np.eye(10)[y_train]

    training_data = [
        (x.reshape(-1, 1), y.reshape(-1, 1)) for x, y in zip(x_train, y_train_one_hot)
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
        net = NN([784, 60, 10], sigmoid, sigmoid_prime, cost_derivative)
        net.train(training_data, 30, 0.3, 10, test_data)
        save_weights_and_biases(net, "weights_and_biases.pkl")
        net.plot()

    elif args.mode == "run":
        net = load_weights_and_biases("weights_and_biases.pkl")

    examples = [f"../data/testSample/img_{i}.jpg" for i in range(1, 350)]
    gr.Interface(
        fn=predict_digit,
        inputs=gr.Image(type="numpy", image_mode="L", sources="upload"),
        outputs=gr.Label(num_top_classes=1),
        examples=examples,
        examples_per_page=50,
        live=True,
        title="NumNet",
        description="A Neural Network trained to classify hand written digits. Click a number to present it to the net.",
        article="",
        allow_flagging="never",
    ).launch()
