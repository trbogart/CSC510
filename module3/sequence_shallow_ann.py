# Module 3 Critical Thinking - Hand-Made Shallow ANN in Python
#
# A 2-layer ANN (1 hidden layer + 1 output layer) trained with backpropagation.
# Predicts the next number in a linear arithmetic sequence.
# Example: given [2, 4, 6, 8], predict 10.
#
# Usage:
#   python shallow_ann.py
# The script prompts the user to enter a sequence of numbers with a constant difference, e.g.:  1.1 1.2 1.3 1.4 1.5 1.6
# It will then predict the next value.
# Default window size for training and prediction is 4.
# Must enter at least window size + 2 values (for at least 2 input windows mapped to the next value)
# Default number of training epochs is 10,000.
import argparse
import re
from typing import Tuple, Callable

from numpy import random, array, mean, exp, ndarray, zeros

default_epochs = 10_000
default_window_size = 4
default_seed = 42


class SequenceShallowANN:
    """
    Shallow ANN to predict next number in numeric sequence.

    Input layer: n_input neurons
    Hidden layer: n_hidden neurons (sigmoid)
    Output layer: 1 neuron (predicted output)

    Weights:
    w2 : (n_hidden, 1)
    b1 : (1, n_hidden)
    w1 : (n_input, n_hidden)
    b2 : (1, 1)
    """

    a1: ndarray
    z1: ndarray
    y_hat: ndarray

    # noinspection PyTypeChecker
    def __init__(self, num_inputs: int, num_hidden: int = 8, learning_rate: float = 0.1):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.lr = learning_rate

        # approximate He initialization for weights
        self.w1: ndarray = random.randn(num_inputs, num_hidden) * 0.5
        self.w2: ndarray = random.randn(num_hidden, 1) * 0.5
        self.b1 = zeros((1, num_hidden))
        self.b2 = zeros((1, 1))

    def feedforward(self, x: ndarray) -> ndarray:
        """
        Forward pass.
        :param x: input data (m, n_input)
        :return: predicted output (m, 1)
        """
        # hidden layer
        self.z1 = x @ self.w1 + self.b1  # (m, n_hidden)
        self.a1 = self.sigmoid(self.z1)  # (m, n_hidden)

        # predicted output
        self.y_hat = self.a1 @ self.w2 + self.b2  # (m, 1)

        return self.y_hat

    def backpropagate(self, x: ndarray, y: ndarray):
        """
        Backward propagation (gradient descent).
        :param x: input value
        :param y: expected value
        """
        m = x.shape[0]

        # Output layer gradient
        y_error = self.y_hat - y  # (m, 1)
        dw2 = (self.a1.T @ y_error) / m  # (n_hidden, 1)
        db2 = mean(y_error, axis=0, keepdims=True)

        # Hidden layer gradient
        da1 = y_error @ self.w2.T  # (m, n_hidden)
        dz1 = da1 * self.sigmoid_derivative(self.z1)  # (m, n_hidden)
        dw1 = (x.T @ dz1) / m  # (n_input, n_hidden)
        db1 = mean(dz1, axis=0, keepdims=True)

        # Update weights and biases
        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2
        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1

    def train(self, x: ndarray, y: ndarray, epochs: int = 10_000, print_every: int | None = None):
        print(f'Training for {epochs:,} epochs...')
        for epoch in range(1, epochs + 1):
            self.feedforward(x)
            self.backpropagate(x, y)
            if print_every and epoch % print_every == 0:
                loss = float(mean((y - self.y_hat) ** 2))
                print(f'- Epoch {epoch:>7,} (loss: {loss:.6f})')

    # Activation function (sigmoid) and its derivative
    @staticmethod
    def sigmoid(x: ndarray) -> ndarray:
        return 1.0 / (1.0 + exp(-x))

    @staticmethod
    def sigmoid_derivative(x: ndarray) -> ndarray:
        s = SequenceShallowANN.sigmoid(x)
        return s * (1.0 - s)


def normalize(arr: ndarray) -> Tuple[ndarray, Callable[[ndarray], ndarray]]:
    """
    Normalize an array with values between 0 and 1.
    :param arr: array to normalize
    :return: normalized array and inverse function
    """
    lo, hi = arr.min(), arr.max()
    span = hi - lo if hi != lo else 1.0
    normalized = (arr - lo) / span

    # inverse transform
    def inverse(a: ndarray):
        return a * span + lo

    return normalized, inverse


def build_training_data(seq: ndarray, window_size: int) -> Tuple[ndarray, ndarray]:
    """
    Builds training data over window size of 4.
    :param seq: sequence
    :param window_size: window size
    :return:
    """
    x_values = []
    y_values = []
    for i in range(len(seq) - window_size):
        x_values.append(seq[i:i + window_size])
        y_values.append([seq[i + window_size]])
    return array(x_values), array(y_values)


def main():
    min_windows = 2  # minimum number of training windows

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--epochs', type=int, default=default_epochs,
                        help=f'Number of training epochs (default: {default_epochs})')
    parser.add_argument('-w', '--window', type=int, default=default_window_size,
                        help=f'Training window size (default: {default_window_size})')
    parser.add_argument('-s', '--seed', type=int, default=default_seed, help='Random seed')
    args = parser.parse_args()

    min_seq = args.window + min_windows

    print(f'Enter an arithmetic sequence (at least {min_seq} values, space or comma-separated, constant step).')
    print('Example: 1.2 1.4 1.6 1.8 2.0 2.2 (expected output ~2.4)')

    def usage():
        print(f'ERROR: Please enter at least {min_seq} comma/space-separated numbers.')

    while True:
        raw = input("Enter sequence: ")
        try:
            seq = array([float(s.strip()) for s in re.split('[, \t]', raw)])
            if len(seq) < min_seq:
                usage()
            else:
                break
        except ValueError:
            usage()

    # normalize sequence to [0, 1] space
    seq_norm, inverse = normalize(seq)
    x, y = build_training_data(seq_norm, args.window)

    assert x.shape[1] == args.window
    assert x.shape[0] == y.shape[0]
    random.seed(args.seed)

    net = SequenceShallowANN(args.window)
    net.train(x, y, epochs=args.epochs, print_every=args.epochs // 10)

    # predict next value from final window
    test_x = seq_norm[len(seq) - args.window:]
    next_pred = inverse(net.feedforward(test_x))[0, 0]
    print(f'Predicted next value: {next_pred:.6f}')


if __name__ == "__main__":
    main()
