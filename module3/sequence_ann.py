# Module 3 Critical Thinking - Hand-Made Shallow ANN in Python
#
# Usage: sequence_ann.py [-h] [-n EPOCHS] [-w WINDOW] [-l LR] [-z HIDDEN] [-s SEED]
#
# Use a 2-layer ANN to predict the next number in a sequence.
#
# Options:
#   -h, --help           show this help message and exit
#   -n, --epochs EPOCHS  Number of training epochs (default: 10,000)
#   -w, --window WINDOW  Window size (default: 3)
#   -l, --lr LR          Learning rate (default: 0.1)
#   -z, --hidden HIDDEN  Size of hidden layer (default: 10)
#   -s, --seed SEED      Random seed
#
# The script prompts the user to enter a sequence of numbers with a constant difference, e.g.:  1.1 1.2 1.3 1.4 1.5 1.6
# It will then train a 2-layer ANN with EPOCHS epochs to predict the next value.
# WINDOW (default 3) is the window size used for training and prediction.
# For example, a sequence of [1,2,3,4,5] with WINDOW=3 will train the ANN with [[1,2,3], [2,3,4]] -> [[4], [5]],
# then print the predicted next value for [3,4,5], which should be approximately 6.
# Must enter at least WINDOW + 3 values (for at least 3 input windows mapped to the next value).
import argparse
import re
from typing import Tuple, Callable

from numpy import random, array, mean, exp, ndarray, zeros

default_epochs = 10_000
default_window_size = 3
default_seed = 42
default_learning_rate = 0.1
default_hidden = 8
min_windows = 3  # minimum number of training windows


class SequenceANN:
    """
    2-layer ANN to predict next number in numeric sequence.

    Input layer: n_input neurons
    Hidden layer: n_hidden neurons (sigmoid)
    Output layer: 1 neuron (predicted output)

    Weights:
    w2 : (n_hidden, 1)
    b1 : (1, n_hidden)
    w1 : (n_input, n_hidden)
    b2 : (1, 1)
    """

    a1: ndarray  # (m, n_hidden)
    z1: ndarray  # (m, n_hidden)
    y_hat: ndarray  # (m, 1)

    # noinspection PyTypeChecker
    def __init__(self, num_inputs: int, num_hidden: int, learning_rate: float):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.lr = learning_rate

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
        s = SequenceANN.sigmoid(x)
        return s * (1.0 - s)


def normalize(arr: ndarray) -> Tuple[ndarray, Callable[[ndarray], ndarray]]:
    """
    Normalize an array to have values between 0 and 1, inclusive.
    :param arr: array to normalize
    :return: normalized array and inverse function
    """
    lo, hi = arr.min(), arr.max()
    span = hi - lo if hi != lo else 1.0
    normalized = (arr - lo) / span

    # inverse transform
    def inverse(n: ndarray):
        return n * span + lo

    return normalized, inverse


def build_training_data(seq: ndarray, window_size: int) -> Tuple[ndarray, ndarray]:
    """
    Builds training data over given window size.
    :param seq: sequence
    :param window_size: window size
    :return: tuple of x and y data
    """
    x_values = []
    y_values = []
    for i in range(len(seq) - window_size):
        x_values.append(seq[i:i + window_size])
        y_values.append([seq[i + window_size]])
    return array(x_values), array(y_values)


def main():
    def positive_int(value):
        i = int(value)
        if i <= 0:
            raise argparse.ArgumentTypeError(f'{value} is an invalid positive int value')
        return i

    parser = argparse.ArgumentParser(description='Use a 2-layer ANN to predict the next number in a sequence.')
    parser.add_argument('-n', '--epochs', type=positive_int, default=default_epochs,
                        help=f'Number of training epochs (default: {default_epochs:,})')
    parser.add_argument('-w', '--window', type=positive_int, default=default_window_size,
                        help=f'Window size (default: {default_window_size})')
    parser.add_argument('-l', '--lr', type=float, default=default_learning_rate,
                        help=f'Learning rate (default: {default_learning_rate})')
    parser.add_argument('-z', '--hidden', type=positive_int, default=default_hidden,
                        help=f'Size of hidden layer (default: {default_hidden})')
    parser.add_argument('-s', '--seed', type=positive_int, default=default_seed, help='Random seed')
    args = parser.parse_args()

    min_seq = args.window + min_windows

    print(f'Enter an arithmetic sequence (at least {min_seq} values, space or comma-separated, constant step).')
    print('Example: 1.2 1.4 1.6 1.8 2.0 2.2 (expected output ~2.4)')

    def usage():
        print(f'ERROR: Please enter at least {min_seq} comma/space-separated numbers.')

    while True:
        raw = input("Enter sequence: ")
        try:
            seq = array([float(s) for s in re.split('[, \t]+', raw)])
            if len(seq) < min_seq:
                usage()
            else:
                break
        except ValueError:
            usage()

    # normalize sequence to [0, 1] space
    seq_norm, inverse = normalize(seq)
    x, y = build_training_data(seq_norm, args.window)

    random.seed(args.seed)

    net = SequenceANN(args.window, args.hidden, args.lr)
    net.train(x, y, epochs=args.epochs, print_every=args.epochs // 10)

    # predict next value from final window
    test_x = seq_norm[len(seq) - args.window:]
    next_pred = inverse(net.feedforward(test_x))[0, 0]
    print(f'Predicted next value: {next_pred:.6f}')


if __name__ == "__main__":
    main()
