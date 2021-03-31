import numpy as np


class NeuralNetwork:
    def __init__(self, shape, learning_rate=1e-2):
        self.size = len(shape)
        self.shape = shape
        self.l_r = learning_rate
        self.biases = []
        self.weights = []
        for prev_layer, layer in zip(self.shape[:-1], self.shape[1:]):
            b = np.random.randn(layer, 1)
            self.biases.append(b)
            w = np.random.randn(layer, prev_layer)
            self.weights.append(w)

    def train(self, x, y):
        y_pred = self.forward(x)
        # TODO Fill me

        return y_pred

    def forward(self, a):
        self.zs = []
        self.activations = [a]
        # TODO Fill me

        return a

    def backprop(self, x, y):
        gradient_bias = [np.zeros(b.shape) for b in self.biases]
        gradient_weights = [np.zeros(w.shape) for w in self.weights]

        # last layer
        delta = cost_derivative(self.activations[-1], y) * sigmoid_derivative(
            self.zs[-1]
        )
        gradient_bias[-1] = delta
        delta_w = np.dot(delta, self.activations[-2].T)
        gradient_weights[-1] = delta_w

        # other layers:
        # TODO: Fill me

        return gradient_bias, gradient_weights

    def update(self, nabla_b, nabla_w):
        self.weights = [w - self.l_r * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - self.l_r * nb for b, nb in zip(self.biases, nabla_b)]

    def evaluate(self, x, y):
        test_results = [
            (np.argmax(self.forward(_x)), np.argmax(_y)) for _x, _y in zip(x, y)
        ]
        result = sum(int(_y_pred == _y) for (_y_pred, _y) in test_results)
        result /= len(x)
        return round(result, 3)


def cost(a, y):
    return (a - y) ** 2


def cost_derivative(a, y):
    return 2 * (a - y)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))
