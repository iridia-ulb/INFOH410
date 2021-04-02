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
        nabla_b, nabla_w = self.backprop(x, y)
        self.update(nabla_b, nabla_w)
        return y_pred

    def forward(self, a):
        self.zs = []
        self.activations = [a]
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            self.zs.append(z)
            a = sigmoid(z)
            self.activations.append(a)
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

        # from before last layer to first layer
        # last layer is self.size-2
        # before last layer is self.size-3
        for l in range(self.size - 3, -1, -1):
            delta = np.dot(self.weights[l + 1].T, delta) * sigmoid_derivative(
                self.zs[l]
            )
            gradient_bias[l] = delta
            # len(activation) == len(weights)+1
            # activation[i] is the previous activations to the layer weights[i]
            delta_w = np.dot(delta, self.activations[l].T)
            gradient_weights[l] = delta_w

        return gradient_bias, gradient_weights

    def update(self, nabla_b, nabla_w):
        self.weights = [w - self.l_r * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - self.l_r * nb for b, nb in zip(self.biases, nabla_b)]

    def train_sgd(self, dataset_x, dataset_y, batch_size=8):
        dataset_x_batches = [
            dataset_x[i : i + batch_size] for i in range(0, len(dataset_x), batch_size)
        ]
        dataset_y_batches = [
            dataset_y[i : i + batch_size] for i in range(0, len(dataset_y), batch_size)
        ]

        for x_batch, y_batch in zip(dataset_x_batches, dataset_y_batches):
            gradient_bias = [np.zeros(b.shape) for b in self.biases]
            gradient_weights = [np.zeros(w.shape) for w in self.weights]
            for x, y in zip(x_batch, y_batch):
                y_pred = self.forward(x)
                delta_grad_b, delta_grad_w = self.backprop(x, y)
                gradient_bias = [
                    nb + dnb for nb, dnb in zip(gradient_bias, delta_grad_b)
                ]
                gradient_weights = [
                    nw + dnw for nw, dnw in zip(gradient_weights, delta_grad_w)
                ]
            gradient_weights = [nw / batch_size for nw in gradient_weights]
            gradient_bias = [nb / batch_size for nb in gradient_bias]
            self.weights = [
                w - self.l_r * nw for w, nw in zip(self.weights, gradient_weights)
            ]
            self.biases = [
                b - self.l_r * nb for b, nb in zip(self.biases, gradient_bias)
            ]

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
