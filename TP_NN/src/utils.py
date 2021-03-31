import numpy as np
import matplotlib.pyplot as plt


def create_dataset(N=100, K=2, D=2):
    """
    :param N: number of points per class
    :param K: number of classes
    :param D: dimension
    :return:
    """
    X = np.zeros((N * K, D))  # data matrix (each row = single example)
    y = np.zeros(N * K)  # class labels

    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0, 1, N)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    # lets visualize the data:
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()

    return X, y


def plot_contour(nn, x, y):
    # plot the resulting classifier
    h = 0.02
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    points = np.c_[xx.ravel(), yy.ravel()]

    # forward prop with our trained parameters & classify into highest prob
    points = np.expand_dims(points, -1)
    Z = np.array([np.argmax(nn.forward(_x)) for _x in points])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)

    # plt the points
    y = np.argmax(y, axis=1)
    plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()
    # fig.savefig('spiral_net.png')
