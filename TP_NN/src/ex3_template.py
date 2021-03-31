from utils import create_dataset, plot_contour
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def main():
    EPOCH = 200
    POINTS = 1000
    DIMENSION = 2
    CLASSES = 3

    x, y = create_dataset(N=POINTS, K=CLASSES, D=DIMENSION)
    y = tf.keras.utils.to_categorical(y, num_classes=CLASSES)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, shuffle=True
    )

    # create, train and evaluate model using keras
    # TODO Fill me

    plot_contour_tf(model, x, y)


def plot_contour_tf(nn, x, y):
    # plot the resulting classifier
    h = 0.02
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    points = np.c_[xx.ravel(), yy.ravel()]
    print(points)

    # forward prop with our trained parameters & classify into highest prob
    Z = np.argmax(nn.predict(points), axis=1)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)

    # plt the points
    y = np.argmax(y, axis=1)
    plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()
    # fig.savefig('spiral_net.png')


if __name__ == "__main__":
    main()
