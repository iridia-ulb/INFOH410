from utils import create_dataset, plot_contour
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm
from nn import NeuralNetwork


def main():
    EPOCH = 200
    POINTS = 1000
    DIMENSION = 2
    CLASSES = 3
    # create a 4 layer NN with 10 neurons per hidden layer
    NN_SHAPE = [DIMENSION, 10, 10, 10, CLASSES]

    x, y = create_dataset(N=POINTS, K=CLASSES, D=DIMENSION)
    x = np.expand_dims(x, -1)
    # transform data to categorical
    y = tf.keras.utils.to_categorical(y, num_classes=CLASSES)
    y = np.expand_dims(y, -1)

    # automatically split dataset in train/test with 20% as test
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, shuffle=True
    )

    nn = NeuralNetwork(NN_SHAPE)
    pbar = tqdm(range(EPOCH))
    for e in pbar:
        shuffle(x_train, y_train)
        for _x, _y in zip(x_train, y_train):
            nn.train(_x, _y)

        # compute train and validation accuracy
        train_accurracy = nn.evaluate(x_train, y_train)
        test_accuracy = nn.evaluate(x_test, y_test)
        pbar.set_description(
            f"Epoch {e:03}/{EPOCH} - Train {train_accurracy:.3f}% - Test {test_accuracy:.3f}% "
        )

    # compute train and validation accuracy
    train_accurracy = nn.evaluate(x_train, y_train)
    test_accuracy = nn.evaluate(x_test, y_test)
    print(f"Train {train_accurracy:.3f}% - Test {test_accuracy}%")

    plot_contour(nn, x, y)


if __name__ == "__main__":
    main()
