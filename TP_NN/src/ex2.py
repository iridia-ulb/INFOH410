from utils import create_dataset, plot_contour
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm
from nn import NeuralNetwork

# number of cycles through the full training dataset
EPOCH = 200

# dataset generation
CLASSES = 3
DIMENSION = 2  # set to 2 for visualisation
POINTS = 1000

# TRAIN VAL TEST SPLIT
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# create a 4 layer NN with 10 neurons per hidden layer
NN_SHAPE = [DIMENSION, 10, 10, 10, CLASSES]
LEARNING_RATE = 1e-2


def ex2():
    x, y, x_train, x_val, x_test, y_train, y_val, y_test = generate_dateset()

    nn = NeuralNetwork(NN_SHAPE, LEARNING_RATE)
    pbar = tqdm(range(EPOCH))
    for e in pbar:
        shuffle(x_train, y_train)
        for _x, _y in zip(x_train, y_train):
            nn.train(_x, _y)

        # compute train and validation accuracy
        train_accurracy = nn.evaluate(x_train, y_train)
        val_accuracy = nn.evaluate(x_val, y_val)
        pbar.set_description(
            f"Epoch {e:03}/{EPOCH} - Train {train_accurracy:.3f}% - Test {val_accuracy:.3f}% "
        )

    # compute test accuracy
    test_accuracy = nn.evaluate(x_test, y_test)
    print(f"Test {test_accuracy:.3f}%")

    # plot NN borders
    plot_contour(nn, x, y)


def generate_dateset():
    x, y = create_dataset(N=POINTS, K=CLASSES, D=DIMENSION)
    x = np.expand_dims(x, -1)
    # transform data to categorical
    y = tf.keras.utils.to_categorical(y, num_classes=CLASSES)
    y = np.expand_dims(y, -1)

    # automatically split dataset in train/test with 20% as test
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SPLIT, random_state=42, shuffle=True
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=VAL_SPLIT / (1 - TEST_SPLIT),
        random_state=42,
        shuffle=True,
    )

    return x, y, x_train, x_val, x_test, y_train, y_val, y_test


if __name__ == "__main__":
    ex2()
