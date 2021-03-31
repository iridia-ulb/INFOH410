from utils import create_dataset, plot_contour_tf
import tensorflow as tf
from sklearn.model_selection import train_test_split

# number of cycles through the full training dataset
EPOCH = 50

# dataset generation
CLASSES = 3
DIMENSION = 2  # set to 2 for visualisation
POINTS = 1000

# TRAIN VAL TEST SPLIT
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

LEARNING_RATE = 1e-2


def ex3():
    x, y = create_dataset(N=POINTS, K=CLASSES, D=DIMENSION)
    y = tf.keras.utils.to_categorical(y, num_classes=CLASSES)

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=TEST_SPLIT,
                                                        random_state=42,
                                                        shuffle=True)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                      test_size=VAL_SPLIT / (1 - TEST_SPLIT),
                                                      random_state=42,
                                                      shuffle=True)

    # create, train and evaluate model using tf.keras
    # TODO Fill me

    plot_contour_tf(model, x, y)


if __name__ == '__main__':
    ex3()
