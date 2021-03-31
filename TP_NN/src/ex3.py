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

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(DIMENSION,)))
    model.add(tf.keras.layers.Dense(10, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(CLASSES, activation=tf.keras.activations.sigmoid))

    opt = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
    # opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=opt,
                  loss=tf.losses.mean_squared_error,
                  metrics=[tf.keras.metrics.categorical_accuracy])
    model.fit(x_train, y_train,
              batch_size=8,
              epochs=EPOCH,
              validation_data=(x_val, y_val))
    print(f"Test accuracy: {model.evaluate(x_test, y_test)[1] * 100:.3f}%")

    plot_contour_tf(model, x, y)


if __name__ == '__main__':
    ex3()
