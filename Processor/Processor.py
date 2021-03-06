# Inspired by https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/
# and https://www.kaggle.com/ashwani07/mnist-classification-using-random-forest

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

from Preprocessing import Preprocessing

tf.random.set_seed(42)
import random

random.seed(42)


def get_baseline(pp):
    """Run the random forest baseline and calculate accuracy

    Parameters
    ==========
    pp : Preprocessor
        Used to get the original MNIST data

    Returns
    =======
    accuracy: boolean
        The accuracy of the RF baseline
    """
    # get MNIST data
    x_train, y_train, x_test, y_test = (
        pp.getMNISTTrainData(),
        pp.getMNISTTrainLabel(),
        pp.getMNISTTestData(),
        pp.getMNISTTestLabel(),
    )
    # define Random forest and fit it
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(x_train.reshape(x_train.shape[0], 28 * 28), y_train)
    # run prediction and return accuracy
    pred = rf.predict(x_test.reshape(x_test.shape[0], 28 * 28))
    return accuracy_score(y_test, pred)


def define_model():
    """Define the sequential layers of the CNN model

    Returns
    =======
    model: Sequential keras model
        The CNN model
    """
    model = Sequential()
    model.add(
        Conv2D(
            32,
            (3, 3),
            activation="relu",
            kernel_initializer="he_uniform",
            input_shape=(28, 28, 1),
        )
    )
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dense(10, activation="softmax"))
    # compile model
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def runProcessor(full=True):
    """Run the CNN model and the RF and determines predictions and accuracies

    Parameters
    ==========
    full : boolean
        Default value is True
        Determines if all the MNIST records should be considered or only
        a subset shall be used for testing.

    Returns
    =======
    acc_baseline: float
        The accuracy of the RF baseline
    acc_CNN: float
        The accuracy of the CNN model
    y_test: numpy.ndarray
        The ground truth test labels
    y_pred: numpy.ndarray
        The predicted test labels
    """
    pp = Preprocessing.Preprocessing(full)
    # Get RF baseline
    acc_baseline = get_baseline(pp)
    # get preprocessed data
    x_train, y_train, x_test, y_test = (
        pp.getMNISTPreprocessedTrainData(),
        pp.getMNISTPreprocessedTrainLabel(),
        pp.getMNISTPreprocessedTestData(),
        pp.getMNISTPreprocessedTestLabel(),
    )
    # define and train the CNN model
    model = define_model()
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)
    # evaluate model on test dataset
    _, acc_CNN = model.evaluate(x_test, y_test, verbose=0)
    y_pred = model.predict(x_test)
    return acc_baseline, acc_CNN, y_test, y_pred
