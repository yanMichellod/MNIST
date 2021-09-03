
from keras.datasets import mnist
import numpy as np
import gzip
import struct
from array import array


def loadFromLocalFile(image_path, label_path):
    """Load the MNIST train and test datasets from local files.
        Code from : https://www.kaggle.com/hojjatk/read-mnist-dataset
        Attributes
        -------
        image_path : string path of the image dataset
        label_path : string path of the label dataset

        Returns
        -------
        x_train : uint8 NumPy array containing training data with shape (60000, 28, 28)
        y_train : uint8 NumPy array containing training digit labels with shape (60000,)
        x_test  : uint8 NumPy array containing test data with shape (10000, 28, 28)
        y_test  : uint8 NumPy array containing test digit labels with shape (10000,)
        """
    #Load label
    with gzip.open(label_path, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
        labels = np.array(array("B", file.read()))
    # Load label
    with gzip.open(image_path, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
        x_train = array("B", file.read())
    images = []
    for i in range(size):
        images.append([0] * rows * cols)
    for i in range(size):
        img = np.array(x_train[i * rows * cols:(i + 1) * rows * cols])
        img = img.reshape(28, 28)
        images[i][:] = img
    images = np.array(images)
    return images, labels


def loadMNISTDatabase():
    """Load the MNIST train and test datasets.
    Returns
    -------
    x_train : uint8 NumPy array containing training data with shape (60000, 28, 28)
    y_train : uint8 NumPy array containing training digit labels with shape (60000,) 
    x_test  : uint8 NumPy array containing test data with shape (10000, 28, 28)
    y_test  : uint8 NumPy array containing test digit labels with shape (10000,) 
    """
    x_train, y_train, x_test, y_test = [], [], [], []
    print('*************************************************')
    print('Load Data')
    print('')
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if len(x_train) == 0:
        print('!! Connection to Keras dataset failed !!') 
        print('Take data from project storage')
        x_train, y_train = loadFromLocalFile("data/train-images-idx3-ubyte.gz", "data/train-labels-idx1-ubyte.gz")
        x_test, y_test = loadFromLocalFile("data/t10k-images-idx3-ubyte.gz", "data/t10k-labels-idx1-ubyte.gz")
    else:
        print('Data comes from Keras data set')
    print('*************************************************')
    return x_train, y_train, x_test, y_test