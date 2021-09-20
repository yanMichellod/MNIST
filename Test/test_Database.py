from Database.MNISTDatabase import loadMNISTDatabase, loadFromLocalFile
import numpy as np
import os

def test_train_data_path():
    """
    Test if the path to the train data exist
    """
    assert os.path.isfile("data/train-images-idx3-ubyte.gz")

def test_train_label_path():
    """
    Test if the path to the train label exist
    """
    assert os.path.isfile("data/train-labels-idx1-ubyte.gz")

def test_test_data_path():
    """
    Test if the path to the test data exist
    """
    assert os.path.isfile("data/t10k-images-idx3-ubyte.gz")

def test_test_label_path():
    """
    Test if the path to the test label exist
    """
    assert os.path.isfile("data/t10k-labels-idx1-ubyte.gz")

def test_load_data():
    x_train, y_train, x_test, y_test = loadMNISTDatabase()
    # Test shape of the arrays
    np.testing.assert_equal(x_train.shape, (60000, 28, 28))
    np.testing.assert_equal(len(y_train), 60000)
    np.testing.assert_equal(x_test.shape, (10000, 28, 28))
    np.testing.assert_equal(len(y_test), 10000)