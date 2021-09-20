from Database.MNISTDatabase import loadMNISTDatabase, loadFromLocalFile
import numpy as np
import gzip
import struct
import os

def test_train_data_path():
    """
    Test if the path to the train data exist
    """
    print(os.getcwd())
    print(os.listdir('../'))
    assert os.path.exists('../Database/data/train-images-idx3-ubyte.gz')

def test_train_label_path():
    """
    Test if the path to the train label exist
    """
    assert os.path.exists('../Database/data/train-labels-idx1-ubyte.gz')

def test_test_data_path():
    """
    Test if the path to the test data exist
    """
    assert os.path.exists('../Database/data/t10k-images-idx3-ubyte.gz')

def test_test_label_path():
    """
    Test if the path to the test label exist
    """
    assert os.path.exists('../Database/data/t10k-labels-idx1-ubyte.gz')

def test_train_data_not_corrupted():
    """
    Test if the train data are not corrupted
    """
    with gzip.open("../Database/data/train-images-idx3-ubyte.gz", 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
    assert magic == 2051

def test_train_label_not_corrupted():
    """
    Test if the train labels are not corrupted
    """
    with gzip.open("../Database/data/train-labels-idx1-ubyte.gz", 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
    assert magic == 2049

def test_test_data_not_corrupted():
    """
    Test if the test data are not corrupted
    """
    with gzip.open("../Database/data/t10k-images-idx3-ubyte.gz", 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
    assert magic == 2051

def test_test_label_not_corrupted():
    """
    Test if the test label are not corrupted
    """
    with gzip.open("../Database/data/t10k-labels-idx1-ubyte.gz", 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
    assert magic == 2049

def test_load_data_shape():
    x_train, y_train, x_test, y_test = loadMNISTDatabase()
    # Test shape of the arrays
    np.testing.assert_equal(x_train.shape, (60000, 28, 28))
    np.testing.assert_equal(len(y_train), 60000)
    np.testing.assert_equal(x_test.shape, (10000, 28, 28))
    np.testing.assert_equal(len(y_test), 10000)