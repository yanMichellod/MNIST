from MNISTDatabase import loadMNISTDatabase
import numpy as np

def test_load_data():
        x_train, y_train, x_test, y_test = loadMNISTDatabase()
        # Test shape of the arrays
        np.testing.assert_equal(x_train.shape, (60000, 28, 28))
        np.testing.assert_equal(len(y_train), 60000)
        np.testing.assert_equal(x_test.shape, (10000, 28, 28))
        np.testing.assert_equal(len(y_test), 10000)