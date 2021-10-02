from Preprocessing import Preprocessing
import numpy as np

def test_preprocessing():
    """Test if the shapes of the original MNIST and the preprocessed datasets are correct
    """
    pp = Preprocessing.Preprocessing(False)
    x_train, y_train, x_test, y_test = pp.getMNISTTrainData(), pp.getMNISTTrainLabel(), pp.getMNISTTestData(), pp.getMNISTTestLabel()
    np.testing.assert_equal(x_train.shape, (100, 28, 28))
    np.testing.assert_equal(y_train.shape, (100,))
    np.testing.assert_equal(x_test.shape, (10, 28, 28))
    np.testing.assert_equal(y_test.shape, (10,))
    x_train, y_train, x_test, y_test = pp.getMNISTPreprocessedTrainData(), pp.getMNISTPreprocessedTrainLabel(), pp.getMNISTPreprocessedTestData(), pp.getMNISTPreprocessedTestLabel()
    np.testing.assert_equal(x_train.shape, (100, 28, 28, 1))
    np.testing.assert_equal(y_train.shape, (100, 10))
    np.testing.assert_equal(x_test.shape, (10, 28, 28, 1))
    np.testing.assert_equal(y_test.shape, (10, 10))
