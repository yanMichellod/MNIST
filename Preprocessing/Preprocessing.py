import sys
sys.path.insert(0, '../Database')
import MNISTDatabase as db
from tensorflow.keras.utils import to_categorical


class Preprocessing:
    """
    Class Preprocessing used to pre-process data before using machine learning algorithm
    """

    def __init__(self):
        """
        Constructor
        Get data from the database
        """
        print('*************************************************')
        print('Preprocess Data')
        print('')
        self.x_train, self.y_train, self.x_test, self.y_test = db.loadMNISTDatabase()
        self.x_train_preprocess, self.y_train_preprocess, self.x_test_preprocess, self.y_test_preprocess =\
            self.preprocess_data(self.x_train, self.y_train, self.x_test, self.y_test)
        print('*************************************************')

    """
    Raw Data
    """
    def getMNISTTrainData(self):
        """

        Returns
        -------
        self.x_train : uint8 numpy array training data with shape (60000, 28, 28)
        """
        return self.x_train

    def getMNISTTrainLabel(self):
        """

        Returns
        -------
        self.y_train : uint8 numpy array training digit labels with shape (60000,)
        """
        return self.y_train

    def getMNISTTestData(self):
        """

        Returns
        -------
        self.x_test : uint8 numpy array testing data with shape (10000, 28, 28)
        """
        return self.x_test

    def getMNISTTestLabel(self):
        """

        Returns
        -------
        self.y_test : uint8 numpy array training digit labels with shape (10000,)
        """
        return self.y_test

    """
    Preprocess Data
    """
    def getMNISTPreprocessedTrainData(self):
        """

        Returns
        -------
        self.x_train : uint8 numpy array training data with shape (60000, 28, 28)
        """
        return self.x_train_preprocess

    def getMNISTPreprocessedTrainLabel(self):
        """

        Returns
        -------
        self.y_train : uint8 numpy array training digit labels with shape (60000,)
        """
        return self.y_train_preprocess

    def getMNISTPreprocessedTestData(self):
        """

        Returns
        -------
        self.x_test : uint8 numpy array testing data with shape (10000, 28, 28)
        """
        return self.x_test_preprocess

    def getMNISTPreprocessedTestLabel(self):
        """

        Returns
        -------
        self.y_test : uint8 numpy array training digit labels with shape (10000,)
        """
        return self.y_test_preprocess

    def preprocess_data(self, x_train, y_train, x_test, y_test):
        """
        Method which scales pixels

        Parameters
        x_train : uint numpy array training digit data with shape (60000, 28, 28)
        y_train : uint numpy array training digit labels with shape (60000,)
        x_test : uint numpy array testing digit labels with shape (10000, 28, 28)
        y_test : uint numpy array testing digit labels with shape (10000,)

        Returns
        x_train : uint numpy preprocess array training digit data with shape (60000, 28, 28)
        y_train : uint numpy preprocess array training digit labels with shape (60000,)
        x_test : uint numpy preprocess array testing digit labels with shape (10000, 28, 28)
        y_test : uint numpy preprocess array testing digit labels with shape (10000,)
        """
        # reshape dataset to have a single channel
        x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
        x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
        # one hot encode target values
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        # convert from integers to floats
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        # normalize to range 0-1
        x_train = x_train / 255.0
        x_test = x_test / 255.0
        # return normalized images
        return x_train, y_train, x_test, y_test

    