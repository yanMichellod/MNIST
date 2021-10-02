from Database import MNISTDatabase as db
from tensorflow.keras.utils import to_categorical


class Preprocessing:
    """Class Preprocessing used to pre-process data before using machine learning algorithm
    """

    def __init__(self, full=True):
        """Constructor which load the data from Database and store the original datasets and also the preprocessed datasets
        
        self.x_train: numpy.ndarray
            The training dataset
        self.y_train: numpy.ndarray
            The training labels
        self.x_test: numpy.ndarray
            The test dataset
        self.y_test: numpy.ndarray
            The test labels (ground truth)
        self.x_train_preprocess: numpy.ndarray
            The preprocessed training dataset
        self.y_train_preprocess: numpy.ndarray
            The preprocessed training labels
        self.x_test_preprocess: numpy.ndarray
            The preprocessed test dataset
        self.y_test_preprocess: numpy.ndarray
            The preprocessed test labels (ground truth)
            
        Parameters
        ==========
        full : boolean
            Default value is True
            Determines if all the MNIST records should be considered or only 
            a subset shall be used for testing.
        """
        print('*************************************************')
        print('Preprocess Data')
        print('')
        self.x_train, self.y_train, self.x_test, self.y_test = db.loadMNISTDatabase()
        if not full:
            print('TESTMODE --> Only subset of data used!')
            self.x_train = self.x_train[0:100]
            self.y_train = self.y_train[0:100]
            self.x_test = self.x_test[0:10]
            self.y_test = self.y_test[0:10]
        self.x_train_preprocess, self.y_train_preprocess, self.x_test_preprocess, self.y_test_preprocess =\
            self.preprocess_data(self.x_train, self.y_train, self.x_test, self.y_test)
        print('*************************************************')

    def getMNISTTrainData(self):
        """Get the training dataset

        Returns
        =======
        self.x_train : uint8 numpy array 
            training data with shape (60000, 28, 28)
        """
        return self.x_train

    def getMNISTTrainLabel(self):
        """Get the training labels

        Returns
        =======
        self.y_train : uint8 numpy array 
            training digit labels with shape (60000,)
        """
        return self.y_train

    def getMNISTTestData(self):
        """Get the test dataset

        Returns
        =======
        self.x_test : uint8 numpy array 
            testing data with shape (10000, 28, 28)
        """
        return self.x_test

    def getMNISTTestLabel(self):
        """Get the test labels

        Returns
        =======
        self.y_test : uint8 numpy array
            training digit labels with shape (10000,)
        """
        return self.y_test

    def getMNISTPreprocessedTrainData(self):
        """Get preprocessed training dataset

        Returns
        =======
        self.x_train_preprocess : uint8 numpy array 
            training data with shape (60000, 28, 28, 1)
        """
        return self.x_train_preprocess

    def getMNISTPreprocessedTrainLabel(self):
        """Get preprocessed training labels

        Returns
        -------
        self.y_train_preprocess : uint8 numpy array 
            training digit labels with shape (60000, 10)
        """
        return self.y_train_preprocess

    def getMNISTPreprocessedTestData(self):
        """Get preprocessed test dataset

        Returns
        -------
        self.x_test_preprocess : uint8 numpy array
            testing data with shape (10000, 28, 28, 1)
        """
        return self.x_test_preprocess

    def getMNISTPreprocessedTestLabel(self):
        """Get preprocessed test labels

        Returns
        -------
        self.y_test_preprocess : uint8 numpy array
            training digit labels with shape (10000, 10)
        """
        return self.y_test_preprocess

    def preprocess_data(self, x_train, y_train, x_test, y_test):
        """Method which preprocess the data to be used by CNN model
        
        Parameters
        ==========
        x_train : uint numpy array 
            training digit data with shape (60000, 28, 28)
        y_train : uint numpy array 
            training digit labels with shape (60000,)
        x_test : uint numpy array 
            testing digit labels with shape (10000, 28, 28)
        y_test : uint numpy array 
            testing digit labels with shape (10000,)

        Returns
        =======
        x_train : uint numpy preprocess array 
            training digit data with shape (60000, 28, 28, 1)
        y_train : uint numpy preprocess array 
            training digit labels with shape (60000, 10)
        x_test : uint numpy preprocess array 
            testing digit labels with shape (10000, 28, 28, 1)
        y_test : uint numpy preprocess array 
            testing digit labels with shape (10000, 10)
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

    
