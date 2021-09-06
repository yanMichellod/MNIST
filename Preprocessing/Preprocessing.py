from Database import MNISTDatabase as db


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
        print('*************************************************')

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