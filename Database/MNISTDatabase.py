from keras.datasets import mnist
from numpy import save, load
import os.path

def loadFromFile(filepath):
    if os.path.exists(filepath):
        ds = load(filepath)
    else:
        raise FileNotFoundError("file {0} not found".format(filepath))
    return ds

def loadMNISTDatabase(method="keras", saveLocal=True):
    """Load the MNIST train and test datasets.

    Parameters
    ----------
    method :    string which contains 'keras' or 'local'
                keras: The datasets getting loaded from the keras library
                local: The datasets getting loaded from the local copy
    saveLocal:  boolean
                For archieving the datasets, they getting stored localy which allows to load them with method "local"

    Returns
    -------
    x_train : uint8 NumPy array containing training data with shape (60000, 28, 28)
    y_train : uint8 NumPy array containing training digit labels with shape (60000,) 
    x_test  : uint8 NumPy array containing test data with shape (10000, 28, 28)
    y_test  : uint8 NumPy array containing test digit labels with shape (10000,) 
    """
    x_train, y_train, x_test, y_test = [], [], [], []
    
    if method=="keras":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        save("Database/x_train.npy", x_train)
        save("Database/y_train.npy", y_train)
        save("Database/x_test.npy", x_test)
        save("Database/y_test.npy", y_test)
    elif method=="local":
        x_train = loadFromFile("Database/x_train.npy")
        y_train = loadFromFile("Database/y_train.npy")
        x_test  = loadFromFile("Database/x_test.npy")
        y_test  = loadFromFile("Database/y_test.npy")
    else:
        raise ValueError("method has to be string which contains 'keras' or 'local'")

    return x_train, y_train, x_test, y_test