# Inspired by https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/
# and https://www.kaggle.com/ashwani07/mnist-classification-using-random-forest

import sys
sys.path.insert(0, '../Database')
import MNISTDatabase as db

from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD


# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(learning_rate=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model
    
def get_baseline(x_train, y_train, x_test, y_test):
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(x_train.reshape(x_train.shape[0], 28*28),y_train)
    pred = rf.predict(x_test.reshape(x_test.shape[0], 28*28))
    return accuracy_score(y_test, pred)
    
def runProcessor():
    # load the datasets
    x_train, y_train, x_test, y_test = db.loadMNISTDatabase()
    # Get RF baseline
    acc_baseline = get_baseline(x_train, y_train, x_test, y_test)
    print('Accuracy RF: %.3f' % (acc_baseline * 100.0))
    # preprocess the data
    x_train, y_train, x_test, y_test = preprocess_data(x_train, y_train, x_test, y_test)
    # train the model
    model = define_model()
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)
	# evaluate model on test dataset
    _, acc_CNN = model.evaluate(x_test, y_test, verbose=0)
    print('Accuracy CNN: %.3f' % (acc_CNN * 100.0))
    return len(x_train)
    
    