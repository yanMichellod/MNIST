import sys
sys.path.insert(0, '../Database')
import MNISTDatabase as db

def runProcessor():
    x_train, y_train, x_test, y_test = db.loadMNISTDatabase()
    return len(x_train)