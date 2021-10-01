import sys
sys.path.insert(0, '../Processor')
import Processor as pro
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from matplotlib.pyplot import figure

figure(figsize=(11, 11), dpi=80)

class Analysis:
    """
    Class Analyses the model results
    """

    def __init__(self, full=True):
        """
        Constructor
        run the processor
        """
        print('*************************************************')
        print('Analyse results')
        print('')
        self.acc_baseline, self.acc_CNN, self.y_test, self.y_pred = pro.runProcessor(full=full)
        print('*************************************************')

    """
    Do the analysis
    """
    def checkVSBaseline(self):
        """
        Returns
        -------
        Boolean: True if CNN accuracy is higher then baseline, false otherwise
        """
        return self.acc_CNN > self.acc_baseline

    def checkHypothesis(self, hypothesis):
        """
        Returns
        -------
        Boolean: True if CNN accuracy is higher then expected by hypothesis, false otherwise
        """
        return self.acc_CNN > hypothesis

    def saveConfusionMatrix(self):
        """
        Create a confusion matrix for CNN showing the confusion for the different digits
        The matrix is getting saved to file cfm.png.
        """
        cfm = confusion_matrix(np.argmax(self.y_test, axis=1), np.argmax(self.y_pred, axis=1))
        sns_plot = sns.heatmap(cfm/np.sum(cfm), annot=True, fmt='.2%', cmap='Blues')
        sns_plot.figure.savefig("cfm.png")
        return True

 
    
