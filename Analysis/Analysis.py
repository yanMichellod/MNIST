from Processor import Processor as pro
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from matplotlib.pyplot import figure

figure(figsize=(11, 11), dpi=80)

class Analysis:
    """Class Analysis for analysing the performance of the model 
    """

    def __init__(self, full=True):
        """Constructor which run the Processor and store the results
        
        self.acc_baseline: float
            The accuracy of the random forest
        self.acc_CNN: float
            The accuracy of the CNN model
        self.y_test: numpy.ndarray
            The ground truth labels
        self.y_pred: numpy.ndarray
            The predicted labels of the CNN model
            
        Parameters
        ==========
        full : boolean
            Default value is True
            Determines if all the MNIST records should be considered or only 
            a subset shall be used for testing.
        """
        print('*************************************************')
        print('Analyse results')
        print('')
        self.acc_baseline, self.acc_CNN, self.y_test, self.y_pred = pro.runProcessor(full=full)
        print('*************************************************')

    def checkVSBaseline(self):
        """Check if the CNN accuracy is higher then the accuracy of the RF baseline
        
        Returns
        =======
        Boolean: True if CNN accuracy is higher then baseline, false otherwise
        """
        return self.acc_CNN > self.acc_baseline

    def checkHypothesis(self, hypothesis):
        """Check if the CNN accuracy is higher then the accuracy expected by hypothesis

        Parameters
        ==========
        hypothesis: float
            The accuracy expected by the hypothesis
        Returns
        =======
        Boolean: True if CNN accuracy is higher then expected by hypothesis, false otherwise
        """
        return self.acc_CNN > hypothesis

    def saveConfusionMatrix(self):
        """Create a confusion matrix for CNN showing the confusion for the different digits
        The matrix is getting saved to file cfm.png.
        
        Returns
        =======
        Boolean: Always True
        """
        cfm = confusion_matrix(np.argmax(self.y_test, axis=1), np.argmax(self.y_pred, axis=1))
        sns_plot = sns.heatmap(cfm/np.sum(cfm), annot=True, fmt='.2%', cmap='Blues')
        sns_plot.figure.savefig("cfm.png")
        return True

 
    
