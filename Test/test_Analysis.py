import numpy as np
from Analysis import Analysis 

def testAnalysis():
    """Run several tests to check if Analysis works correct
    
    Test if accuracy of CNN is higher then accuracy of baseline
    Test if accuracy of CNN is higher or equal to the expected accuracy of hypothesis
    Test if generation and saving of confusion matrix works
    """
    an = Analysis.Analysis(full=False)
    assert an.checkVSBaseline()
    assert an.checkHypothesis(0.7)
    assert an.saveConfusionMatrix()
