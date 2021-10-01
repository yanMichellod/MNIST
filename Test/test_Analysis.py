import numpy as np
from Analysis import Analysis 

def testAnalysis():
    '''
    Run several tests to check if Analysis works correct
    '''
    an = Analysis.Analysis(full=False)
    assert an.checkVSBaseline()
    assert an.checkHypothesis(0.7)
    assert an.saveConfusionMatrix()
