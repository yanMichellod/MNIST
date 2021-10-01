import numpy as np
from Analysis import Analysis 

def testAnalysis():
    '''
    Run several tests to check if Analysis works correct
    '''
    an = Analysis.Analysis(full=False)
    assert self.an.checkVSBaseline()
    assert self.an.checkHypothesis(0.7)
    assert self.an.saveConfusionMatrix()
