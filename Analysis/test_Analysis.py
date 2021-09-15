import unittest
import numpy as np
import Analysis 
import math


class Test(unittest.TestCase):

    an = Analysis.Analysis(full=False)
        
    def test_checkVSBaseline(self):
       assert self.an.checkVSBaseline()

    def test_checkHypothesis(self):
       assert self.an.checkHypothesis(0.7)

    def test_saveConfusionMatrix(self):
       assert self.an.saveConfusionMatrix()


if __name__ == "__main__":

    unittest.main()