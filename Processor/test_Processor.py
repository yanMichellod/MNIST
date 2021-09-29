import unittest
import numpy as np
import Processor as pro
import math


class Test(unittest.TestCase):
    def test_processor(self):
        acc_baseline, acc_CNN, y_test, y_pred = pro.runProcessor(full=False)
        assert acc_CNN > 0


if __name__ == "__main__":

    unittest.main()
