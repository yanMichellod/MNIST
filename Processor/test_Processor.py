import unittest
import numpy as np
import Processor as pro
import math


class Test(unittest.TestCase):

    def test_processor(self):
        out = pro.runProcessor()
        np.testing.assert_equal(out, 60000)


if __name__ == "__main__":

    unittest.main()