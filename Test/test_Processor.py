from Processor import Processor
import numpy as np

def test_processor():
    """Test if the processor runs and returns expected values
    """
    acc_baseline, acc_CNN, y_test, y_pred = Processor.runProcessor(full=False)
    assert acc_baseline > 0
    assert acc_CNN > 0
    np.testing.assert_equal(y_test.shape, (10, 10))
    np.testing.assert_equal(y_pred.shape, (10, 10))

