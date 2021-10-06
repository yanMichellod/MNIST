import os
    
def test_main():
    """Run the cli command to test the whole analysis in test mode (full=False)
    """
    assert os.system("pytest --cov main/mnist.py --full=False")
