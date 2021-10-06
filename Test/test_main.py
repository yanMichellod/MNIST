from argparse import Namespace
from main import mnist
    
def test_main():
    """Run the cli main function to test the whole analysis in test mode (full=False)
    """
    assert mnist.main(['--full=False'])
