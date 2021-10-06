import os

def test_main():
    assert os.system("python main/mnist.py --full=False")
